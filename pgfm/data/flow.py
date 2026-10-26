import os
from collections import OrderedDict
from bisect import bisect_right
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


from .daily2d import Daily2D, daily_days, daily_path

PHYS_PREFIX = "PHYS_"

DEFAULT_PHYS_TERM_KEYS = [
    "op_prod_tecu_rate",
    "op_loss_tecu_rate",
    "op_drift_tecu_rate",
    "op_wind_tecu_rate",
]
DEFAULT_PHYS_UV_KEYS = [
    "ui_exb_layer_mean",
    "vi_exb_layer_mean",
]


def list_phys_cached_days(cache_dir: str) -> List[str]:
    cache_dir = os.path.expanduser(cache_dir)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")

    new_days = daily_days(cache_dir)
    if new_days:
        return new_days
    days: List[str] = []
    for fn in os.listdir(cache_dir):
        if fn.startswith(PHYS_PREFIX) and fn.endswith(".npz"):
            days.append(fn[len(PHYS_PREFIX):-4])
    days = sorted(set(days))
    if not days:
        raise RuntimeError(f"No cached PHYS files found in {cache_dir}")
    return days


def _load_phys_npz(cache_dir: str, day: str):
    if daily_path(os.path.expanduser(cache_dir), day).exists():
        return Daily2D(os.path.expanduser(cache_dir), day)
    path = os.path.join(os.path.expanduser(cache_dir), f"{PHYS_PREFIX}{day}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def estimate_key_mean_std(cache_dir: str, days: Sequence[str], key: str) -> Tuple[float, float]:
    count, mean, m2 = 0, 0., 0.
    for day in days:
        with _load_phys_npz(cache_dir, day) as z:
            if key not in z:
                raise KeyError(f"Missing normalization field {key} on {day}")
            arr = np.asarray(z[key], dtype=np.float64).reshape(-1)
            arr = arr[np.isfinite(arr)]
            n = arr.size
            if not n: continue
            mu = float(arr.mean()); delta = mu - mean
            m2 += float(np.square(arr - mu).sum()) + delta**2 * count*n/(count+n)
            mean += delta*n/(count+n); count += n
    if not count: raise ValueError(f"No training values for {key}")
    std = float(np.sqrt(m2/count))
    return mean, std if std >= 1e-8 else 1.


def estimate_many_mean_std(cache_dir: str, days: Sequence[str], keys: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    return {key: estimate_key_mean_std(cache_dir, days, key) for key in keys}


def _finite_zero(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _preview_keys(keys: Sequence[str], max_items: int = 32) -> str:
    keys = sorted(str(k) for k in keys)
    shown = keys[:max_items]
    suffix = "" if len(keys) <= max_items else f", ... (+{len(keys) - max_items} more)"
    return ", ".join(shown) + suffix


class HourlyPiecewiseFlowPhysicsDataset(Dataset):
    """
    One-hour piecewise flow-matching dataset with 15-minute internal knots.

    Final target remains hourly TEC. The 15-minute frames are only used to build:
      - the piecewise path knots for the next hour
      - quarter-hour alignment of the selected physics source
    """

    def __init__(
        self,
        cache_dir: str,
        days: Sequence[str],
        past_hours: int,
        rollout_hours: int,
        tec_mean: float,
        tec_std: float,
        bg_scalar_keys: Sequence[str],
        bg_map_keys: Sequence[str],
        bg_stats: Dict[str, Tuple[float, float]],
        uv_stats: Dict[str, Tuple[float, float]],
        phys_term_keys: Optional[Sequence[str]] = None,
        phys_uv_keys: Optional[Sequence[str]] = None,
        strict_required_physics: bool = True,
        step_per_hour: int = 4,
        max_cached_days: int = 3,
        include_rollout: bool = True,
        future_physics_mode: str = "truth",
        future_physics_blend_last_hour_weight: float = 0.5,
        future_physics_convgru_path: Optional[str] = None,
    ):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.origin_days = getattr(days, "origin_days", None)
        self.days = sorted(list(days))
        self.past_hours = int(past_hours)
        self.rollout_hours = int(rollout_hours)
        self.step_per_hour = int(step_per_hour)
        self.tec_mean = float(tec_mean)
        self.tec_std = float(tec_std) if tec_std > 1e-8 else 1.0
        self.bg_scalar_keys = list(bg_scalar_keys)
        self.bg_map_keys = list(bg_map_keys)
        self.bg_stats = dict(bg_stats)
        self.uv_stats = dict(uv_stats)
        self.strict_required_physics = bool(strict_required_physics)
        self.max_cached_days = max(int(max_cached_days), 1)
        self.include_rollout = bool(include_rollout)
        self.future_physics_mode = self._normalize_future_physics_mode(future_physics_mode)
        self.future_physics_blend_last_hour_weight = float(np.clip(future_physics_blend_last_hour_weight, 0.0, 1.0))
        self.future_physics_convgru_path = os.path.expanduser(str(future_physics_convgru_path)) if future_physics_convgru_path else None
        self._convgru_forecast = None
        self._convgru_anchor_to_idx: Dict[int, int] = {}
        self._convgru_npz = None

        self.phys_term_keys = list(phys_term_keys or DEFAULT_PHYS_TERM_KEYS)
        self.phys_uv_keys = list(phys_uv_keys or DEFAULT_PHYS_UV_KEYS)

        self.lat = None
        self.lon = None
        self.day_infos: List[Dict[str, object]] = []
        time_list: List[np.ndarray] = []
        utc_hour_list: List[np.ndarray] = []
        offset = 0

        for day in self.days:
            z = _load_phys_npz(self.cache_dir, day)
            tec_shape = tuple(z.shape("tec") if hasattr(z, "shape") else np.asarray(z["tec"]).shape)
            t_len, height, width = tec_shape
            if self.lat is None:
                self.lat = np.asarray(z["lat"], dtype=np.float32)
            if self.lon is None:
                self.lon = np.asarray(z["lon"], dtype=np.float32)
            time_arr = np.asarray(z["time"])
            utc_hour_arr = np.asarray(z["utc_hour"], dtype=np.float32)
            time_list.append(time_arr)
            utc_hour_list.append(utc_hour_arr)
            self.day_infos.append(
                {
                    "day": day,
                    "path": os.path.join(self.cache_dir, f"{PHYS_PREFIX}{day}.npz"),
                    "start": offset,
                    "end": offset + t_len,
                    "length": t_len,
                    "shape": (height, width),
                }
            )
            offset += t_len
            z.close()

        self.time = np.concatenate(time_list, axis=0)
        self.utc_hour = np.concatenate(utc_hour_list, axis=0).astype(np.float32)
        self.T_total = int(offset)
        self.H = int(self.lat.shape[0])
        self.W = int(self.lon.shape[0])
        self.day_starts = [int(info["start"]) for info in self.day_infos]
        self._cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()
        self._validate_required_physics_keys()
        self.anchor_indices = self._build_anchor_indices()
        if self.origin_days is not None:
            allowed = set(self.origin_days)
            self.anchor_indices = [a for a in self.anchor_indices
                if str(np.datetime64(int(self.time[a]), "s").astype("datetime64[D]")) in allowed]
        if self.future_physics_mode == "convgru":
            self._load_convgru_forecast()
            self._filter_anchor_indices_to_convgru_forecast()

    @staticmethod
    def _normalize_future_physics_mode(mode: str) -> str:
        mode = str(mode or "truth").strip().lower().replace("-", "_")
        aliases = {
            "oracle": "truth",
            "future_truth": "truth",
            "persistence": "last_hour",
            "last": "last_hour",
            "last_hour_persistence": "last_hour",
            "diurnal": "diurnal_24h",
            "diurnal_persistence": "diurnal_24h",
            "diurnal_persistence_24h": "diurnal_24h",
            "causal_blend": "blend",
            "history_blend": "blend",
            "conv_gru": "convgru",
            "convgru_forecast": "convgru",
        }
        mode = aliases.get(mode, mode)
        allowed = {"truth", "last_hour", "diurnal_24h", "blend", "convgru"}
        if mode not in allowed:
            raise ValueError(f"Unknown future_physics_mode={mode!r}; expected one of {sorted(allowed)}")
        return mode

    def _anchor_time_key(self, anchor: int) -> int:
        t = np.asarray(self.time[int(anchor)])
        if np.issubdtype(t.dtype, np.datetime64):
            return int(t.astype("datetime64[s]").astype(np.int64))
        return int(t.astype(np.int64))

    def _load_convgru_forecast(self) -> None:
        if not self.future_physics_convgru_path:
            raise ValueError("future_physics_mode='convgru' requires future_physics_convgru_path")
        if not os.path.exists(self.future_physics_convgru_path):
            raise FileNotFoundError(self.future_physics_convgru_path)

        path = self.future_physics_convgru_path
        if path.endswith(".npy"):
            rhs_path = path
            anchor_path = path[:-4] + "_anchor_time.npy"
            if not os.path.exists(anchor_path):
                raise FileNotFoundError(f"Missing ConvGRU anchor sidecar for {rhs_path}: {anchor_path}")
            anchor_time = np.asarray(np.load(anchor_path, mmap_mode="r"), dtype=np.int64).reshape(-1)
            future_rhs_rollout = np.load(rhs_path, mmap_mode="r")
        else:
            z = np.load(path, mmap_mode="r")
            required = {"anchor_time"}
            missing = sorted(required.difference(z.files))
            if missing:
                raise KeyError(f"ConvGRU forecast file missing keys {missing}: {path}")
            anchor_time = np.asarray(z["anchor_time"], dtype=np.int64).reshape(-1)
            sidecar = path[:-4] + "_future_rhs_rollout.npy" if path.endswith(".npz") else path + "_future_rhs_rollout.npy"
            if os.path.exists(sidecar):
                future_rhs_rollout = np.load(sidecar, mmap_mode="r")
            else:
                raise FileNotFoundError(
                    "ConvGRU forecast sidecar is required for memory-safe DDP training. "
                    f"Expected {sidecar}. Create it from {path} or rerun pgfm.cli.predict_physics "
                    "with the updated code."
                )

        self._convgru_forecast = {
            "anchor_time": anchor_time,
            "future_rhs_rollout": future_rhs_rollout,
        }
        self._convgru_anchor_to_idx = {int(t): i for i, t in enumerate(anchor_time.tolist())}

    def _filter_anchor_indices_to_convgru_forecast(self) -> None:
        if self._convgru_forecast is None:
            return
        rhs = self._convgru_forecast["future_rhs_rollout"]
        required_hours = int(self.rollout_hours if self.include_rollout else 1)
        if int(rhs.shape[1]) < required_hours:
            raise ValueError(
                f"ConvGRU forecast horizon={rhs.shape[1]} is shorter than required hours={required_hours}. "
                f"Regenerate {self.future_physics_convgru_path} with a longer physics_forecaster.horizon_hours."
            )

        before = len(self.anchor_indices)
        self.anchor_indices = [
            int(anchor)
            for anchor in self.anchor_indices
            if self._anchor_time_key(int(anchor)) in self._convgru_anchor_to_idx
        ]
        if self.origin_days is not None and len(self.anchor_indices) != before:
            raise ValueError("New-data ConvGRU forecasts must cover every eligible origin; regenerate forecasts")
        if before > 0 and not self.anchor_indices:
            raise ValueError(
                "No dataset anchors are covered by the ConvGRU forecast file. "
                f"Check split/days and regenerate forecasts: {self.future_physics_convgru_path}"
            )

    def _validate_required_physics_keys(self) -> None:
        if not self.strict_required_physics:
            return

        required = list(self.phys_term_keys) + list(self.phys_uv_keys)
        if not required:
            return

        missing_by_day: Dict[str, List[str]] = {}
        first_available: Optional[Sequence[str]] = None
        for info in self.day_infos:
            day = str(info["day"])
            z = _load_phys_npz(self.cache_dir, day)
            try:
                available = set(z.files)
                if first_available is None:
                    first_available = list(z.files)
                missing = [key for key in required if key not in available]
            finally:
                z.close()
            if missing:
                missing_by_day[day] = missing

        if not missing_by_day:
            return

        first_day = next(iter(missing_by_day))
        missing_preview = ", ".join(missing_by_day[first_day])
        raise KeyError(
            "Missing required physics cache fields for piecewise_condphys_fm. "
            f"First failing day={first_day}, missing=[{missing_preview}]. "
            f"Available keys include: {_preview_keys(first_available or [])}. "
            "These keys feed u_phys; silently replacing them with zeros makes "
            "u_phys identically zero. Rebuild the cache with integral_vars "
            "OP_PROD/OP_LOSS/OP_DRIFT/OP_WIND and layer_vars UI_ExB/VI_ExB, "
            "or set data.strict_required_physics=false only for a deliberate ablation."
        )

    def _build_anchor_indices(self) -> List[int]:
        anchors: List[int] = []
        hist_span = max((self.past_hours - 1) * self.step_per_hour, 24 * self.step_per_hour - 1)
        future_span = self.rollout_hours * self.step_per_hour
        total_required = hist_span + future_span
        if self.T_total <= total_required:
            return anchors

        time_sec = np.asarray(self.time).reshape(-1)
        if np.issubdtype(time_sec.dtype, np.datetime64):
            time_sec = time_sec.astype("datetime64[s]").astype(np.int64)
        else:
            time_sec = time_sec.astype(np.int64)

        for idx in range(hist_span, self.T_total - future_span):
            if not np.isclose(self.utc_hour[idx] % 1.0, 0.0, atol=1e-4):
                continue
            start = idx - hist_span
            stop = idx + future_span
            seg = time_sec[start:stop + 1]
            if seg.size < 2:
                continue
            diffs = np.diff(seg)
            if not np.all(diffs == diffs[0]):
                continue
            if int(diffs[0]) != 900:
                continue
            anchors.append(idx)
        return anchors

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _norm(self, arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        out = (np.asarray(arr, dtype=np.float32) - float(mean)) / (float(std) + 1e-6)
        return _finite_zero(out)

    def _day_local_from_global(self, global_idx: int) -> Tuple[int, int]:
        day_idx = bisect_right(self.day_starts, int(global_idx)) - 1
        day_idx = max(0, min(day_idx, len(self.day_infos) - 1))
        local_idx = int(global_idx) - int(self.day_infos[day_idx]["start"])
        return day_idx, local_idx

    def _load_day_arrays(self, day_idx: int) -> Dict[str, np.ndarray]:
        if day_idx in self._cache:
            self._cache.move_to_end(day_idx)
            return self._cache[day_idx]

        info = self.day_infos[day_idx]
        z = _load_phys_npz(self.cache_dir, str(info["day"]))
        tec = _finite_zero(z["tec"])
        t_len = int(tec.shape[0])
        day_data: Dict[str, np.ndarray] = {"tec": tec}

        for key in self.bg_scalar_keys:
            if key in z:
                day_data[key] = _finite_zero(z[key])
            else:
                day_data[key] = np.zeros((t_len,), dtype=np.float32)

        for key in self.bg_map_keys:
            if key in z:
                day_data[key] = _finite_zero(z[key])
            else:
                day_data[key] = np.zeros((t_len, self.H, self.W), dtype=np.float32)

        for key in self.phys_term_keys:
            if key in z:
                day_data[key] = _finite_zero(z[key])
            elif self.strict_required_physics:
                raise KeyError(f"Missing required physics term {key!r} in {info['path']}")
            else:
                day_data[key] = np.zeros((t_len, self.H, self.W), dtype=np.float32)

        for key in self.phys_uv_keys:
            if key in z:
                day_data[key] = _finite_zero(z[key])
            elif self.strict_required_physics:
                raise KeyError(f"Missing required physics UV field {key!r} in {info['path']}")
            else:
                day_data[key] = np.zeros((t_len, self.H, self.W), dtype=np.float32)

        self._cache[day_idx] = day_data
        self._cache.move_to_end(day_idx)
        while len(self._cache) > self.max_cached_days:
            self._cache.popitem(last=False)
        return day_data

    def _slice_global_key(self, key: str, start: int, stop: int) -> np.ndarray:
        parts: List[np.ndarray] = []
        cursor = int(start)
        stop = int(stop)
        while cursor < stop:
            day_idx, local_idx = self._day_local_from_global(cursor)
            info = self.day_infos[day_idx]
            local_stop = min(stop, int(info["end"])) - int(info["start"])
            day_data = self._load_day_arrays(day_idx)
            parts.append(day_data[key][local_idx:local_stop])
            cursor = int(info["start"]) + int(local_stop)
        return _finite_zero(np.concatenate(parts, axis=0))

    def _slice_global_key_padded(self, key: str, start: int, stop: int) -> np.ndarray:
        start = int(start)
        stop = int(stop)
        if stop <= start:
            raise ValueError(f"Invalid slice for {key!r}: start={start}, stop={stop}")

        parts: List[np.ndarray] = []
        if start < 0:
            first = self._slice_global_key(key, 0, 1)
            pad_stop = min(stop, 0)
            if pad_stop > start:
                parts.append(np.repeat(first, repeats=pad_stop - start, axis=0))

        valid_start = max(start, 0)
        valid_stop = min(stop, self.T_total)
        if valid_start < valid_stop:
            parts.append(self._slice_global_key(key, valid_start, valid_stop))

        if stop > self.T_total:
            last = self._slice_global_key(key, self.T_total - 1, self.T_total)
            parts.append(np.repeat(last, repeats=stop - self.T_total, axis=0))

        if not parts:
            idx = max(0, min(self.T_total - 1, start))
            base = self._slice_global_key(key, idx, idx + 1)
            parts.append(np.repeat(base, repeats=stop - start, axis=0))

        out = _finite_zero(np.concatenate(parts, axis=0))
        expected = stop - start
        if out.shape[0] != expected:
            raise RuntimeError(f"Padded slice for {key!r} returned length={out.shape[0]}, expected={expected}")
        return out

    def _last_observed_quarter_window(self, anchor: int) -> Tuple[int, int]:
        return int(anchor) - self.step_per_hour + 1, int(anchor) + 1

    def _diurnal_quarter_window(self, anchor: int, hour: int) -> Tuple[int, int]:
        day_steps = 24 * self.step_per_hour
        q0 = int(anchor) + int(hour) * self.step_per_hour + 1 - day_steps
        return q0, q0 + self.step_per_hour

    def _future_truth_quarter_window(self, anchor: int, hour: int) -> Tuple[int, int]:
        q0 = int(anchor) + int(hour) * self.step_per_hour + 1
        return q0, q0 + self.step_per_hour

    def _is_causal_window(self, start: int, stop: int, anchor: int) -> bool:
        return int(start) >= 0 and int(stop) <= int(anchor) + 1

    def _physics_source_specs(self, anchor: int, hour: int) -> List[Tuple[int, int, float]]:
        if self.future_physics_mode == "truth":
            q0, q1 = self._future_truth_quarter_window(anchor, hour)
            return [(q0, q1, 1.0)]

        last_q0, last_q1 = self._last_observed_quarter_window(anchor)
        if self.future_physics_mode == "last_hour":
            return [(last_q0, last_q1, 1.0)]

        day_q0, day_q1 = self._diurnal_quarter_window(anchor, hour)
        day_is_valid = self._is_causal_window(day_q0, day_q1, anchor)
        if self.future_physics_mode == "diurnal_24h":
            return [(day_q0, day_q1, 1.0)] if day_is_valid else [(last_q0, last_q1, 1.0)]

        last_w = self.future_physics_blend_last_hour_weight
        if not day_is_valid:
            return [(last_q0, last_q1, 1.0)]
        return [(last_q0, last_q1, last_w), (day_q0, day_q1, 1.0 - last_w)]

    def _weighted_source_slice(self, key: str, specs: Sequence[Tuple[int, int, float]]) -> np.ndarray:
        out = None
        total_weight = 0.0
        for start, stop, weight in specs:
            weight = float(weight)
            if weight <= 0.0:
                continue
            arr = self._slice_global_key_padded(key, start, stop)
            out = arr * weight if out is None else out + arr * weight
            total_weight += weight
        if out is None or total_weight <= 0.0:
            start, stop, _ = specs[0]
            return self._slice_global_key_padded(key, start, stop)
        return _finite_zero(out / total_weight)

    def build_future_physics_hour(self, anchor: int, hour: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if self.future_physics_mode == "convgru":
            rhs = self.build_future_rhs_rollout(anchor=anchor, rollout_hours=int(hour) + 1)
            zeros_terms = np.zeros((len(self.phys_term_keys), self.step_per_hour, self.H, self.W), dtype=np.float32)
            zeros_uv = np.zeros((len(self.phys_uv_keys), self.step_per_hour, self.H, self.W), dtype=np.float32)
            zeros_terms[0:1] = rhs[int(hour)]
            return zeros_terms, zeros_uv

        specs = self._physics_source_specs(anchor=anchor, hour=hour)
        future_terms = np.stack(
            [self._weighted_source_slice(key, specs) for key in self.phys_term_keys],
            axis=0,
        ).astype(np.float32)

        future_uv = []
        for key in self.phys_uv_keys:
            mean, std = self.uv_stats.get(key, (0.0, 1.0))
            future_uv.append(self._norm(self._weighted_source_slice(key, specs), mean, std))
        future_uv = np.stack(future_uv, axis=0).astype(np.float32)
        return future_terms, future_uv

    def build_future_physics_rollout(self, anchor: int, rollout_hours: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        hours = self.rollout_hours if rollout_hours is None else int(rollout_hours)
        if self.future_physics_mode == "convgru":
            rhs = self.build_future_rhs_rollout(anchor=anchor, rollout_hours=hours)
            terms = np.zeros((hours, len(self.phys_term_keys), self.step_per_hour, self.H, self.W), dtype=np.float32)
            uv = np.zeros((hours, len(self.phys_uv_keys), self.step_per_hour, self.H, self.W), dtype=np.float32)
            terms[:, 0:1] = rhs
            return terms, uv

        future_terms_rollout = []
        future_uv_rollout = []
        for hour in range(hours):
            terms, uv = self.build_future_physics_hour(anchor=anchor, hour=hour)
            future_terms_rollout.append(terms)
            future_uv_rollout.append(uv)
        return (
            np.stack(future_terms_rollout, axis=0).astype(np.float32),
            np.stack(future_uv_rollout, axis=0).astype(np.float32),
        )

    def build_future_rhs_rollout(self, anchor: int, rollout_hours: Optional[int] = None) -> np.ndarray:
        if self.future_physics_mode != "convgru":
            raise ValueError("build_future_rhs_rollout is only available for future_physics_mode='convgru'")
        hours = self.rollout_hours if rollout_hours is None else int(rollout_hours)
        if self._convgru_forecast is None:
            self._load_convgru_forecast()
        key = self._anchor_time_key(anchor)
        if key not in self._convgru_anchor_to_idx:
            raise KeyError(
                f"No ConvGRU forecast for anchor_time={key}. "
                f"Regenerate forecast file for the same split/days: {self.future_physics_convgru_path}"
            )
        idx = self._convgru_anchor_to_idx[key]
        rhs = self._convgru_forecast["future_rhs_rollout"][idx]
        if rhs.shape[0] < hours:
            raise ValueError(f"ConvGRU forecast horizon={rhs.shape[0]} is shorter than requested hours={hours}")
        return _finite_zero(np.asarray(rhs[:hours], dtype=np.float32))

    def _hist_hour_indices(self, anchor: int) -> np.ndarray:
        return np.arange(anchor - (self.past_hours - 1) * self.step_per_hour, anchor + 1, self.step_per_hour, dtype=np.int64)

    def _rollout_hour_indices(self, anchor: int) -> np.ndarray:
        return np.arange(anchor + self.step_per_hour, anchor + self.rollout_hours * self.step_per_hour + 1, self.step_per_hour, dtype=np.int64)

    def __getitem__(self, idx: int):
        anchor = int(self.anchor_indices[idx])
        hist_idx = self._hist_hour_indices(anchor)

        tec_hist = self._norm(self._slice_global_key("tec", int(hist_idx[0]), int(hist_idx[-1]) + 1)[::self.step_per_hour], self.tec_mean, self.tec_std)
        current_tec = tec_hist[-1]

        next_hour_knots = self._norm(self._slice_global_key("tec", anchor, anchor + self.step_per_hour + 1), self.tec_mean, self.tec_std)

        bg_scalar_hist = []
        for key in self.bg_scalar_keys:
            mean, std = self.bg_stats.get(key, (0.0, 1.0))
            bg_scalar_hist.append(
                self._norm(
                    self._slice_global_key(key, int(hist_idx[0]), int(hist_idx[-1]) + 1)[::self.step_per_hour],
                    mean,
                    std,
                )
            )
        bg_scalar_hist = np.stack(bg_scalar_hist, axis=0).astype(np.float32) if bg_scalar_hist else np.zeros((0, self.past_hours), dtype=np.float32)

        bg_map_hist = []
        for key in self.bg_map_keys:
            mean, std = self.bg_stats.get(key, (0.0, 1.0))
            bg_map_hist.append(
                self._norm(
                    self._slice_global_key(key, int(hist_idx[0]), int(hist_idx[-1]) + 1)[::self.step_per_hour],
                    mean,
                    std,
                )
            )
        bg_map_hist = np.stack(bg_map_hist, axis=0).astype(np.float32) if bg_map_hist else np.zeros((0, self.past_hours, self.H, self.W), dtype=np.float32)

        future_terms_hour1, future_uv_hour1 = self.build_future_physics_hour(anchor=anchor, hour=0)
        future_rhs_hour1 = None
        if self.future_physics_mode == "convgru":
            future_rhs_hour1 = self.build_future_rhs_rollout(anchor=anchor, rollout_hours=1)[0]

        sample = {
            "tec_hist": torch.from_numpy(tec_hist),
            "current_tec": torch.from_numpy(current_tec[None, ...]),
            "knots_hour1": torch.from_numpy(next_hour_knots),
            "bg_scalar_hist": torch.from_numpy(bg_scalar_hist),
            "bg_map_hist": torch.from_numpy(bg_map_hist),
            "future_terms_hour1": torch.from_numpy(future_terms_hour1),
            "future_uv_hour1": torch.from_numpy(future_uv_hour1),
            "lat": torch.from_numpy(self.lat.copy()),
            "lon": torch.from_numpy(self.lon.copy()),
        }
        if future_rhs_hour1 is not None:
            sample["future_rhs_hour1"] = torch.from_numpy(future_rhs_hour1)
        if self.include_rollout:
            rollout_hour_idx = self._rollout_hour_indices(anchor)
            target_hourly = self._norm(
                self._slice_global_key("tec", int(rollout_hour_idx[0]), int(rollout_hour_idx[-1]) + 1)[::self.step_per_hour],
                self.tec_mean,
                self.tec_std,
            )
            future_terms_rollout, future_uv_rollout = self.build_future_physics_rollout(anchor=anchor, rollout_hours=self.rollout_hours)
            future_rhs_rollout = None
            if self.future_physics_mode == "convgru":
                future_rhs_rollout = self.build_future_rhs_rollout(anchor=anchor, rollout_hours=self.rollout_hours)

            sample["target_hourly"] = torch.from_numpy(target_hourly)
            sample["future_terms_rollout"] = torch.from_numpy(future_terms_rollout)
            sample["future_uv_rollout"] = torch.from_numpy(future_uv_rollout)
            if future_rhs_rollout is not None:
                sample["future_rhs_rollout"] = torch.from_numpy(future_rhs_rollout)

        return sample
