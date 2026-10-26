import os
from collections import OrderedDict
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .flow import (
    PHYS_PREFIX,
    DEFAULT_PHYS_TERM_KEYS,
    DEFAULT_PHYS_UV_KEYS,
    _finite_zero,
    _load_phys_npz,
)


def _as_stats(stats: Dict[str, Tuple[float, float]], key: str) -> Tuple[float, float]:
    mean, std = stats.get(key, (0.0, 1.0))
    std = float(std)
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    mean = float(mean) if np.isfinite(float(mean)) else 0.0
    return mean, std


class PhysicsConvGRUForecastDataset(Dataset):
    """
    Causal physics forecaster dataset.

    The model predicts RHS residuals relative to a diurnal-24h baseline:
      normalized_future_rhs = normalized_diurnal_rhs + predicted_residual

    Inputs intentionally exclude TEC history. They include historical cached PDE
    terms, historical UV fields, background scalar/map drivers, and UTC phase.
    """

    def __init__(
        self,
        cache_dir: str,
        days: Sequence[str],
        history_hours: int,
        horizon_hours: int,
        tec_mean: float,
        tec_std: float,
        rhs_stats: Tuple[float, float],
        term_stats: Dict[str, Tuple[float, float]],
        uv_stats: Dict[str, Tuple[float, float]],
        bg_stats: Dict[str, Tuple[float, float]],
        bg_scalar_keys: Sequence[str],
        bg_map_keys: Sequence[str],
        phys_term_keys: Optional[Sequence[str]] = None,
        phys_uv_keys: Optional[Sequence[str]] = None,
        step_per_hour: int = 4,
        sequence_stride: int = 4,
        max_cached_days: int = 2,
        anchor_stride_hours: int = 1,
        max_samples: int = 0,
    ):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.origin_days = getattr(days, "origin_days", None)
        self.days = sorted(list(days))
        self.history_hours = int(history_hours)
        self.horizon_hours = int(horizon_hours)
        self.step_per_hour = int(step_per_hour)
        self.sequence_stride = int(sequence_stride)
        if self.sequence_stride <= 0 or self.step_per_hour % self.sequence_stride != 0:
            raise ValueError(f"sequence_stride={self.sequence_stride} must be a positive divisor of step_per_hour={self.step_per_hour}")
        self.forecast_steps_per_hour = self.step_per_hour // self.sequence_stride
        self.history_steps = self.history_hours * self.step_per_hour
        self.horizon_steps = self.horizon_hours * self.step_per_hour
        self.tec_mean = float(tec_mean)
        self.tec_std = float(tec_std) if float(tec_std) > 1e-8 else 1.0
        self.rhs_stats = tuple(rhs_stats)
        self.term_stats = dict(term_stats)
        self.uv_stats = dict(uv_stats)
        self.bg_stats = dict(bg_stats)
        self.bg_scalar_keys = list(bg_scalar_keys)
        self.bg_map_keys = list(bg_map_keys)
        self.phys_term_keys = list(phys_term_keys or DEFAULT_PHYS_TERM_KEYS)
        self.phys_uv_keys = list(phys_uv_keys or DEFAULT_PHYS_UV_KEYS)
        self.physics_input_keys = self.phys_term_keys + self.phys_uv_keys
        self.max_cached_days = max(int(max_cached_days), 1)
        self.anchor_stride_hours = max(int(anchor_stride_hours), 1)
        self.max_samples = max(int(max_samples), 0)

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
            time_list.append(np.asarray(z["time"]))
            utc_hour_list.append(np.asarray(z["utc_hour"], dtype=np.float32))
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
        self.anchor_indices = self._build_anchor_indices()
        if self.origin_days is not None:
            allowed = set(self.origin_days)
            self.anchor_indices = [a for a in self.anchor_indices
                if str(np.datetime64(int(self.time[a]), "s").astype("datetime64[D]")) in allowed]
        if self.anchor_stride_hours > 1:
            self.anchor_indices = self.anchor_indices[:: self.anchor_stride_hours]
        if self.max_samples > 0:
            self.anchor_indices = self.anchor_indices[: self.max_samples]

    def _build_anchor_indices(self) -> List[int]:
        anchors: List[int] = []
        if self.T_total <= self.history_steps + self.horizon_steps:
            return anchors
        time_sec = np.asarray(self.time).reshape(-1)
        if np.issubdtype(time_sec.dtype, np.datetime64):
            time_sec = time_sec.astype("datetime64[s]").astype(np.int64)
        else:
            time_sec = time_sec.astype(np.int64)
        day_steps = 24 * self.step_per_hour
        for idx in range(self.history_steps - 1, self.T_total - self.horizon_steps):
            if not np.isclose(self.utc_hour[idx] % 1.0, 0.0, atol=1e-4):
                continue
            hist_start = idx - self.history_steps + 1
            future_stop = idx + self.horizon_steps + 1
            diurnal_start = idx + 1 - day_steps
            if diurnal_start < 0:
                continue
            seg = time_sec[hist_start:future_stop]
            if seg.size < 2:
                continue
            diffs = np.diff(seg)
            if np.all(diffs == diffs[0]) and int(diffs[0]) == 900:
                anchors.append(idx)
        return anchors

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _day_local_from_global(self, global_idx: int) -> Tuple[int, int]:
        day_idx = bisect_right(self.day_starts, int(global_idx)) - 1
        day_idx = max(0, min(day_idx, len(self.day_infos) - 1))
        return day_idx, int(global_idx) - int(self.day_infos[day_idx]["start"])

    def _load_day_arrays(self, day_idx: int) -> Dict[str, np.ndarray]:
        if day_idx in self._cache:
            self._cache.move_to_end(day_idx)
            return self._cache[day_idx]
        info = self.day_infos[day_idx]
        z = _load_phys_npz(self.cache_dir, str(info["day"]))
        tec = _finite_zero(z["tec"])
        t_len = int(tec.shape[0])
        day_data: Dict[str, np.ndarray] = {"tec": tec}
        for key in self.physics_input_keys + self.bg_scalar_keys + self.bg_map_keys:
            if key in z:
                day_data[key] = _finite_zero(z[key])
            else:
                shape = (t_len,) if key in self.bg_scalar_keys else (t_len, self.H, self.W)
                day_data[key] = np.zeros(shape, dtype=np.float32)
        z.close()
        self._cache[day_idx] = day_data
        self._cache.move_to_end(day_idx)
        while len(self._cache) > self.max_cached_days:
            self._cache.popitem(last=False)
        return day_data

    def _slice(self, key: str, start: int, stop: int) -> np.ndarray:
        parts: List[np.ndarray] = []
        cursor = int(start)
        stop = int(stop)
        while cursor < stop:
            day_idx, local_idx = self._day_local_from_global(cursor)
            info = self.day_infos[day_idx]
            local_stop = min(stop, int(info["end"])) - int(info["start"])
            parts.append(self._load_day_arrays(day_idx)[key][local_idx:local_stop])
            cursor = int(info["start"]) + int(local_stop)
        return _finite_zero(np.concatenate(parts, axis=0))

    def _norm_key(self, key: str, arr: np.ndarray) -> np.ndarray:
        if key == "tec":
            mean, std = self.tec_mean, self.tec_std
        elif key == "rhs":
            mean, std = self.rhs_stats
        elif key in self.phys_term_keys:
            mean, std = _as_stats(self.term_stats, key)
        elif key in self.phys_uv_keys:
            mean, std = _as_stats(self.uv_stats, key)
        else:
            mean, std = _as_stats(self.bg_stats, key)
        return _finite_zero((np.asarray(arr, dtype=np.float32) - mean) / (std + 1e-6))

    def _broadcast_scalar(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        return np.broadcast_to(arr[:, None, None], (arr.shape[0], self.H, self.W)).astype(np.float32)

    def _time_features(self, indices: np.ndarray) -> np.ndarray:
        utc = np.asarray(self.utc_hour[indices], dtype=np.float32)
        phase = 2.0 * np.pi * utc / 24.0
        sin_utc = np.sin(phase).astype(np.float32)
        cos_utc = np.cos(phase).astype(np.float32)
        feats = [
            self._broadcast_scalar(sin_utc),
            self._broadcast_scalar(cos_utc),
        ]
        return np.stack(feats, axis=1).astype(np.float32)

    def _sample_window(self, key: str, start: int, stop: int) -> np.ndarray:
        return self._slice(key, start, stop)[::self.sequence_stride]

    def _rhs_window(self, start: int, stop: int) -> np.ndarray:
        rhs = None
        for key in self.phys_term_keys:
            arr = self._sample_window(key, start, stop)
            rhs = arr if rhs is None else rhs + arr
        if rhs is None:
            n_steps = len(np.arange(start, stop, self.sequence_stride))
            rhs = np.zeros((n_steps, self.H, self.W), dtype=np.float32)
        return _finite_zero(rhs)

    def _stack_feature_window(self, start: int, stop: int, include_physics: bool) -> np.ndarray:
        chans: List[np.ndarray] = []
        if include_physics:
            for key in self.physics_input_keys:
                arr = self._sample_window(key, start, stop)
                chans.append(self._norm_key(key, arr))
        for key in self.bg_scalar_keys:
            arr = self._broadcast_scalar(self._norm_key(key, self._sample_window(key, start, stop)))
            chans.append(arr)
        for key in self.bg_map_keys:
            chans.append(self._norm_key(key, self._sample_window(key, start, stop)))
        indices = np.arange(start, stop, self.sequence_stride, dtype=np.int64)
        chans.extend([self._time_features(indices)[:, i] for i in range(2)])
        return np.stack(chans, axis=1).astype(np.float32)

    def _stack_rhs_window(self, start: int, stop: int) -> np.ndarray:
        rhs = self._norm_key("rhs", self._rhs_window(start, stop))
        return rhs[:, None].astype(np.float32)

    def _diurnal_window(self, anchor: int) -> Tuple[int, int]:
        day_steps = 24 * self.step_per_hour
        start = int(anchor) + 1 - day_steps
        return start, start + self.horizon_steps

    def __getitem__(self, idx: int):
        anchor = int(self.anchor_indices[idx])
        hist_start = anchor - self.history_steps + 1
        hist_stop = anchor + 1
        fut_start = anchor + 1
        fut_stop = fut_start + self.horizon_steps
        base_start, base_stop = self._diurnal_window(anchor)

        enc = self._stack_feature_window(hist_start, hist_stop, include_physics=True)
        baseline = self._stack_rhs_window(base_start, base_stop)
        future_aux = self._stack_feature_window(base_start, base_stop, include_physics=False)
        dec = np.concatenate([baseline, future_aux], axis=1).astype(np.float32)
        target = self._stack_rhs_window(fut_start, fut_stop)
        residual = _finite_zero(target - baseline)
        anchor_time = np.asarray(self.time[anchor]).astype(np.int64)

        return {
            "enc": torch.from_numpy(np.transpose(enc, (1, 0, 2, 3))),
            "dec": torch.from_numpy(np.transpose(dec, (1, 0, 2, 3))),
            "baseline": torch.from_numpy(np.transpose(baseline, (1, 0, 2, 3))),
            "target": torch.from_numpy(np.transpose(target, (1, 0, 2, 3))),
            "residual": torch.from_numpy(np.transpose(residual, (1, 0, 2, 3))),
            "anchor_index": torch.tensor(anchor, dtype=torch.long),
            "anchor_time": torch.tensor(int(anchor_time), dtype=torch.long),
        }

    @property
    def enc_channels(self) -> int:
        return len(self.physics_input_keys) + len(self.bg_scalar_keys) + len(self.bg_map_keys) + 2

    @property
    def dec_channels(self) -> int:
        return 1 + len(self.bg_scalar_keys) + len(self.bg_map_keys) + 2

    @property
    def out_channels(self) -> int:
        return 1
