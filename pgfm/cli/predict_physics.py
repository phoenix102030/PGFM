"""Export physical tendency forecasts indexed by forecast origin."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pgfm.data.flow import list_phys_cached_days
from pgfm.data.physics import PhysicsConvGRUForecastDataset
from pgfm.data.split import configured_splits
from pgfm.models.physics import PhysicsConvGRUForecaster
from pgfm.utils.device import pick_device
from pgfm.utils.io import load_yaml


def _cfg_first(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _stats_tuple(stats, key):
    mean, std = stats.get(key, (0.0, 1.0))
    return float(mean), float(std if float(std) > 1e-8 else 1.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    fore_cfg = cfg.get("physics_forecaster", {}) or {}
    ckpt = torch.load(args.ckpt, map_location="cpu")

    cache_dir = os.path.expanduser(str(data_cfg["cache_dir"]))
    all_days = list_phys_cached_days(cache_dir)
    days = configured_splits(all_days, data_cfg)[args.split]

    phys_term_keys = list(ckpt["phys_term_keys"])
    phys_uv_keys = list(ckpt["phys_uv_keys"])
    bg_scalar_keys = list(ckpt["bg_scalar_keys"])
    bg_map_keys = list(ckpt["bg_map_keys"])
    term_stats = {k: tuple(v) for k, v in ckpt["term_stats"].items()}
    uv_stats = {k: tuple(v) for k, v in ckpt["uv_stats"].items()}
    bg_stats = {k: tuple(v) for k, v in ckpt["bg_stats"].items()}
    rhs_stats = tuple(ckpt.get("rhs_stats", (0.0, 1.0)))

    ds = PhysicsConvGRUForecastDataset(
        cache_dir=cache_dir,
        days=days,
        history_hours=int(ckpt["history_hours"]),
        horizon_hours=int(ckpt["horizon_hours"]),
        tec_mean=float(ckpt["tec_mean"]),
        tec_std=float(ckpt["tec_std"]),
        rhs_stats=rhs_stats,
        term_stats=term_stats,
        uv_stats=uv_stats,
        bg_stats=bg_stats,
        bg_scalar_keys=bg_scalar_keys,
        bg_map_keys=bg_map_keys,
        phys_term_keys=phys_term_keys,
        phys_uv_keys=phys_uv_keys,
        sequence_stride=int(ckpt.get("sequence_stride", fore_cfg.get("sequence_stride", 4))),
        max_cached_days=int(fore_cfg.get("max_cached_days", 1)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(_cfg_first(args.batch_size, fore_cfg.get("eval_batch_size"), fore_cfg.get("batch_size"), default=1)),
        shuffle=False,
        num_workers=int(fore_cfg.get("num_workers", 0)),
        pin_memory=bool(fore_cfg.get("pin_memory", False)),
    )

    model = PhysicsConvGRUForecaster(
        enc_channels=int(ckpt["enc_channels"]),
        dec_channels=int(ckpt["dec_channels"]),
        out_channels=int(ckpt["out_channels"]),
        hidden_channels=list(ckpt["hidden_channels"]),
        kernel_size=int(ckpt["kernel_size"]),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = pick_device(str(fore_cfg.get("device", (cfg.get("train", {}) or {}).get("device", "auto"))))
    model = model.to(device).eval()

    out_path = Path(os.path.expanduser(str(args.out or (Path(fore_cfg.get("outdir", "outputs/pgfm/physics")) / f"forecasts_{args.split}.npz"))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rhs_sidecar = out_path.with_name(out_path.stem + "_future_rhs_rollout.npy")
    partial = rhs_sidecar.with_suffix(".npy.partial")
    if len(ds) == 0: raise ValueError("No eligible forecast origins")
    all_anchor_time = []
    storage = None
    written = 0
    horizon_hours = int(ckpt["horizon_hours"])
    forecast_steps_per_hour = int(ckpt.get("forecast_steps_per_hour", 1))

    for batch in loader:
        batch = _move(batch, device)
        enc = batch["enc"].permute(0, 2, 1, 3, 4).contiguous()
        dec = batch["dec"].permute(0, 2, 1, 3, 4).contiguous()
        baseline = batch["baseline"].permute(0, 2, 1, 3, 4).contiguous()
        pred_norm = baseline + model(enc, dec)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
        rhs_mean, rhs_std = rhs_stats
        rhs_raw = (pred_norm[:, :, :1] * (float(rhs_std) + 1e-6) + float(rhs_mean)).astype(np.float32)
        bsz, steps, _, height, width = rhs_raw.shape
        expected_steps = horizon_hours * forecast_steps_per_hour
        if steps != expected_steps:
            raise ValueError(f"Predicted steps={steps}, expected={expected_steps}")
        rhs_seq = rhs_raw.reshape(bsz, horizon_hours, forecast_steps_per_hour, 1, height, width)
        if forecast_steps_per_hour == 1:
            rhs_quarter = np.repeat(rhs_seq, repeats=4, axis=2)
        else:
            repeat = 4 // forecast_steps_per_hour
            rhs_quarter = np.repeat(rhs_seq, repeats=repeat, axis=2)
        rhs_hour = rhs_quarter[:, :, :4].transpose(0, 1, 3, 2, 4, 5)
        if storage is None:
            storage = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=(len(ds), *rhs_hour.shape[1:]))
        storage[written:written+bsz] = rhs_hour
        written += bsz
        all_anchor_time.append(batch["anchor_time"].detach().cpu().numpy().astype(np.int64))

    if written != len(ds): raise RuntimeError("Incomplete forecast export")
    storage.flush()
    del storage
    with open(partial, "rb") as f: os.fsync(f.fileno())
    check = np.load(partial, mmap_mode="r")
    for i in range(len(check)):
        if not np.isfinite(check[i]).all(): raise ValueError("Nonfinite forecast export")
    del check
    os.replace(partial, rhs_sidecar)
    anchor_time = np.concatenate(all_anchor_time, axis=0)
    metadata_partial = out_path.with_suffix(".npz.partial")
    with metadata_partial.open("wb") as f:
        np.savez_compressed(f, anchor_time=anchor_time,
            phys_term_keys=np.asarray(phys_term_keys), phys_uv_keys=np.asarray(phys_uv_keys),
            days=np.asarray(days), horizon_hours=np.asarray([int(ckpt["horizon_hours"])], dtype=np.int64))
    with np.load(metadata_partial) as z: np.testing.assert_array_equal(z["anchor_time"], anchor_time)
    os.replace(metadata_partial, out_path)
    meta = {
        "split": args.split,
        "days": days,
        "num_anchors": int(anchor_time.shape[0]),
        "phys_term_keys": phys_term_keys,
        "phys_uv_keys": phys_uv_keys,
        "horizon_hours": int(ckpt["horizon_hours"]),
        "future_rhs_rollout_npy": str(rhs_sidecar),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
