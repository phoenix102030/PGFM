"""Evaluate PGFM TEC rollouts and save model-only forecast metrics."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pgfm.data.flow import (
    DEFAULT_PHYS_TERM_KEYS,
    DEFAULT_PHYS_UV_KEYS,
    HourlyPiecewiseFlowPhysicsDataset,
    list_phys_cached_days,
)
from pgfm.data.split import configured_splits
from pgfm.models.flow import PiecewisePhysFlowMatchingUNet
from pgfm.training import PiecewiseFlowMatchingPhysicsTrainer
from pgfm.utils.device import pick_device
from pgfm.utils.metrics import per_lead_metrics, point_metrics
from pgfm.utils.io import load_yaml


def _cfg_first(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _select_days(cfg: Dict, split: str, start_day_idx: int, num_days: int) -> List[str]:
    cache_dir = cfg["data"]["cache_dir"]
    all_days = list_phys_cached_days(cache_dir)
    split_days = configured_splits(all_days, cfg["data"])[split]
    if cfg["data"].get("monthly_origin_split"):
        if start_day_idx != 0 or num_days > 0:
            raise ValueError("Monthly split requires all origin days: start_day_idx=0, days=0")
        return split_days
    s = max(int(start_day_idx), 0)
    e = len(split_days) if int(num_days) <= 0 else min(s + int(num_days), len(split_days))
    sel_days = split_days[s:e]
    if not sel_days:
        raise RuntimeError("No selected days. Check split/start_day_idx/days.")
    return sel_days


def _build_eval_batch(
    ds: HourlyPiecewiseFlowPhysicsDataset,
    anchor: int,
    horizon: int,
):
    hist_idx = ds._hist_hour_indices(anchor)
    tec_hist = ds._norm(
        ds._slice_global_key("tec", int(hist_idx[0]), int(hist_idx[-1]) + 1)[::ds.step_per_hour],
        ds.tec_mean,
        ds.tec_std,
    )
    current_tec = tec_hist[-1]

    bg_scalar_hist = []
    for key in ds.bg_scalar_keys:
        mean, std = ds.bg_stats.get(key, (0.0, 1.0))
        bg_scalar_hist.append(
            ds._norm(
                ds._slice_global_key(key, int(hist_idx[0]), int(hist_idx[-1]) + 1)[::ds.step_per_hour],
                mean,
                std,
            )
        )
    bg_scalar_hist = np.stack(bg_scalar_hist, axis=0).astype(np.float32) if bg_scalar_hist else np.zeros((0, ds.past_hours), dtype=np.float32)

    bg_map_hist = []
    for key in ds.bg_map_keys:
        mean, std = ds.bg_stats.get(key, (0.0, 1.0))
        bg_map_hist.append(
            ds._norm(
                ds._slice_global_key(key, int(hist_idx[0]), int(hist_idx[-1]) + 1)[::ds.step_per_hour],
                mean,
                std,
            )
        )
    bg_map_hist = np.stack(bg_map_hist, axis=0).astype(np.float32) if bg_map_hist else np.zeros((0, ds.past_hours, ds.H, ds.W), dtype=np.float32)

    future_terms_rollout, future_uv_rollout = ds.build_future_physics_rollout(anchor=anchor, rollout_hours=int(horizon))
    future_rhs_rollout = None
    if ds.future_physics_mode == "convgru":
        future_rhs_rollout = ds.build_future_rhs_rollout(anchor=anchor, rollout_hours=int(horizon))

    rollout_hour_idx = np.arange(anchor + ds.step_per_hour, anchor + horizon * ds.step_per_hour + 1, ds.step_per_hour, dtype=np.int64)
    truth_hourly_norm = ds._norm(
        ds._slice_global_key("tec", int(rollout_hour_idx[0]), int(rollout_hour_idx[-1]) + 1)[::ds.step_per_hour],
        ds.tec_mean,
        ds.tec_std,
    )

    batch = {
        "tec_hist": torch.from_numpy(tec_hist[None, ...]),
        "current_tec": torch.from_numpy(current_tec[None, None, ...]),
        "bg_scalar_hist": torch.from_numpy(bg_scalar_hist[None, ...]),
        "bg_map_hist": torch.from_numpy(bg_map_hist[None, ...]),
        "future_terms_rollout": torch.from_numpy(future_terms_rollout[None, ...]),
        "future_uv_rollout": torch.from_numpy(future_uv_rollout[None, ...]),
        "target_hourly": torch.from_numpy(truth_hourly_norm[None, ...]),
    }
    if future_rhs_rollout is not None:
        batch["future_rhs_rollout"] = torch.from_numpy(future_rhs_rollout[None, ...])
    return batch


def _compute_per_lead_summary(
    pred_blocks: List[np.ndarray],
    truth_blocks: List[np.ndarray],
    horizon: int,
    percentage_eps_tecu: float = 1.0,
) -> List[Dict[str, float]]:
    return per_lead_metrics(
        pred_blocks,
        truth_blocks,
        horizon=horizon,
        percentage_eps_tecu=percentage_eps_tecu,
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--future-physics-mode", default=None, choices=["truth", "last_hour", "diurnal_24h", "blend", "convgru"])
    ap.add_argument("--future-physics-blend-last-hour-weight", type=float, default=None)
    ap.add_argument("--future-physics-convgru-path", default=None)
    ap.add_argument("--percentage-eps-tecu", type=float, default=None, help="Denominator floor in TECU for MAPE/sMAPE. Defaults to eval.percentage_eps_tecu or 1.0.")
    ap.add_argument("--save-full-pred-truth", action="store_true", help="Save full pred/truth arrays for offline metric analysis.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_cfg = cfg.get("train", {})
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    fm_cfg = cfg.get("fm", {})
    eval_cfg = cfg.get("eval", {}) or {}
    percentage_eps_tecu = float(_cfg_first(args.percentage_eps_tecu, eval_cfg.get("percentage_eps_tecu"), default=1.0))

    backbone = str((fm_cfg.get("backbone", "piecewise_condphys_fm")))
    mode = str(eval_cfg.get("mode", "truth")).lower()
    if mode != "truth":
        raise ValueError("This eval script currently supports only eval.mode='truth'.")

    start_day_idx = int(eval_cfg.get("start_day_idx", 0))
    days_requested = int(eval_cfg.get("days", 5))
    start_hour = int(eval_cfg.get("start_hour", 24))
    horizon = int(_cfg_first(eval_cfg.get("horizon_hours"), data_cfg.get("horizon_hours"), data_cfg.get("rollout_hours"), default=24))
    block_step_hours = int(eval_cfg.get("block_step_hours", horizon))
    if block_step_hours <= 0:
        raise ValueError("eval.block_step_hours must be positive")
    eval_steps = int(_cfg_first(eval_cfg.get("eval_steps"), fm_cfg.get("ode_steps"), default=24))
    future_physics_mode = str(_cfg_first(args.future_physics_mode, eval_cfg.get("future_physics_mode"), data_cfg.get("future_physics_mode"), ckpt.get("future_physics_mode"), default="truth"))
    future_physics_blend_last_hour_weight = float(
        _cfg_first(
            args.future_physics_blend_last_hour_weight,
            eval_cfg.get("future_physics_blend_last_hour_weight"),
            data_cfg.get("future_physics_blend_last_hour_weight"),
            ckpt.get("future_physics_blend_last_hour_weight"),
            default=0.5,
        )
    )
    future_physics_convgru_path = _cfg_first(
        args.future_physics_convgru_path,
        eval_cfg.get("future_physics_convgru_path"),
        data_cfg.get(f"future_physics_convgru_{args.split}_path"),
        data_cfg.get("future_physics_convgru_path"),
        ckpt.get("future_physics_convgru_val_path"),
        default=None,
    )

    sel_days = _select_days(cfg, args.split, start_day_idx, days_requested)

    bg_scalar_keys = ckpt["bg_scalar_keys"]
    bg_map_keys = ckpt["bg_map_keys"]
    phys_term_keys = list(ckpt.get("phys_term_keys", DEFAULT_PHYS_TERM_KEYS))
    phys_uv_keys = list(ckpt.get("phys_uv_keys", DEFAULT_PHYS_UV_KEYS))
    bg_stats = {k: tuple(v) for k, v in ckpt["bg_stats"].items()}
    uv_stats = {k: tuple(v) for k, v in ckpt["uv_stats"].items()}
    tec_mean = float(ckpt["tec_mean"])
    tec_std = float(ckpt["tec_std"])

    ds = HourlyPiecewiseFlowPhysicsDataset(
        cache_dir=os.path.expanduser(str(data_cfg["cache_dir"])),
        days=sel_days,
        past_hours=int(data_cfg.get("past_hours", 24)),
        rollout_hours=horizon,
        tec_mean=tec_mean,
        tec_std=tec_std,
        bg_scalar_keys=bg_scalar_keys,
        bg_map_keys=bg_map_keys,
        bg_stats=bg_stats,
        uv_stats=uv_stats,
        phys_term_keys=phys_term_keys,
        phys_uv_keys=phys_uv_keys,
        strict_required_physics=bool(eval_cfg.get("strict_required_physics", data_cfg.get("strict_required_physics", True))),
        include_rollout=False,
        max_cached_days=int(train_cfg.get("max_cached_days", 8)),
        future_physics_mode=future_physics_mode,
        future_physics_blend_last_hour_weight=future_physics_blend_last_hour_weight,
        future_physics_convgru_path=future_physics_convgru_path,
    )

    model = PiecewisePhysFlowMatchingUNet(
        past_hours=int(data_cfg.get("past_hours", 24)),
        bg_scalar_channels=len(bg_scalar_keys),
        bg_map_channels=len(bg_map_keys),
        tec_std=tec_std,
        hidden=int(_cfg_first(ckpt.get("model_hidden"), (ckpt.get("cfg", {}) or {}).get("model", {}).get("hidden"), model_cfg.get("hidden"), default=64)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)

    device = pick_device(str(train_cfg.get("device", "auto")))
    trainer = PiecewiseFlowMatchingPhysicsTrainer(
        model=model,
        device=device,
        lr=float(_cfg_first(train_cfg.get("lr"), fm_cfg.get("lr"), default=1e-4)),
        use_amp=bool(_cfg_first(train_cfg.get("use_amp"), fm_cfg.get("use_amp"), default=True)),
        ode_steps=eval_steps,
    )

    hourly_anchor_positions = [int(a) for a in ds.anchor_indices if np.isclose(ds.utc_hour[int(a)] % 1.0, 0.0, atol=1e-4)]
    if start_hour < 0 or start_hour >= len(hourly_anchor_positions):
        raise RuntimeError(f"start_hour={start_hour} out of range for available anchors={len(hourly_anchor_positions)}")

    block_starts: List[int] = []
    pos = int(start_hour)
    while pos < len(hourly_anchor_positions):
        anchor = hourly_anchor_positions[pos]
        if anchor + horizon * ds.step_per_hour < ds.T_total:
            block_starts.append(anchor)
        pos += int(block_step_hours)
    if not block_starts:
        raise RuntimeError("No evaluation blocks available with current start_hour/horizon/block_step_hours.")

    pred_blocks: List[np.ndarray] = []
    truth_blocks: List[np.ndarray] = []

    for anchor in block_starts:
        batch = _build_eval_batch(ds, anchor=anchor, horizon=horizon)
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_norm = trainer.rollout_hours(batch, rollout_hours=horizon)
        truth_norm = batch["target_hourly"][:, :horizon]

        pred = (pred_norm.cpu().numpy()[0] * tec_std + tec_mean).astype(np.float32)
        truth = (truth_norm.cpu().numpy()[0] * tec_std + tec_mean).astype(np.float32)

        pred_blocks.append(pred)
        truth_blocks.append(truth)

    pred_all = np.stack(pred_blocks, axis=0)
    truth_all = np.stack(truth_blocks, axis=0)

    overall = point_metrics(pred_all, truth_all, percentage_eps_tecu=percentage_eps_tecu)
    per_lead = _compute_per_lead_summary(
        pred_blocks,
        truth_blocks,
        horizon=horizon,
        percentage_eps_tecu=percentage_eps_tecu,
    )

    total_hours = int(len(block_starts) * block_step_hours)
    out_root = Path(os.path.expanduser(str(_cfg_first(train_cfg.get("outdir"), eval_cfg.get("outdir"), default="outputs/pgfm")))) / "eval"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "header": {
            "backbone": backbone,
            "mode": mode,
            "future_physics_mode": ds.future_physics_mode,
            "future_physics_blend_last_hour_weight": float(ds.future_physics_blend_last_hour_weight),
            "future_physics_convgru_path": ds.future_physics_convgru_path,
            "split": args.split,
            "days_requested": days_requested,
            "days_used": len(sel_days),
            "day_list": sel_days,
            "start_hour": start_hour,
            "past_used": int(data_cfg.get("past_hours", 24)),
            "phys_term_keys": phys_term_keys,
            "phys_uv_keys": phys_uv_keys,
            "horizon": horizon,
            "block_step_hours": block_step_hours,
            "num_blocks": len(block_starts),
            "total_hours": total_hours,
            "percentage_eps_tecu": percentage_eps_tecu,
        },
        "overall": overall,
        "per_lead": per_lead,
    }
    with open(out_root / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_root / f"samples_{args.split}.npz",
        pred=pred_all[: min(4, len(pred_all))],
        truth=truth_all[: min(4, len(truth_all))],
    )
    if args.save_full_pred_truth:
        np.savez_compressed(
            out_root / f"full_pred_truth_{args.split}.npz",
            pred=pred_all.astype(np.float32),
            truth=truth_all.astype(np.float32),
        )

    print(
        f"[eval {args.split}] rmse={overall['rmse']:.6f} "
        f"mae={overall['mae']:.6f} metrics={out_root / f'metrics_{args.split}.json'}"
    )


if __name__ == "__main__":
    main()
