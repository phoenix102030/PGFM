"""Train PGFM from prepared daily grids and exported physics forecasts."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pgfm.data.flow import (
    DEFAULT_PHYS_TERM_KEYS,
    DEFAULT_PHYS_UV_KEYS,
    HourlyPiecewiseFlowPhysicsDataset,
    estimate_key_mean_std,
    estimate_many_mean_std,
    list_phys_cached_days,
)
from pgfm.data.split import configured_splits
from pgfm.models.flow import PiecewisePhysFlowMatchingUNet
from pgfm.training import PiecewiseFlowMatchingPhysicsTrainer
from pgfm.utils.device import pick_device
from pgfm.utils.io import load_yaml
from pgfm.utils.seed import set_seed
from pgfm.data.normalization import load_normalization


def ddp_is_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def _cfg_first(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _atomic_torch_save(obj, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="Full training resume checkpoint. Restores model, optimizer/scaler when available, epoch, and best metric.")
    ap.add_argument("--init-ckpt", default=None, help="Weights-only initialization checkpoint. Starts a fresh training run from the checkpoint state_dict.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    fm_cfg = cfg.get("fm", {})

    use_ddp = ddp_is_enabled()
    rank, world_size, local_rank = 0, 1, 0
    if use_ddp:
        rank, world_size, local_rank = ddp_setup()

    try:
        set_seed(int(train_cfg.get("seed", 1234)) + rank)
        device = pick_device(str(train_cfg.get("device", "auto")))
        if use_ddp and torch.cuda.is_available():
            device = f"cuda:{local_rank}"

        cache_dir = os.path.expanduser(str(data_cfg["cache_dir"]))
        days = list_phys_cached_days(cache_dir)
        splits = configured_splits(days, data_cfg)
        train_days, val_days = splits["train"], splits["val"]

        past_hours = int(data_cfg.get("past_hours", 24))
        rollout_hours = int(_cfg_first(data_cfg.get("rollout_hours"), data_cfg.get("horizon_hours"), default=24))
        bg_scalar_keys = list(data_cfg.get("bg_scalar_keys", ["kp", "f107d", "f107a", "hpower", "ctpoten", "bximf", "byimf", "bzimf"]))
        bg_map_keys = list(data_cfg.get("bg_map_keys", ["tlbc", "ulbc", "vlbc"]))
        phys_term_keys = list(data_cfg.get("phys_term_keys", DEFAULT_PHYS_TERM_KEYS))
        phys_uv_keys = list(data_cfg.get("phys_uv_keys", DEFAULT_PHYS_UV_KEYS))
        strict_required_physics = bool(data_cfg.get("strict_required_physics", True))
        future_physics_mode = str(data_cfg.get("future_physics_mode", "truth"))
        future_physics_blend_last_hour_weight = float(data_cfg.get("future_physics_blend_last_hour_weight", 0.5))
        convgru_train_path = data_cfg.get("future_physics_convgru_train_path", data_cfg.get("future_physics_convgru_path"))
        convgru_val_path = data_cfg.get("future_physics_convgru_val_path", data_cfg.get("future_physics_convgru_path"))

        normalization = load_normalization(cfg)
        if normalization is None:
            tec_mean, tec_std = estimate_key_mean_std(cache_dir, train_days, key="tec")
            bg_stats = estimate_many_mean_std(cache_dir, train_days, keys=list(bg_scalar_keys) + list(bg_map_keys))
            uv_stats = estimate_many_mean_std(cache_dir, train_days, keys=phys_uv_keys)
        else:
            tec_mean, tec_std = normalization["tec"]
            bg_stats = {k: normalization[k] for k in bg_scalar_keys + bg_map_keys}
            uv_stats = {k: normalization[k] for k in phys_uv_keys}
        train_ds = HourlyPiecewiseFlowPhysicsDataset(
            cache_dir=cache_dir,
            days=train_days,
            past_hours=past_hours,
            rollout_hours=rollout_hours,
            tec_mean=tec_mean,
            tec_std=tec_std,
            bg_scalar_keys=bg_scalar_keys,
            bg_map_keys=bg_map_keys,
            bg_stats=bg_stats,
            uv_stats=uv_stats,
            phys_term_keys=phys_term_keys,
            phys_uv_keys=phys_uv_keys,
            strict_required_physics=strict_required_physics,
            max_cached_days=int(train_cfg.get("max_cached_days", 3)),
            include_rollout=False,
            future_physics_mode=future_physics_mode,
            future_physics_blend_last_hour_weight=future_physics_blend_last_hour_weight,
            future_physics_convgru_path=convgru_train_path,
        )
        val_ds = HourlyPiecewiseFlowPhysicsDataset(
            cache_dir=cache_dir,
            days=val_days,
            past_hours=past_hours,
            rollout_hours=rollout_hours,
            tec_mean=tec_mean,
            tec_std=tec_std,
            bg_scalar_keys=bg_scalar_keys,
            bg_map_keys=bg_map_keys,
            bg_stats=bg_stats,
            uv_stats=uv_stats,
            phys_term_keys=phys_term_keys,
            phys_uv_keys=phys_uv_keys,
            strict_required_physics=strict_required_physics,
            max_cached_days=int(train_cfg.get("max_cached_days", 2)),
            include_rollout=False,
            future_physics_mode=future_physics_mode,
            future_physics_blend_last_hour_weight=future_physics_blend_last_hour_weight,
            future_physics_convgru_path=convgru_val_path,
        )
        val_rollout_ds = None
        val_rollout_loader = None
        rollout_eval_hours = int(train_cfg.get("rollout_eval_hours", rollout_hours))
        if int(train_cfg.get("rollout_eval_every", 1)) > 0:
            val_rollout_ds = HourlyPiecewiseFlowPhysicsDataset(
                cache_dir=cache_dir,
                days=val_days,
                past_hours=past_hours,
                rollout_hours=rollout_eval_hours,
                tec_mean=tec_mean,
                tec_std=tec_std,
                bg_scalar_keys=bg_scalar_keys,
                bg_map_keys=bg_map_keys,
                bg_stats=bg_stats,
                uv_stats=uv_stats,
                phys_term_keys=phys_term_keys,
                phys_uv_keys=phys_uv_keys,
                strict_required_physics=strict_required_physics,
                max_cached_days=int(train_cfg.get("max_cached_days", 2)),
                include_rollout=True,
                future_physics_mode=future_physics_mode,
                future_physics_blend_last_hour_weight=future_physics_blend_last_hour_weight,
                future_physics_convgru_path=convgru_val_path,
            )
            if len(val_rollout_ds) == 0:
                val_rollout_ds = None

        if data_cfg.get("monthly_origin_split") and (val_rollout_ds is None or rollout_eval_hours != 24 or int(train_cfg.get("rollout_eval_every", 1)) != 1):
            raise ValueError("New-data selection requires 24h rollout validation each epoch")

        batch_size = int(train_cfg.get("batch_size", 4))
        num_workers = int(train_cfg.get("num_workers", 0))
        pin_memory = bool(train_cfg.get("pin_memory", False))
        persistent_workers = bool(train_cfg.get("persistent_workers", False)) if num_workers > 0 else False
        prefetch_factor = int(train_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
        loader_extra = {}
        if num_workers > 0:
            loader_extra["persistent_workers"] = persistent_workers
            loader_extra["prefetch_factor"] = prefetch_factor
        train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **loader_extra,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **loader_extra,
        )
        if val_rollout_ds is not None:
            val_rollout_sampler = DistributedSampler(val_rollout_ds, shuffle=False) if use_ddp else None
            val_rollout_loader = DataLoader(
                val_rollout_ds,
                batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
                shuffle=False,
                sampler=val_rollout_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **loader_extra,
            )
        else:
            val_rollout_loader = None

        model = PiecewisePhysFlowMatchingUNet(
            past_hours=past_hours,
            bg_scalar_channels=len(bg_scalar_keys),
            bg_map_channels=len(bg_map_keys),
            tec_std=tec_std,
            hidden=int(model_cfg.get("hidden", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            grad_checkpoint=bool(model_cfg.get("grad_checkpoint", False)),
        )
        model = model.to(device)
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        trainer = PiecewiseFlowMatchingPhysicsTrainer(
            model=model,
            device=device,
            lr=float(_cfg_first(train_cfg.get("lr"), fm_cfg.get("lr"), default=1e-4)),
            grad_clip_norm=float(_cfg_first(train_cfg.get("grad_clip_norm"), fm_cfg.get("grad_clip_norm"), default=1.0)),
            use_amp=bool(_cfg_first(train_cfg.get("use_amp"), fm_cfg.get("use_amp"), default=True)),
            ode_steps=int(_cfg_first(fm_cfg.get("ode_steps"), (cfg.get("eval", {}) or {}).get("eval_steps"), default=24)),
            lambda_balance=float(fm_cfg.get("lambda_balance", 1e-4)),
            lambda_phys=float(fm_cfg.get("lambda_phys", 1e-4)),
        )

        run_name = str(train_cfg.get("run_name", "pgfm"))
        eval_cfg = cfg.get("eval", {}) or {}
        out_root = Path(os.path.expanduser(str(_cfg_first(train_cfg.get("outdir"), eval_cfg.get("outdir"), default=f"outputs/{run_name}"))))
        model_dir = out_root / "models"
        if is_main_process(rank):
            out_root.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

        best_metric = float("inf")
        best_path = model_dir / "best.pt"
        last_path = model_dir / "last.pt"
        history = []
        start_epoch = 1

        resume_path = _cfg_first(args.resume, train_cfg.get("resume"), train_cfg.get("resume_from"), default=None)
        init_ckpt_path = _cfg_first(args.init_ckpt, train_cfg.get("init_ckpt"), train_cfg.get("init_from"), default=None)
        if resume_path and init_ckpt_path:
            raise ValueError("Use only one of --resume/train.resume or --init-ckpt/train.init_ckpt.")

        if resume_path:
            resume_path = os.path.expanduser(str(resume_path))
            ckpt_resume = torch.load(resume_path, map_location=device)
            trainer.load_state_dict(ckpt_resume["state_dict"], strict=True)
            if "optimizer_state_dict" in ckpt_resume:
                trainer.load_optimizer_state_dict(ckpt_resume["optimizer_state_dict"])
            elif is_main_process(rank):
                print(f"[piecewise_phys] resume checkpoint has no optimizer state; continuing with a fresh optimizer: {resume_path}")
            if "scaler_state_dict" in ckpt_resume:
                trainer.load_scaler_state_dict(ckpt_resume["scaler_state_dict"])
            start_epoch = int(ckpt_resume.get("epoch", 0)) + 1
            best_metric = float(ckpt_resume.get("best_metric", best_metric))
            history = list(ckpt_resume.get("history", []))
            if is_main_process(rank):
                print(f"[piecewise_phys] resumed from {resume_path} at epoch {start_epoch}")
        elif init_ckpt_path:
            init_ckpt_path = os.path.expanduser(str(init_ckpt_path))
            ckpt_init = torch.load(init_ckpt_path, map_location=device)
            trainer.load_state_dict(ckpt_init["state_dict"], strict=True)
            if is_main_process(rank):
                print(f"[piecewise_phys] initialized model weights from {init_ckpt_path}")

        epochs = int(train_cfg.get("epochs", 20))
        rollout_eval_every = int(train_cfg.get("rollout_eval_every", 1))
        rollout_eval_max_batches = int(train_cfg.get("rollout_eval_max_batches", 0))
        save_every = int(train_cfg.get("save_every", 1))
        if start_epoch > epochs:
            if is_main_process(rank):
                print(f"[piecewise_phys] start_epoch={start_epoch} > epochs={epochs}; nothing to train.")
        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = trainer.train_one_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            if rollout_eval_every > 0 and epoch % rollout_eval_every == 0 and val_rollout_loader is not None:
                rollout_stats = trainer.validate_rollout(
                    val_rollout_loader,
                    rollout_hours=rollout_eval_hours,
                    max_batches=rollout_eval_max_batches,
                )
                metric = float(rollout_stats["rollout_mse"])
                if not math.isfinite(metric):
                    if data_cfg.get("monthly_origin_split"):
                        raise ValueError("Nonfinite 24h rollout selection metric")
                    metric = float(val_loss)
            else:
                rollout_stats = {"rollout_mse": float("nan"), "rollout_mae": float("nan")}
                metric = float(val_loss)

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                **{k: float(v) for k, v in rollout_stats.items()},
            }
            if is_main_process(rank):
                history.append(row)
                print(
                    f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} "
                    f"rollout_mse={rollout_stats['rollout_mse']:.6f} rollout_mae={rollout_stats['rollout_mae']:.6f}"
                )

            if is_main_process(rank):
                is_best = metric < best_metric
                if is_best:
                    best_metric = metric
                ckpt = {
                    "cfg": cfg,
                    "epoch": epoch,
                    "state_dict": trainer.get_state_dict(),
                    "tec_mean": float(tec_mean),
                    "tec_std": float(tec_std),
                    "bg_scalar_keys": bg_scalar_keys,
                    "bg_map_keys": bg_map_keys,
                    "phys_term_keys": phys_term_keys,
                    "phys_uv_keys": phys_uv_keys,
                    "future_physics_mode": future_physics_mode,
                    "future_physics_blend_last_hour_weight": float(future_physics_blend_last_hour_weight),
                    "future_physics_convgru_train_path": str(convgru_train_path) if convgru_train_path else None,
                    "future_physics_convgru_val_path": str(convgru_val_path) if convgru_val_path else None,
                    "model_hidden": int(model_cfg.get("hidden", 64)),
                    "model_grad_checkpoint": bool(model_cfg.get("grad_checkpoint", False)),
                    "bg_stats": {k: [float(v[0]), float(v[1])] for k, v in bg_stats.items()},
                    "uv_stats": {k: [float(v[0]), float(v[1])] for k, v in uv_stats.items()},
                    "metrics": row,
                    "history": history,
                    "best_metric": float(best_metric),
                    "optimizer_state_dict": trainer.get_optimizer_state_dict(),
                    "scaler_state_dict": trainer.get_scaler_state_dict(),
                    "world_size": world_size,
                }
                _atomic_torch_save(ckpt, last_path)
                if save_every > 0 and epoch % save_every == 0:
                    _atomic_torch_save(ckpt, model_dir / f"epoch_{epoch:03d}.pt")
                if is_best:
                    _atomic_torch_save(ckpt, best_path)

        if is_main_process(rank):
            with open(out_root / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            print(f"[done] best_ckpt={best_path}")
    finally:
        if use_ddp:
            ddp_cleanup()


if __name__ == "__main__":
    main()
