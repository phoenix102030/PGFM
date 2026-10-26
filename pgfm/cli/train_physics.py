"""Train the causal physics forecaster separately from PGFM."""

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pgfm.data.flow import (
    _load_phys_npz,
    DEFAULT_PHYS_TERM_KEYS,
    DEFAULT_PHYS_UV_KEYS,
    estimate_key_mean_std,
    estimate_many_mean_std,
    list_phys_cached_days,
)
from pgfm.data.physics import PhysicsConvGRUForecastDataset
from pgfm.data.split import configured_splits
from pgfm.models.physics import PhysicsConvGRUForecaster
from pgfm.utils.device import pick_device
from pgfm.utils.io import load_yaml
from pgfm.utils.seed import set_seed
from pgfm.data.normalization import load_normalization


def _cfg_first(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _as_hidden_channels(value) -> Sequence[int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (32, 32)


def estimate_rhs_mean_std(cache_dir: str, days, phys_term_keys) -> tuple[float, float]:
    count, mean, m2 = 0, 0., 0.
    for day in days:
        with _load_phys_npz(cache_dir, day) as data:
            rhs = sum(np.asarray(data[key], dtype=np.float64) for key in phys_term_keys)
            x = rhs[np.isfinite(rhs)]
            n = x.size
            if not n: continue
            mu = float(x.mean()); delta = mu-mean
            m2 += float(np.square(x-mu).sum()) + delta**2*count*n/(count+n)
            mean += delta*n/(count+n); count += n
    if not count: raise ValueError("No training RHS values")
    return mean, max(float(np.sqrt(m2/count)), 1e-8)


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
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def _move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _prep(batch):
    enc = batch["enc"].permute(0, 2, 1, 3, 4).contiguous()
    dec = batch["dec"].permute(0, 2, 1, 3, 4).contiguous()
    target_residual = batch["residual"].permute(0, 2, 1, 3, 4).contiguous()
    target = batch["target"].permute(0, 2, 1, 3, 4).contiguous()
    baseline = batch["baseline"].permute(0, 2, 1, 3, 4).contiguous()
    return enc, dec, target_residual, target, baseline


def _loss_fn(pred_residual, target_residual, baseline, target, residual_weight: float):
    pred = baseline + pred_residual
    loss_direct = torch.mean((pred - target) ** 2)
    loss_residual = torch.mean((pred_residual - target_residual) ** 2)
    return loss_direct + float(residual_weight) * loss_residual, loss_direct.detach(), loss_residual.detach()


def _sync_metric(total: float, count: int, device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return total / max(count, 1)
    stats = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float((stats[0] / stats[1].clamp_min(1.0)).item())


def _run_epoch(model, loader, device, opt, scaler, use_amp: bool, residual_weight: float, train: bool):
    model.train(train)
    total, count = 0.0, 0
    for batch in loader:
        batch = _move(batch, device)
        enc, dec, target_residual, target, baseline = _prep(batch)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_residual = model(enc, dec)
            loss, _, _ = _loss_fn(pred_residual, target_residual, baseline, target, residual_weight=residual_weight)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        bsz = enc.shape[0]
        total += float(loss.detach().item()) * bsz
        count += bsz
    return _sync_metric(total, count, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("physics_forecaster", {}) or {}
    base_train_cfg = cfg.get("train", {}) or {}

    use_ddp = ddp_is_enabled()
    rank, world_size, local_rank = 0, 1, 0
    if use_ddp:
        rank, world_size, local_rank = ddp_setup()

    try:
        set_seed(int(train_cfg.get("seed", base_train_cfg.get("seed", 1234))) + rank)
        device = pick_device(str(train_cfg.get("device", base_train_cfg.get("device", "auto"))))
        if use_ddp and torch.cuda.is_available():
            device = f"cuda:{local_rank}"

        cache_dir = os.path.expanduser(str(data_cfg["cache_dir"]))
        all_days = list_phys_cached_days(cache_dir)
        splits = configured_splits(all_days, data_cfg)
        train_days, val_days = splits["train"], splits["val"]
        bg_scalar_keys = list(data_cfg.get("bg_scalar_keys", []))
        bg_map_keys = list(data_cfg.get("bg_map_keys", []))
        phys_term_keys = list(data_cfg.get("phys_term_keys", DEFAULT_PHYS_TERM_KEYS))
        phys_uv_keys = list(data_cfg.get("phys_uv_keys", DEFAULT_PHYS_UV_KEYS))

        normalization = load_normalization(cfg)
        if normalization is None:
            tec_mean, tec_std = estimate_key_mean_std(cache_dir, train_days, "tec")
            rhs_stats = estimate_rhs_mean_std(cache_dir, train_days, phys_term_keys)
            term_stats = estimate_many_mean_std(cache_dir, train_days, phys_term_keys)
            uv_stats = estimate_many_mean_std(cache_dir, train_days, phys_uv_keys)
            bg_stats = estimate_many_mean_std(cache_dir, train_days, list(bg_scalar_keys) + list(bg_map_keys))
        else:
            tec_mean, tec_std = normalization["tec"]
            bg_stats = {k: normalization[k] for k in bg_scalar_keys + bg_map_keys}
            uv_stats = {k: normalization[k] for k in phys_uv_keys}
            rhs_stats = normalization["rhs"]
            term_stats = {k: normalization[k] for k in phys_term_keys}

        history_hours = int(train_cfg.get("history_hours", data_cfg.get("past_hours", 24)))
        horizon_hours = int(train_cfg.get("horizon_hours", data_cfg.get("horizon_hours", 24)))
        sequence_stride = int(train_cfg.get("sequence_stride", 4))
        max_cached_days = int(train_cfg.get("max_cached_days", 1))
        anchor_stride_hours = int(train_cfg.get("anchor_stride_hours", 1))
        max_train_samples = int(train_cfg.get("max_train_samples", 0))
        max_val_samples = int(train_cfg.get("max_val_samples", 0))
        train_ds = PhysicsConvGRUForecastDataset(
            cache_dir=cache_dir,
            days=train_days,
            history_hours=history_hours,
            horizon_hours=horizon_hours,
            tec_mean=tec_mean,
            tec_std=tec_std,
            rhs_stats=rhs_stats,
            term_stats=term_stats,
            uv_stats=uv_stats,
            bg_stats=bg_stats,
            bg_scalar_keys=bg_scalar_keys,
            bg_map_keys=bg_map_keys,
            phys_term_keys=phys_term_keys,
            phys_uv_keys=phys_uv_keys,
            sequence_stride=sequence_stride,
            max_cached_days=max_cached_days,
            anchor_stride_hours=anchor_stride_hours,
            max_samples=max_train_samples,
        )
        val_ds = PhysicsConvGRUForecastDataset(
            cache_dir=cache_dir,
            days=val_days,
            history_hours=history_hours,
            horizon_hours=horizon_hours,
            tec_mean=tec_mean,
            tec_std=tec_std,
            rhs_stats=rhs_stats,
            term_stats=term_stats,
            uv_stats=uv_stats,
            bg_stats=bg_stats,
            bg_scalar_keys=bg_scalar_keys,
            bg_map_keys=bg_map_keys,
            phys_term_keys=phys_term_keys,
            phys_uv_keys=phys_uv_keys,
            sequence_stride=sequence_stride,
            max_cached_days=max_cached_days,
            anchor_stride_hours=anchor_stride_hours,
            max_samples=max_val_samples,
        )
        shuffle_train = bool(train_cfg.get("shuffle", False))
        train_sampler = DistributedSampler(train_ds, shuffle=shuffle_train) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None
        batch_size = int(train_cfg.get("batch_size", 1))
        num_workers = int(train_cfg.get("num_workers", 0))
        loader_extra = {}
        if num_workers > 0:
            loader_extra["persistent_workers"] = bool(train_cfg.get("persistent_workers", False))
            loader_extra["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None and shuffle_train),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=bool(train_cfg.get("pin_memory", False)),
            **loader_extra,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=bool(train_cfg.get("pin_memory", False)),
            **loader_extra,
        )

        model = PhysicsConvGRUForecaster(
            enc_channels=train_ds.enc_channels,
            dec_channels=train_ds.dec_channels,
            out_channels=train_ds.out_channels,
            hidden_channels=_as_hidden_channels(train_cfg.get("hidden_channels", [32, 32])),
            kernel_size=int(train_cfg.get("kernel_size", 3)),
            dropout=float(train_cfg.get("dropout", 0.0)),
        ).to(device)
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
        use_amp = bool(train_cfg.get("use_amp", True)) and str(device).startswith("cuda") and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        residual_weight = float(train_cfg.get("residual_weight", 0.25))

        out_root = Path(os.path.expanduser(str(train_cfg.get("outdir", "outputs/pgfm/physics"))))
        model_dir = out_root / "models"
        if is_main_process(rank):
            model_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        history = []
        start_epoch = 1
        if args.resume:
            resume = torch.load(args.resume, map_location="cpu", weights_only=False)
            if resume["cfg"] != cfg: raise ValueError("Resume configuration differs")
            (model.module if hasattr(model, "module") else model).load_state_dict(resume["state_dict"])
            opt.load_state_dict(resume["optimizer_state_dict"])
            scaler.load_state_dict(resume["scaler_state_dict"])
            history = resume["history"]
            best_val = float(resume["best_val"])
            start_epoch = int(resume["epoch"]) + 1
            print(f"[resume] next_epoch={start_epoch}", flush=True)
        for epoch in range(start_epoch, int(train_cfg.get("epochs", 20)) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = _run_epoch(model, train_loader, device, opt, scaler, use_amp, residual_weight, train=True)
            with torch.no_grad():
                val_loss = _run_epoch(model, val_loader, device, opt, scaler, use_amp, residual_weight, train=False)
            if is_main_process(rank):
                row = {"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)}
                history.append(row)
                print(f"[physics_convgru epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f}")
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                is_best = val_loss < best_val
                best_val = min(best_val, val_loss)
                ckpt = {
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "history": history,
                    "best_val": best_val,
                    "cfg": cfg,
                    "state_dict": state_dict,
                    "epoch": epoch,
                    "enc_channels": train_ds.enc_channels,
                    "dec_channels": train_ds.dec_channels,
                    "out_channels": train_ds.out_channels,
                    "history_hours": history_hours,
                    "horizon_hours": horizon_hours,
                    "sequence_stride": sequence_stride,
                    "forecast_steps_per_hour": train_ds.forecast_steps_per_hour,
                    "hidden_channels": list(_as_hidden_channels(train_cfg.get("hidden_channels", [32, 32]))),
                    "kernel_size": int(train_cfg.get("kernel_size", 3)),
                    "tec_mean": float(tec_mean),
                    "tec_std": float(tec_std),
                    "rhs_stats": [float(rhs_stats[0]), float(rhs_stats[1])],
                    "term_stats": {k: [float(v[0]), float(v[1])] for k, v in term_stats.items()},
                    "uv_stats": {k: [float(v[0]), float(v[1])] for k, v in uv_stats.items()},
                    "bg_stats": {k: [float(v[0]), float(v[1])] for k, v in bg_stats.items()},
                    "bg_scalar_keys": bg_scalar_keys,
                    "bg_map_keys": bg_map_keys,
                    "phys_term_keys": phys_term_keys,
                    "phys_uv_keys": phys_uv_keys,
                    "metrics": row,
                    "world_size": world_size,
                }
                def save_atomic(path):
                    temporary = path.with_suffix(".pt.partial")
                    torch.save(ckpt, temporary)
                    os.replace(temporary, path)
                save_atomic(model_dir / "last.pt")
                if epoch % 5 == 0: save_atomic(model_dir / f"epoch_{epoch:03d}.pt")
                if is_best: save_atomic(model_dir / "best.pt")
                tmp = out_root / "history.json.partial"
                tmp.write_text(json.dumps(history, indent=2)+"\n")
                os.replace(tmp, out_root / "history.json")

        if is_main_process(rank):
            with open(out_root / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            print(f"[done] best_ckpt={model_dir / 'best.pt'}")
    finally:
        if use_ddp:
            ddp_cleanup()


if __name__ == "__main__":
    main()
