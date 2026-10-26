"""Piecewise flow-matching objective, optimization and autoregressive integration."""

from typing import Dict
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist


class PiecewiseFlowMatchingPhysicsTrainer:
    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-4,
        grad_clip_norm=1.0,
        use_amp=True,
        ode_steps=24,
        lambda_balance=1e-4,
        lambda_phys=1e-4,
    ):
        self.device = device
        self.model = model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.grad_clip_norm = float(grad_clip_norm)
        self.ode_steps = int(ode_steps)
        self.lambda_balance = float(lambda_balance)
        self.lambda_phys = float(lambda_phys)
        self.use_amp = bool(use_amp) and str(device).startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _is_dist(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _sync_sum_count(self, total: float, count: int) -> float:
        if not self._is_dist():
            return total / max(count, 1)
        stats = torch.tensor([total, float(count)], device=self.device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return float((stats[0] / stats[1].clamp_min(1.0)).item())

    def _sync_rollout_metrics(self, mse_sum: float, mae_sum: float, count: int) -> Dict[str, float]:
        if not self._is_dist():
            return {
                "rollout_mse": mse_sum / max(count, 1),
                "rollout_mae": mae_sum / max(count, 1),
            }
        stats = torch.tensor([mse_sum, mae_sum, float(count)], device=self.device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        denom = stats[2].clamp_min(1.0)
        return {
            "rollout_mse": float((stats[0] / denom).item()),
            "rollout_mae": float((stats[1] / denom).item()),
        }

    def get_state_dict(self):
        if hasattr(self.model, "module"):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        if hasattr(self.model, "module"):
            return self.model.module.load_state_dict(state_dict, strict=strict)
        return self.model.load_state_dict(state_dict, strict=strict)

    def get_optimizer_state_dict(self):
        return self.opt.state_dict()

    def load_optimizer_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def get_scaler_state_dict(self):
        return self.scaler.state_dict()

    def load_scaler_state_dict(self, state_dict):
        if state_dict:
            self.scaler.load_state_dict(state_dict)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    def _count_nonfinite(self, batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                count = int((~torch.isfinite(value)).sum().item())
                if count > 0:
                    out[key] = count
        return out

    def _sample_tau(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand((batch_size, 1, 1, 1), device=self.device, dtype=dtype)

    def _finite_zero(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_piecewise_target(self, knots: torch.Tensor, tau: torch.Tensor):
        """
        knots: (B, 5, H, W)
        tau:   (B, 1, 1, 1)
        """
        seg = torch.clamp((tau.view(-1) * 4.0).long(), min=0, max=3)
        alpha = (tau.view(-1) * 4.0 - seg.float()).view(-1, 1, 1)
        left = knots[torch.arange(knots.shape[0], device=knots.device), seg]
        right = knots[torch.arange(knots.shape[0], device=knots.device), seg + 1]
        x_tau = self._finite_zero((1.0 - alpha) * left + alpha * right)
        u_target = self._finite_zero(4.0 * (right - left))
        return x_tau.unsqueeze(1), u_target.unsqueeze(1)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]):
        tau = self._sample_tau(batch["tec_hist"].shape[0], dtype=batch["tec_hist"].dtype)
        x_tau, u_target = self._build_piecewise_target(batch["knots_hour1"], tau=tau)

        pred_u, aux = self.model(
            x_tau=x_tau,
            tau=tau,
            tec_hist=batch["tec_hist"],
            bg_scalar_hist=batch["bg_scalar_hist"],
            bg_map_hist=batch["bg_map_hist"],
            future_terms_hour=batch["future_terms_hour1"],
            future_uv_hour=batch["future_uv_hour1"],
            future_rhs_hour=batch.get("future_rhs_hour1"),
            return_aux=True,
        )
        pred_u = self._finite_zero(pred_u)
        u_target = self._finite_zero(u_target)
        aux["u_phys"] = self._finite_zero(aux["u_phys"])
        aux["lambda_map"] = self._finite_zero(aux["lambda_map"])

        loss_main = self.loss_fn(pred_u, u_target)
        loss_balance = ((aux["lambda_map"] - 0.5) ** 2).mean()
        loss_phys = self.loss_fn(pred_u, aux["u_phys"])
        total = loss_main + self.lambda_balance * loss_balance + self.lambda_phys * loss_phys
        metrics = {
            "loss": total.detach(),
            "loss_main": loss_main.detach(),
            "loss_balance": loss_balance.detach(),
            "loss_phys": loss_phys.detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_one_hour(
        self,
        current_tec: torch.Tensor,
        tec_hist: torch.Tensor,
        bg_scalar_hist: torch.Tensor,
        bg_map_hist: torch.Tensor,
        future_terms_hour: torch.Tensor,
        future_uv_hour: torch.Tensor,
        future_rhs_hour: torch.Tensor = None,
        steps: int = None,
    ):
        self.model.eval()
        steps = int(self.ode_steps if steps is None else steps)
        x = current_tec
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            tau = torch.full((x.shape[0], 1, 1, 1), i * dt, device=x.device, dtype=x.dtype)
            v1 = self.model(
                x_tau=x,
                tau=tau,
                tec_hist=tec_hist,
                bg_scalar_hist=bg_scalar_hist,
                bg_map_hist=bg_map_hist,
                future_terms_hour=future_terms_hour,
                future_uv_hour=future_uv_hour,
                future_rhs_hour=future_rhs_hour,
            )
            x_euler = x + dt * v1
            tau2 = torch.full((x.shape[0], 1, 1, 1), min((i + 1) * dt, 1.0), device=x.device, dtype=x.dtype)
            v2 = self.model(
                x_tau=x_euler,
                tau=tau2,
                tec_hist=tec_hist,
                bg_scalar_hist=bg_scalar_hist,
                bg_map_hist=bg_map_hist,
                future_terms_hour=future_terms_hour,
                future_uv_hour=future_uv_hour,
                future_rhs_hour=future_rhs_hour,
            )
            x = x + dt * 0.5 * (v1 + v2)
        return x

    @torch.no_grad()
    def rollout_hours(self, batch: Dict[str, torch.Tensor], rollout_hours: int):
        self.model.eval()
        tec_hist = batch["tec_hist"]
        bg_scalar_hist = batch["bg_scalar_hist"]
        bg_map_hist = batch["bg_map_hist"]
        current = batch["current_tec"]
        preds = []

        for hour in range(int(rollout_hours)):
            future_terms_hour = batch["future_terms_rollout"][:, hour]
            future_uv_hour = batch["future_uv_rollout"][:, hour]
            future_rhs_hour = batch["future_rhs_rollout"][:, hour] if "future_rhs_rollout" in batch else None
            pred_next = self.sample_one_hour(
                current_tec=current,
                tec_hist=tec_hist,
                bg_scalar_hist=bg_scalar_hist,
                bg_map_hist=bg_map_hist,
                future_terms_hour=future_terms_hour,
                future_uv_hour=future_uv_hour,
                future_rhs_hour=future_rhs_hour,
            )
            preds.append(pred_next)
            tec_hist = torch.cat([tec_hist[:, 1:], pred_next], dim=1)
            current = pred_next
            if bg_scalar_hist.shape[1] > 0:
                bg_scalar_hist = torch.cat([bg_scalar_hist[:, :, 1:], bg_scalar_hist[:, :, -1:]], dim=2)
            if bg_map_hist.shape[1] > 0:
                bg_map_hist = torch.cat([bg_map_hist[:, :, 1:], bg_map_hist[:, :, -1:]], dim=2)

        return torch.cat(preds, dim=1)

    def train_one_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(loader):
            batch = self._move_batch(batch)
            if step == 0:
                nonfinite = self._count_nonfinite(batch)
                if nonfinite:
                    warnings.warn(
                        f"Non-finite entries in first training batch: {nonfinite}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                loss, _ = self._compute_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            bsz = batch["tec_hist"].shape[0]
            total += float(loss.item()) * bsz
            n += bsz
        return self._sync_sum_count(total, n)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            batch = self._move_batch(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                loss, _ = self._compute_loss(batch)
            bsz = batch["tec_hist"].shape[0]
            total += float(loss.item()) * bsz
            n += bsz
        return self._sync_sum_count(total, n)

    @torch.no_grad()
    def validate_rollout(self, loader, rollout_hours: int, max_batches: int = 0):
        self.model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        n = 0
        for step, batch in enumerate(loader):
            if max_batches > 0 and step >= int(max_batches):
                break
            batch = self._move_batch(batch)
            pred = self.rollout_hours(batch, rollout_hours=rollout_hours)
            target = batch["target_hourly"][:, :rollout_hours]
            mse_sum += float(((pred - target) ** 2).mean().item()) * pred.shape[0]
            mae_sum += float((pred - target).abs().mean().item()) * pred.shape[0]
            n += pred.shape[0]
        return self._sync_rollout_metrics(mse_sum, mae_sum, n)
