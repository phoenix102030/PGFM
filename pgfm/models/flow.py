"""One-hour flow model with a learned spatial data–physics mixing weight."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import Down2D, ResBlock2D, Up2D, _match_size_2d


def _finite_zero_t(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _select_quarter_field(x: torch.Tensor, seg_idx: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, 4, H, W)
    seg_idx: (B,) int64 in [0, 3]
    returns: (B, C, H, W)
    """
    gather_idx = seg_idx.view(-1, 1, 1, 1, 1).expand(-1, x.shape[1], 1, x.shape[-2], x.shape[-1])
    return torch.gather(x, dim=2, index=gather_idx).squeeze(2)


class _SeqProj3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = max(out_ch, 8)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=(3, 3, 3), padding=1),
            nn.SiLU(),
            nn.Conv3d(mid, out_ch, kernel_size=(3, 3, 3), padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PiecewisePhysFlowMatchingUNet(nn.Module):
    """
    One-hour conditional flow-matching model.

    u = lambda * u_data + (1 - lambda) * u_phys
    lambda = (1 + exp(logit))^-1

    ``lambda_map`` weights the data branch; ``1 - lambda_map`` weights physics.
    Both heads share the U-Net trunk and are optimized by the same loss.
    """

    def __init__(
        self,
        past_hours: int,
        bg_scalar_channels: int,
        bg_map_channels: int,
        tec_std: float,
        hidden: int = 64,
        groups: int = 8,
        dropout: float = 0.0,
        hist_proj_ch: int = 16,
        bg_map_proj_ch: int = 12,
        bg_scalar_proj_ch: int = 8,
        quarter_dt_seconds: float = 900.0,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.past_hours = int(past_hours)
        self.bg_scalar_channels = int(bg_scalar_channels)
        self.bg_map_channels = int(bg_map_channels)
        self.bg_map_proj_ch = int(bg_map_proj_ch)
        self.bg_scalar_proj_ch = int(bg_scalar_proj_ch)
        self.hidden = int(hidden)
        self.tec_std = float(tec_std) if tec_std > 1e-8 else 1.0
        self.quarter_dt_seconds = float(quarter_dt_seconds)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.tec_hist_proj = _SeqProj3D(1, hist_proj_ch)
        if self.bg_map_channels > 0:
            self.bg_map_proj = _SeqProj3D(self.bg_map_channels, bg_map_proj_ch)
        else:
            self.bg_map_proj = None

        if self.bg_scalar_channels > 0:
            self.bg_scalar_proj = nn.Sequential(
                nn.Conv1d(self.bg_scalar_channels, bg_scalar_proj_ch, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(bg_scalar_proj_ch, bg_scalar_proj_ch, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        else:
            self.bg_scalar_proj = None

        base_in = 3 + hist_proj_ch + bg_map_proj_ch + bg_scalar_proj_ch
        h = self.hidden

        self.in_conv = nn.Conv2d(base_in, h, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(ResBlock2D(h, h, groups, dropout), ResBlock2D(h, h, groups, dropout))
        self.down1 = Down2D(h)
        self.enc2 = nn.Sequential(ResBlock2D(h, 2 * h, groups, dropout), ResBlock2D(2 * h, 2 * h, groups, dropout))
        self.down2 = Down2D(2 * h)
        self.enc3 = nn.Sequential(ResBlock2D(2 * h, 4 * h, groups, dropout), ResBlock2D(4 * h, 4 * h, groups, dropout))
        self.down3 = Down2D(4 * h)
        self.mid = nn.Sequential(ResBlock2D(4 * h, 4 * h, groups, dropout), ResBlock2D(4 * h, 4 * h, groups, dropout))
        self.up3 = Up2D(4 * h)
        self.dec3 = nn.Sequential(ResBlock2D(8 * h, 4 * h, groups, dropout), ResBlock2D(4 * h, 4 * h, groups, dropout))
        self.up2 = Up2D(4 * h)
        self.dec2 = nn.Sequential(ResBlock2D(6 * h, 2 * h, groups, dropout), ResBlock2D(2 * h, 2 * h, groups, dropout))
        self.up1 = Up2D(2 * h)
        self.dec1 = nn.Sequential(ResBlock2D(3 * h, h, groups, dropout), ResBlock2D(h, h, groups, dropout))
        self.out_norm = nn.GroupNorm(num_groups=min(groups, h), num_channels=h)

        self.u_data_head = nn.Conv2d(h, 1, kernel_size=1)
        self.lambda_head = nn.Conv2d(h, 1, kernel_size=1)
        self.uv_gate = nn.Sequential(
            nn.Conv2d(2, max(h // 4, 8), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(max(h // 4, 8), 1, kernel_size=1),
            nn.Tanh(),
        )

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _encode_context(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        tec_hist: torch.Tensor,
        bg_scalar_hist: torch.Tensor,
        bg_map_hist: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, height, width = x_tau.shape
        x_tau = _finite_zero_t(x_tau)
        tec_hist = _finite_zero_t(tec_hist)
        bg_scalar_hist = _finite_zero_t(bg_scalar_hist)
        bg_map_hist = _finite_zero_t(bg_map_hist)
        last_tec = tec_hist[:, -1:, :, :]
        tau_map = _finite_zero_t(tau.view(bsz, 1, 1, 1).expand(-1, 1, height, width))

        tec_hist_feat = self._maybe_checkpoint(self.tec_hist_proj, tec_hist.unsqueeze(1)).mean(dim=2)

        if self.bg_map_channels > 0:
            bg_map_feat = self._maybe_checkpoint(self.bg_map_proj, bg_map_hist).mean(dim=2)
        else:
            bg_map_feat = torch.zeros((bsz, self.bg_map_proj_ch, height, width), device=x_tau.device, dtype=x_tau.dtype)

        if self.bg_scalar_proj is not None:
            scalar_feat = self.bg_scalar_proj(bg_scalar_hist).mean(dim=-1, keepdim=True)
            scalar_feat = scalar_feat.view(bsz, -1, 1, 1).expand(-1, -1, height, width)
        else:
            scalar_feat = torch.zeros((bsz, self.bg_scalar_proj_ch, height, width), device=x_tau.device, dtype=x_tau.dtype)

        return _finite_zero_t(torch.cat([x_tau, last_tec, tau_map, tec_hist_feat, bg_map_feat, scalar_feat], dim=1))

    def _run_unet(self, x: torch.Tensor) -> torch.Tensor:
        x = _finite_zero_t(x)
        x0 = self.in_conv(x)
        e1 = self._maybe_checkpoint(self.enc1, x0)
        d1 = self.down1(e1)
        e2 = self._maybe_checkpoint(self.enc2, d1)
        d2 = self.down2(e2)
        e3 = self._maybe_checkpoint(self.enc3, d2)
        d3 = self.down3(e3)
        m = self._maybe_checkpoint(self.mid, d3)
        u3 = self.up3(m)
        u3 = _match_size_2d(u3, e3)
        u3 = self._maybe_checkpoint(self.dec3, torch.cat([u3, e3], dim=1))
        u2 = self.up2(u3)
        u2 = _match_size_2d(u2, e2)
        u2 = self._maybe_checkpoint(self.dec2, torch.cat([u2, e2], dim=1))
        u1 = self.up1(u2)
        u1 = _match_size_2d(u1, e1)
        u1 = self._maybe_checkpoint(self.dec1, torch.cat([u1, e1], dim=1))
        return _finite_zero_t(F.silu(self.out_norm(u1)))

    def _build_u_phys(
        self,
        future_terms_hour: torch.Tensor,
        future_uv_hour: torch.Tensor,
        tau: torch.Tensor,
        future_rhs_hour: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        seg_idx = torch.clamp((tau.view(-1) * 4.0).long(), min=0, max=3)

        if future_rhs_hour is not None:
            rhs = _finite_zero_t(_select_quarter_field(future_rhs_hour, seg_idx))
            unused_param_anchor = rhs.new_zeros(())
            for param in self.uv_gate.parameters():
                unused_param_anchor = unused_param_anchor + param.sum() * 0.0
            u_phys = _finite_zero_t(rhs * (4.0 * self.quarter_dt_seconds / self.tec_std))
            zeros = torch.zeros_like(rhs)
            return {
                "u_phys": u_phys + unused_param_anchor,
                "rhs": rhs + unused_param_anchor,
                "gamma": torch.ones_like(rhs) + unused_param_anchor,
                "prod": rhs,
                "loss": zeros,
                "drift": zeros,
                "wind": zeros,
            }

        term_sel = _finite_zero_t(_select_quarter_field(future_terms_hour, seg_idx))
        uv_sel = _finite_zero_t(_select_quarter_field(future_uv_hour, seg_idx))

        prod = term_sel[:, 0:1]
        loss = term_sel[:, 1:2]
        drift = term_sel[:, 2:3]
        wind = term_sel[:, 3:4]

        gamma = 1.0 + self.uv_gate(uv_sel)
        rhs = prod + loss + gamma * drift + wind
        u_phys = _finite_zero_t(rhs * (4.0 * self.quarter_dt_seconds / self.tec_std))

        return {
            "u_phys": u_phys,
            "rhs": rhs,
            "gamma": gamma,
            "prod": prod,
            "loss": loss,
            "drift": drift,
            "wind": wind,
        }

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        tec_hist: torch.Tensor,
        bg_scalar_hist: torch.Tensor,
        bg_map_hist: torch.Tensor,
        future_terms_hour: torch.Tensor,
        future_uv_hour: torch.Tensor,
        future_rhs_hour: torch.Tensor = None,
        return_aux: bool = False,
    ):
        context = self._encode_context(
            x_tau=x_tau,
            tau=tau,
            tec_hist=tec_hist,
            bg_scalar_hist=bg_scalar_hist,
            bg_map_hist=bg_map_hist,
        )
        trunk = self._run_unet(context)
        u_data = _finite_zero_t(self.u_data_head(trunk))
        lambda_logits = self.lambda_head(trunk).clamp(min=-30.0, max=30.0)
        lambda_map = _finite_zero_t(1.0 / (1.0 + torch.exp(lambda_logits)))

        phys_out = self._build_u_phys(
            future_terms_hour=future_terms_hour,
            future_uv_hour=future_uv_hour,
            tau=tau,
            future_rhs_hour=future_rhs_hour,
        )
        u = _finite_zero_t(lambda_map * u_data + (1.0 - lambda_map) * phys_out["u_phys"])

        if return_aux:
            aux = {
                "u_data": u_data,
                "u_phys": phys_out["u_phys"],
                "lambda_map": lambda_map,
                "lambda_logits": lambda_logits,
                **phys_out,
            }
            return u, aux
        return u
