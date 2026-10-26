"""Recurrent residual forecasting of the physical TEC tendency."""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = int(kernel_size) // 2
        in_ch = int(input_channels) + int(hidden_channels)
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(in_ch, 2 * self.hidden_channels, kernel_size=kernel_size, padding=padding)
        self.candidate = nn.Conv2d(in_ch, self.hidden_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        z, r = gates.chunk(2, dim=1)
        cand = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * cand


class ConvGRUStack(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: Sequence[int], kernel_size: int = 3):
        super().__init__()
        hidden_channels = [int(ch) for ch in hidden_channels]
        if not hidden_channels:
            raise ValueError("hidden_channels must not be empty")
        cells: List[ConvGRUCell] = []
        in_ch = int(input_channels)
        for h_ch in hidden_channels:
            cells.append(ConvGRUCell(in_ch, h_ch, kernel_size=kernel_size))
            in_ch = h_ch
        self.cells = nn.ModuleList(cells)
        self.hidden_channels = hidden_channels

    def init_state(self, batch: int, height: int, width: int, device, dtype) -> List[torch.Tensor]:
        return [
            torch.zeros((batch, ch, height, width), device=device, dtype=dtype)
            for ch in self.hidden_channels
        ]

    def step(self, x: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        new_states: List[torch.Tensor] = []
        out = x
        for cell, h in zip(self.cells, states):
            h_new = cell(out, h)
            new_states.append(h_new)
            out = h_new
        return out, new_states

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: (B,T,C,H,W)
        returns:
          outputs: (B,T,H_last,H,W)
          states: final hidden states
        """
        bsz, steps, _, height, width = x.shape
        states = self.init_state(bsz, height, width, x.device, x.dtype)
        outputs = []
        for t in range(steps):
            out, states = self.step(x[:, t], states)
            outputs.append(out)
        return torch.stack(outputs, dim=1), states


class PhysicsConvGRUForecaster(nn.Module):
    """
    Forecasts future normalized physics residuals around a causal baseline.

    Inputs:
      enc_seq: (B,Tenc,Cenc,H,W), historical TEC/physics/background/time maps
      dec_seq: (B,Tdec,Cdec,H,W), future baseline physics + causal aux maps

    Output:
      residual: (B,Tdec,Cout,H,W), added to the normalized baseline physics.
    """

    def __init__(
        self,
        enc_channels: int,
        dec_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int] = (32, 32),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_channels = tuple(int(ch) for ch in hidden_channels)
        self.encoder = ConvGRUStack(enc_channels, hidden_channels, kernel_size=kernel_size)
        self.decoder = ConvGRUStack(dec_channels, hidden_channels, kernel_size=kernel_size)
        head_in = int(hidden_channels[-1])
        self.head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(head_in, int(out_channels), kernel_size=1),
        )

    def forward(self, enc_seq: torch.Tensor, dec_seq: torch.Tensor) -> torch.Tensor:
        _, states = self.encoder(enc_seq)
        outputs = []
        dec_states = states
        for t in range(dec_seq.shape[1]):
            out, dec_states = self.decoder.step(dec_seq[:, t], dec_states)
            outputs.append(self.head(out))
        return torch.stack(outputs, dim=1)
