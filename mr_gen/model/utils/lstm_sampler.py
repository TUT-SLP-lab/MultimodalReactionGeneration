from typing import Tuple
import torch
from torch import nn


class LSTMSampler(nn.Module):
    def __init__(
        self,
        hiddn_size: int,
        num_layers: int,
        dropout: float,
        decline_rate: int,
        bidirectional=False,
    ):
        super().__init__()
        self.sampler = nn.LSTM(
            input_size=hiddn_size,
            hidden_size=hiddn_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.decline_rate = decline_rate

    def forward(
        self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        h, hx = self.sampler(x, hx)

        # [batch_size, seq_len, hiddn_size] -> [batch_size, seq_len // decline_rate, hiddn_size]
        sampled_h: torch.Tensor = h[:, self.decline_rate - 1 :: self.decline_rate, :]

        return sampled_h.contiguous(), hx
