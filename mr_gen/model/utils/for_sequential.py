import torch
from torch import nn

from mr_gen.model.utils.nonlinearity import set_nonlinearity
from mr_gen.model.utils.types import MhaIoType, NonlinearityType


class MHAforSequentail(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = False,
        nonlinearity: NonlinearityType = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        mha_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **mha_conf,
        )
        self.nonlinearity = set_nonlinearity(nonlinearity)
        if self.nonlinearity is not None:
            self.nonlinearity = set_nonlinearity(nonlinearity)()

    def forward(self, x: MhaIoType) -> torch.Tensor:
        x = list(x)
        x[4] = False  # need_weights
        x[6] = False  # average_attn_weights

        x = self.mha(*x)
        if self.nonlinearity is not None:
            x = self.nonlinearity(x[0])

        return x
