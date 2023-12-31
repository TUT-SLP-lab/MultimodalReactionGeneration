from torch import nn
import pytorch_lightning as pl


class ResidualConnection(pl.LightningModule):
    def __init__(
        self, module: nn.Module, use_layer_norm=True, num_nodes: int = -1, dropout=0.0
    ):
        super().__init__()
        if use_layer_norm and num_nodes == -1:
            raise ValueError(
                "num_nodes must be specified when use_layer_norm is set to True."
            )

        self.module = module
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(num_nodes) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        y = self.module(x, *args, **kwargs)

        # for LSTM, and so on
        others = None
        if isinstance(y, (tuple, list)):
            others = y[1:]
            y = y[0]

        y = y + x
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        y = self.dropout(y)

        if others is not None:
            y = (y, *others)

        return y
