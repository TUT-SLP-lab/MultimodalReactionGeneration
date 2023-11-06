from collections import OrderedDict
import torch
from torch import nn

from mr_gen.model.utils import ResidualConnection


class LSTMModule(nn.Module):
    def __init__(
        self,
        input_size=256,
        hidden_size=128,
        num_layers=1,
        output_size=256,
        dropout=0.0,
        bidirectional=True,
        use_mixing=True,
    ):
        super().__init__()
        self.lstm_module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.mixer = nn.Linear(lstm_out_size, output_size) if use_mixing else None

        # check feature size
        if not use_mixing and lstm_out_size != output_size:
            raise ValueError(
                "lstm_out_size must be equal to output_size when use_mixing is False."
            )

    def forward(self, input_tensor) -> torch.Tensor:
        hs, (_h, _c) = self.lstm_module(input_tensor)
        if self.mixer is not None:
            y = self.mixer(hs)
        else:
            y = hs
        return y


class LSTMBlock(nn.Module):
    def __init__(
        self,
        input_size=256,
        hidden_size=128,
        lstm_out_size=256,
        num_layers=1,
        bottleneck_size=64,
        output_size=256,
        dropout=0.0,
        bidirectional=True,
        use_layer_norm=True,
        use_relu=True,
        use_mixing=False,
        use_residual=True,
    ) -> None:
        super().__init__()
        if use_residual and input_size != lstm_out_size or lstm_out_size != output_size:
            raise ValueError(
                "input_size must be equal to lstm_out_size and output_size when use_residuals."
            )

        self.lstm_module = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=lstm_out_size,
            dropout=dropout,
            bidirectional=bidirectional,
            use_mixing=use_mixing,
        )

        module_list = OrderedDict()
        module_list["input"] = nn.Linear(lstm_out_size, bottleneck_size)
        if use_relu:
            module_list["relu"] = nn.ReLU()
        module_list["mapping"] = nn.Linear(bottleneck_size, output_size)
        self.feed_forward_module = nn.Sequential(module_list)

        if use_residual:
            self.lstm_module = ResidualConnection(
                self.lstm_module, use_layer_norm, lstm_out_size, dropout
            )
            self.feed_forward_module = ResidualConnection(
                self.feed_forward_module, use_layer_norm, output_size, dropout
            )

    def forward(self, input_tenor: torch.Tensor) -> torch.Tensor:
        h = self.lstm_module(input_tenor)
        y = self.feed_forward_module(h)
        return y


class LSTMLayerd(nn.Module):
    def __init__(
        self,
        input_size=256,
        lstm_hidden_size=128,
        affine_hidden_size=256,
        bottleneck_size=64,
        num_layers=2,
        num_layers_per_block=1,
        output_size=256,
        dropout=0.0,
        bidirectional=True,
        use_layer_norm=True,
        use_relu=True,
        use_mixing=False,
        use_residual=True,
    ):
        super().__init__()
        self.lstm_layered = nn.ModuleList()

        for i in range(num_layers):
            _input_size = input_size if i == 0 else affine_hidden_size
            _output_size = output_size if i == num_layers - 1 else affine_hidden_size

            self.lstm_layered.append(
                LSTMBlock(
                    input_size=_input_size,
                    hidden_size=lstm_hidden_size,
                    lstm_out_size=affine_hidden_size,
                    num_layers=num_layers_per_block,
                    bottleneck_size=bottleneck_size,
                    output_size=_output_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    use_layer_norm=use_layer_norm,
                    use_relu=use_relu,
                    use_mixing=use_mixing,
                    use_residual=use_residual,
                )
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        for lstm_module in self.lstm_layered:
            input_tensor = lstm_module(input_tensor)
        return input_tensor
