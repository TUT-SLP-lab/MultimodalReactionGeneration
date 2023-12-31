from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union, Any
from collections import OrderedDict
import torch
from torch import nn

from mr_gen.model.utils.residual_connection import ResidualConnection
from mr_gen.model.utils.for_sequential import MHAforSequentail
from mr_gen.model.utils.types import (
    LSTMStateType,
    NonlinearityType,
    MixerType,
    MHABlockReturnType,
    MHAStateType,
    MHALayerdReturnType,
)
from mr_gen.model.utils.types import MixerReturnType, BlockReturnType, LayerdReturnType
from mr_gen.model.utils.nonlinearity import set_nonlinearity


def split_state(state: list, prev_state: list):
    # for load state in nn.Sequential or when use model in auto-regressive
    _state, state = (None, None) if state is None else (state[:1], state[1:])
    if _state is not None:
        _state = None if _state == [] and _state else _state[0]
    if state is not None:
        state = None if state == [] and state else state
    prev_state = [] if prev_state is None else prev_state

    return _state, state, prev_state


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for each block's self.feed_forward
class FeedForward(nn.Module):
    """FeedForward layer that have input and output linear & nonlinearity between them."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = None,
        output_size: int = None,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        device = device if device is not None else DEVICE
        ff_conf = {"bias": bias, "device": device, "dtype": dtype}

        bottleneck_size = hidden_size if bottleneck_size is None else bottleneck_size
        output_size = hidden_size if output_size is None else output_size

        if hidden_size != output_size and residual:
            raise ValueError(
                "hidden_size must be equal to output_size when residual is True."
            )
        if nonlinearity is None or nonlinearity == "none":
            architecure = OrderedDict(
                [
                    ("feedforward", nn.Linear(hidden_size, output_size, **ff_conf)),
                ]
            )
        else:
            architecure = OrderedDict(
                [
                    ("input", nn.Linear(hidden_size, bottleneck_size, **ff_conf)),
                    ("activation", set_nonlinearity(nonlinearity)()),
                    ("output", nn.Linear(bottleneck_size, output_size, **ff_conf)),
                ]
            )

        self.feed_forward = nn.Sequential(architecure)

        if residual:
            self.feed_forward = ResidualConnection(
                self.feed_forward, residual_layer_norm, hidden_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.feed_forward(x)
        return y


class Mixer(nn.Module, metaclass=ABCMeta):
    """Mixer interface."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> MixerReturnType:
        raise NotImplementedError


class MixerBlock(nn.Module, metaclass=ABCMeta):
    """MixerBlock interface."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> BlockReturnType:
        raise NotImplementedError


class MixerLayerd(nn.Module, metaclass=ABCMeta):
    """MixerLayerd interface."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> LayerdReturnType:
        raise NotImplementedError


class MLPMixer(Mixer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layer: int = 1,
        nonlinearity: NonlinearityType = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        if num_layer < 1:
            raise ValueError("num_layer must be greater than 0.")
        device = device if device is not None else DEVICE
        linear_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.nonlinearity = set_nonlinearity(nonlinearity)

        # input block
        self.input_block = OrderedDict()
        self.input_block["input"] = nn.Linear(input_size, hidden_size, **linear_conf)
        if self.nonlinearity is not None:
            self.input_block["activation"] = self.nonlinearity()
        self.input_block = nn.Sequential(self.input_block)

        # hidden block
        mixer_hidden = []
        for _ in range(1, num_layer):
            self.hidden_block = OrderedDict()
            self.hidden_block["hidden"] = nn.Linear(
                hidden_size, hidden_size, **linear_conf
            )
            if self.nonlinearity is not None:
                self.hidden_block["activation"] = self.nonlinearity()
            self.hidden_block = nn.Sequential(self.hidden_block)
            mixer_hidden.append(self.hidden_block)

        # output block
        self.output_block = nn.Linear(hidden_size, hidden_size, **linear_conf)

        # Integrate all blocks
        self.mixer = OrderedDict()
        self.mixer["input"] = self.input_block
        for i, hidden_block in enumerate(mixer_hidden):
            self.mixer[f"hidden[{i+1}]"] = hidden_block
        self.mixer["output"] = self.output_block
        self.mixer = nn.Sequential(self.mixer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mixer(x)
        return y


class GRUMixer(Mixer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be greater than 0.")
        if bidirectional:
            if hidden_size % 2 != 0:
                raise ValueError(
                    "hidden_size must be even number when bidirectional is True."
                )
            hidden_size = hidden_size // 2
        device = device if device is not None else DEVICE
        gru_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            **gru_conf,
        )

    def forward(
        self, x: torch.Tensor, hn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y, hn = self.mixer(x, hn)
        return y, hn


class LSTMMixer(Mixer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be greater than 0.")
        if bidirectional:
            if hidden_size % 2 != 0:
                raise ValueError(
                    "hidden_size must be even number when bidirectional is True."
                )
            hidden_size = hidden_size // 2
        device = device if device is not None else DEVICE
        lstm_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            **lstm_conf,
        )

    def forward(
        self, x: torch.Tensor, hn: LSTMStateType
    ) -> Tuple[torch.Tensor, LSTMStateType]:
        y, hn = self.mixer(x, hn)
        return y, hn


class MHAMixer(Mixer):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = False,
        nonlinearity: NonlinearityType = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be greater than 0.")
        device = device if device is not None else DEVICE
        mha_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = nn.ModuleList()
        for _ in range(num_layers):
            self.mixer.append(
                MHAforSequentail(
                    embed_dim=input_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    kdim=kdim,
                    vdim=vdim,
                    batch_first=batch_first,
                    nonlinearity=nonlinearity,
                    **mha_conf,
                )
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = (q, k, v, None, False, attn_mask, False, False)
        for mixer in self.mixer:
            x = mixer(x)
        return x[0]


class MLPMixerBlock(MixerBlock):
    """Supported by nn.Sequential & auto-regressive & ResidualConnection"""

    def __init__(
        self,
        hidden_size: int,
        num_layer: int = 1,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        block_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = MLPMixer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layer=num_layer,
            nonlinearity=nonlinearity,
            **block_conf,
        )

        if residual:
            self.mixer = ResidualConnection(
                self.mixer, residual_layer_norm, hidden_size
            )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            nonlinearity=nonlinearity,
            residual=residual,
            residual_layer_norm=residual_layer_norm,
            **block_conf,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mixer(x)
        y = self.feed_forward(y)
        return y


class GRUMixerBlock(MixerBlock):
    """Supported by nn.Sequential & auto-regressive & ResidualConnection"""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        bidirectional: bool = False,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        block_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = GRUMixer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            **block_conf,
        )

        if residual:
            self.mixer = ResidualConnection(
                self.mixer, residual_layer_norm, hidden_size
            )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            nonlinearity=nonlinearity,
            residual=residual,
            residual_layer_norm=residual_layer_norm,
            **block_conf,
        )

    def forward(
        self,
        x: Union[torch.Tensor, tuple],
        hx: List[torch.Tensor] = None,
        prev_hx: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # unpack x when x is tuple
        if isinstance(x, tuple):
            x, hx, prev_hx = x
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x must be torch.Tensor or tuple or list, but got {type(x)}."
            )

        # for load cell state in nn.Sequential or when use model in auto-regressive
        _hx, hx = (None, None) if hx is None else (hx[:1], hx[1:])
        _hx = None if _hx == [] else _hx
        hx = None if hx == [] else hx
        prev_hx = [] if prev_hx is None else prev_hx

        # forward
        y, _hx = self.mixer(x, _hx)
        y = self.feed_forward(y)

        # record cell state
        prev_hx.append(_hx)

        return (y, hx, prev_hx)


class LSTMMixerBlock(MixerBlock):
    """Supported by nn.Sequential & auto-regressive & ResidualConnection"""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        bidirectional: bool = False,
        proj_size: int = 0,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        block_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = LSTMMixer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            **block_conf,
        )

        if residual:
            self.mixer = ResidualConnection(
                self.mixer, residual_layer_norm, hidden_size
            )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            nonlinearity=nonlinearity,
            residual=residual,
            residual_layer_norm=residual_layer_norm,
            **block_conf,
        )

    def forward(
        self,
        x: Union[torch.Tensor, tuple],
        hx: List[LSTMStateType] = None,
        prev_hx: List[LSTMStateType] = None,
    ) -> Tuple[torch.Tensor, List[LSTMStateType], List[LSTMStateType]]:
        # unpack x when x is tuple
        if isinstance(x, (tuple, list)):
            x, hx, prev_hx = x
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x must be torch.Tensor or tuple or list, but got {type(x)}."
            )

        # for load cell state in nn.Sequential or when use model in auto-regressive
        _hx, hx, prev_hx = split_state(hx, prev_hx)
        # _hx, hx = (None, None) if hx is None else (hx[:1], hx[1:])
        # _hx = None if _hx == [] else _hx
        # hx = None if hx == [] else hx
        # prev_hx = [] if prev_hx is None else prev_hx

        # forward
        y, _hx = self.mixer(x, _hx)
        y = self.feed_forward(y)

        # record cell state
        prev_hx.append(_hx)

        return (y, hx, prev_hx)


class MHAMixerBlock(MixerBlock):
    """Supported by nn.Sequential & auto-regressive & ResidualConnection"""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        max_context_len: int = 125,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        block_conf = {"bias": bias, "device": device, "dtype": dtype}

        self.mixer = MHAMixer(
            input_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            nonlinearity=nonlinearity,
            **block_conf,
        )

        if residual:
            self.mixer = ResidualConnection(
                self.mixer, residual_layer_norm, hidden_size
            )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            nonlinearity=nonlinearity,
            residual=residual,
            residual_layer_norm=residual_layer_norm,
            **block_conf,
        )

        self.max_context_len = max_context_len

    def forward(
        self,
        query: Union[torch.Tensor, tuple],
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        hx: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_hx: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> MHABlockReturnType:
        # unpack query when query is tuple
        if isinstance(query, (tuple, list)):
            query, key, value, attn_mask, hx, prev_hx = query
        elif not isinstance(query, torch.Tensor):
            raise TypeError(
                f"query must be torch.Tensor or tuple or list, but got {type(query)}."
            )

        # for load state in nn.Sequential or when use model in auto-regressive
        _hx, hx, prev_hx = split_state(hx, prev_hx)
        # _hx, hx = (None, None) if hx is None else (hx[:1], hx[1:])
        # _hx = None if _hx == [] else _hx[0]
        # hx = None if hx == [] else hx
        # prev_hx = [] if prev_hx is None else prev_hx

        # unpack hx when hx is tuple
        if isinstance(_hx, (tuple, list)) and not self.training:
            prev_key, prev_val = _hx
            key = torch.cat([prev_key, key], dim=1)[-self.max_context_len :]
            value = torch.cat([prev_val, value], dim=1)[-self.max_context_len :]

        # forward
        x = self.mixer(query, key, value, attn_mask)
        x = self.feed_forward(x)

        prev_hx.append((key, value))

        return (x, key, value, attn_mask, hx, prev_hx)


class MLPMixerLayerd(MixerLayerd):
    def __init__(
        self,
        hidden_size: int,
        input_projection: bool = False,
        input_projection_size: int = None,
        output_projection: bool = False,
        output_projection_size: int = None,
        num_layerd: int = 1,
        num_internal_layer: int = 1,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        layerd_conf = {"bias": bias, "device": device, "dtype": dtype}

        # build input projection
        self.input_projection = None
        if input_projection and input_projection_size is None:
            raise ValueError(
                "input_projection_size must be specified when input_projection is True."
            )
        if input_projection:
            self.input_projection = nn.Linear(
                input_projection_size, hidden_size, **layerd_conf
            )

        # build output projection
        self.output_projection = None
        if output_projection and output_projection_size is None:
            raise ValueError(
                "output_projection_size must be specified when output_projection is True."
            )
        if output_projection:
            self.output_projection = nn.Linear(
                output_projection_size, hidden_size, **layerd_conf
            )

        blocks = []
        for _ in range(num_layerd):
            blocks.append(
                MLPMixerBlock(
                    hidden_size=hidden_size,
                    num_layer=num_internal_layer,
                    nonlinearity=nonlinearity,
                    residual=residual,
                    residual_layer_norm=residual_layer_norm,
                    bottleneck_size=bottleneck_size,
                    **layerd_conf,
                )
            )
        self.mixer = OrderedDict()
        for i, block in enumerate(blocks):
            self.mixer[f"block[{i+1}]"] = block
        self.mixer = nn.Sequential(self.mixer)

    def forward(
        self, x: torch.Tensor, hx=None, other=(None,)
    ) -> Tuple[torch.Tensor, Any, Tuple[Any]]:
        if self.input_projection is not None:
            x = self.input_projection(x)
        x = self.mixer(x)
        if self.output_projection is not None:
            x = self.output_projection(x)
        return (x, hx, other)


class GRUMixerLayerd(MixerLayerd):
    def __init__(
        self,
        hidden_size: int,
        input_projection: bool = False,
        input_projection_size: int = None,
        output_projection: bool = False,
        output_projection_size: int = None,
        num_layerd: int = 1,
        num_internal_layer: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        bidirectional: bool = False,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        layerd_conf = {"bias": bias, "device": device, "dtype": dtype}

        # build input projection
        self.input_projection = None
        if input_projection and input_projection_size is None:
            raise ValueError(
                "input_projection_size must be specified when input_projection is True."
            )
        if input_projection:
            self.input_projection = nn.Linear(
                input_projection_size, hidden_size, **layerd_conf
            )

        # build output projection
        self.output_projection = None
        if output_projection and output_projection_size is None:
            raise ValueError(
                "output_projection_size must be specified when output_projection is True."
            )
        if output_projection:
            self.output_projection = nn.Linear(
                output_projection_size, hidden_size, **layerd_conf
            )

        blocks = []
        for _ in range(num_layerd):
            blocks.append(
                GRUMixerBlock(
                    hidden_size=hidden_size,
                    num_layers=num_internal_layer,
                    dropout=dropout,
                    batch_first=batch_first,
                    bidirectional=bidirectional,
                    nonlinearity=nonlinearity,
                    residual=residual,
                    residual_layer_norm=residual_layer_norm,
                    bottleneck_size=bottleneck_size,
                    **layerd_conf,
                )
            )
        self.mixer = nn.ModuleList(blocks)
        # self.mixer = OrderedDict()
        # for i, block in enumerate(blocks):
        #     self.mixer[f"block[{i+1}]"] = block
        # self.mixer = nn.Sequential(self.mixer)

    def forward(
        self, x: torch.Tensor, hx: List[torch.Tensor] = None, other=(None,)
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[Any]]:
        if self.input_projection is not None:
            x = self.input_projection(x)
        phx = None
        for block in self.mixer:
            x, hx, phx = block(x, hx, phx)
        # x, _, hx = self.mixer(x, hx)
        if self.output_projection is not None:
            x = self.output_projection(x)
        return (x, hx, other)


class LSTMMixerLayerd(MixerLayerd):
    def __init__(
        self,
        hidden_size: int,
        input_projection: bool = False,
        input_projection_size: int = None,
        output_projection: bool = False,
        output_projection_size: int = None,
        num_layerd: int = 1,
        num_internal_layer: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        bidirectional: bool = False,
        proj_size: int = 0,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        layerd_conf = {"bias": bias, "device": device, "dtype": dtype}

        # build input projection
        self.input_projection = None
        if input_projection and input_projection_size is None:
            raise ValueError(
                "input_projection_size must be specified when input_projection is True."
            )
        if input_projection:
            self.input_projection = nn.Linear(
                input_projection_size, hidden_size, **layerd_conf
            )

        # build output projection
        self.output_projection = None
        if output_projection and output_projection_size is None:
            raise ValueError(
                "output_projection_size must be specified when output_projection is True."
            )
        if output_projection:
            self.output_projection = nn.Linear(
                output_projection_size, hidden_size, **layerd_conf
            )

        blocks = []
        for _ in range(num_layerd):
            blocks.append(
                LSTMMixerBlock(
                    hidden_size=hidden_size,
                    num_layers=num_internal_layer,
                    dropout=dropout,
                    batch_first=batch_first,
                    bidirectional=bidirectional,
                    proj_size=proj_size,
                    nonlinearity=nonlinearity,
                    residual=residual,
                    residual_layer_norm=residual_layer_norm,
                    bottleneck_size=bottleneck_size,
                    **layerd_conf,
                )
            )
        self.mixer = nn.ModuleList(blocks)
        # self.mixer = OrderedDict()
        # for i, block in enumerate(blocks):
        #     self.mixer[f"block[{i+1}]"] = block
        # self.mixer = nn.Sequential(self.mixer)

    def forward(
        self, x: torch.Tensor, hx: List[LSTMStateType] = None, other=(None,)
    ) -> Tuple[torch.Tensor, List[LSTMStateType], Tuple[Any]]:
        if self.input_projection is not None:
            x = self.input_projection(x)
        phx = None
        for block in self.mixer:
            x, hx, phx = block(x, hx, phx)
        if self.output_projection is not None:
            x = self.output_projection(x)
        return (x, hx, other)


class MHAMixerLayerd(MixerLayerd):
    def __init__(
        self,
        hidden_size: int,
        input_projection: bool = False,
        input_projection_size: int = None,
        self_attention: bool = False,
        output_projection: bool = False,
        output_projection_size: int = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        max_context_len: int = 125,
        num_layerd: int = 1,
        num_internal_layer: int = 1,
        nonlinearity: NonlinearityType = None,
        residual: bool = False,
        residual_layer_norm: bool = False,
        bottleneck_size: int = None,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        layerd_conf = {"bias": bias, "device": device, "dtype": dtype}

        # build input projection
        self.input_projection = None
        if input_projection and input_projection_size is None:
            raise ValueError(
                "input_projection_size must be specified when input_projection is True."
            )
        if input_projection:
            self.input_projection = nn.Linear(
                input_projection_size, hidden_size, **layerd_conf
            )

        # when self_attention is True, key and value are same as query
        # so, we need to apply input projection to key and value
        self.self_attention = self_attention

        # build output projection
        self.output_projection = None
        if output_projection and output_projection_size is None:
            raise ValueError(
                "output_projection_size must be specified when output_projection is True."
            )
        if output_projection:
            self.output_projection = nn.Linear(
                output_projection_size, hidden_size, **layerd_conf
            )

        blocks = []
        for _ in range(num_layerd):
            blocks.append(
                MHAMixerBlock(
                    hidden_size=hidden_size,
                    num_layers=num_internal_layer,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=batch_first,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    kdim=kdim,
                    vdim=vdim,
                    nonlinearity=nonlinearity,
                    residual=residual,
                    residual_layer_norm=residual_layer_norm,
                    bottleneck_size=bottleneck_size,
                    **layerd_conf,
                )
            )
        self.mixer = nn.ModuleList(blocks)
        # self.mixer = OrderedDict()
        # for i, block in enumerate(blocks):
        #     self.mixer[f"block[{i+1}]"] = block
        # self.mixer = nn.Sequential(self.mixer)

        self.max_context_len = max_context_len

    def forward(
        self,
        x: torch.Tensor,
        hx: List[MHAStateType] = None,
        key: Union[List[torch.Tensor], torch.Tensor] = None,
        value: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> MHALayerdReturnType:
        if isinstance(key, (tuple, list)):
            key, value, attn_mask = key
        elif not isinstance(key, torch.Tensor):
            raise TypeError(
                f"key must be torch.Tensor or tuple or list, but got {type(key)}."
            )

        query = x
        if self.input_projection is not None:
            query = self.input_projection(query)

        # when self_attention is True, key and value are same as query
        key, value = (query, query) if self.self_attention else (key, value)
        if key is None or value is None:
            raise ValueError(
                "key and value must be specified when self_attention is False."
            )

        phx = None
        for block in self.mixer:
            query, *_, hx, phx = block(query, key, value, attn_mask, hx, phx)

        if self.output_projection is not None:
            query = self.output_projection(query)
        return (query, hx, (key, value, attn_mask))


class MixerBlockFactory:
    def __init__(self):
        self._factory = {
            "mlp": self._mlp_mixer_block,
            "gru": self._gru_mixer_block,
            "lstm": self._lstm_mixer_block,
            "mha": self._mha_mixer_block,
        }

    def _mlp_mixer_block(self, **kwargs):
        return MLPMixerBlock(**kwargs)

    def _gru_mixer_block(self, **kwargs):
        return GRUMixerBlock(**kwargs)

    def _lstm_mixer_block(self, **kwargs):
        return LSTMMixerBlock(**kwargs)

    def _mha_mixer_block(self, **kwargs):
        return MHAMixerBlock(**kwargs)

    def build(self, mixer_type: MixerType, configs) -> MixerBlock:
        if mixer_type not in self._factory:
            raise ValueError(f"mixer_type must be in {self._factory.keys()}.")
        return self._factory[mixer_type](**configs)


class MixerLayerdFactory:
    def __init__(self):
        self._factory = {
            "mlp": self._mlp_mixer_layerd,
            "gru": self._gru_mixer_layerd,
            "lstm": self._lstm_mixer_layerd,
            "mha": self._mha_mixer_layerd,
        }

    def _mlp_mixer_layerd(self, **kwargs):
        return MLPMixerLayerd(**kwargs)

    def _gru_mixer_layerd(self, **kwargs):
        return GRUMixerLayerd(**kwargs)

    def _lstm_mixer_layerd(self, **kwargs):
        return LSTMMixerLayerd(**kwargs)

    def _mha_mixer_layerd(self, **kwargs):
        return MHAMixerLayerd(**kwargs)

    def build(self, mixer_type: MixerType, configs) -> MixerLayerd:
        if mixer_type not in self._factory:
            raise ValueError(f"mixer_type must be in {self._factory.keys()}.")
        return self._factory[mixer_type](**configs)
