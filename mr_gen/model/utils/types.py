from typing import Tuple, Optional, Union, Iterable, List, Dict
from typing_extensions import Literal, TypeAlias
import torch
from torch import nn

MhaIoType: TypeAlias = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    bool,
    Optional[torch.Tensor],
    bool,
    bool,
]

AvailableMixers = ["mlp", "gru", "lstm", "mha"]
MixerType: TypeAlias = Literal["mlp", "gru", "lstm", "mha"]
MLPMixerType: TypeAlias = Literal["mlp"]
GRUMixerType: TypeAlias = Literal["gru"]
LSTMMixerType: TypeAlias = Literal["lstm"]
MHAMixerType: TypeAlias = Literal["mha"]

AllowedNonlinearity = ["relu", "swish", "tanh", None]
NonlinearityType: TypeAlias = Literal["relu", "swish", "tanh", "none", None]
Nonlinearity: TypeAlias = Union[nn.ReLU, nn.SiLU, nn.Tanh, None]

LSTMStateType = Tuple[torch.Tensor, torch.Tensor]
LSTMLayerStateType = List[LSTMStateType]

KeyStateType: TypeAlias = torch.Tensor
ValStateType: TypeAlias = torch.Tensor
MHAStateType: TypeAlias = Tuple[KeyStateType, ValStateType]
MHALayerStateType: TypeAlias = List[MHAStateType]
MHALayerdReturnType: TypeAlias = Tuple[
    torch.Tensor, List[MHAStateType], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]

ModalsDimType: TypeAlias = Union[int, Iterable[int]]
ModalsMixerType: TypeAlias = Union[MixerType, Iterable[MixerType]]
ModalsHyperParamsType: TypeAlias = Union[dict, Iterable[dict]]
Modals_MixerHParams: TypeAlias = List[Tuple[MixerType, dict]]

# forward return types
RNNMixerReturnType: TypeAlias = Tuple[torch.Tensor, Union[LSTMStateType, torch.Tensor]]
MixerReturnType: TypeAlias = Union[torch.Tensor, RNNMixerReturnType]

RNNBlockReturnType: TypeAlias = Tuple[
    torch.Tensor,
    Union[LSTMLayerStateType, List[torch.Tensor]],
    Union[LSTMLayerStateType, List[torch.Tensor]],
]
MHABlockReturnType: TypeAlias = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool
]
BlockReturnType: TypeAlias = Union[torch.Tensor, RNNBlockReturnType, MHABlockReturnType]

RNNLayerdReturnType: TypeAlias = Tuple[
    torch.Tensor, Union[LSTMLayerStateType, torch.Tensor]
]

LayerdStateType: TypeAlias = Union[
    List[LSTMStateType], List[torch.Tensor], List[MHAStateType], None
]

LayerdReturnType = Tuple[torch.Tensor, LayerdStateType, tuple]

MetaformerBlocks: TypeAlias = Literal["emb", "crm"]
MetaformerBlockStateType: TypeAlias = Dict[MetaformerBlocks, List[LayerdStateType]]
MetaformerStateType: TypeAlias = Dict[MetaformerBlocks, LayerdStateType]
MetaformerBlockReturnType: TypeAlias = Tuple[
    torch.Tensor,
    torch.Tensor,
    List[MetaformerStateType],
    List[MetaformerStateType],
    Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None],
    List[Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]],
    List[torch.Tensor],
]
