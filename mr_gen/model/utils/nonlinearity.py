from torch import nn

from mr_gen.model.utils.types import NonlinearityType, Nonlinearity, AllowedNonlinearity


def set_nonlinearity(nonlinearity: NonlinearityType) -> Nonlinearity:
    if nonlinearity == "relu":
        return nn.ReLU
    elif nonlinearity == "swish":
        return nn.SiLU
    elif nonlinearity == "tanh":
        return nn.Tanh
    elif nonlinearity is None or nonlinearity == "none":
        return None
    else:
        raise ValueError(f"nonlinearity must be in {AllowedNonlinearity}")
