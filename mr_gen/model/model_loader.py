import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from mr_gen.model.simple_lstm.simple_lstm import SimpleLSTM

# from mr_gen.model.lstmformer.lstmformer import LSTMFormer


def load_model(model_type: str, model_path: str, cfg: DictConfig) -> pl.LightningModule:
    if model_type == "simple_lstm":
        model = SimpleLSTM(cfg.model, cfg.optim, cfg.metrics)
        state_dict = torch.load(model_path)["state_dict"]
        model.load_state_dict(state_dict)
    elif model_type == "lstmformer":
        raise NotImplementedError
    else:
        raise ValueError("model_type must be one of [simple_lstm]")

    return model
