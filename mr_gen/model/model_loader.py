import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from mr_gen.model.simple_lstm.simple_lstm import SimpleLSTM

from mr_gen.model.lstmformer import Metaformer
from mr_gen.model.lstm_with_sampling import LSTMwithSample

MODEL_TYPE = ["simple_lstm", "lstmformer", "lstm_with_sampling"]


def load_model(model_type: str, model_path: str, cfg: DictConfig) -> pl.LightningModule:
    if model_type == "simple_lstm":
        model = SimpleLSTM(cfg.model, cfg.optim, cfg.metrics)
    elif model_type == "lstmformer":
        model = Metaformer(cfg.model, cfg.optim, cfg.metrics)
    elif model_type == "lstm_with_sampling":
        model = LSTMwithSample(cfg.model, cfg.optim, cfg.metrics)
    else:
        raise ValueError(f"model_type must be one of {MODEL_TYPE}")

    state_dict = torch.load(model_path)["state_dict"]
    model.load_state_dict(state_dict)

    return model
