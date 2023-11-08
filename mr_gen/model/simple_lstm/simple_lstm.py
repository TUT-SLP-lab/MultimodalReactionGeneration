from collections import OrderedDict
from typing import Any, Dict, Tuple, Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from mr_gen.model.utils.lstm_block import LSTMLayerd
from mr_gen.model.utils.multi_modal_att import MultimodalAttention
from mr_gen.utils.metrics import MultiTargetMetrics


def gen_target_dict(cfg: DictConfig) -> Dict[str, Tuple[int, int]]:
    target_dict = {
        "centroid": (0, 3),
        "angle": (3, 6),
    }
    tail_index = 6

    if not cfg.use_centroid:
        target_dict.pop("centroid")
        target_dict["angle"] = (0, 3)
        tail_index = 3
    elif not cfg.use_angle:
        target_dict.pop("angle")
        tail_index = 3

    if cfg.delta_order > 0:
        if cfg.use_centroid:
            target_dict["delta1-centroid"] = (tail_index, tail_index + 3)
            tail_index += 3
        if cfg.use_angle:
            target_dict["delta1-angle"] = (tail_index, tail_index + 3)
            tail_index += 3
    if cfg.delta_order > 1:
        if cfg.use_centroid:
            target_dict["delta2-centroid"] = (tail_index, tail_index + 3)
            tail_index += 3
        if cfg.use_angle:
            target_dict["delta2-angle"] = (tail_index, tail_index + 3)
            tail_index += 3

    return target_dict


class AcousticEncoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.embed_layer = nn.Linear(cfg.acostic_feat_size, cfg.acostic_affine_size)

        self.acostic_lstm = LSTMLayerd(
            input_size=cfg.acostic_affine_size,
            lstm_hidden_size=cfg.acostic_lstm_size,
            affine_hidden_size=cfg.acostic_affine_size,
            num_layers=cfg.acostic_num_layers,
            num_layers_per_block=cfg.acostic_num_lstm,
            output_size=cfg.acostic_output_size,
            dropout=cfg.dropout_rate,
            bidirectional=cfg.bidirectional,
            use_layer_norm=cfg.use_layer_norm,
            use_relu=cfg.use_relu,
            use_mixing=cfg.use_mixing,
            use_residual=cfg.use_residual,
        )

    def forward(self, acoustic_feature: torch.Tensor) -> torch.Tensor:
        emb = self.embed_layer(acoustic_feature)
        return self.acostic_lstm(emb)


class MotionEncoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.embed_layer = nn.Linear(cfg.motion_feat_size, cfg.motion_affine_size)

        self.motion_lstm = LSTMLayerd(
            input_size=cfg.motion_affine_size,
            lstm_hidden_size=cfg.motion_lstm_size,
            affine_hidden_size=cfg.motion_affine_size,
            num_layers=cfg.motion_num_layers,
            num_layers_per_block=cfg.motion_num_lstm,
            output_size=cfg.motion_output_size,
            dropout=cfg.dropout_rate,
            bidirectional=cfg.bidirectional,
            use_layer_norm=cfg.use_layer_norm,
            use_relu=cfg.use_relu,
            use_mixing=cfg.use_mixing,
            use_residual=cfg.use_residual,
        )

    def forward(self, head_feature: torch.Tensor) -> torch.Tensor:
        emb = self.embed_layer(head_feature)
        return self.motion_lstm(emb)


class MotionDecoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.decoder_lstm = LSTMLayerd(
            input_size=cfg.motion_output_size,
            lstm_hidden_size=cfg.decoder_lstm_size,
            affine_hidden_size=cfg.decoder_affine_size,
            bottleneck_size=cfg.decoder_bottleneck_size,
            num_layers=cfg.decoder_num_layers,
            num_layers_per_block=cfg.decoder_num_lstm,
            output_size=cfg.decoder_output_size,
            dropout=cfg.dropout_rate,
            bidirectional=cfg.decoder_bidirectional,
            use_layer_norm=cfg.decoder_use_layer_norm,
            use_relu=cfg.decoder_use_relu,
            use_mixing=cfg.decoder_use_mixing,
            use_residual=cfg.decoder_use_residual,
        )

        ffn = OrderedDict()
        ffn["input"] = nn.Linear(cfg.decoder_output_size, cfg.decoder_mapping_size)
        if cfg.decoder_use_relu:
            ffn["relu"] = nn.ReLU()
        ffn["output"] = nn.Linear(cfg.decoder_mapping_size, cfg.output_size)
        self.mapping = nn.Sequential(ffn)

    def seq_reshape(self, x: torch.Tensor) -> torch.Tensor:
        # x: [sumone, batch_size, seq_len, feat_size]
        org_shape = list(x.size())
        x = x.view(-1, x.size(-2), x.size(-1))
        # x: [someone * batch_size, seq_len, feat_size]
        x = x[:, -1:, :]
        # x: [someone * batch_size, 1, feat_size]
        new_shape = org_shape
        new_shape[-2] = 1
        x = x.view(new_shape)
        # x: [someone, batch_size, 1, feat_size]
        return x

    def forward(self, att_embedded: torch.Tensor) -> torch.Tensor:
        y = self.decoder_lstm(att_embedded)
        y = self.mapping(self.seq_reshape(y))
        return y


class SimpleLSTM(pl.LightningModule):
    def __init__(self, cfg: DictConfig, optim: DictConfig, metrics: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.optim = optim
        self.metrics = metrics

        self.acoustic_encoder = AcousticEncoder(cfg)
        self.motion_encoder = MotionEncoder(cfg)
        self.multimodal_att = MultimodalAttention(
            modal1_feat_size=cfg.acostic_output_size,
            modal2_feat_size=cfg.motion_output_size,
            num_head=cfg.att_heads,
            num_layers=cfg.att_num_layers,
            dropout=cfg.dropout_rate,
            use_residual=cfg.att_use_residual,
            use_layer_norm=cfg.att_use_layer_norm,
        )
        self.motion_decoder = MotionDecoder(cfg)

        self.train_metrics = MultiTargetMetrics(
            target_range=gen_target_dict(metrics),
            prefix="train_",
        )
        self.valid_metrics = MultiTargetMetrics(
            target_range=gen_target_dict(metrics),
            prefix="valid_",
        )

        self.optimizer = None
        self.lr_scheduler = None

    def forward(
        self, acoustic_feature: torch.Tensor, motion_feature: torch.Tensor
    ) -> torch.Tensor:
        acoustic_embedded = self.acoustic_encoder(acoustic_feature)
        motion_embedded = self.motion_encoder(motion_feature)
        att_embedded = self.multimodal_att(motion_embedded, acoustic_embedded)
        y = self.motion_decoder(att_embedded)
        return y

    def lossfun(self):
        return nn.MSELoss(reduction="mean")

    def configure_optimizers(self) -> Union[Dict, Dict[str, LRScheduler]]:
        if self.optim.use_optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optim.lr,
                weight_decay=self.optim.weight_decay,
            )
        elif self.optim.use_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.optim.lr,
                momentum=self.optim.momentum,
                weight_decay=self.optim.weight_decay,
            )
        else:
            raise ValueError("invalid optimizer type")

        if self.optim.use_lr_sched:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.optim.max_epochs
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": "val_loss",
                },
            }
        return {"optimizer": self.optimizer}

    def training_step(self, batch, *args) -> STEP_OUTPUT:
        acoustic_feature, motion_feature, motion_target = batch
        y = self.forward(acoustic_feature, motion_feature)

        loss = self.lossfun()(y, motion_target)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        self.train_metrics(y, motion_target)
        self.log_dict(self.train_metrics, logger=True, on_epoch=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch, *args: Any) -> STEP_OUTPUT:
        acoustic_feature, motion_feature, motion_target = batch
        y = self.forward(acoustic_feature, motion_feature)

        loss = self.lossfun()(y, motion_target)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        self.valid_metrics(y, motion_target)
        self.log_dict(self.valid_metrics, logger=True, on_epoch=True, on_step=True)
        return {"loss": loss}
