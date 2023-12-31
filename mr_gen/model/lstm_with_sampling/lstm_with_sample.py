from collections import OrderedDict
from typing import Any, Dict, Tuple, Union, List
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from mr_gen.model.utils import LSTMSampler
from mr_gen.model.utils.lstm_block import LSTMLayerd
from mr_gen.utils.metrics import MultiTargetMetrics
from mr_gen.model.lstm_with_sampling.dataloader import PADDING_VALUE


DataType = torch.Tensor
LenType = torch.Tensor
InputTypes = Tuple[DataType, LenType]

RNNStateType = Tuple[torch.Tensor, torch.Tensor]
ForwardReturnType = Tuple[
    torch.Tensor, Tuple[int, int, int], Tuple[RNNStateType, List[RNNStateType]]
]


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


class LSTMwithSample(pl.LightningModule):
    def __init__(self, model: DictConfig, optim: DictConfig, metrics: DictConfig):
        super().__init__()
        self.model = model
        self.optim = optim
        self.metrics = metrics

        # for use_scheduled_sampling
        self.max_epochs = model.max_epochs
        self.use_scheduled_sampling = model.use_scheduled_sampling

        # check loss type
        if model.loss_type not in ["mse", "mae", "huber", "smoothl1"]:
            raise ValueError("invalid loss type")
        self.huber_delta = model.get("huber_delta", 1.0)
        self.smoothl1_beta = model.get("smoothl1_beta", 1.0)

        # compute ratio of pred_fps : acoustic_fps
        pred_fps = model.pred_fps
        acoustic_fps = model.sampling_rate / model.shift
        self.ratio = int(acoustic_fps / pred_fps)

        # compute input size of motion feature
        motion_base_size = (model.use_centroid + model.use_angle) * 3
        motion_input_size = motion_base_size * (model.delta_order + 1)
        motion_input_size *= 2  # because of participant number, 'self' & 'partner'

        # compute input size of acoustic feature
        acoustic_input_size = (model.nmels + 1) * (model.delta_order + 1)

        # compute input size of prediction
        prediction_input_size = motion_input_size + model.sampler_hidden_size

        self.acoustic_projection = nn.Linear(
            acoustic_input_size, model.sampler_hidden_size
        )

        self.sampling_lstm = LSTMSampler(
            model.sampler_hidden_size,
            model.sampler_num_layers,
            model.sampler_dropout_rate,
            self.ratio,
            bidirectional=False,
        )

        self.feature_projection = nn.Linear(prediction_input_size, model.hidden_size)

        self.layerd_lstm = LSTMLayerd(
            input_size=model.hidden_size,
            lstm_hidden_size=model.hidden_size,
            affine_hidden_size=model.hidden_size,
            bottleneck_size=model.bottleneck_size,
            num_layers=model.num_layers,
            num_layers_per_block=model.num_lstm,
            output_size=model.hidden_size,
            dropout=model.dropout_rate,
            bidirectional=False,
            use_layer_norm=model.use_layer_norm,
            use_mixing=model.use_mixing,
            use_residual=model.use_residual,
            use_feed_forward=False,
        )
        module_list = OrderedDict()
        module_list["input"] = nn.Linear(model.hidden_size, model.bottleneck_size)
        if model.use_relu:
            module_list["relu"] = nn.ReLU()
        module_list["mapping"] = nn.Linear(
            model.bottleneck_size, motion_input_size // 2
        )
        self.feed_forward = nn.Sequential(module_list)

        self.train_metrics = MultiTargetMetrics(
            target_range=gen_target_dict(metrics),
            prefix="train_",
        )
        self.valid_metrics = MultiTargetMetrics(
            target_range=gen_target_dict(metrics),
            prefix="valid_",
        )
        self.genrt_metrics = MultiTargetMetrics(
            target_range=gen_target_dict(metrics),
            prefix="genrt_",
        )

        self.optimizer = None
        self.lr_scheduler = None
        self.delta_loss_scale = model.get("delta_loss_scale", 1.0)
        self.all_static = model.get("all_static", False)
        self.delta_order = metrics.delta_order

    def forward(
        self,
        acoustic_partner: InputTypes,
        motion_partner: InputTypes,
        motion_self: InputTypes,
        leading_acoustic_partner: InputTypes,
        leading_motion_partner: InputTypes,
        leading_motion_self: InputTypes,
        cell_state: Tuple[RNNStateType, List[RNNStateType]] = None,
    ) -> ForwardReturnType:
        acoustic_partner, _ = acoustic_partner
        motion_partner, _ = motion_partner
        motion_self, ms_len = motion_self
        leading_acoustic_partner, _ = leading_acoustic_partner
        leading_motion_partner, _ = leading_motion_partner
        leading_motion_self, _ = leading_motion_self

        # set cell state
        if cell_state is None:
            hx_sampler, hxs = None, None
        else:
            hx_sampler, hxs = cell_state

        # concat acoustic feature
        leading_acoustic_len = leading_acoustic_partner.shape[1]
        org_acoustic_len = acoustic_partner.shape[1]
        acoustic_partner = torch.cat(
            [leading_acoustic_partner, acoustic_partner], dim=1
        )

        ashape = acoustic_partner.shape
        leading_motion_len = leading_motion_partner.shape[1]
        org_motion_len = motion_partner.shape[1]
        org_motion_self_len = motion_self.shape[1]

        # concat motion feature
        motion_partner = torch.cat([leading_motion_partner, motion_partner], dim=1)
        motion_self = torch.cat([leading_motion_self, motion_self], dim=1)
        motion_len = motion_partner.shape[1]

        # sampling acoustic feature
        acoustic_partner = self.acoustic_projection(acoustic_partner)
        acoustic_partner, hx_sampler = self.sampling_lstm(acoustic_partner, hx_sampler)

        acoustic_len = acoustic_partner.shape[1]
        if not (acoustic_len == motion_partner.shape[1] == motion_self.shape[1]):
            msg = f"acoustic: {ashape} -> {acoustic_partner.shape}: "
            msg += f"{leading_acoustic_len} + {org_acoustic_len}\n"
            msg += f"motion_p: {leading_motion_len} + {org_motion_len}\n"
            msg += f"motion_s: {leading_motion_self.shape[1]} + {org_motion_self_len}\n"
            msg += f"ratio: {self.ratio}"
            raise RuntimeError(msg)

        # concat acoustic feature and motion feature
        try:
            features = torch.cat(
                [
                    acoustic_partner,
                    motion_partner,
                    motion_self,
                ],
                dim=-1,
            )
        except RuntimeError as exc:
            msg = f"acoustic: {acoustic_partner.shape}, "
            msg += f"motion_p: {motion_partner.shape}, "
            msg += f"motion_s: {motion_self.shape}"
            raise RuntimeError(msg) from exc

        # predict motion
        features = self.feature_projection(features)
        h, hxs = self.layerd_lstm(features, hxs)
        y = self.feed_forward(h)

        return y, (leading_motion_len, motion_len, ms_len), (hx_sampler, hxs)

    def lossfun(self):
        if self.model.loss_type == "mse":
            return nn.MSELoss(reduction=self.model.loss_reduction)
        elif self.model.loss_type == "mae":
            return nn.L1Loss(reduction=self.model.loss_reduction)
        elif self.model.loss_type == "huber":
            return nn.HuberLoss(
                reduction=self.model.loss_reduction, delta=self.huber_delta
            )
        elif self.model.loss_type == "smoothl1":
            return nn.SmoothL1Loss(
                reduction=self.model.loss_reduction, beta=self.smoothl1_beta
            )

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

    def training_step(self, batch: List[InputTypes], *args) -> STEP_OUTPUT:
        if self.use_scheduled_sampling:
            rate = self.current_epoch / self.max_epochs
            self.log("scheduled_sampling_rate", rate, logger=True)
            y, target = self.prediction(batch, use_scheduled_sampling=True)
        else:
            y, (lead_len, _, _), _ = self.forward(*batch[:-1])
            target, _ = batch[-1]
            y = y[:, lead_len:]

        loss_mask = (target != PADDING_VALUE).int()
        y = y * loss_mask
        target = target * loss_mask

        scaler = torch.ones_like(y)
        delta_start = y.shape[2] // (self.delta_order + 1)
        scaler[:, :, delta_start:] = torch.sqrt(torch.tensor(self.delta_loss_scale))

        loss = self.lossfun()(y * scaler, target * scaler)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        self.train_metrics(y * scaler, target * scaler)
        self.log_dict(self.train_metrics, logger=True, on_epoch=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch: List[InputTypes], *args: Any) -> STEP_OUTPUT:
        ################# validation phase #################
        y, (lead_len, _, _), _ = self.forward(*batch[:-1])
        target, _ = batch[-1]

        loss_mask = (target != PADDING_VALUE).int()
        y = y[:, lead_len:] * loss_mask
        target = target * loss_mask

        loss = self.lossfun()(y, target)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        self.valid_metrics(y, target)
        self.log_dict(self.valid_metrics, logger=True, on_epoch=True, on_step=True)

        ################# generation phase #################
        gen_loss = self.generation_step(batch)["loss"]

        return {"loss": loss, "gen_loss": gen_loss}

    def generation_step(self, batch: List[InputTypes]) -> STEP_OUTPUT:
        prediction, target = self.prediction(batch)

        # padding mask
        loss_mask = (target != PADDING_VALUE).int()
        prediction = prediction * loss_mask
        target = target * loss_mask

        loss = self.lossfun()(prediction, target)
        self.log("genrt_loss", loss, prog_bar=False, logger=True)

        self.genrt_metrics(prediction, target)
        self.log_dict(self.genrt_metrics, logger=True, on_epoch=True, on_step=True)

        return {"loss": loss}

    def prediction(
        self, batch: List[InputTypes], use_scheduled_sampling: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize
        formed_batch, dummy_input, length = self.batch_forming(batch)
        target, _ = batch[-1]

        # warmup model
        cell_state = self.warmup_model(dummy_input, batch)

        # head motion generation
        prediction = self.head_motion_generation(
            formed_batch, dummy_input, length, cell_state, use_scheduled_sampling
        )

        return prediction, target

    def batch_forming(
        self, batch: List[InputTypes]
    ) -> Tuple[List[InputTypes], List[InputTypes], int]:
        """form batch for training"""
        formed_batch, length = self.form_generation_init(batch)
        dummy_input = self.gen_dummy_input(batch)

        return formed_batch, dummy_input, length

    def warmup_model(self, dummy_input: List[InputTypes], batch: List[InputTypes]):
        """warmup model"""
        _, _, cell_state = self.forward(*dummy_input[:3], *batch[3:6], cell_state=None)
        return cell_state

    def head_motion_generation(
        self,
        formed_batch: List[InputTypes],
        dummy_input: List[InputTypes],
        length: int,
        cell_state: RNNStateType = None,
        use_scheduled_sampling: bool = False,
    ):
        if use_scheduled_sampling:
            sampling_mask = torch.rand(length) < (self.current_epoch / self.max_epochs)
        else:
            sampling_mask = torch.ones(length, dtype=torch.bool)

        motion_s = formed_batch[2][0]
        y = motion_s[0]
        prediction = y.clone()
        for step, mask in enumerate(sampling_mask):
            y, cell_state = self.generate_one_step(
                step, formed_batch, y, dummy_input, cell_state
            )
            prediction = torch.cat([prediction, y], dim=1)
            y = y if mask else motion_s[step]

        prediction = prediction[:, 1:].contiguous()

        return prediction

    def generate_one_step(
        self,
        step,
        formed_batch: List[InputTypes],
        previous: torch.Tensor,
        dummy_input: List[InputTypes],
        cell_state: RNNStateType,
    ):
        fbank = formed_batch[0]
        motion_p = formed_batch[1]
        batch_size = motion_p[0][step].shape[0]
        previous_len = torch.ones(batch_size, dtype=torch.long).to(self.device)

        step_batch = (
            (fbank[0][step], fbank[1]),
            (motion_p[0][step], motion_p[1]),
            (previous, previous_len),
            dummy_input[3],
            dummy_input[4],
            dummy_input[5],
        )

        y, _, cell_state = self.forward(*step_batch, cell_state)
        return y, cell_state

    def form_generation_init(self, batch) -> Tuple[List[InputTypes], int]:
        (fbank, lf), (motion_p, lp), (motion_s, ls), _, _, _, _ = batch

        length: int = motion_p.shape[1]
        batch_size: int = motion_p.shape[0]

        # [batch_size, seq_len*self.ratio, feature_dim] -> [seq_len, batch_size, self.ratio, feature_dim]
        fbank: torch.Tensor
        fbank = fbank.view(batch_size, length, self.ratio, fbank.shape[-1])
        fbank = fbank.transpose(0, 1).contiguous()
        # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, 1, feature_dim]
        motion_p: torch.Tensor
        motion_p = motion_p.transpose(0, 1).unsqueeze(2).contiguous()
        motion_s: torch.Tensor
        motion_s = motion_s.transpose(0, 1).unsqueeze(2).contiguous()

        return [(fbank, lf), (motion_p, lp), (motion_s, ls)], length

    def gen_dummy_input(self, batch) -> List[InputTypes]:
        """create dummy input for generation (for sequence length)"""
        new_batch = []
        for modal in batch:
            (modal, l) = modal
            modal_shape = modal.shape
            dummy_modal = torch.empty(
                (modal_shape[0], 0, modal_shape[2]), dtype=modal.dtype
            ).to(self.device)
            new_batch.append((dummy_modal, l))
        return new_batch
