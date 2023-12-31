import os
from typing import Any, Dict, Tuple, Union, List
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from mr_gen.utils.metrics import MultiTargetMetrics
from mr_gen.model.lstmformer.dataloader import PADDING_VALUE

from mr_gen.model.utils.multi_modal_metaformer import (
    MultiModalMetaformer,
    gen_attention_mask,
)
from mr_gen.model.utils.argparser import (
    mixer_layerd_argments_select,
    feedforward_block_argments,
)

os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # avoid deterministic error

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


class Metaformer(pl.LightningModule):
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
        self.ratio = acoustic_fps / pred_fps
        if self.ratio != int(self.ratio):
            raise ValueError(
                "pred_fps must be a divisor of acoustic_fps",
                f"pred_fps: {pred_fps}, acoustic_fps: {acoustic_fps}, ratio: {self.ratio}",
            )
        self.ratio = int(self.ratio)

        # compute input size of acoustic feature
        self.acoustic_input_size = (model.nmels + 1) * (model.delta_order + 1)

        # compute input size of motion feature
        self.motion_base_size = (model.use_centroid + model.use_angle) * 3
        self.motion_input_size = self.motion_base_size * (model.delta_order + 1)

        # prepare metaformer configs
        self.modalities = model.modalities
        self.modal_num = len(model.modalities)
        self.hidden_dim = model.hidden_size
        self.num_block = model.num_block
        self.num_heads = model.num_heads

        self.main_mixer_type = model.emb_mixers[model.main_modal_idx]
        self.other_mixer_type: list = list(model.emb_mixers)
        self.other_mixer_type.pop(model.main_modal_idx)

        self.repeat_with_encoder = model.repeat_with_encoder
        self.interlayer_residual = model.interlayer_residual
        self.interlayer_residual_norm = model.interlayer_residual_norm

        self.use_devise = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.common_configs = dict(
            hidden_size=self.hidden_dim,
            input_projection=False,
            output_projection=False,
            self_attention=True,
            num_heads=model.num_heads,
            dropout=model.dropout,
            batch_first=True,
            bidirectional=False,
            proj_size=model.proj_size,
            add_bias_kv=model.add_bias_kv,
            add_zero_attn=model.add_zero_attn,
            kdim=self.hidden_dim,
            vdim=self.hidden_dim,
            max_context_len=model.max_context_len,
            num_layerd=model.num_layerd,
            num_internal_layer=model.num_internal_layer,
            nonlinearity=model.nonlinearity,
            bottleneck_size=model.bottleneck_size,
            residual=model.residual,
            residual_layer_norm=model.residual_layer_norm,
            bias=model.bias,
            device=self.use_devise,
        )

        # architecture
        # embeddin configs
        self.main_mixer_configs = mixer_layerd_argments_select(
            self.main_mixer_type, **self.common_configs
        )
        out_mixer_configs = self.common_configs.copy()
        out_mixer_configs["num_layerd"] = model.encoder_num_layer
        self.other_mixer_configs = [
            mixer_layerd_argments_select(mixer_type, **out_mixer_configs)
            for mixer_type in self.other_mixer_type
        ]
        # cross-attention configs
        self.integrate_mixer_configs = self.common_configs.copy()
        self.integrate_mixer_configs["self_attention"] = False
        self.integrate_mixer_configs = [
            mixer_layerd_argments_select("mha", **self.integrate_mixer_configs)
        ] * (self.modal_num - 1)
        # feedforward configs
        self.feedforward_configs = feedforward_block_argments(
            hidden_size=self.hidden_dim,
            bottleneck_size=model.bottleneck_size,
            nonlinearity=model.ffn_nonlinearity,
            residual=model.residual,
            residual_layer_norm=model.residual_layer_norm,
            bias=model.bias,
            device=self.use_devise,
        )
        # output feedforward configs
        self.output_feedforward_configs = feedforward_block_argments(
            hidden_size=self.hidden_dim,
            bottleneck_size=model.bottleneck_size,
            output_size=self.motion_input_size,
            nonlinearity=model.ffn_nonlinearity,
            residual=False,
            bias=model.bias,
            device=self.use_devise,
        )
        self.metaformer = MultiModalMetaformer(
            modal_num=self.modal_num,
            hidden_dim=self.hidden_dim,
            num_layer=self.num_block,
            main_modal_feature_dim=self.motion_input_size,
            main_mixer_type=self.main_mixer_type,
            main_mixer_configs=self.main_mixer_configs,
            integrate_mixer_configs=self.integrate_mixer_configs,
            feedforward_configs=self.feedforward_configs,
            output_feedforward_configs=self.output_feedforward_configs,
            other_modal_feature_dim=[self.acoustic_input_size, self.motion_input_size],
            other_mixer_type=self.other_mixer_type,
            other_mixer_configs=self.other_mixer_configs,
            repeat_with_encoder=self.repeat_with_encoder,
            interlayer_residual=self.interlayer_residual,
            interlayer_residual_norm=self.interlayer_residual_norm,
        )

        # metrics
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
        self.delta_order = metrics.delta_order

    def forward(
        self,
        acoustic_partner: InputTypes,
        motion_partner: InputTypes,
        motion_self: InputTypes,
        leading_acoustic_partner: InputTypes,
        leading_motion_partner: InputTypes,
        leading_motion_self: InputTypes,
        hxs: Tuple[RNNStateType, List[RNNStateType]] = None,
    ) -> ForwardReturnType:
        acoustic_partner, _ = acoustic_partner
        motion_partner, _ = motion_partner
        motion_self, _ = motion_self
        leading_acoustic_partner, _ = leading_acoustic_partner
        leading_motion_partner, _ = leading_motion_partner
        leading_motion_self, _ = leading_motion_self

        # concat acoustic feature
        acoustic_partner = torch.cat(
            [leading_acoustic_partner, acoustic_partner], dim=1
        )

        # concat motion feature
        motion_partner = torch.cat([leading_motion_partner, motion_partner], dim=1)
        motion_self = torch.cat([leading_motion_self, motion_self], dim=1)
        motion_len = motion_partner.shape[1]

        mm_attention_mask = gen_attention_mask(
            motion_self, motion_partner, self.num_heads, PADDING_VALUE
        ).view(-1, motion_len, motion_len)
        ma_attention_mask = gen_attention_mask(
            motion_self, acoustic_partner, self.num_heads, PADDING_VALUE
        ).view(-1, motion_len, acoustic_partner.shape[1])
        ac_self_attention_mask = None
        mp_self_attention_mask = None
        ms_self_attention_mask = None
        if self.other_mixer_type[0] == "mha":
            ac_self_attention_mask = gen_attention_mask(
                acoustic_partner, acoustic_partner, self.num_heads, PADDING_VALUE
            ).view(-1, acoustic_partner.shape[1], acoustic_partner.shape[1])
        if self.main_mixer_type[1] == "mha":
            mp_self_attention_mask = gen_attention_mask(
                motion_partner, motion_partner, self.num_heads, PADDING_VALUE
            ).view(-1, motion_len, motion_len)
        if self.main_mixer_type[2] == "mha":
            ms_self_attention_mask = gen_attention_mask(
                motion_self, motion_self, self.num_heads, PADDING_VALUE
            ).view(-1, motion_len, motion_len)

        main_modal = motion_self
        other_modals = [acoustic_partner, motion_partner]
        main_modal_others = (None, None, ms_self_attention_mask)
        other_modals_others = [
            (None, None, ac_self_attention_mask),
            (None, None, mp_self_attention_mask),
        ]
        integrate_attn_mask = [ma_attention_mask, mm_attention_mask]

        main_modal, other_modals, hxs = self.metaformer(
            main_modal,
            other_modals,
            hxs,
            main_modal_others,
            other_modals_others,
            integrate_attn_mask,
        )
        del other_modals

        return main_modal, hxs

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
            lead_len = batch[4][0].shape[1]
            y, _ = self.forward(*batch[:-1])
            target, _ = batch[-1]
            y = y[:, lead_len:]
            del _

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
        lead_len = batch[4][0].shape[1]

        ################# validation phase #################
        y, _ = self.forward(*batch[:-1])
        target, _ = batch[-1]
        del _

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
        _, cell_state = self.forward(*dummy_input[:3], *batch[3:6], hxs=None)
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

        y, cell_state = self.forward(*step_batch, cell_state)
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
