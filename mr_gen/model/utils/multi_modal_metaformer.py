from typing import Tuple, Union, List, Any
import torch
from torch import nn

from mr_gen.model.utils.residual_connection import ResidualConnection
from mr_gen.model.utils.types import (
    MixerType,
    Modals_MixerHParams,
    LayerdReturnType,
    LayerdStateType,
    MetaformerBlockStateType,
    MetaformerBlockReturnType,
)
from mr_gen.model.utils.mixer_block import MixerLayerdFactory, FeedForward, split_state
from mr_gen.model.utils.values import PADDING_VALUE


def check_form_modal_num(
    modal_num: int, sameone: Union[list, tuple], msg: str = None
) -> list:
    if not isinstance(sameone, (list, tuple)):
        raise TypeError(f"must be list or tuple. but got {type(sameone)}")
    if not modal_num == len(sameone):
        if len(sameone) == 1:
            sameone = sameone * modal_num
        else:
            msg = "" if msg is None else msg
            raise ValueError(msg)
    return sameone


def gen_attention_mask(
    main_modal: torch.Tensor,
    other_modal: torch.Tensor,
    head_num: int,
    padding_value: float = PADDING_VALUE,
) -> torch.Tensor:
    # generate attention mask for rectangular attention
    main_modal_len = main_modal.shape[1]
    other_modal_len = other_modal.shape[1]
    if other_modal_len % main_modal_len != 0 and main_modal_len % other_modal_len != 0:
        raise ValueError(
            f"other_modal_len must be divisible by main_modal_len. "
            f"main_modal_len: {main_modal_len}, other_modal_len: {other_modal_len}"
        )
    batch_size = main_modal.shape[0]

    if other_modal_len % main_modal_len == 0:
        rate = other_modal_len // main_modal_len
        attn_mask = torch.ones(main_modal_len, main_modal_len, dtype=torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = torch.tile(attn_mask, (1, rate))
        attn_mask = attn_mask.view(main_modal_len, rate, main_modal_len)
        attn_mask = attn_mask.transpose(1, 2).contiguous()
        attn_mask = attn_mask.view(main_modal_len, other_modal_len)
    else:
        rate = main_modal_len // other_modal_len
        attn_mask = torch.ones(other_modal_len, other_modal_len, dtype=torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = torch.tile(attn_mask, (rate, 1))
        attn_mask = attn_mask.view(rate, other_modal_len, other_modal_len)
        attn_mask = attn_mask.transpose(1, 0).contiguous()
        attn_mask = attn_mask.view(main_modal_len, other_modal_len)
    attn_mask = attn_mask.to(main_modal.device).unsqueeze(0).unsqueeze(0)
    attn_mask = attn_mask.repeat(batch_size, head_num, 1, 1)

    # padding mask
    main_modal_padding = main_modal[:, :, 0] == padding_value
    other_modal_padding = other_modal[:, :, 0] == padding_value
    main_modal_padding = main_modal_padding.float().unsqueeze(-1)
    other_modal_padding = other_modal_padding.float().unsqueeze(1)
    padding_mask = torch.matmul(main_modal_padding, other_modal_padding)
    padding_mask = padding_mask.bool().unsqueeze(1)
    padding_mask = padding_mask.repeat(1, head_num, 1, 1)

    # merge
    merged_mask = attn_mask + padding_mask

    return merged_mask


class MultiModalEmbedding(nn.Module):
    def __init__(self, modal_num: int, mixer_configs: Modals_MixerHParams):
        super().__init__()

        # matching modal_num and mixer_configs
        self.modal_num = modal_num
        self.mixer_configs = check_form_modal_num(
            modal_num,
            mixer_configs,
            "modal_num must be equal to the length of mixer_configs."
            f"modal_num: {modal_num}, mixer_configs: {len(mixer_configs)}",
        )

        # build each modal embeddings
        self.modal_embeddings = nn.ModuleList()
        layerd_factory = MixerLayerdFactory()
        for mixer_config in self.mixer_configs:
            modal_embedding = layerd_factory.build(*mixer_config)
            self.modal_embeddings.append(modal_embedding)

    def forward(
        self,
        x: List[torch.Tensor],
        hx: List[LayerdStateType] = None,
        other: List[tuple] = None,
    ) -> Tuple[List[torch.Tensor], List[LayerdStateType]]:
        x = check_form_modal_num(
            self.modal_num,
            x,
            "The length of x must be equal to modal_num."
            f"modal_num: {self.modal_num}, x: {len(x)}",
        )

        hx = [None] * self.modal_num if hx is None else hx
        other = [(None,)] * self.modal_num if other is None else other

        y = []
        states = []
        for i, modal_embedding in enumerate(self.modal_embeddings):
            modal_out: LayerdReturnType = modal_embedding(x[i], hx[i], other[i])
            y.append(modal_out[0])
            states.append(modal_out[1])

        return (y, states)


class IntegrateModalBlock(nn.Module):
    def __init__(
        self, modal_num: int, mixer_configs: Modals_MixerHParams, output_dim: int
    ):
        super().__init__()

        # matching modal_num and mixer_configs
        self.modal_num = modal_num
        self.mixer_configs = check_form_modal_num(
            modal_num - 1,
            mixer_configs,
            "modal_num must be equal to the length + 1 of mixer_configs. "
            f"modal_num: {modal_num}, mixer_configs: {len(mixer_configs)} (+1)",
        )

        # build each modal integrator
        self.integrators = nn.ModuleList()
        layerd_factory = MixerLayerdFactory()
        for mixer_config in self.mixer_configs:
            if mixer_config[0] != "mha":
                raise TypeError("IntegrateModalBlock only supports mha mixer.")
            modal_integrator = layerd_factory.build(*mixer_config)
            self.integrators.append(modal_integrator)

        # build cat_linear
        hidden_dim_sum = 0
        for _, mixer_config in self.mixer_configs:
            hidden_dim = mixer_config["hidden_size"]
            if mixer_config["output_projection"]:
                hidden_dim = mixer_config["output_projection_size"]
            hidden_dim_sum += hidden_dim
        self.cat_linear = nn.Linear(hidden_dim_sum, output_dim)

    def check_form_input(
        self,
        other_modals: List[torch.Tensor],
        attn_mask: List[torch.Tensor],
        hx: List[LayerdStateType] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[bool]]:
        other_modals = (
            [other_modals] if isinstance(other_modals, torch.Tensor) else other_modals
        )
        other_modals = check_form_modal_num(
            self.modal_num - 1,
            other_modals,
            "The length of other_modals must be equal to modal_num - 1. "
            f"modal_num: {self.modal_num}, other_modals: {len(other_modals)}",
        )
        attn_mask = [None] * (self.modal_num - 1) if attn_mask is None else attn_mask
        attn_mask = [attn_mask] if isinstance(attn_mask, torch.Tensor) else attn_mask
        attn_mask = check_form_modal_num(
            self.modal_num - 1,
            attn_mask,
            "The length of attn_mask must be equal to modal_num - 1. "
            f"modal_num: {self.modal_num}, attn_mask: {len(attn_mask)}",
        )
        hx = [None] * (self.modal_num - 1) if hx is None else hx

        return other_modals, attn_mask, hx

    def forward(
        self,
        main_modal: torch.Tensor,
        other_modals: List[torch.Tensor],
        attn_mask: List[torch.Tensor] = None,
        hxs: List[LayerdStateType] = None,
    ) -> LayerdReturnType:
        (other_modals, attn_mask, hxs) = self.check_form_input(
            other_modals, attn_mask, hxs
        )

        # integrate other modals
        y = []
        state = []
        for i, integrator in enumerate(self.integrators):
            res: LayerdReturnType = integrator(
                main_modal,
                hxs[i],
                other_modals[i],
                other_modals[i],
                attn_mask[i],
            )
            _y, _state, _ = res
            y.append(_y)
            state.append(_state)

        main_modal = torch.cat(y, dim=-1)
        main_modal = self.cat_linear(main_modal)

        return (main_modal, state)


class MultiModalMetaformerBlock(nn.Module):
    def __init__(
        self,
        num_modal: int,
        main_modal_embedding_config: Modals_MixerHParams,
        integrate_configs: Modals_MixerHParams,
        feedforward_configs: dict,
        encode_other_modal: bool = False,
        other_modal_embedding_config: Modals_MixerHParams = None,
    ):
        super().__init__()

        if not encode_other_modal:
            other_modal_embedding_config = []
        if other_modal_embedding_config is None:
            other_modal_embedding_config = []
        if isinstance(main_modal_embedding_config, tuple):
            main_modal_embedding_config = [main_modal_embedding_config]

        # check num_modal and configs
        if encode_other_modal:
            other_modal_embedding_config = check_form_modal_num(
                num_modal - 1,
                other_modal_embedding_config,
                "The length of other_modal_embedding_config must be equal to num_modal - 1. "
                f"num_modal: {num_modal}, other_modal_embedding_config: {len(other_modal_embedding_config)}",
            )
        integrate_configs = check_form_modal_num(
            num_modal - 1,
            integrate_configs,
            "The length of integrate_configs must be equal to num_modal - 1. "
            f"num_modal: {num_modal}, integrate_configs: {len(integrate_configs)}",
        )

        self.num_modal = num_modal
        self.emb_num_modal = num_modal if encode_other_modal else 1
        self.main_modal_embedding_config = main_modal_embedding_config
        self.integrate_configs = integrate_configs
        self.encode_other_modal = encode_other_modal
        self.other_modal_embedding_config = other_modal_embedding_config

        self.embedding_configs = (
            main_modal_embedding_config + other_modal_embedding_config
        )
        self.emb_mixer_type = [config[0] for config in self.embedding_configs]
        ffn_dim = feedforward_configs["hidden_size"]

        # build architecture
        self.embedding = MultiModalEmbedding(self.emb_num_modal, self.embedding_configs)
        self.integrator = IntegrateModalBlock(num_modal, integrate_configs, ffn_dim)
        self.feedforward = FeedForward(**feedforward_configs)

    def forward(
        self,
        main_modal: Union[torch.Tensor, tuple],
        other_modals: List[torch.Tensor],
        hx: List[MetaformerBlockStateType] = None,
        prev_hx: List[MetaformerBlockStateType] = None,
        main_modal_others: Tuple[Any] = None,
        other_modals_others: List[Tuple[Any]] = None,
        integrate_attn_mask: List[torch.Tensor] = None,
    ) -> MetaformerBlockReturnType:
        # unpack main_modal for nn.Sequential
        if isinstance(main_modal, tuple):
            other_modals = main_modal[1]
            hx = main_modal[2]
            prev_hx = main_modal[3]
            main_modal_others = main_modal[4]
            other_modals_others = main_modal[5]
            integrate_attn_mask = main_modal[6]
            main_modal = main_modal[0]

        # for load state in nn.Sequential or when use model in auto-regressive
        _hx, hx, prev_hx = split_state(hx, prev_hx)
        if _hx is None:
            _hx = {"emb": None, "crm": None}

        other_modals_others = other_modals_others if other_modals_others else [None]
        other_modals_others = check_form_modal_num(
            self.num_modal - 1,
            other_modals_others,
            "The length of other_modals_others must be equal to num_modal - 1."
            f"num_modal: {self.num_modal}, other_modals_others: {len(other_modals_others)}",
        )

        if self.encode_other_modal:
            embeded_modal = [main_modal] + other_modals
        else:
            embeded_modal = [main_modal]

        states: MetaformerBlockStateType = {"emb": [], "crm": []}
        others = [main_modal_others] + other_modals_others

        # embedding
        embeded_modal = self.embedding(embeded_modal, _hx["emb"], others)
        modals, emb_states = embeded_modal

        main_modal = modals[0]
        states["emb"] = emb_states
        if self.encode_other_modal:
            other_modals = modals[1:]

        main_modal, other_state = self.integrator.forward(
            main_modal, other_modals, integrate_attn_mask, _hx["crm"]
        )
        states["crm"] = other_state
        prev_hx.append(states)

        main_modal = self.feedforward(main_modal)

        return (
            main_modal,
            other_modals,
            hx,
            prev_hx,
            main_modal_others,
            other_modals_others,
            integrate_attn_mask,
        )


class MultiModalMetaformer(nn.Module):
    def __init__(
        self,
        modal_num: int,
        hidden_dim: int,
        num_layer: int,
        # main modal configs
        main_modal_feature_dim: Union[List[int], int],
        main_mixer_type: Union[List[MixerType], MixerType],
        main_mixer_configs: Union[List[dict], dict],
        # integrate configs
        integrate_mixer_configs: Union[List[dict], dict],
        # feedforward configs
        feedforward_configs: dict,
        # output feedforward configs
        output_feedforward_configs: dict,
        # other modal configs
        other_modal_feature_dim: Union[List[int], int] = None,
        other_mixer_type: Union[List[MixerType], MixerType] = "mha",
        other_mixer_configs: Union[List[dict], dict] = None,
        # connection configs
        repeat_with_encoder: bool = False,
        interlayer_residual: bool = False,
        interlayer_residual_norm: bool = True,
    ):
        super().__init__()

        ############### check modal_num and configs ###############
        if isinstance(main_modal_feature_dim, (list, tuple)):
            main_modal_feature_dim = main_modal_feature_dim[0]
        if isinstance(main_mixer_type, (list, tuple)):
            main_mixer_type = main_mixer_type[0]
        if isinstance(main_mixer_configs, (list, tuple)):
            main_mixer_configs = main_mixer_configs[0]
        main_mixer_configs = [(main_mixer_type, main_mixer_configs)]
        if isinstance(integrate_mixer_configs, dict):
            integrate_mixer_configs = [integrate_mixer_configs]
        integrate_mixer_configs = check_form_modal_num(
            modal_num - 1,
            integrate_mixer_configs,
            "The length of integrate_mixer_configs must be equal to modal_num - 1."
            f"modal_num: {modal_num}, integrate_mixer_configs: {len(integrate_mixer_configs)}",
        )
        integrate_mixer_configs = [
            ("mha", integrate_mixer_configs[i]) for i in range(modal_num - 1)
        ]
        if isinstance(other_modal_feature_dim, int):
            other_modal_feature_dim = [other_modal_feature_dim]
        other_modal_feature_dim = check_form_modal_num(
            modal_num - 1,
            other_modal_feature_dim,
            "The length of other_modal_feature_dim must be equal to modal_num - 1."
            f"modal_num: {modal_num}, other_modal_feature_dim: {len(other_modal_feature_dim)}",
        )
        if isinstance(other_mixer_type, str):
            other_mixer_type = [other_mixer_type]
        other_mixer_type = check_form_modal_num(
            modal_num - 1,
            other_mixer_type,
            "The length of other_mixer_type must be equal to modal_num - 1."
            f"modal_num: {modal_num}, other_mixer_type: {len(other_mixer_type)}",
        )
        if isinstance(other_mixer_configs, dict):
            other_mixer_configs = [other_mixer_configs]
        other_mixer_configs = check_form_modal_num(
            modal_num - 1,
            other_mixer_configs,
            "The length of other_mixer_configs must be equal to modal_num - 1."
            f"modal_num: {modal_num}, other_mixer_configs: {len(other_mixer_configs)}",
        )
        other_mixer_configs = [
            (other_mixer_type[i], other_mixer_configs[i]) for i in range(modal_num - 1)
        ]
        other_modal_num = 0
        if other_modal_feature_dim is not None:
            other_modal_num = len(other_modal_feature_dim)
        if other_modal_num != modal_num - 1:
            raise ValueError(
                "The length of other_modal_feature_dim must be equal to modal_num - 1."
                f"modal_num: {modal_num} (-1), other_modal_feature_dim: {len(other_modal_feature_dim)}"
            )

        self.modal_num = modal_num
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.repeat_with_encoder = repeat_with_encoder
        self.interlayer_residual = interlayer_residual

        self.embedding_mixer_type = [main_mixer_type] + other_mixer_type

        ############### build architecture ###############
        # feature projection
        self.feature_embedding = nn.ModuleList()
        for dim in [main_modal_feature_dim] + other_modal_feature_dim:
            self.feature_embedding.append(nn.Linear(dim, hidden_dim))

        metaformer_blocks = []

        # 1st layer metaformer block
        first_block = MultiModalMetaformerBlock(
            modal_num,
            main_mixer_configs,
            integrate_mixer_configs,
            feedforward_configs,
            encode_other_modal=True,
            other_modal_embedding_config=other_mixer_configs,
        )
        metaformer_blocks.append(first_block)
        if not repeat_with_encoder:
            other_mixer_configs = None

        # 2nd ~ last layer metaformer block
        for _ in range(num_layer - 1):
            block = MultiModalMetaformerBlock(
                modal_num,
                main_mixer_configs,
                integrate_mixer_configs,
                feedforward_configs,
                encode_other_modal=repeat_with_encoder,
                other_modal_embedding_config=other_mixer_configs,
            )
            metaformer_blocks.append(block)

        self.metaformer_blocks = nn.ModuleList()
        for block in metaformer_blocks:
            if interlayer_residual:
                self.metaformer_blocks.append(
                    ResidualConnection(block, interlayer_residual_norm, hidden_dim)
                )
            else:
                self.metaformer_blocks.append(block)

        # output feedforward
        self.output_feedforward = FeedForward(**output_feedforward_configs)

    def forward(
        self,
        main_modal: torch.Tensor,
        other_modals: List[torch.Tensor],
        hx: List[MetaformerBlockStateType] = None,
        main_modal_others: Tuple[Any] = None,
        other_modals_others: List[Tuple[Any]] = None,
        integrate_attn_mask: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[MetaformerBlockStateType]]:
        # prepare for forward
        main_modal = self.feature_embedding[0](main_modal)
        other_modals = [
            self.feature_embedding[i + 1](other_modal)
            for i, other_modal in enumerate(other_modals)
        ]

        args = [
            other_modals,
            hx,
            None,
            main_modal_others,
            other_modals_others,
            integrate_attn_mask,
        ]
        # main process
        for block in self.metaformer_blocks:
            main_modal, *args = block(main_modal, *args)
        # output feedforward
        main_modal = self.output_feedforward(main_modal)

        other_modals, phx, hx, mmo, omo, iam = args
        del phx, mmo, omo, iam

        return main_modal, other_modals, hx
