from collections import OrderedDict
import torch
from torch import nn
from mr_gen.model.utils import ResidualConnection


class MultimodalAttentionBlock(nn.Module):
    def __init__(
        self,
        modal1_feat_size=256,
        modal2_feat_size=256,
        num_head=1,
        dropout=0.0,
        use_residual=True,
        use_layer_norm=True,
    ) -> None:
        super().__init__()

        att_module = OrderedDict()
        att_module["cross_modal_att"] = nn.MultiheadAttention(
            embed_dim=modal1_feat_size,
            num_heads=num_head,
            dropout=dropout,
            batch_first=True,
            kdim=modal2_feat_size,
            vdim=modal2_feat_size,
        )
        att_module["proj"] = nn.Linear(modal1_feat_size, modal1_feat_size)
        self.att_module = nn.Sequential(att_module)

        if use_residual:
            self.att_module = ResidualConnection(
                self.att_module, use_layer_norm, modal1_feat_size
            )

    def forward(self, modal1: torch.Tensor, modal2: torch.Tensor) -> torch.Tensor:
        return self.att_module(modal1, modal2, modal2)


class MultimodalAttention(nn.Module):
    def __init__(
        self,
        modal1_feat_size=256,
        modal2_feat_size=256,
        num_head=1,
        num_layers=1,
        dropout=0.0,
        use_residual=True,
        use_layer_norm=True,
    ) -> None:
        super().__init__()

        self.att_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.att_layers.append(
                MultimodalAttentionBlock(
                    modal1_feat_size=modal1_feat_size,
                    modal2_feat_size=modal2_feat_size,
                    num_head=num_head,
                    dropout=dropout,
                    use_residual=use_residual,
                    use_layer_norm=use_layer_norm,
                )
            )

    def forward(self, modal1: torch.Tensor, modal2: torch.Tensor) -> torch.Tensor:
        for att_layer in self.att_layers:
            modal1 = att_layer(modal1, modal2)
        return modal1
