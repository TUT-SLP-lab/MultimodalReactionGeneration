import torch
from torch import nn
from mr_gen.model.utils import ResidualConnection


class MultiModalAttentionBlockSequential(nn.Module):
    def __init__(
        self, modal1_feat_size=256, modal2_feat_size=256, num_head=1, dropout=0.0
    ) -> None:
        super().__init__()

        self.cross_modal_att = nn.MultiheadAttention(
            embed_dim=modal1_feat_size,
            num_heads=num_head,
            dropout=dropout,
            batch_first=True,
            kdim=modal2_feat_size,
            vdim=modal2_feat_size,
        )
        self.projection = nn.Linear(modal1_feat_size, modal1_feat_size)

    def forward(self, modal1: torch.Tensor, modal2: torch.Tensor) -> torch.Tensor:
        # modal1: [batch_size, seq_len, modal1_feat_size]
        # modal2: [batch_size, seq_len, modal2_feat_size]
        modal1, _att_weight = self.cross_modal_att(
            query=modal1, key=modal2, value=modal2, need_weights=False
        )
        # modal1: [batch_size, seq_len, modal1_feat_size]
        modal1 = self.projection(modal1)
        # modal1: [batch_size, seq_len, modal1_feat_size]
        return modal1


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

        self.att_module = MultiModalAttentionBlockSequential(
            modal1_feat_size=modal1_feat_size,
            modal2_feat_size=modal2_feat_size,
            num_head=num_head,
            dropout=dropout,
        )

        if use_residual:
            self.att_module = ResidualConnection(
                self.att_module, use_layer_norm, modal1_feat_size
            )

    def forward(self, modal1: torch.Tensor, modal2: torch.Tensor) -> torch.Tensor:
        return self.att_module(modal1, modal2)


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
