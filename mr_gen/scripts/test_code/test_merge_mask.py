import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional, Union, Iterable, List, Dict


def merge_masks(
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor],
    query: Tensor,
    num_heads: int,
) -> Tuple[Optional[Tensor], Optional[int]]:
    r"""
    Determine mask type and combine masks if necessary. If only one mask is provided, that mask
    and the corresponding mask type will be returned. If both masks are provided, they will be both
    expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
    and mask type 2 will be returned
    Args:
        attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
        key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
        query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
    Returns:
        merged_mask: merged mask
        mask_type: merged mask type (0, 1, or 2)
    """
    mask_type: Optional[int] = None
    merged_mask: Optional[Tensor] = None

    if key_padding_mask is not None:
        mask_type = 1
        merged_mask = key_padding_mask

    if attn_mask is not None:
        # In this branch query can't be a nested tensor, so it has a shape
        batch_size, seq_len, _ = query.shape
        mask_type = 2

        # Always expands attn_mask to 4D
        if attn_mask.dim() == 3:
            attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
        else:  # attn_mask.dim() == 2:
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                batch_size, num_heads, -1, -1
            )
        merged_mask = attn_mask_expanded

        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.view(
                batch_size, 1, 1, seq_len
            ).expand(-1, num_heads, -1, -1)
            merged_mask = attn_mask_expanded + key_padding_mask_expanded

    # no attn_mask and no key_padding_mask, returns None, None
    return merged_mask, mask_type


def gen_attention_mask(
    main_modal: torch.Tensor,
    other_modal: torch.Tensor,
    head_num: int,
    padding_value: float = 0,
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
    main_modal_padding = main_modal_padding.int().unsqueeze(-1)
    other_modal_padding = other_modal_padding.int().unsqueeze(1)
    padding_mask = torch.matmul(main_modal_padding, other_modal_padding)
    padding_mask = padding_mask.bool().unsqueeze(1)
    padding_mask = padding_mask.repeat(1, head_num, 1, 1)

    # merge
    merged_mask = attn_mask + padding_mask

    return merged_mask


def main():
    query = torch.rand(1, 3, 5)
    value = torch.rand(1, 3, 5)
    # attn_mask = torch.ones(3, 3).bool()
    # key_padding_mask = torch.ones(2, 3).bool()
    # num_heads = 2
    # mask, _ = merge_masks(attn_mask, key_padding_mask, query, num_heads)
    mask = gen_attention_mask(query, value, 2).float().bool()
    print(mask.shape)
    print(mask)


if __name__ == "__main__":
    main()
