import torch
import torch.nn as nn
import numpy as np
from math import isqrt


class HierarchicalZigzagEmbedding(nn.Module):
    """
    Hierarchical 2D patch embedding using Conv2D patch extractors at multiple scales.

    Args:
        img_size (int): Image size (assumes square).
        in_channels (int): Input channels (e.g. 3 for RGB).
        patch_size_list (List[int]): Group sizes per level (e.g., [4, 16, 64]).
        embed_dim (int): Output dim per scale.
    """

    def __init__(self, img_size, in_channels, patch_size_list, embed_dim):
        super().__init__()
        self.levels = nn.ModuleList()
        pre_patch_size = 1
        pre_patch_list = []

        for group_patch_size in patch_size_list:
            self.levels.append(ZigzagEmbedding2DGrouped(
                img_size, pre_patch_size, group_patch_size, in_channels, embed_dim
            ))
            pre_patch_list.append(pre_patch_size)
            pre_patch_size *= 2

        self.patch_list = [
            int(((img_size // pre) // isqrt(gp)) ** 2)
            for pre, gp in zip(pre_patch_list, patch_size_list)
        ]

        self.embed_dim = embed_dim * len(patch_size_list)
        self.depth = len(patch_size_list)
        self.n_patches = self.patch_list[0]  # token count of finest level

    def forward(self, x):
        patches = [level(x) for level in self.levels]  # [B, N_i, D]

        # Upsample to match finest resolution
        target_len = self.n_patches
        for i in range(1, len(patches)):
            patches[i] = nn.functional.interpolate(
                patches[i].transpose(1, 2), size=target_len, mode='linear', align_corners=False
            ).transpose(1, 2)

        return torch.cat(patches, dim=-1)  # [B, n_tokens, embed_dim * depth]