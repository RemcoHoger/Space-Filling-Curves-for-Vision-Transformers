import torch
import numpy as np
import torch.nn as nn
from functools import cache
from einops import rearrange
from src.curves.space_filling_curves import embed_and_prune_sfc, hilbert_curve


class HierarchicalHilbertEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, patch_size_list, embed_dim, curve_fn=hilbert_curve):
        super().__init__()
        self.levels = nn.ModuleList()
        pre_patch_size = 1
        pre_patch_list = []

        for patch_size in patch_size_list:
            self.levels.append(SFCEmbedding1D(
                img_size, pre_patch_size, patch_size, in_channels, embed_dim, curve_fn))
            pre_patch_list.append(pre_patch_size)
            pre_patch_size *= 2

        self.patch_list = [int(((img_size // pre_size) // np.sqrt(patch_size)) ** 2)
                           for pre_size, patch_size in zip(pre_patch_list, patch_size_list)]

        self.embed_dim = embed_dim * len(patch_size_list)
        self.depth = len(patch_size_list)
        self.n_patches = self.patch_list[0]
        self.fusion = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        patches = [level(x) for level in self.levels]  # each [B, N_i, D]
        n_tokens = self.patch_list[0]

        for i in range(1, len(patches)):
            patches[i] = torch.nn.functional.interpolate(
                patches[i].transpose(1, 2), size=n_tokens, mode='linear', align_corners=False
            ).transpose(1, 2)

        x = torch.cat(patches, dim=-1)  # [B, n_tokens, D * depth]
        return self.fusion(x)  # [B, n_tokens, D * depth]


class SFCEmbedding1D(nn.Module):
    def __init__(self, img_size, pre_patch_size, group_patch_size, in_channels, embed_dim, curve_fn=hilbert_curve):
        super().__init__()

        assert img_size % pre_patch_size == 0, "Image size must be divisible by pre_patch_size"

        self.img_size = img_size
        self.pre_patch_size = pre_patch_size
        self.group_patch_size = group_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.curve_fn = curve_fn

        self.grid_size = img_size // pre_patch_size
        self.n_pre_patches = self.grid_size * self.grid_size
        self.n_final_patches = self.n_pre_patches // group_patch_size

        self.pre_patch_dim = in_channels * pre_patch_size * pre_patch_size
        self.input_dim = self.pre_patch_dim * group_patch_size

        # Register buffer for curve-based indices
        self.register_buffer("sfc_indices", self._sfc_indices(self.grid_size).long())

        self.proj = nn.Linear(self.input_dim, embed_dim)

    @cache
    def _sfc_indices(self, n):
        curve = embed_and_prune_sfc(self.curve_fn, n, n)
        flat_indices = [r * n + c for r, c in curve]
        return torch.tensor(flat_indices, dtype=torch.long)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.pre_patch_size

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = x[:, self.sfc_indices]  # Reorder using SFC

        gp = self.group_patch_size
        x = rearrange(x, f'b (n g) d -> b n (g d)', g=gp)

        return self.proj(x)