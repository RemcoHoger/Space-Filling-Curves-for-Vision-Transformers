import torch
import numpy as np
import torch.nn as nn
from functools import cache
from einops import rearrange
from src.curves.space_filling_curves import embed_and_prune_sfc, z_curve


class HierarchicalMortonEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, patch_size_list, embed_dim_list):
        super().__init__()
        self.levels = nn.ModuleList()
        pre_patch_size = 1
        pre_patch_list = []
        for patch_size, embed_dim in zip(patch_size_list, embed_dim_list):
            self.levels.append(MortonEmbedding1D(
                img_size, pre_patch_size, patch_size, in_channels, embed_dim))
            pre_patch_list.append(pre_patch_size)
            pre_patch_size *= 2

        # calculate the total number of patches
        print(f"Patch sizes: {patch_size_list}, Pre-patch sizes: {pre_patch_list}")
        self.n_patches = int(sum(
            ((img_size // pre_size) // np.sqrt(patch_size)) ** 2
            for pre_size, patch_size in zip(pre_patch_list, patch_size_list)
        ))
        self.input_dim = in_channels * (patch_size_list[0] ** 2)

    def forward(self, x):
        return [level(x) for level in self.levels]


class MortonEmbedding1D(nn.Module):
    def __init__(self, img_size, pre_patch_size, group_patch_size, in_channels, embed_dim):
        super().__init__()

        assert img_size % pre_patch_size == 0, "Image size must be divisible by pre_patch_size"

        self.img_size = img_size
        self.pre_patch_size = pre_patch_size
        self.group_patch_size = group_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = img_size // pre_patch_size
        self.n_pre_patches = self.grid_size * self.grid_size
        self.n_final_patches = self.n_pre_patches // group_patch_size

        self.pre_patch_dim = in_channels * pre_patch_size * pre_patch_size
        self.input_dim = self.pre_patch_dim * group_patch_size

        # Register Z-order indices over the grid of pre-patches
        self.register_buffer("z_indices", self._z_order(self.grid_size).long())

        # Final projection layer
        self.proj = nn.Linear(self.input_dim, embed_dim)

    @cache
    def _z_order(self, n):
        # Uses your z-order logic
        curve = embed_and_prune_sfc(z_curve, n, n)  # Returns (row, col) pairs
        # Flatten 2D grid index to 1D
        flat_indices = [r * n + c for r, c in curve]
        return torch.tensor(flat_indices, dtype=torch.long)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D]
        """
        B, C, H, W = x.shape
        print(f"Input shape: {x.shape}")
        p = self.pre_patch_size

        # Step 0.5: Create small p×p pre-patches
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # shape: [B, num_pre_patches, pre_patch_dim] where pre_patch_dim = p1 * p2 * C

        # Step 1: Reorder pre-patches via Z-order
        x = x[:, self.z_indices]  # [B, num_pre_patches, pre_patch_dim]

        # Step 2: Group pre-patches into final patches
        gp = self.group_patch_size
        x = rearrange(x, f'b (n g) d -> b n (g d)', g=gp)
        # shape: [B, num_final_patches, input_dim]

        # Step 3: Project
        x = self.proj(x)  # [B, num_final_patches, embed_dim]

        return x
