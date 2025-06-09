import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class HierarchicalRasterScanEmbedding(nn.Module):
    """
    Hierarchical version of 1D Raster Scan embedding.
    At each level, the image is split into small pre-patches,
    which are flattened in raster-scan order and grouped into larger patches.

    Args:
        img_size (int): Size of the square input image.
        in_channels (int): Number of channels (e.g. 3 for RGB).
        patch_size_list (List[int]): Number of pre-patches to group at each level.
        embed_dim (int): Embedding dimension per level.
    """

    def __init__(self, img_size, in_channels, patch_size_list, embed_dim):
        super().__init__()
        self.levels = nn.ModuleList()
        pre_patch_size = 1
        pre_patch_list = []

        for group_patch_size in patch_size_list:
            self.levels.append(RasterScan1DGroupedEmbedding(
                img_size, pre_patch_size, group_patch_size, in_channels, embed_dim))
            pre_patch_list.append(pre_patch_size)
            pre_patch_size *= 2  # double patch resolution each level

        self.patch_list = [
            int(((img_size // pre_size) // np.sqrt(group_size)) ** 2)
            for pre_size, group_size in zip(pre_patch_list, patch_size_list)
        ]

        self.embed_dim = embed_dim * len(patch_size_list)
        self.depth = len(patch_size_list)
        self.n_patches = self.patch_list[0]
        self.fusion = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        patches = [level(x) for level in self.levels]  # [B, N_i, D]
        n_tokens = self.n_patches

        for i in range(1, len(patches)):
            patches[i] = nn.functional.interpolate(
                patches[i].transpose(1, 2), size=n_tokens, mode='linear', align_corners=False
            ).transpose(1, 2)

        x = torch.cat(patches, dim=-1)  # [B, n_tokens, D * depth]
        return self.fusion(x)  # [B, n_tokens, D * depth]


class RasterScan1DGroupedEmbedding(nn.Module):
    """
    Single level of raster scan 1D embedding with grouping over pre-patches.

    Args:
        img_size (int): Input image height/width (square).
        pre_patch_size (int): Size of smallest square patch (e.g., 1, 2, 4).
        group_patch_size (int): How many pre-patches to group into final token.
        in_channels (int): Input channel count.
        embed_dim (int): Final projection dim.
    """
    def __init__(self, img_size, pre_patch_size, group_patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % pre_patch_size == 0, "Image must be divisible by pre_patch_size"

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

        self.proj = nn.Linear(self.input_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.pre_patch_size

        # Step 0.5: Break into square pre-patches (raster scan order)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # [B, n_pre_patches, pre_patch_dim]

        # Step 1: Group pre-patches linearly (no reordering)
        g = self.group_patch_size
        x = rearrange(x, f'b (n g) d -> b n (g d)', g=g)  # [B, n_final_patches, group_patch_dim]

        return self.proj(x)  # [B, N, D]