import torch
import numpy as np
import torch.nn as nn
from functools import cache
from einops import rearrange
from src.curves.space_filling_curves import embed_and_prune_sfc, hilbert_curve
from ..base_patch_embedding import BasePatchEmbedding


class OnionEmbedding1D(BasePatchEmbedding):
    """
    Converts an image into a sequence of 1D raster-scan patch embeddings.

    Args:
        img_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each 1D patch (in pixels, not 2D square).
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        embed_dim (int): Dimensionality of the projected patch embeddings.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        num_pixels = img_size * img_size
        assert num_pixels % patch_size == 0, "Image must be divisible into 1D patches"
        self.n_patches = num_pixels // patch_size
        self.input_dim = patch_size * in_channels

        self.proj = nn.Linear(self.input_dim, embed_dim)

    @cache
    def onion_indices(self, c, d):
        visited = np.zeros((c, d), dtype=bool)
        result = []
        dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # right, up, left, down
        dir_idx = 0
        i, j = c - 1, 0  # Start at bottom-left

        for _ in range(c * d):
            result.append((i, j))
            visited[i, j] = True
            # Try to move in current direction
            ni, nj = i + dirs[dir_idx][0], j + dirs[dir_idx][1]
            if 0 <= ni < c and 0 <= nj < d and not visited[ni, nj]:
                i, j = ni, nj
            else:
                dir_idx = (dir_idx + 1) % 4
                i, j = i + dirs[dir_idx][0], j + dirs[dir_idx][1]
        return tuple(np.array(result).T)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D]
        """
        B, C, H, W = x.shape

        idx_h, idx_w = self.onion_indices(H, W)  # Get indices in onion order

        # Reorder pixels spatially in spiral order
        x = x[:, :, idx_h, idx_w]  # shape: [B, C, H*W] in spiral order

        # Rearrange to [B, H*W, C]
        x = x.permute(0, 2, 1)  # [B, H*W, C]

        # Chunk into 1D patches
        # [B, N, patch_size * C]
        x = x.reshape(B, self.n_patches, self.input_dim)

        # Project
        x = self.proj(x)  # [B, N, D]
        return x
