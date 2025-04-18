import numpy as np
import torch
import torch.nn as nn
from functools import cache
from .base_patch_embedding import BasePatchEmbedding


class HilbertEmbedding(BasePatchEmbedding):
    """
    Converts an input image into a sequence of patch embeddings
    using a Conv2D with stride = patch_size. This results
    in a sequence of flattened patches.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.hilbert_indices = self._get_hilbert_indices(self.grid_size)

    def _get_hilbert_indices(self, grid_size):
        """
        Computes the flattened Hilbert curve indices for a given grid size.

        Args:
            grid_size (int): The size of the grid (H/patch_size or W/patch_size).

        Returns:
            torch.Tensor: Flattened Hilbert curve indices.
        """
        order = int(np.log2(grid_size))  # Assuming grid_size is a power of 2
        hilbert_points = self.hilbert_curve(order)
        hilbert_indices = [(int(x * grid_size), int(y * grid_size))
                           for x, y in hilbert_points]
        hilbert_flat_indices = [i * grid_size + j for i, j in hilbert_indices]
        return torch.tensor(hilbert_flat_indices, dtype=torch.long)

    def hilbert_curve(self, order, size=1.0):
        """
        Generate points for a Hilbert curve of a given order
        on the unit square.

        Args:
            order (int): Recursion depth of the Hilbert curve.
            size (float): Length of one side of the entire curve's square.

        Returns:
            List[Tuple[float, float]]: The list of (x, y) points.
        """
        points = []

        @cache
        def hilbert(x0, y0, xi, xj, yi, yj, n):
            if n <= 0:
                x = x0 + (xi + yi) / 2
                y = y0 + (xj + yj) / 2
                points.append((x, y))
            else:
                hilbert(x0, y0,               yi/2, yj /
                        2,               xi/2, xj/2, n-1)
                hilbert(x0 + xi/2, y0 + xj/2, xi/2, xj /
                        2,               yi/2, yj/2, n-1)
                hilbert(x0 + xi/2 + yi/2, y0 + xj/2 +
                        yj/2, xi/2, xj/2, yi/2, yj/2, n-1)
                hilbert(x0 + xi/2 + yi, y0 + xj/2 + yj, -
                        yi/2, -yj/2, -xi/2, -xj/2, n-1)

        hilbert(0, 0, size, 0, 0, size, order)
        return points

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D]
        """
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        B, C, H, W = x.shape

        # Rearrange patches according to the precomputed Hilbert curve
        x = x.permute(0, 2, 3, 1)  # [B, H/patch_size, W/patch_size, embed_dim]
        x = x.reshape(B, H * W, C)  # Flatten spatial dimensions
        x = x[:, self.hilbert_indices, :]  # Reorder patches
        return x
