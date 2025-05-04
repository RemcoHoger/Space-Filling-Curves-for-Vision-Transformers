import math
import torch
import torch.nn as nn
from functools import cache
from einops import rearrange
from ..base_patch_embedding import BasePatchEmbedding


class HilbertEmbedding1D(BasePatchEmbedding):
    """
    Converts an input image into a sequence of patch embeddings
    using a Conv2D with stride = patch_size. This results
    in a sequence of flattened patches.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.n_patches = (img_size * img_size) // patch_size
        self.input_dim = in_channels * patch_size
        self.hilbert_indices = self._hilbert_order(img_size)
        self.embed_dim = embed_dim
        self.proj = nn.Linear(self.input_dim, self.embed_dim)

    @cache
    def _hilbert_order(self, n):
        coords = []

        def hilbert(x0, y0, xi, xj, yi, yj, n):
            if n <= 0:
                x = x0 + (xi + yi) // 2
                y = y0 + (xj + yj) // 2
                coords.append((x, y))
            else:
                hilbert(x0, y0, yi//2, yj//2, xi//2, xj//2, n-1)
                hilbert(x0 + xi//2, y0 + xj//2, xi //
                        2, xj//2, yi//2, yj//2, n-1)
                hilbert(x0 + xi//2 + yi//2, y0 + xj//2 + yj//2,
                        xi//2, xj//2, yi//2, yj//2, n-1)
                hilbert(x0 + xi//2 + yi, y0 + xj//2 + yj,
                        -yi//2, -yj//2, -xi//2, -xj//2, n-1)

        hilbert(0, 0, n, 0, 0, n, int(math.log2(n)))
        return torch.tensor(coords, dtype=torch.long)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D]
        """
        B, C, H, W = x.shape
        x = x[:, :, self.hilbert_indices[:, 0],
              self.hilbert_indices[:, 1]]
        x = rearrange(x, 'b c n -> b n c')  # (B, N, C)
        # Reshape to patch-level input
        x = x.reshape(B, self.n_patches, self.input_dim)

        # Apply linear projection
        x = self.proj(x)
        return x
