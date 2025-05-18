import torch
import torch.nn as nn
from functools import cache
from einops import rearrange
from src.curves.space_filling_curves import embed_and_prune_sfc, peano_curve
from ..base_patch_embedding import BasePatchEmbedding


class PeanoEmbedding1D(BasePatchEmbedding):
    """
    Converts an input image into a sequence of patch embeddings
    using a Conv2D with stride = patch_size. This results
    in a sequence of flattened patches.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.n_patches = (img_size * img_size) // patch_size
        self.input_dim = in_channels * patch_size
        self.register_buffer(
            "peano_indices", self._peano_order(img_size).long())
        self.embed_dim = embed_dim
        self.proj = nn.Linear(self.input_dim, self.embed_dim)

    @cache
    def _peano_order(self, n):
        curve = embed_and_prune_sfc(peano_curve, n, n)
        return torch.tensor(curve, dtype=torch.long)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D]
        """
        B, C, H, W = x.shape
        x = x[:, :, self.peano_indices[:, 0],
              self.peano_indices[:, 1]]
        x = rearrange(x, 'b c n -> b n c')  # (B, N, C)
        # Reshape to patch-level input
        x = x.reshape(B, self.n_patches, self.input_dim)

        # Apply linear projection
        x = self.proj(x)
        return x
