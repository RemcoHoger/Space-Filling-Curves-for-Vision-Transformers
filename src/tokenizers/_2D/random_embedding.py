import torch
import torch.nn as nn
from ..base_patch_embedding import BasePatchEmbedding


class RandomEmbedding(BasePatchEmbedding):
    """
    Converts an input image into a sequence of patch embeddings
    using a Conv2D with stride = patch_size. This results
    in a sequence of flattened patches, which are then shuffled randomly.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D] with patches shuffled randomly
        """
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # -> [B, N, D]

        # Shuffle patches randomly
        B, N, D = x.shape
        # Generate random permutation of patch indices
        indices = torch.randperm(N)
        x = x[:, indices, :]  # Apply the random permutation

        return x
