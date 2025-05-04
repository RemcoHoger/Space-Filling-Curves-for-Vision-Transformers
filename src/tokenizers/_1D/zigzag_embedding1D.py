import torch.nn as nn
from ..base_patch_embedding import BasePatchEmbedding


class RasterScan1DEmbedding(BasePatchEmbedding):
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

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, N, D]
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        # [B, N, patch_size*C]
        x = x.reshape(B, self.n_patches, self.input_dim)
        return self.proj(x)  # [B, N, D]
