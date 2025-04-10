import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings.

    Args:
        img_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        embed_dim (int): Dimensionality of the patch embeddings.

    Attributes:
        proj (nn.Conv2d): Convolutional layer to extract patch embeddings.
        n_patches (int): Total number of patches in the image.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        """
        Forward pass to compute patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Patch embeddings of shape [B, N, D].
        """
        x = self.proj(x)               # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
