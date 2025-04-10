import torch
import torch.nn as nn

# Import the "frontend" patch tokenizer
from src.tokenizers.patch_embedding import PatchEmbedding

########################################################################
# TransformerSeqEncoder
########################################################################


class TransformerSeqEncoder(nn.Module):
    """
    Transformer-based sequence encoder for processing patch embeddings.

    Args:
        input_dim (int): Dimensionality of the input embeddings.
        max_len (int): Maximum sequence length (number of patches).
        n_head (int): Number of attention heads.
        hidden_dim (int): Dimensionality of the feedforward network.
        dropout_p (float): Dropout probability.
        n_layers (int): Number of Transformer encoder layers.
    """

    def __init__(self, input_dim, max_len, n_head, hidden_dim,
                 dropout_p=0.1, n_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, input_dim))

    def forward(self, x):
        """
        Forward pass to encode the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D].

        Returns:
            torch.Tensor: Encoded [CLS] token of shape [B, D].
        """
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]
        # add positional embeddings
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)                        # transformer encoding
        return x[:, 0]  # Return the [CLS] token


########################################################################
# MultiLayerPredictor
########################################################################
class MultiLayerPredictor(nn.Sequential):
    """
    Multi-layer perceptron (MLP) for classification.

    Args:
        hidden_dim (int): Dimensionality of the hidden layers.
        n_layers (int): Number of layers in the MLP.
        dropout_p (float): Dropout probability.
        num_classes (int): Number of output classes.
    """

    def __init__(self, hidden_dim, n_layers=3, dropout_p=0.1, num_classes=10):
        super().__init__()
        for _ in range(n_layers - 1):
            self.append(nn.Linear(hidden_dim, hidden_dim))
            self.append(nn.GELU())
            self.append(nn.Dropout(dropout_p))
        self.append(nn.Linear(hidden_dim, num_classes))


########################################################################
# VisionTransformer
########################################################################
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    Args:
        img_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        embed_dim (int): Dimensionality of the patch embeddings.
        depth (int): Number of Transformer encoder layers.
        n_heads (int): Number of attention heads.
        mlp_dim (int): Dimensionality of the feedforward network.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        depth=6,
        n_heads=4,
        mlp_dim=256,
        num_classes=10
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        self.encoder = TransformerSeqEncoder(
            input_dim=embed_dim,
            max_len=self.patch_embed.n_patches,
            n_head=n_heads,
            hidden_dim=mlp_dim,
            n_layers=depth
        )
        self.mlp_head = MultiLayerPredictor(
            embed_dim, n_layers=2, num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes].
        """
        x = self.patch_embed(x)  # Convert image to patch embeddings
        x = self.encoder(x)      # Encode patches using Transformer
        x = self.mlp_head(x)     # Classify using MLP head
        return x
