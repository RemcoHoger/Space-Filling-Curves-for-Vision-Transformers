import torch
import math
import torch.nn as nn
from einops import rearrange
# from flash_attn import flash_attn_func
# from flash_attn.modules.mha import FlashMHA

# Import the "frontend" patch tokenizer
from src.tokenizers.base_patch_embedding import BasePatchEmbedding
# from src.tokenizers.hilbert_embedding import HilbertEmbedding

########################################################################
# TokenAggregator
########################################################################


class TokenAggregator(nn.Module):
    """
    Aggregates neighbouring tokens with a depth‑wise separable Conv‑1d.
    Based on the paper which introduced the 'localformer'.
    ----
    dim : int           # embedding dimension
    k   : int = 3       # kernel size (paper uses 3)
    s   : int = 1       # stride   (paper uses 1)
    """

    def __init__(self, dim: int, k: int = 3, s: int = 1):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, k, s, padding=k//2, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1, 1)     # point‑wise
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                       # x: [B, N, D]
        # move channel to Conv1d’s expected position
        x = x.transpose(1, 2)                   # [B, D, N]
        x = self.pw(self.dw(x))                 # depthwise --> pointwise
        x = x.transpose(1, 2)                   # [B, N, D]
        return self.norm(self.act(x))


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

    def __init__(self, input_dim, max_len, n_head, hidden_dim, method,
                 dropout_p=0.1, n_layers=1):
        super().__init__()
        self.max_len = max_len
        self.grid_size = int(math.sqrt(max_len))
        # assert self.grid_size ** 2 == max_len, "max_len must be a perfect square."

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            dropout=dropout_p,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        # self.pos_embed = nn.Parameter(torch.randn(1, max_len, input_dim))

        # # initialize the learnable [CLS] token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        # initialize the positional encoding
        # patch_encoding = self._sinusoidal_positional_embedding(
        #     torch.arange(max_len), input_dim, method)
        # # make sure the [CLS] token is at the beginning and is initialized to 0
        # cls_row = torch.zeros(1, input_dim, device=patch_encoding.device,
        #                       dtype=patch_encoding.dtype)
        # self.register_buffer("pos_embed", torch.cat(
        #     [cls_row, patch_encoding], dim=0))

        self.to_patch_embedding = method

    def forward(self, x):
        """
        Forward pass to encode the input sequence.
        B = batch size,
        N = Number of patches,
        D = Embedding dimension.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D].

        Returns:
            torch.Tensor: Encoded [CLS] token of shape [B, D].
        """
        B = x.size(0)
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        # x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]
        # add positional embeddings
        # x = x + self.pos_embed[:x.size(1), :]  # [B, N, D]
        x = self.transformer(x)  # transformer encoding
        return x


########################################################################
# MultiLayerPredictor
########################################################################

class FactorisedLinear(nn.Module):
    """
    Factorised linear: (N·D) -> out_dim
      W ≈ (W_seq ⊗ W_emb) so   y = W_seq · (X · W_embᵀ)
    """

    def __init__(self, seq_len, embed_dim, rank, out_dim):
        super().__init__()
        self.W_emb = nn.Parameter(torch.empty(rank, embed_dim))
        self.W_seq = nn.Parameter(torch.empty(out_dim, seq_len, rank))
        nn.init.xavier_normal_(self.W_emb)
        nn.init.xavier_normal_(self.W_seq)

    def forward(self, x):                 # x: [B, N, D]
        h = torch.einsum('bnd, rd -> bnr', x, self.W_emb)    # mix embedding
        y = torch.einsum('bnr, onr -> bo',  h, self.W_seq)   # mix tokens
        return y


class MultiLayerPredictor(nn.Sequential):
    def __init__(self, embed_dim, seq_len,
                 n_layers=2, rank=64, dropout_p=0.5, num_classes=10):
        super().__init__()
        self.append(nn.LayerNorm(embed_dim))

        # factorised first layer
        fact_out = (seq_len * embed_dim) // 2
        self.append(FactorisedLinear(seq_len, embed_dim, rank, fact_out))
        self.append(nn.GELU())
        self.append(nn.Dropout(dropout_p))
        prev_dim = fact_out

        for _ in range(n_layers - 2):
            next_dim = prev_dim // 2
            self.append(nn.Linear(prev_dim, next_dim))
            self.append(nn.GELU())
            self.append(nn.Dropout(dropout_p))
            prev_dim = next_dim

        self.append(nn.Linear(prev_dim, num_classes))


########################################################################
# Regular VisionTransformer
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
        patch_embed: BasePatchEmbedding,
        embed_dim=128,
        depth=6,
        n_heads=4,
        mlp_dim=256,
        num_classes=10
    ):
        super().__init__()
        self.patch_embed = patch_embed
        embed_dim = patch_embed.embed_dim
        self.encoder = TransformerSeqEncoder(
            input_dim=embed_dim,
            max_len=self.patch_embed.n_patches,
            method=self.patch_embed,
            n_head=n_heads,
            hidden_dim=mlp_dim,
            n_layers=depth
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.n_patches, embed_dim))
        # self.ta = TokenAggregator(embed_dim)
        self.mlp_head = MultiLayerPredictor(
            embed_dim, self.patch_embed.n_patches, n_layers=2, num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                B = batch size,
                C = number of channels,
                H = height,
                W = width.

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes].
        """
        x = self.patch_embed(x)  # Convert image to patch embeddings
        # x = self.ta(x)           # Aggregate tokens
        x = x + self.pos_embed[:x.size(1), :]  # Add positional embeddings
        x = self.encoder(x)      # Encode patches using Transformer
        x = self.mlp_head(x)  # Classify using linear head
        return x

########################################################################
# 1D VisionTransformer
########################################################################


class VisionTransformer1D(nn.Module):
    """
    Vision Transformer (ViT) model for 1D data classification.

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
        patch_embed: BasePatchEmbedding,
        embed_dim=128,
        depth=6,
        n_heads=4,
        mlp_dim=256,
        num_classes=10
    ):
        super().__init__()
        self.patch_embed = patch_embed

        embed_dim = patch_embed.embed_dim
        self.encoder = TransformerSeqEncoder(
            input_dim=embed_dim,
            max_len=self.patch_embed.n_patches,
            n_head=n_heads,
            hidden_dim=mlp_dim,
            n_layers=depth,
            method=self.patch_embed
        )

        self.mlp_head = MultiLayerPredictor(
            embed_dim,
            self.patch_embed.n_patches,
            n_layers=2,
            dropout_p=0.5,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the RasterScan1DViT.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes].
        """
        x = self.patch_embed(x)  # [B, N, D]
        x = self.encoder(x)      # [B, N*D]
        return self.mlp_head(x)  # [B, num_classes]
