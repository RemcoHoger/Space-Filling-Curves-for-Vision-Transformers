import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# class CustomTransformerEncoderLayer(nn.Module):
#     """
#     Exactly like nn.TransformerEncoderLayer (with batch_first=True), but
#     forward(…) returns (output, attn_weights) instead of just output.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         nhead: int,
#         dim_feedforward: int = 2048,
#         dropout: float = 0.1,
#         activation: str = "relu",
#         layer_norm_eps: float = 1e-5,
#         batch_first: bool = True,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.nhead = nhead
#         self.dim_feedforward = dim_feedforward
#         self.dropout = dropout
#         self.activation = activation
#         self.batch_first = batch_first

#         # Self‐Attention
#         # (PyTorch 1.11+ supports batch_first in MultiheadAttention)
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=nhead,
#             dropout=dropout,
#             batch_first=batch_first,
#         )
#         # Feed‐forward
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout1 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.dropout2 = nn.Dropout(dropout)

#         # LayerNorm layers
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

#         # Activation: either relu or gelu
#         if activation == "relu":
#             self.activation_fn = F.relu
#         elif activation == "gelu":
#             self.activation_fn = F.gelu
#         else:
#             raise ValueError(
#                 f"activation must be 'relu' or 'gelu', got {activation}")

#     def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None):
#         """
#         src: (batch, seq_len, d_model)   [because batch_first=True]
#         src_mask:          (seq_len, seq_len) or None
#         src_key_padding_mask: (batch, seq_len) or None

#         Returns:
#             output:        (batch, seq_len, d_model)
#             attn_weights:  (batch, num_heads, seq_len, seq_len)
#         """
#         # 1) Self‐attention block
#         # multihead_attn returns: (attn_output, attn_output_weights)
#         # attn_output: (batch, seq_len, d_model)
#         # attn_output_weights: (batch, num_heads, seq_len, seq_len)  (since batch_first=True)
#         attn_output, attn_weights = self.self_attn(
#             query=src,
#             key=src,
#             value=src,
#             attn_mask=src_mask,
#             key_padding_mask=src_key_padding_mask,
#             need_weights=True,
#             average_attn_weights=False,  # so we keep per-head maps
#         )

#         # Residual + LayerNorm
#         src2 = attn_output
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         # 2) Feedforward block
#         ff = self.linear2(self.dropout1(self.activation_fn(self.linear1(src))))
#         src2 = ff
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)

#         return src, attn_weights


# class CustomTransformerEncoder(nn.Module):
#     """
#     A stack of CustomTransformerEncoderLayer’s. In forward(x), returns:
#        (final_output, [attn_weights_layer0, attn_weights_layer1, … ])
#     exactly like nn.TransformerEncoder, but we collect all attn‐maps.
#     """

#     def __init__(self, encoder_layer: CustomTransformerEncoderLayer, num_layers: int, norm: nn.LayerNorm = None):
#         super().__init__()
#         # Clone the layer num_layers times
#         self.layers = nn.ModuleList(
#             [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src: torch.Tensor, mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None):
#         """
#         src: (batch, seq_len, d_model)
#         mask: (seq_len, seq_len) or None
#         src_key_padding_mask: (batch, seq_len) or None

#         Returns:
#             output: (batch, seq_len, d_model)
#             all_attn_weights: List[Tensor] of length num_layers, each
#                               of shape (batch, num_heads, seq_len, seq_len)
#         """
#         output = src
#         all_attn_weights = []

#         for mod in self.layers:
#             output, attn_w = mod(output, src_mask=mask,
#                                  src_key_padding_mask=src_key_padding_mask)
#             all_attn_weights.append(attn_w)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output, all_attn_weights


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


class MixerBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.token_mix_ln = nn.LayerNorm(embed_dim)
        self.channel_mix_ln = nn.LayerNorm(embed_dim)

        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_len),
        )

        self.channel_mix = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):  # x: [B, N, D]
        # x = x + \
        #     self.token_mix(self.token_mix_ln(
        #         x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(self.channel_mix_ln(x))
        return x


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
                 n_layers=2, rank=64, dropout_p=0.5, num_classes=10, mix=False):
        super().__init__()

        if mix:
            self.append(MixerBlock(seq_len, embed_dim, embed_dim * 2))
        else:
            self.append(nn.LayerNorm(embed_dim))

        # factorised first layer
        fact_out = embed_dim * 2
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
        # self.pos_embed = nn.Parameter(
        #     torch.randn(1, self.patch_embed.n_patches, embed_dim))
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
        # x = x + self.pos_embed[:x.size(1), :]  # Add positional embeddings
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

        self.mlp_mixer = MixerBlock(
            seq_len=self.patch_embed.n_patches,
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            out_dim=embed_dim
        )

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
        x = self.mlp_mixer(x)  # [B, N, D]
        x = self.encoder(x)      # [B, N*D]
        return self.mlp_head(x)  # [B, num_classes]

########################################################################
# Hierarchical Vision Transformer
########################################################################


class HierarchicalVisionTransformer1D(nn.Module):
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
        n_encoders = patch_embed.depth
        # create a total of n_encoders Transformer encoders
        self.encoder = nn.ModuleList([
            TransformerSeqEncoder(
                input_dim=embed_dim,
                max_len=patch_embed.patch_list[i],
                n_head=n_heads,
                hidden_dim=mlp_dim,
                n_layers=depth,
                method=self.patch_embed.levels[i]
            ) for i in range(n_encoders)
        ])

        # self.scale_embeddings = nn.ParameterList([
        #     nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in range(n_encoders)
        # ])

        self.fusion_encoder = TransformerSeqEncoder(
            input_dim=embed_dim,
            max_len=patch_embed.n_patches,
            n_head=n_heads,
            hidden_dim=mlp_dim,
            n_layers=2,
            method=self.patch_embed
        )

        self.mlp_head = MultiLayerPredictor(
            embed_dim,
            self.patch_embed.n_patches,
            n_layers=2,
            dropout_p=0.5,
            num_classes=num_classes,
            mix=True
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
        # Process each level of the hierarchical embedding
        for i, encoder in enumerate(self.encoder):
            x[i] = encoder(x[i])
            # x[i] = x[i] + self.scale_embeddings[i]  # Scale embeddings
        x = torch.cat(x, dim=1)
        x = self.fusion_encoder(x)
        return self.mlp_head(x)  # [B, num_classes]
