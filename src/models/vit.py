import torch
import math
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func
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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class FlashAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, flash=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.flash = flash
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            qkv = self.to_qkv(x)
            qkv = rearrange(
                qkv, 'b n (three h d) -> b n three h d', three=3, h=self.heads)
            q, k, v = qkv.unbind(dim=2)

            if self.flash and x.is_cuda:
                out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
                out = rearrange(out, 'b n h d -> b n (h d)')
            else:
                # Fallback to standard attention
                scale = self.dim_head ** -0.5
                q, k, v = map(lambda t: rearrange(
                    t, 'b n h d -> b h n d'), (q, k, v))
                dots = torch.matmul(q, k.transpose(-1, -2)) * scale
                attn = dots.softmax(dim=-1)
                out = torch.matmul(attn, v)
                out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Attention(dim, heads=heads, dim_head=dim_head),
                FlashAttention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


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

        # Use transformer with flash attention to utilize the H100 architecture
        self.transformer = Transformer(
            dim=input_dim,
            depth=n_layers,
            heads=n_head,
            dim_head=input_dim // n_head,
            mlp_dim=hidden_dim
        )

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

    def _sinusoidal_positional_embedding(self, pos, dim, method,
                                         temperature=10000, dtype=torch.float32,
                                         T=4, h_param=3.0):
        h = w = int(math.sqrt(pos.size(0)))
        assert h * \
            w == pos.size(0), "pos must be a perfect square for 2D embedding"
        assert dim % 2 == 0, "feature dimension must be even for GFPE sincos embedding"

        # # Use encoding as introduced in the GFPE-ViT paper
        # if isinstance(method, HilbertEmbedding):
        #     hilbert_indices = method.hilbert_indices
        #     n = hilbert_indices.numel()
        #     N = int(math.sqrt(n))
        #     assert N*N == n
        #     assert dim % 2 == 0

        #     pos = hilbert_indices.to(torch.float32).unsqueeze(1)  # (n,1)
        #     i_ar = torch.arange(
        #         dim//2, dtype=torch.float32).unsqueeze(0)  # (1, d/2)
        #     two_pi = 2 * math.pi

        #     # scale  = (2*i * N^2 * pos * 2pi) / (T * n * d)
        #     scale = (2.0 * i_ar * N ** 2 * pos * two_pi) / (T * n * dim)

        #     # phase = h * (2*i * pos * 2pi) / d
        #     phase = h_param * (2.0 * i_ar * pos * two_pi) / dim

        #     arg = scale + phase  # (n, d/2)
        #     pe_sin = torch.sin(arg)
        #     pe_cos = torch.cos(arg)
        #     pe = torch.cat([pe_sin, pe_cos], dim=1)  # (n, d)

        #     return pe.type(dtype)

        # Default: classic sincos
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y = y.flatten()[:, None]
        x = x.flatten()[:, None]
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)
        y = y * omega
        x = x * omega
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        return pe.type(dtype)

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
        return x.flatten(1, 2)  # [B, N*D]


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

    def __init__(self, embed_dim, n_layers=2, dropout_p=0.5, num_classes=10):
        super().__init__()
        self.append(nn.LayerNorm(embed_dim))
        self.append(nn.Flatten())
        self.append(nn.Dropout(dropout_p))
        prev_dim = embed_dim
        for _ in range(n_layers - 1):
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
        # self.ta = TokenAggregator(embed_dim)
        self.mlp_head = MultiLayerPredictor(
            embed_dim * self.patch_embed.n_patches, n_layers=2, num_classes=num_classes)

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
            embed_dim * self.patch_embed.n_patches,
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
