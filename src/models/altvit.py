import torch
import math
from torch import nn
import numpy as np
from functools import cache
from einops import rearrange
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_1d(n_pos, dim, temperature: float = 10000.0, dtype=torch.float32):
    """
    Vaswani et al. sinusoidal positional embedding (1D).

    Args:
        n_pos (int): number of positions (e.g. sequence length).
        dim   (int): embedding dimension (must be even).
        temperature (float): base of the frequency schedule.
    Returns:
        pe : (n_pos, dim) tensor of sin/cos embeddings.
    """
    # Allocate
    pe = torch.zeros(n_pos, dim, dtype=dtype)
    # Position indices [0,1,...,n_pos-1]
    position = torch.arange(n_pos, dtype=dtype).unsqueeze(1)  # (n_pos,1)
    # Compute the inverse frequency terms: 1/temperature^(2i/dim)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=dtype) *
        (-math.log(temperature) / dim)
    )  # (dim/2,)

    # Apply sin to even indices, cos to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)   # even dims
    pe[:, 1::2] = torch.cos(position * div_term)   # odd dims

    return pe  # Shape: (n_pos, dim)

# classes


class HilbertPatchEmbedding(nn.Module):
    def __init__(self, *, image_size, patch_size, channels, dim):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width
        assert self.grid_h == self.grid_w and (self.grid_h & (self.grid_h - 1)) == 0, \
            "Hilbert curve requires square grid size that is a power of 2."

        patch_dim = channels * patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels

        self.layernorm1 = nn.LayerNorm(patch_dim)
        self.linear = nn.Linear(patch_dim, dim)
        self.layernorm2 = nn.LayerNorm(dim)

        self.hilbert_indices = self._hilbert_order(self.grid_h)

    @cache
    def _hilbert_order(self, n):
        coords = []

        def hilbert(x0, y0, xi, xj, yi, yj, n):
            if n <= 0:
                x = x0 + (xi + yi) // 2
                y = y0 + (xj + yj) // 2
                coords.append((x, y))
            else:
                hilbert(x0, y0,           yi//2, yj//2,           xi//2, xj//2, n-1)
                hilbert(x0 + xi//2, y0 + xj//2, xi//2, xj//2,     yi//2, yj//2, n-1)
                hilbert(x0 + xi//2 + yi//2, y0 + xj//2 + yj//2,
                        xi//2, xj//2, yi//2, yj//2, n-1)
                hilbert(x0 + xi//2 + yi, y0 + xj//2 + yj,
                        -yi//2, -yj//2, -xi//2, -xj//2, n-1)

        hilbert(0, 0, n, 0, 0, n, int(np.log2(n)))
        flat_indices = [y * n + x for x, y in coords]
        return torch.tensor(flat_indices, dtype=torch.long)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p1, p2 = self.patch_height, self.patch_width
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=p1, p2=p2)

        x = x[:, self.hilbert_indices.to(x.device)]  # reorder according to hilbert curve
        x = self.layernorm1(x)
        x = self.linear(x)
        return self.layernorm2(x)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

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
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.posemb = posemb_sincos_1d(
            n_pos=(image_height // patch_height) * (image_width // patch_width),
            dim=dim,
        )
        self.register_buffer("pos_embedding", self.posemb)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


class HilbertViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, T=4, h_param=3.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width

        assert self.grid_h == self.grid_w and (self.grid_h & (self.grid_h - 1)) == 0, \
            "Hilbert embedding requires square grid size that is a power of 2."

        self.to_patch_embedding = HilbertPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            dim=dim,
        )

        # Create 2D positional embedding using the new encoding scheme
        hilbert_indices = self.to_patch_embedding.hilbert_indices
        n = hilbert_indices.numel()
        N = int(math.sqrt(n))
        assert N * N == n, "Hilbert indices must form a square grid."
        assert dim % 2 == 0, "Feature dimension must be even."

        pos = hilbert_indices.to(torch.float32).unsqueeze(1)  # (n, 1)
        i_ar = torch.arange(dim // 2, dtype=torch.float32).unsqueeze(0)  # (1, d/2)
        two_pi = 2 * math.pi

        # scale = (2*i * N^2 * pos * 2π) / (T * n * d)
        scale = (2.0 * i_ar * N ** 2 * pos * two_pi) / (T * n * dim)

        # phase = h * (2*i * pos * 2π) / d
        phase = h_param * (2.0 * i_ar * pos * two_pi) / dim

        arg = scale + phase  # (n, d/2)
        pe_sin = torch.sin(arg)
        pe_cos = torch.cos(arg)
        pe = torch.cat([pe_sin, pe_cos], dim=1)  # (n, d)

        self.register_buffer("pos_embedding", pe.type(torch.float32))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(x.device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
