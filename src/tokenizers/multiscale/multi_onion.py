import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from functools import cache


class HierarchicalOnionEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, patch_size_list, embed_dim):
        super().__init__()
        self.levels = nn.ModuleList()
        pre_patch_size = 1
        pre_patch_list = []

        for patch_size in patch_size_list:
            self.levels.append(OnionEmbedding1D(
                img_size, pre_patch_size, patch_size, in_channels, embed_dim))
            pre_patch_list.append(pre_patch_size)
            pre_patch_size *= 2  # doubly larger patches at each level

        self.patch_list = [
            int(((img_size // pre_size) // np.sqrt(group_size)) ** 2)
            for pre_size, group_size in zip(pre_patch_list, patch_size_list)
        ]

        self.embed_dim = embed_dim * len(patch_size_list)
        self.depth = len(patch_size_list)
        self.n_patches = self.patch_list[0]  # from finest scale
        self.fusion = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        patches = [level(x) for level in self.levels]  # [B, N_i, D]

        # Upsample all coarser token streams to match finest scale
        target_len = self.n_patches
        for i in range(1, len(patches)):
            patches[i] = nn.functional.interpolate(
                patches[i].transpose(1, 2), size=target_len, mode='linear', align_corners=False
            ).transpose(1, 2)

        # Concatenate all patches along the feature dimension
        x = torch.cat(patches, dim=-1)  # [B, n_tokens, D * depth]
        return self.fusion(x)  # [B, n_tokens, D * depth]


class OnionEmbedding1D(nn.Module):
    def __init__(self, img_size, pre_patch_size, group_patch_size, in_channels, embed_dim):
        super().__init__()

        assert img_size % pre_patch_size == 0, "Image size must be divisible by pre_patch_size"
        self.img_size = img_size
        self.pre_patch_size = pre_patch_size
        self.group_patch_size = group_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = img_size // pre_patch_size
        self.n_pre_patches = self.grid_size * self.grid_size
        self.n_final_patches = self.n_pre_patches // group_patch_size

        self.pre_patch_dim = in_channels * pre_patch_size * pre_patch_size
        self.input_dim = self.pre_patch_dim * group_patch_size

        self.register_buffer("onion_indices", self._onion_indices(self.grid_size).long())

        self.proj = nn.Linear(self.input_dim, embed_dim)

    @cache
    def _onion_indices(self, size):
        H = W = size
        visited = np.zeros((H, W), dtype=bool)
        order = []
        dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # right, up, left, down
        dir_idx = 0
        i, j = H - 1, 0  # Start at bottom-left

        for _ in range(H * W):
            order.append((i, j))
            visited[i, j] = True
            ni, nj = i + dirs[dir_idx][0], j + dirs[dir_idx][1]
            if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                i, j = ni, nj
            else:
                dir_idx = (dir_idx + 1) % 4
                i, j = i + dirs[dir_idx][0], j + dirs[dir_idx][1]

        flat_indices = [r * W + c for r, c in order]
        return torch.tensor(flat_indices, dtype=torch.long)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.pre_patch_size

        # Step 0.5: Divide into small p×p pre-patches
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # [B, n_pre, d]

        # Step 1: Onion reorder of pre-patches
        x = x[:, self.onion_indices]  # [B, n_pre, d]

        # Step 2: Group pre-patches into final patches
        g = self.group_patch_size
        x = rearrange(x, f'b (n g) d -> b n (g d)', g=g)

        # Step 3: Projection
        x = self.proj(x)  # [B, n_final_patches, embed_dim]
        return x