import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BasePatchEmbedding(nn.Module, ABC):
    """
    Abstract base class for patch embedding.
    It defines how images are turned into sequences of patch embeddings.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input image of shape [B, C, H, W].

        Returns:
            torch.Tensor: A sequence of patch embeddings of shape [B, N, D].
        """
        pass
