"""
    Name : Remco Hogerwerf
    UvAnetID : 14348462
    Study : Bsc Computer Science

    main.py

    This is an example script which trains a Vision Transformer on
    the CIFAR-10 dataset using a Zigzag embedding layer for patch
    extraction. The script includes data loading, model definition,
    training, and evaluation.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.tokenizers.hilbert_embedding import HilbertEmbedding
from src.tokenizers.zigzag_embedding import ZigzagEmbedding
from src.tokenizers.random_embedding import RandomEmbedding
from src.models.vit import VisionTransformer
from src.training.scheduler import WarmupCosineScheduler
from src.training.train import train_with_scheduler, evaluate


def plot_all_positional_encodings(pos_embed, title="GFPE-ViT positional encodings"):
    """
    Plot all positional encodings (excluding CLS token).

    Args:
        pos_embed (torch.Tensor): (num_patches + 1, dim) positional encodings.
    """
    num_patches_plus_cls, dim = pos_embed.shape
    num_patches = num_patches_plus_cls - 1

    plt.figure(figsize=(10, 6))
    for i in range(1, num_patches_plus_cls):  # skip CLS at index 0
        plt.plot(pos_embed[i].cpu().numpy(), alpha=0.4, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Encoding Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    embedding = HilbertEmbedding(
        img_size=32, patch_size=4, in_channels=3, embed_dim=128)

    model = VisionTransformer(
        patch_embed=embedding,
        embed_dim=128,
        depth=6,
        n_heads=4,
        mlp_dim=256,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 10
    warmup_epochs = 1
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    for epoch in range(epochs):
        train_loss, train_acc = train_with_scheduler(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
            f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}"
        )


if __name__ == '__main__':
    main()
