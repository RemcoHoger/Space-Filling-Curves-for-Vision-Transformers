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

from src.tokenizers.hilbert_embedding import HilbertEmbedding
from src.tokenizers.zigzag_embedding import ZigzagEmbedding
from src.models.vit import VisionTransformer
from src.training.scheduler import WarmupCosineScheduler
from src.training.train import train_with_scheduler, evaluate


def main():
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

    patch_embed = HilbertEmbedding(
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128
    )

    model = VisionTransformer(
        patch_embed=patch_embed,
        depth=6,
        n_heads=4,
        mlp_dim=256,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 100
    warmup_epochs = 5
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
