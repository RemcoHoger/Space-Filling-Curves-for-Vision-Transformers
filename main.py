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
from torchvision import datasets
from torchvision.transforms import v2 as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from src.tokenizers._2D.zigzag_embedding import ZigzagEmbedding
from src.tokenizers._1D.zigzag_embedding1D import RasterScan1DEmbedding
from src.tokenizers._1D.hilbert_embedding1D import HilbertEmbedding1D
# from src.models.vit import VisionTransformer
from src.models.vit import VisionTransformer1D, VisionTransformer
from src.training.scheduler import WarmupCosineScheduler
from src.training.train import train_with_scheduler, evaluate, train_with_mixup_or_cutmix


def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomErasing(0.2),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageFolder(
        root="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder/train",
        transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(
        root="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder/val",
        transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=512,
                             shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    patch_embed = HilbertEmbedding1D(
        img_size=224,
        patch_size=16 ** 2,
        in_channels=3,
        embed_dim=512
    )

    model = VisionTransformer1D(
        patch_embed=patch_embed,
        depth=8,
        n_heads=8,
        mlp_dim=2048,
        num_classes=1000
    ).to(device)

    model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 100
    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    checkpoint_path = "/projects/0/prjs1528/vit_checkpoints/hilbert"

    for epoch in range(epochs):
        train_loss, train_acc = train_with_mixup_or_cutmix(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
            f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}"
        )

        if (epoch + 1) % 5 == 0:
            checkpoint_file = f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, checkpoint_file)


if __name__ == '__main__':
    main()
