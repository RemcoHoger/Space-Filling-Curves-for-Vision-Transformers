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
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.tokenizers._2D.zigzag_embedding import ZigzagEmbedding
from src.tokenizers._1D.zigzag_embedding1D import RasterScan1DEmbedding
from src.tokenizers._1D.hilbert_embedding1D import HilbertEmbedding1D
# from src.models.vit import VisionTransformer
from src.models.vit import VisionTransformer1D, VisionTransformer
from src.training.scheduler import WarmupCosineScheduler
from src.training.train import train_with_scheduler, evaluate


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
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageFolder(
        root="/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/train",
        transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(
        root="/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/val",
        transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    patch_embed = HilbertEmbedding1D(
        img_size=224,
        patch_size=16 ** 2,
        in_channels=3,
        embed_dim=768
    )

    model = VisionTransformer1D(
        patch_embed=patch_embed,
        depth=12,
        n_heads=12,
        mlp_dim=3072,
        num_classes=1000
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 1
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
