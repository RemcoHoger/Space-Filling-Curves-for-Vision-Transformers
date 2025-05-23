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

import os
import torch
import numpy as np
from torchvision.transforms import v2 as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from src.tokenizers._2D.zigzag_embedding import ZigzagEmbedding
from src.tokenizers._1D.zigzag_embedding1D import RasterScan1DEmbedding
from src.tokenizers._1D.hilbert_embedding1D import HilbertEmbedding1D
from src.tokenizers._1D.peano_embedding1D import PeanoEmbedding1D
from src.tokenizers._1D.moore_embedding1D import MooreEmbedding1D
from src.tokenizers._1D.onion_embedding1D import OnionEmbedding1D
from src.tokenizers._1D.morton_embedding1D import MortonEmbedding1D
# from src.models.vit import VisionTransformer
from transformers import get_cosine_schedule_with_warmup
from src.models.vit import VisionTransformer1D, VisionTransformer
from src.training.train import evaluate, train_with_mixup_or_cutmix


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomErasing(0.2),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='/projects/0/prjs1528/',
        train=True,
        download=True,
        transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root='/projects/0/prjs1528/',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)


    patch_embed_dict = {
        "raster": RasterScan1DEmbedding,
        "hilbert": HilbertEmbedding1D,
        "peano": PeanoEmbedding1D,
        "moore": MooreEmbedding1D,
        "onion": OnionEmbedding1D,
        "morton": MortonEmbedding1D,
        "zigzag": ZigzagEmbedding
    }

    # patch_embed = ZigzagEmbedding(
    #     img_size=32,
    #     patch_size=4,
    #     in_channels=3,
    #     embed_dim=256
    # )

    for patch_embed_name, patch_embed_class in patch_embed_dict.items():
        print(f"Using {patch_embed_name} embedding")
        if patch_embed_name == "zigzag":
            patch_embed = patch_embed_class(
                img_size=32,
                patch_size=4,
                in_channels=3,
                embed_dim=256
            )
            model = VisionTransformer(
                patch_embed=patch_embed,
                depth=8,
                n_heads=4,
                mlp_dim=512,
                num_classes=10
            ).to(device)
        else:
            patch_embed = patch_embed_class(
                img_size=32,
                patch_size=16,
                in_channels=3,
                embed_dim=256
            )
            model = VisionTransformer1D(
                patch_embed=patch_embed,
                depth=8,
                n_heads=4,
                mlp_dim=512,
                num_classes=10
            ).to(device)

        model = torch.compile(model, mode="reduce-overhead")

        train_criterion = SoftTargetCrossEntropy()
        test_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        epochs = 300
        warmup_epochs = 10
        warmup_steps = warmup_epochs * len(train_loader)
        total_steps = epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # checkpoint_path = "/projects/0/prjs1528/vit_checkpoints/hilbert"
        # os.makedirs(checkpoint_path, exist_ok=True)
        # checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")

        for epoch in range(epochs):
            train_loss, train_acc = train_with_mixup_or_cutmix(
                model, train_loader, train_criterion, optimizer, scheduler, device
            )
            test_loss, test_acc = evaluate(model, test_loader, test_criterion, device)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}"
            )

            # if (epoch + 1) % 5 == 0:
            #     checkpoint_file = f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pt"
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler_state_dict': scheduler.state_dict(),
            #         'train_loss': train_loss,
            #         'train_acc': train_acc,
            #         'test_loss': test_loss,
            #         'test_acc': test_acc,
            #     }, checkpoint_file)


if __name__ == '__main__':
    main()
