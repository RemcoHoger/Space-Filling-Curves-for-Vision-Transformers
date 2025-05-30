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
from torch.utils.data import DataLoader, Dataset, Subset
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


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def _load_train_data(self):
        classes = sorted(os.listdir(os.path.join(self.root, 'train')))
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_path = os.path.join(self.root, 'train', cls_name, 'images')
            for img_name in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def _load_val_data(self):
        val_annotations_file = os.path.join(
            self.root, 'val', 'val_annotations.txt')
        with open(val_annotations_file, 'r') as f:
            lines = f.readlines()

        # Get all class IDs from training set to maintain consistent indexing
        train_classes = sorted(os.listdir(os.path.join(self.root, 'train')))
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(train_classes)}

        img_to_cls = {}
        for line in lines:
            parts = line.strip().split('\t')
            img_to_cls[parts[0]] = parts[1]

        for img_name, cls_name in img_to_cls.items():
            img_path = os.path.join(self.root, 'val', 'images', img_name)
            if cls_name in self.class_to_idx:
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class ImageNet100Dataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._load_dataset()

    def _load_dataset(self):
        classes = sorted(os.listdir(self.root))
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_dir = os.path.join(self.root, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def filter_by_class(dataset, class_indices):
    selected_indices = [i for i, (_, label) in enumerate(
        dataset.samples) if label in class_indices]
    return Subset(dataset, selected_indices)


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

    # Imagenet
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])

    # CIFAR
    # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
    #                     std=(0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(64),
        # transforms.CenterCrop(128),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])

    # train_set = torchvision.datasets.ImageFolder(
    #     root="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder/train",
    #     transform=train_transform)
    # test_set = torchvision.datasets.ImageFolder(
    #     root="/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder/val",
    #     transform=test_transform)

    # first_100_classes = list(sorted(train_set.class_to_idx.values()))[:100]
    # train_set = filter_by_class(train_set, first_100_classes)
    # test_set = filter_by_class(test_set, first_100_classes)

    # train_set = torchvision.datasets.Imagenette(
    #     root='/projects/0/prjs1528/',
    #     split='train',
    #     download=True,
    #     transform=train_transform
    # )

    # test_set = torchvision.datasets.Imagenette(
    #     root='/projects/0/prjs1528/',
    #     split='val',
    #     download=True,
    #     transform=test_transform
    # )

    train_set = TinyImageNetDataset(
        root='/projects/0/prjs1528/tiny-imagenet-200',
        split='train',
        transform=train_transform
    )

    test_set = TinyImageNetDataset(
        root='/projects/0/prjs1528/tiny-imagenet-200',
        split='val',
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

    for patch_embed_name, patch_embed_class in patch_embed_dict.items():
        print(f"Using {patch_embed_name} embedding")
        if patch_embed_name == "zigzag":
            patch_embed = patch_embed_class(
                img_size=64,
                patch_size=4,
                in_channels=3,
                embed_dim=384
            )
            model = VisionTransformer(
                patch_embed=patch_embed,
                depth=12,
                n_heads=6,
                mlp_dim=1536,
                num_classes=200
            ).to(device)
        else:
            patch_embed = patch_embed_class(
                img_size=64,
                patch_size=4 ** 2,
                in_channels=3,
                embed_dim=384,
            )
            model = VisionTransformer1D(
                patch_embed=patch_embed,
                depth=12,
                n_heads=6,
                mlp_dim=1536,
                num_classes=200
            ).to(device)

        model = torch.compile(model, mode="reduce-overhead")

        train_criterion = SoftTargetCrossEntropy()
        test_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.00005)

        epochs = 300
        warmup_epochs = 10
        warmup_steps = warmup_epochs * len(train_loader)
        total_steps = epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        suffix = "_img200_definitive"
        checkpoint_path = f"/projects/0/prjs1528/vit_checkpoints/{patch_embed_name}{suffix}"
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_path, f"checkpoint{suffix}.pt")

        best_test_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = train_with_mixup_or_cutmix(
                model, train_loader, train_criterion, optimizer, scheduler, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, test_criterion, device)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}"
            )

            if test_acc > best_test_acc:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                checkpoint_file = f"{checkpoint_path}/checkpoint_epoch_{epoch+1}{suffix}.pt"
                print(
                    f"New best test accuracy: {test_acc:.4f} at epoch {epoch+1}. Checkpoint saved.")
                # First delete the old checkpoint if it exists
                # Save the model checkpoint
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
                best_test_acc = test_acc


if __name__ == '__main__':
    main()
