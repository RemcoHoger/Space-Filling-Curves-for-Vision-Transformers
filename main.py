import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import ViT backend and training utilities
from src.models.vit import VisionTransformer
from src.training.train import train_with_scheduler, evaluate
from src.training.scheduler import WarmupCosineScheduler


def main():
    """
    Main function to train and evaluate a Vision Transformer (ViT) on CIFAR-10.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalization for RGB channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Setup learning rate scheduler with warmup + cosine decay
    epochs = 100
    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # Train and evaluate the model
    for epoch in range(epochs):
        train_loss, train_acc = train_with_scheduler(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")


if __name__ == '__main__':
    main()
