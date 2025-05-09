from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute MixUp loss = lam * L(pred, y_a) + (1-lam) * L(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch with progress tracking using tqdm.
    """
    model.train()
    total_loss, correct = 0, 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    total_loss, correct = 0, 0
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy


def train_with_scheduler(model, train_loader, criterion,
                         optimizer, scheduler, device):
    """
    Train the model for one epoch with scheduler and progress tracking.
    """
    model.train()
    scaler = torch.amp.grad_scaler()
    total_loss, correct = 0, 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        current_lr = scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy


def train_with_mixup(model, train_loader, criterion,
                     optimizer, scheduler, device, mixup_alpha=0.2):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            num_classes = outputs.size(1)
            y_a_1hot = F.one_hot(y_a, num_classes).float()
            y_b_1hot = F.one_hot(y_b, num_classes).float()
            soft_targets = lam * y_a_1hot + (1 - lam) * y_b_1hot
            loss = criterion(outputs, soft_targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = outputs.argmax(dim=1)
        total_correct += (lam * (preds == y_a).float() +
                          (1 - lam) * (preds == y_b).float()).sum().item()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    progress_bar.close()
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc