from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    idx = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(H, W, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[idx]
    # Adjust lambda based on actual area used
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    return x, y_a, y_b, lam


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
    total_loss, correct = 0, 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Update learning rate
        current_lr = scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy


def train_with_mixup_or_cutmix(model, train_loader, criterion,
                               optimizer, scheduler, device,
                               mixup_alpha=0.2, cutmix_alpha=1.0,
                               mix_prob=0.5):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Randomly choose between Mixup and CutMix
        if np.random.rand() < mix_prob:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
        else:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)

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
