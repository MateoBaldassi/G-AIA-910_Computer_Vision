"""
ocr/cnn_model.py
=================
Lightweight CNN for Sudoku digit classification (classes 0–9).
Class 0 = empty cell, classes 1–9 = corresponding digit.

Architecture:
  Input: (B, 1, 64, 64) grayscale
  → Conv3×3(32) → BN → ReLU → MaxPool2
  → Conv3×3(64) → BN → ReLU → MaxPool2
  → Conv3×3(128) → BN → ReLU → MaxPool2
  → Flatten → FC(256) → Dropout(0.4) → FC(10)

~280k parameters — trains in < 5 min on CPU for 10k samples.
"""

import torch
import torch.nn as nn


class SudokuDigitCNN(nn.Module):
    """
    3-block CNN for digit classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (10: 0=empty, 1–9).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → (B, 32, 32, 32)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → (B, 64, 16, 16)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → (B, 128, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train_cnn(
    train_dir: str = "data/processed/train",
    val_dir: str = "data/processed/val",
    output_path: str = "models/cnn_digits.pth",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
):
    """
    Full training loop.

    Dataset structure (ImageFolder compatible):
        data/processed/train/
            0/  ← empty cell crops
            1/  ← crops of digit 1
            ...
            9/
        data/processed/val/
            0/ ... 9/
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import logging

    logger = logging.getLogger(__name__)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),   # augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    if device:
        selected_device = torch.device(device)
    else:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = selected_device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    if selected_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Training on CUDA GPU: {gpu_name}")
    else:
        if torch.version.cuda is None:
            logger.warning(
                "CUDA unavailable because the installed PyTorch build is CPU-only. "
                "Install a CUDA-enabled torch wheel to use your GPU."
            )
        logger.info("Training on CPU")
    logger.info(f"Dataset size — {len(train_ds)} train / {len(val_ds)} val samples")

    model = SudokuDigitCNN(num_classes=10).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, selected_device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, selected_device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info(f"  ✓ New best model saved ({val_acc:.4f})")

    logger.info(f"Training complete — best val acc: {best_val_acc:.4f}")
    return best_val_acc
