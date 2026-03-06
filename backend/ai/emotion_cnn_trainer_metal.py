import os
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support


# ==========================
#  Device & Logging Setup
# ==========================

logger = logging.getLogger("EmotionCNNTrainer")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


def get_device() -> torch.device:
    """Select MPS (Metal) on Apple Silicon if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal/MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


# ==========================
#  Dataset: FER2013 CSV
# ==========================


class FER2013Dataset(Dataset):
    """Simple FER2013 dataset loader from CSV.

    Expects CSV with columns: 'emotion', 'pixels', 'Usage'.
    """

    def __init__(
        self,
        csv_path: str,
        usage: str = "Training",
        transform: Union[transforms.Compose, None] = None,
    ) -> None:
        import pandas as pd

        self.transform = transform
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == usage].reset_index(drop=True)

        self.emotions = df["emotion"].astype(int).values
        self.pixels = df["pixels"].values

    def __len__(self) -> int:
        return len(self.emotions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pixels_str = self.pixels[idx]
        img = np.fromstring(pixels_str, dtype=np.uint8, sep=" ").reshape(48, 48)
        img = np.expand_dims(img, axis=0)  # (1, 48, 48)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float() / 255.0

        label = int(self.emotions[idx])
        return img, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Face‑specific data augmentations for train/val."""

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(size=48, scale=(0.9, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    return train_transform, val_transform


# ==========================
#  Model Architecture
# ==========================


class ImprovedEmotionCNN(nn.Module):
    """Depthwise‑separable CNN with dilated conv and spatial attention."""

    def __init__(self, num_classes: int = 7, in_channels: int = 1) -> None:
        super().__init__()

        # Block 1: preserve high‑frequency details
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: dilated conv, no further spatial downsampling
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        attn = self.spatial_attn(x)
        x = x * attn
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==========================
#  Training Utilities
# ==========================


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes balanced class weights for FER2013 default 7 emotions.

    Args:
        labels (np.ndarray): labels of the training dataset
        num_classes (int): number of classes
        device (torch.device): device to store the tensor

    Returns:
        torch.Tensor: tensor containing the class weights
    """
    classes = np.arange(num_classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
    return weights_tensor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Union[torch.cuda.amp.GradScaler, None],
    max_grad_norm: float = 5.0,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        use_amp = scaler is not None and device.type in {"cuda", "mps"}
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

    val_loss = running_loss / total
    val_acc = correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "loss": val_loss,
        "acc": val_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ==========================
#  Main Training Loop
# ==========================


def train_fer2013(
    csv_path: str,
    output_dir: str = "./checkpoints",
    batch_size: int = 256,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 8,
    num_workers: int = 4,
) -> None:
    device = get_device()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_transform, val_transform = get_transforms()

    train_dataset = FER2013Dataset(csv_path=csv_path, usage="Training", transform=train_transform)
    val_dataset = FER2013Dataset(csv_path=csv_path, usage="PublicTest", transform=val_transform)

    num_classes = len(np.unique(train_dataset.emotions))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type != "mps"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "mps"),
    )

    # Compute class weights
    weights_tensor = compute_class_weights(
        labels=train_dataset.emotions,
        num_classes=num_classes,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    model = ImprovedEmotionCNN(num_classes=num_classes, in_channels=1).to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type in {"cuda", "mps"})

    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history: Dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch [%d/%d] "
            "Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f | "
            "Precision: %.4f | Recall: %.4f | F1: %.4f | "
            "LR: %.2e | Time: %.1fs",
            epoch + 1,
            num_epochs,
            train_loss,
            train_acc,
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            current_lr,
            elapsed,
        )

        # Early stopping by best F1
        if val_metrics["f1"] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_no_improve = 0

            ckpt_path = Path(output_dir) / "best_emotion_cnn.pt"
            to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(to_save.state_dict(), ckpt_path)
            logger.info("New best model saved at epoch %d with F1=%.4f", epoch + 1, best_val_f1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (best epoch %d, best F1=%.4f)",
                epoch + 1,
                best_epoch + 1,
                best_val_f1,
            )
            break

    # Save training history as numpy file
    history_path = Path(output_dir) / "training_history.npy"
    np.save(history_path, history, allow_pickle=True)
    logger.info("Training history saved to %s", history_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotion CNN Trainer (Metal/MPS‑optimized)")
    parser.add_argument("--csv", type=str, required=True, help="Path to FER2013 CSV file")
    parser.add_argument("--out", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    train_fer2013(
        csv_path=args.csv,
        output_dir=args.out,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.workers,
    )
