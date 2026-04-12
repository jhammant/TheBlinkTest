#!/usr/bin/env python3
"""Train a CNN to classify eye crops as open (0) or closed (1).

Uses the RT-BENE dataset (https://zenodo.org/records/3685316).
Expects data at /tmp/blinkcounter_training/rt_bene/ with extracted subject
folders (e.g., s012_noglasses/natural/left/) and label CSVs.

Usage:
    python train_eye_classifier.py [--data-dir DIR] [--epochs N] [--batch-size N]

The trained model is saved to blinkcounter/models/eye_state_classifier.pth.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class EyeStateCNN(nn.Module):
    """Small CNN for binary eye-state classification (open vs closed).

    Input: 3x64x64 RGB image
    Output: single logit (apply sigmoid for probability)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                   # -> 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RTBeneEyeDataset(Dataset):
    """Loads RT-BENE eye crops with open/closed labels.

    Each subject has:
      - sXXX_blink_labels.csv  (filename, label) where label 0.0=open, 1.0=closed
      - sXXX_noglasses/natural/left/  and  .../right/  with PNG eye images
    Both left and right eye images share the same label per frame.
    """

    def __init__(
        self,
        data_dir: str,
        subject_ids: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[str, float]] = []  # (image_path, label)

        for sid in subject_ids:
            label_file = self.data_dir / f"{sid}_blink_labels.csv"
            if not label_file.exists():
                print(f"Warning: label file {label_file} not found, skipping")
                continue

            # Read labels
            labels: dict[str, float] = {}
            with open(label_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        labels[row[0].strip()] = float(row[1])

            # Find image directories for this subject
            base = sid.replace("_blink_labels", "")
            # Try noglasses variant
            for variant in ["noglasses"]:
                for eye_side in ["left", "right"]:
                    eye_dir = self.data_dir / f"{base}_{variant}" / "natural" / eye_side
                    if not eye_dir.exists():
                        continue
                    for img_name in sorted(os.listdir(eye_dir)):
                        if not img_name.endswith(".png"):
                            continue
                        # Map right_XXXXXX_rgb.png -> left_XXXXXX_rgb.png for label lookup
                        label_key = img_name
                        if eye_side == "right":
                            label_key = img_name.replace("right_", "left_")
                        if label_key in labels:
                            img_path = str(eye_dir / img_name)
                            self.samples.append((img_path, labels[label_key]))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {data_dir} for subjects {subject_ids}. "
                "Make sure the data is extracted correctly."
            )

        # Report class balance
        n_closed = sum(1 for _, lbl in self.samples if lbl >= 0.5)
        n_open = len(self.samples) - n_closed
        print(f"Loaded {len(self.samples)} samples: {n_open} open, {n_closed} closed")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # Convert BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 64x64
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img, label_tensor

    def get_labels(self) -> list[float]:
        """Return all labels (used for building a weighted sampler)."""
        return [lbl for _, lbl in self.samples]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_weighted_sampler(dataset: RTBeneEyeDataset) -> WeightedRandomSampler:
    """Create a sampler that oversamples the minority class (closed eyes)."""
    labels = dataset.get_labels()
    n_closed = sum(1 for lbl in labels if lbl >= 0.5)
    n_open = len(labels) - n_closed

    weight_open = 1.0 / max(n_open, 1)
    weight_closed = 1.0 / max(n_closed, 1)

    sample_weights = [weight_closed if lbl >= 0.5 else weight_open for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Precision/recall bookkeeping (positive = closed)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
    }


# ---------------------------------------------------------------------------
# Discover available subjects from data directory
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: str) -> list[str]:
    """Find all subject IDs that have both label CSV and image directory."""
    data_path = Path(data_dir)
    subjects = []
    for csv_file in sorted(data_path.glob("s*_blink_labels.csv")):
        sid = csv_file.stem.replace("_blink_labels", "")
        # Check that image directory exists
        img_dir = data_path / f"{sid}_noglasses" / "natural" / "left"
        if img_dir.exists():
            subjects.append(sid)
    return subjects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train eye-state classifier on RT-BENE")
    parser.add_argument(
        "--data-dir",
        default="/tmp/blinkcounter_training/rt_bene",
        help="Path to extracted RT-BENE data",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--output",
        default=None,
        help="Output model path (default: blinkcounter/models/eye_state_classifier.pth)",
    )
    args = parser.parse_args()

    # Resolve output path
    project_root = Path(__file__).resolve().parent
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "blinkcounter" / "models" / "eye_state_classifier.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    subjects = discover_subjects(args.data_dir)
    if not subjects:
        print(f"ERROR: No subjects found in {args.data_dir}")
        print("Download RT-BENE data first. See download_rt_bene_data.sh")
        sys.exit(1)
    print(f"Found subjects: {subjects}")

    # Split: last subject for validation, rest for training
    val_subjects = subjects[-1:]
    train_subjects = subjects[:-1]
    print(f"Training on: {train_subjects}")
    print(f"Validating on: {val_subjects}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build datasets
    train_dataset = RTBeneEyeDataset(args.data_dir, train_subjects, transform=train_transform)
    val_dataset = RTBeneEyeDataset(args.data_dir, val_subjects, transform=val_transform)

    # Weighted sampler to handle class imbalance
    sampler = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model, loss, optimizer
    device = get_device()
    print(f"Using device: {device}")

    model = EyeStateCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True,
    )

    # Training loop
    best_val_acc = 0.0
    print(f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>8}  "
          f"{'ValAcc':>7}  {'ValPrec':>7}  {'ValRec':>7}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["accuracy"])

        print(
            f"{epoch:5d}  "
            f"{train_metrics['loss']:9.4f}  "
            f"{train_metrics['accuracy']:8.4f}  "
            f"{val_metrics['loss']:8.4f}  "
            f"{val_metrics['accuracy']:7.4f}  "
            f"{val_metrics['precision']:7.4f}  "
            f"{val_metrics['recall']:7.4f}"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "EyeStateCNN",
                    "input_size": 64,
                    "val_accuracy": best_val_acc,
                    "epoch": epoch,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                output_path,
            )
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f}) to {output_path}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
