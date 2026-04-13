"""Train a vision-temporal blink model on UBFC eye crop sequences.

Instead of EAR values, feeds actual eye crop IMAGES through a small
CNN feature extractor + temporal aggregation. Sees what eyes look like,
not just geometric ratios.

Architecture:
  - MobileNetV2 (pretrained, frozen) extracts features per frame
  - Temporal 1D CNN aggregates across 7-frame window
  - Binary classifier: blink at center frame

Usage:
    python train_vision_temporal_model.py
    python train_vision_temporal_model.py --window 7 --epochs 30
"""

import argparse
import math
import os
from collections import deque
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


class VisionTemporalBlink(nn.Module):
    """Eye crop sequence → blink probability.

    Uses a frozen MobileNetV2 to extract per-frame features,
    then a 1D CNN for temporal classification.
    """

    def __init__(self, window_size: int = 7, feature_dim: int = 1280):
        super().__init__()
        self.window_size = window_size

        # Frozen feature extractor (MobileNetV2)
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        for param in self.features.parameters():
            param.requires_grad = False

        # Temporal classifier on feature sequences
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def extract_features(self, x):
        """Extract features from a batch of eye crops. x: (B, 3, H, W)"""
        with torch.no_grad():
            feat = self.features(x)
            feat = self.pool(feat).flatten(1)  # (B, 1280)
        return feat

    def forward(self, x):
        """x: (batch, window, 3, H, W) — sequence of eye crops."""
        B, W, C, H, Wd = x.shape
        # Extract features for all frames at once
        x_flat = x.view(B * W, C, H, Wd)
        feats = self.extract_features(x_flat)  # (B*W, 1280)
        feats = feats.view(B, W, -1)  # (B, W, 1280)
        feats = feats.permute(0, 2, 1)  # (B, 1280, W) for Conv1d
        temporal_out = self.temporal(feats)  # (B, 128, 1)
        temporal_out = temporal_out.squeeze(-1)  # (B, 128)
        return self.classifier(temporal_out)  # (B, 1)


def extract_eye_crop_sequences(
    video_path: str,
    gt_frames: list[int],
    window_size: int,
    predictor_path: str,
    crop_size: tuple = (64, 64),
) -> tuple[list[np.ndarray], list[int]]:
    """Extract sequences of eye crop images with labels."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)

    # Collect all eye crops
    all_crops = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        crop = np.zeros((*crop_size, 3), dtype=np.uint8)  # Default blank
        for face in faces:
            shape = predictor(gray, face)
            # Combined eye region
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 48)])
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            w, h = x2 - x1, y2 - y1
            px, py = int(w * 0.3), int(h * 0.6)
            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(frame.shape[1], x2 + px), min(frame.shape[0], y2 + py)
            if x2 > x1 and y2 > y1:
                crop = cv2.resize(frame[y1:y2, x1:x2], crop_size)
            break

        all_crops.append(crop)
        frame_num += 1

    cap.release()

    if len(all_crops) < window_size:
        return [], []

    # Build labels
    blink_frames = set()
    for gf in gt_frames:
        for off in range(-2, 3):
            blink_frames.add(gf + off)

    # Create windowed sequences
    half = window_size // 2
    sequences = []
    labels = []

    for i in range(half, len(all_crops) - half, 3):  # Step 3 for speed
        seq = all_crops[i - half: i + half + 1]
        label = 1 if i in blink_frames else 0
        sequences.append(np.stack(seq))  # (window, H, W, 3)
        labels.append(label)

    return sequences, labels


class EyeCropSequenceDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (window, H, W, 3) BGR
        frames = []
        for crop in seq:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if self.transform:
                tensor = self.transform(rgb)
            else:
                tensor = transforms.ToTensor()(rgb)
            frames.append(tensor)
        x = torch.stack(frames)  # (window, 3, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, label

    @property
    def num_positive(self):
        return sum(self.labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=7, help="Temporal window")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ann-dir", default="/tmp/blinkcounter_test/ubfc_ground_truth")
    parser.add_argument("--vid-dir", default="/tmp/blinkcounter_test/ubfc_videos/DATASET_2")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    output_path = project_root / "blinkcounter" / "models" / "vision_temporal_model.pth"
    predictor_path = str(project_root / "blinkcounter" / "models" / "shape_predictor_68_face_landmarks.dat")

    # Load annotations
    ann_dir = Path(args.ann_dir)
    vid_dir = Path(args.vid_dir)
    annotations = {}
    for f in sorted(ann_dir.glob("*_blink_groundtruth.txt")):
        subject = f.stem.replace("_blink_groundtruth", "")
        with open(f) as fh:
            frames = [int(l.strip()) for l in fh if l.strip().isdigit()]
        annotations[subject] = frames

    subjects = [(s, str(vid_dir / s / "vid.avi"), f)
                for s, f in annotations.items()
                if (vid_dir / s / "vid.avi").exists() and f]

    np.random.seed(42)
    np.random.shuffle(subjects)
    split = int(len(subjects) * 0.8)
    train_subjects = subjects[:split]
    val_subjects = subjects[split:]

    print(f"Train: {len(train_subjects)} subjects, Val: {len(val_subjects)} subjects")
    print(f"Extracting eye crop sequences (window={args.window})...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_seqs, train_labels = [], []
    for subject, vid_path, gt_frames in train_subjects:
        print(f"  {subject}...", end=" ", flush=True)
        seqs, labels = extract_eye_crop_sequences(vid_path, gt_frames, args.window, predictor_path)
        train_seqs.extend(seqs)
        train_labels.extend(labels)
        print(f"{len(seqs)} sequences ({sum(labels)} pos)")

    val_seqs, val_labels = [], []
    for subject, vid_path, gt_frames in val_subjects:
        print(f"  {subject} (val)...", end=" ", flush=True)
        seqs, labels = extract_eye_crop_sequences(vid_path, gt_frames, args.window, predictor_path)
        val_seqs.extend(seqs)
        val_labels.extend(labels)
        print(f"{len(seqs)} sequences ({sum(labels)} pos)")

    train_ds = EyeCropSequenceDataset(train_seqs, train_labels, transform)
    val_ds = EyeCropSequenceDataset(val_seqs, val_labels, transform)

    n_pos = train_ds.num_positive
    n_neg = len(train_ds) - n_pos
    print(f"\nTrain: {len(train_ds)} ({n_pos} pos, {n_neg} neg)")
    print(f"Val: {len(val_ds)} ({val_ds.num_positive} pos)")

    weights = [1.0 / n_neg if l == 0 else 1.0 / n_pos for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = VisionTemporalBlink(window_size=args.window).to(device)
    pos_weight = torch.tensor([math.sqrt(n_neg / max(n_pos, 1))], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"Pos weight: {pos_weight.item():.1f}x")
    print(f"\n{'Epoch':>5} {'Loss':>8} {'Acc':>7} {'ValP':>6} {'ValR':>6} {'ValF1':>6}")
    print("-" * 45)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = total = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs).squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state_dict": model.state_dict(), "window_size": args.window,
                         "val_f1": f1, "val_precision": p, "val_recall": r, "epoch": epoch},
                        output_path)
            marker = " *"

        print(f"{epoch:5d} {total_loss/len(train_loader):8.4f} {correct/total:7.3f} {p:6.3f} {r:6.3f} {f1:6.3f}{marker}")

    print(f"\nBest F1: {best_f1:.4f}")
    print(f"Model saved: {output_path}")


if __name__ == "__main__":
    main()
