"""Train a temporal blink detection model on UBFC ground truth data.

Instead of classifying single EAR values, this model looks at a SEQUENCE
of 13 EAR values and predicts whether a blink occurs at the center frame.

This captures the temporal pattern: gradual close → hold → gradual open
that distinguishes real blinks from noise.

Usage:
    python train_temporal_blink_model.py
    python train_temporal_blink_model.py --window 13 --epochs 50
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ── Model ────────────────────────────────────────────────────────────────

class TemporalBlinkCNN(nn.Module):
    """1D CNN that classifies a sequence of EAR values as blink/no-blink.

    Input: (batch, 1, window_size) — sequence of EAR values
    Output: (batch, 1) — logit for blink probability
    """

    def __init__(self, window_size: int = 13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Data extraction ──────────────────────────────────────────────────────

def extract_ear_sequences_from_video(
    video_path: str,
    gt_frames: list[int],
    window_size: int = 13,
    predictor_path: str = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract EAR sequences and labels from a video with ground truth.

    For each frame, creates a window of EAR values centered on that frame.
    Label = 1 if the center frame is within ±3 frames of a GT blink, else 0.

    Returns (sequences, labels) where each sequence is shape (window_size,).
    """
    from blinkcounter.core.blink_detector import calculate_ear

    if predictor_path is None:
        predictor_path = str(Path(__file__).parent / "blinkcounter" / "models" / "shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Extract EAR for every frame
    all_ears = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        ear_val = 0.28  # Default if no face
        for face in faces:
            shape = predictor(gray, face)
            left = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)], dtype=np.float64)
            right = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)], dtype=np.float64)
            ear_val = (calculate_ear(left) + calculate_ear(right)) / 2.0
            break  # Use first face

        all_ears.append(ear_val)
        frame_num += 1

    cap.release()

    if len(all_ears) < window_size:
        return [], []

    # Create ground truth set (frames within ±3 of annotated blink)
    blink_frames = set()
    for gf in gt_frames:
        for offset in range(-3, 4):
            blink_frames.add(gf + offset)

    # Create windowed sequences
    half = window_size // 2
    sequences = []
    labels = []

    for i in range(half, len(all_ears) - half):
        seq = all_ears[i - half: i + half + 1]
        label = 1 if i in blink_frames else 0
        sequences.append(np.array(seq, dtype=np.float32))
        labels.append(label)

    return sequences, labels


class EARSequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], labels: list[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(0)  # (1, window)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label

    @property
    def num_positive(self):
        return sum(self.labels)

    @property
    def num_negative(self):
        return len(self.labels) - sum(self.labels)


# ── Training ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train temporal blink detection model")
    parser.add_argument("--window", type=int, default=13, help="Window size (frames)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--ann-dir", default="/tmp/blinkcounter_test/ubfc_ground_truth")
    parser.add_argument("--vid-dir", default="/tmp/blinkcounter_test/ubfc_videos/DATASET_2")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    output_path = project_root / "blinkcounter" / "models" / "temporal_blink_model.pth"

    # Load annotations
    ann_dir = Path(args.ann_dir)
    vid_dir = Path(args.vid_dir)

    annotations = {}
    for f in sorted(ann_dir.glob("*_blink_groundtruth.txt")):
        subject = f.stem.replace("_blink_groundtruth", "")
        with open(f) as fh:
            frames = [int(l.strip()) for l in fh if l.strip().isdigit()]
        annotations[subject] = frames

    # Find subjects with videos
    subjects = []
    for subject, frames in annotations.items():
        vid = vid_dir / subject / "vid.avi"
        if vid.exists() and frames:
            subjects.append((subject, str(vid), frames))

    print(f"Found {len(subjects)} subjects with videos + annotations")

    # Split: 80% train, 20% val
    np.random.seed(42)
    np.random.shuffle(subjects)
    split = int(len(subjects) * 0.8)
    train_subjects = subjects[:split]
    val_subjects = subjects[split:]

    print(f"Train: {len(train_subjects)} subjects, Val: {len(val_subjects)} subjects")

    # Extract EAR sequences
    print("\nExtracting EAR sequences from videos...")
    predictor_path = str(project_root / "blinkcounter" / "models" / "shape_predictor_68_face_landmarks.dat")

    train_seqs, train_labels = [], []
    for subject, vid_path, gt_frames in train_subjects:
        print(f"  {subject}...", end=" ", flush=True)
        seqs, labels = extract_ear_sequences_from_video(vid_path, gt_frames, args.window, predictor_path)
        train_seqs.extend(seqs)
        train_labels.extend(labels)
        pos = sum(labels)
        print(f"{len(seqs)} windows ({pos} positive)")

    val_seqs, val_labels = [], []
    for subject, vid_path, gt_frames in val_subjects:
        print(f"  {subject} (val)...", end=" ", flush=True)
        seqs, labels = extract_ear_sequences_from_video(vid_path, gt_frames, args.window, predictor_path)
        val_seqs.extend(seqs)
        val_labels.extend(labels)
        pos = sum(labels)
        print(f"{len(seqs)} windows ({pos} positive)")

    train_dataset = EARSequenceDataset(train_seqs, train_labels)
    val_dataset = EARSequenceDataset(val_seqs, val_labels)

    print(f"\nTrain: {len(train_dataset)} samples ({train_dataset.num_positive} pos, {train_dataset.num_negative} neg)")
    print(f"Val:   {len(val_dataset)} samples ({val_dataset.num_positive} pos, {val_dataset.num_negative} neg)")

    # Weighted sampler for imbalanced data
    n_pos = train_dataset.num_positive
    n_neg = train_dataset.num_negative
    weights = [1.0 / n_neg if l == 0 else 1.0 / n_pos for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = TemporalBlinkCNN(window_size=args.window).to(device)
    import math
    pos_weight = torch.tensor([math.sqrt(n_neg / max(n_pos, 1))], dtype=torch.float32).to(device)
    print(f"Pos weight: {pos_weight.item():.1f}x")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Train
    best_f1 = 0.0
    print(f"\n{'Epoch':>5} {'TrnLoss':>8} {'TrnAcc':>8} {'ValP':>6} {'ValR':>6} {'ValF1':>6}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

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

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validate
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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "window_size": args.window,
                "epoch": epoch,
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
            }, output_path)
            marker = " *"
        else:
            marker = ""

        print(f"{epoch:5d} {train_loss:8.4f} {train_acc:8.4f} {precision:6.3f} {recall:6.3f} {f1:6.3f}{marker}")

    print(f"\nBest F1: {best_f1:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
