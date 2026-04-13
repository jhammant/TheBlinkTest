"""Validate blink detector against multiple ground truth datasets.

Supports:
- UBFC-rPPG: 42 subjects, frame-level blink annotations
- EyeBlink8: 4 subjects (8 videos), 408 blinks, frame-level .tag annotations
- Talking Face: 1 subject, 61 blinks, frame-level .tag annotations

Usage:
    python run_multi_dataset_validation.py
    python run_multi_dataset_validation.py --dataset eyeblink8
    python run_multi_dataset_validation.py --dataset all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def load_ubfc_annotations(annotation_dir: str) -> dict[str, list[int]]:
    """Load UBFC blink annotations (frame numbers per subject)."""
    annotations = {}
    ann_path = Path(annotation_dir)
    if not ann_path.exists():
        return {}
    for f in sorted(ann_path.glob("*_blink_groundtruth.txt")):
        subject = f.stem.replace("_blink_groundtruth", "")
        frames = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line and line.isdigit():
                    frames.append(int(line))
        annotations[subject] = frames
    return annotations


def load_eyeblink8_annotations(dataset_dir: str) -> list[tuple[str, str, list[int]]]:
    """Load EyeBlink8 .tag annotations.

    Format: frame:blink_id:left_state:...:landmarks
    blink_id > 0 means frame is part of that blink event.
    We extract the center frame of each blink event as the ground truth.

    Returns list of (name, video_path, gt_center_frames).
    """
    results = []
    ds_path = Path(dataset_dir)
    if not ds_path.exists():
        return []

    for subdir in sorted(ds_path.iterdir()):
        if not subdir.is_dir():
            continue

        tag_files = list(subdir.glob("*.tag"))
        vid_files = list(subdir.glob("*.avi"))
        if not tag_files or not vid_files:
            continue

        tag_file = tag_files[0]
        vid_file = vid_files[0]

        # Parse blink events from .tag file
        blink_events = {}  # blink_id -> list of frame numbers
        with open(tag_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                try:
                    frame_num = int(parts[0])
                    blink_id = int(parts[1])
                except ValueError:
                    continue
                if blink_id > 0:
                    if blink_id not in blink_events:
                        blink_events[blink_id] = []
                    blink_events[blink_id].append(frame_num)

        # Get center frame of each blink event
        gt_frames = []
        for blink_id in sorted(blink_events.keys()):
            frames = sorted(blink_events[blink_id])
            center = frames[len(frames) // 2]
            gt_frames.append(center)

        name = f"eb8_{subdir.name}"
        results.append((name, str(vid_file), gt_frames))

    return results


def load_talking_face_annotations(dataset_dir: str) -> list[tuple[str, str, list[int]]]:
    """Load Talking Face .tag annotations (same format as EyeBlink8).

    Returns list of (name, video_path, gt_center_frames).
    """
    ds_path = Path(dataset_dir)
    tag_file = ds_path / "talkingFace" / "talking.tag"
    vid_file = ds_path / "talkingFace" / "talking.avi"

    if not tag_file.exists() or not vid_file.exists():
        return []

    blink_events = {}
    with open(tag_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            try:
                frame_num = int(parts[0])
                blink_id = int(parts[1])
            except ValueError:
                continue
            if blink_id > 0:
                if blink_id not in blink_events:
                    blink_events[blink_id] = []
                blink_events[blink_id].append(frame_num)

    gt_frames = []
    for blink_id in sorted(blink_events.keys()):
        frames = sorted(blink_events[blink_id])
        center = frames[len(frames) // 2]
        gt_frames.append(center)

    return [("talking_face", str(vid_file), gt_frames)]


def match_blinks(
    detected_timestamps: list[float],
    ground_truth_timestamps: list[float],
    tolerance_seconds: float = 0.5,
) -> dict:
    """Match detected blinks to ground truth within tolerance window."""
    gt_matched = set()
    det_matched = set()

    for i, det_t in enumerate(detected_timestamps):
        best_dist = float("inf")
        best_gt_idx = -1
        for j, gt_t in enumerate(ground_truth_timestamps):
            if j in gt_matched:
                continue
            dist = abs(det_t - gt_t)
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = j

        if best_gt_idx >= 0 and best_dist <= tolerance_seconds:
            gt_matched.add(best_gt_idx)
            det_matched.add(i)

    tp = len(gt_matched)
    fp = len(detected_timestamps) - len(det_matched)
    fn = len(ground_truth_timestamps) - len(gt_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "gt_count": len(ground_truth_timestamps),
        "det_count": len(detected_timestamps),
    }


def analyze_subject(video_path: str, gt_frames: list[int], fps: float = 30.0, tolerance: float = 0.5, use_cnn: bool = False) -> dict:
    """Run blink detection on a video and compare against ground truth."""
    from blinkcounter.core.video_analyzer import VideoAnalyzer

    analyzer = VideoAnalyzer(use_cnn=use_cnn)
    result = analyzer.analyze(video_path, lambda v, m="": None)

    if not result.persons:
        return {"error": "No faces detected", "gt_count": len(gt_frames)}

    main = max(result.persons, key=lambda p: p.total_visible_duration)
    detected_timestamps = [e.timestamp for e in main.blink_events]
    gt_timestamps = [f / fps for f in gt_frames]

    metrics = match_blinks(detected_timestamps, gt_timestamps, tolerance)
    metrics["bpm_detected"] = main.blinks_per_minute
    metrics["visible_seconds"] = main.total_visible_duration
    return metrics


def run_dataset(name: str, subjects: list[tuple[str, str, list[int]]], fps: float, tolerance: float, use_cnn: bool = False) -> dict:
    """Run validation on a dataset and return aggregate metrics."""
    if not subjects:
        print(f"\n  {name}: No data found, skipping.\n")
        return {}

    total_gt = sum(len(gt) for _, _, gt in subjects)
    print(f"\n{'='*70}")
    print(f"  {name}: {len(subjects)} videos, {total_gt} ground truth blinks")
    print(f"  FPS: {fps}, Tolerance: ±{tolerance}s")
    print(f"{'='*70}\n")

    total_tp = total_fp = total_fn = 0
    all_metrics = []

    for subject_name, video_path, gt_frames in subjects:
        print(f"  {subject_name}: {len(gt_frames)} GT blinks...", end=" ", flush=True)
        metrics = analyze_subject(video_path, gt_frames, fps, tolerance, use_cnn=use_cnn)

        if "error" in metrics:
            print(f"ERROR: {metrics['error']}")
            continue

        all_metrics.append(metrics)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        print(f"det={metrics['det_count']} TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
              f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    if all_metrics:
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        print(f"\n  {name} TOTALS: TP={total_tp} FP={total_fp} FN={total_fn}")
        print(f"  Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

        return {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": p, "recall": r, "f1": f1,
                "subjects": len(all_metrics), "per_subject": all_metrics}

    return {}


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset blink detection validation")
    parser.add_argument("--dataset", default="all", choices=["all", "ubfc", "eyeblink8", "talking_face"],
                        help="Which dataset to validate against")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Match tolerance in seconds")
    parser.add_argument("--ubfc-ann-dir", default="/tmp/blinkcounter_test/ubfc_ground_truth")
    parser.add_argument("--ubfc-vid-dir", default="/tmp/blinkcounter_test/ubfc_videos")
    parser.add_argument("--eyeblink8-dir", default="/tmp/blinkcounter_test/eyeblink8/eyeblink8")
    parser.add_argument("--talking-face-dir", default="/tmp/blinkcounter_test/talking_face")
    parser.add_argument("--use-cnn", action="store_true", help="Enable CNN eye-state confirmation gate")
    args = parser.parse_args()

    results = {}

    if args.dataset in ("all", "eyeblink8"):
        subjects = load_eyeblink8_annotations(args.eyeblink8_dir)
        if subjects:
            results["eyeblink8"] = run_dataset("EyeBlink8", subjects, fps=30.0, tolerance=args.tolerance, use_cnn=args.use_cnn)

    if args.dataset in ("all", "talking_face"):
        subjects = load_talking_face_annotations(args.talking_face_dir)
        if subjects:
            # Talking Face video is 30 fps (5000 frames, 166.7s duration)
            results["talking_face"] = run_dataset("Talking Face", subjects, fps=30.0, tolerance=args.tolerance, use_cnn=args.use_cnn)

    if args.dataset in ("all", "ubfc"):
        annotations = load_ubfc_annotations(args.ubfc_ann_dir)
        if annotations:
            vid_dir = Path(args.ubfc_vid_dir)
            subjects = []
            for subject, frames in annotations.items():
                for pattern in [f"DATASET_2/{subject}/vid.avi", f"{subject}/vid.avi"]:
                    candidate = vid_dir / pattern
                    if candidate.exists():
                        subjects.append((subject, str(candidate), frames))
                        break
            if subjects:
                results["ubfc"] = run_dataset("UBFC-rPPG", subjects, fps=30.0, tolerance=args.tolerance, use_cnn=args.use_cnn)

    # Grand total across all datasets
    if len(results) > 1:
        grand_tp = sum(r.get("tp", 0) for r in results.values())
        grand_fp = sum(r.get("fp", 0) for r in results.values())
        grand_fn = sum(r.get("fn", 0) for r in results.values())
        p = grand_tp / (grand_tp + grand_fp) if (grand_tp + grand_fp) > 0 else 0
        r = grand_tp / (grand_tp + grand_fn) if (grand_tp + grand_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        print(f"\n{'='*70}")
        print(f"  GRAND TOTAL ACROSS ALL DATASETS")
        print(f"{'='*70}")
        print(f"  TP={grand_tp} FP={grand_fp} FN={grand_fn}")
        print(f"  Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")
        print(f"{'='*70}")

    # Save results
    output = Path(__file__).parent / "multi_dataset_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
