"""Validate blink detector against UBFC-rPPG ground truth annotations.

Uses frame-level blink annotations from UBFC dataset (42 subjects, 258 blinks)
to measure precision, recall, and F1 score.

A detected blink is "correct" if it falls within ±0.5s of an annotated blink frame.

Usage:
    python run_ground_truth_validation.py
    python run_ground_truth_validation.py --video-dir /path/to/ubfc/videos
    python run_ground_truth_validation.py --optimize  # Try different thresholds
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_annotations(annotation_dir: str) -> dict[str, list[int]]:
    """Load UBFC blink annotations (frame numbers per subject)."""
    annotations = {}
    ann_path = Path(annotation_dir)
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


def frames_to_timestamps(frames: list[int], fps: float = 30.0) -> list[float]:
    """Convert frame numbers to timestamps in seconds."""
    return [f / fps for f in frames]


def match_blinks(
    detected_timestamps: list[float],
    ground_truth_timestamps: list[float],
    tolerance_seconds: float = 0.5,
) -> dict:
    """Match detected blinks to ground truth within tolerance window.

    Returns dict with TP, FP, FN counts and matched pairs.
    """
    gt_matched = set()
    det_matched = set()

    # For each detected blink, find closest ground truth blink
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

    tp = len(gt_matched)  # True positives: correctly detected blinks
    fp = len(detected_timestamps) - len(det_matched)  # False positives: detected but no GT match
    fn = len(ground_truth_timestamps) - len(gt_matched)  # False negatives: GT blinks not detected

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_count": len(ground_truth_timestamps),
        "det_count": len(detected_timestamps),
    }


def analyze_subject(video_path: str, gt_frames: list[int], fps: float = 30.0) -> dict:
    """Run blink detection on a video and compare against ground truth."""
    from blinkcounter.core.video_analyzer import VideoAnalyzer

    analyzer = VideoAnalyzer()
    result = analyzer.analyze(video_path, lambda v, m="": None)

    if not result.persons:
        return {"error": "No faces detected", "gt_count": len(gt_frames)}

    # Get the main person
    main = max(result.persons, key=lambda p: p.total_visible_duration)
    detected_timestamps = [e.timestamp for e in main.blink_events]
    gt_timestamps = frames_to_timestamps(gt_frames, fps)

    metrics = match_blinks(detected_timestamps, gt_timestamps)
    metrics["bpm_detected"] = main.blinks_per_minute
    metrics["visible_seconds"] = main.total_visible_duration
    metrics["analyzable_seconds"] = main.analyzable_duration

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate against UBFC ground truth")
    parser.add_argument(
        "--annotation-dir",
        default="/tmp/blinkcounter_test/ubfc_ground_truth",
        help="Directory containing UBFC blink annotation files",
    )
    parser.add_argument(
        "--video-dir",
        default="/tmp/blinkcounter_test/ubfc_videos",
        help="Directory containing UBFC video files",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Tolerance window in seconds for matching blinks (default: 0.5)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Try different threshold parameters to find optimal settings",
    )
    args = parser.parse_args()

    # Load annotations
    annotations = load_annotations(args.annotation_dir)
    print(f"Loaded annotations for {len(annotations)} subjects ({sum(len(v) for v in annotations.values())} total blinks)")

    # Find available videos
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"\nVideo directory not found: {video_dir}")
        print("Download UBFC-rPPG videos from:")
        print("  https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk")
        print(f"\nPlace videos in: {video_dir}")
        print("\nShowing annotation statistics instead:\n")
        _show_annotation_stats(annotations)
        return

    # Find matching video files
    subjects_with_videos = []
    for subject, frames in annotations.items():
        # Try different naming patterns
        patterns = [
            f"{subject}/vid.avi",
            f"{subject}.avi",
            f"{subject}/video.avi",
            f"{subject}.mp4",
        ]
        video_path = None
        for pattern in patterns:
            candidate = video_dir / pattern
            if candidate.exists():
                video_path = str(candidate)
                break

        if video_path:
            subjects_with_videos.append((subject, video_path, frames))

    if not subjects_with_videos:
        # Try to find any video files and match by name
        for vid_file in video_dir.rglob("*.avi"):
            parent = vid_file.parent.name
            if parent in annotations:
                subjects_with_videos.append((parent, str(vid_file), annotations[parent]))
        for vid_file in video_dir.rglob("*.mp4"):
            parent = vid_file.parent.name
            if parent in annotations:
                subjects_with_videos.append((parent, str(vid_file), annotations[parent]))

    print(f"Found {len(subjects_with_videos)} subjects with matching videos")

    if not subjects_with_videos:
        print("No matching videos found. Showing annotation stats instead.\n")
        _show_annotation_stats(annotations)
        return

    # Run validation
    print(f"\n{'='*70}")
    print(f"  UBFC GROUND TRUTH VALIDATION")
    print(f"  {len(subjects_with_videos)} subjects, tolerance ±{args.tolerance}s")
    print(f"{'='*70}\n")

    all_metrics = []
    total_tp = total_fp = total_fn = 0

    for subject, video_path, gt_frames in subjects_with_videos:
        print(f"  {subject}: {len(gt_frames)} GT blinks...", end=" ", flush=True)
        metrics = analyze_subject(video_path, gt_frames)

        if "error" in metrics:
            print(f"ERROR: {metrics['error']}")
            continue

        all_metrics.append(metrics)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
              f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    # Summary
    if all_metrics:
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        print(f"\n{'='*70}")
        print(f"  OVERALL RESULTS ({len(all_metrics)} subjects)")
        print(f"{'='*70}")
        print(f"  True Positives:  {total_tp}")
        print(f"  False Positives: {total_fp}")
        print(f"  False Negatives: {total_fn}")
        print(f"  Precision:       {overall_precision:.3f}")
        print(f"  Recall:          {overall_recall:.3f}")
        print(f"  F1 Score:        {overall_f1:.3f}")
        print(f"{'='*70}")

        # Save results
        results = {
            "subjects_tested": len(all_metrics),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "tolerance_seconds": args.tolerance,
            "per_subject": all_metrics,
        }
        results_path = Path(__file__).parent / "ground_truth_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


def _show_annotation_stats(annotations: dict) -> None:
    """Show statistics from annotations without videos."""
    total_blinks = sum(len(v) for v in annotations.values())
    blink_counts = [len(v) for v in annotations.values()]

    print(f"Annotation Statistics:")
    print(f"  Subjects: {len(annotations)}")
    print(f"  Total blinks: {total_blinks}")
    print(f"  Mean blinks/subject: {np.mean(blink_counts):.1f}")
    print(f"  Min: {min(blink_counts)}, Max: {max(blink_counts)}")
    print(f"  Median: {np.median(blink_counts):.0f}")
    print()

    # Estimate blink rates (assuming ~60s videos at 30fps)
    print("  Estimated blink rates (assuming 60s videos at 30fps):")
    for subject, frames in sorted(annotations.items(), key=lambda x: len(x[1])):
        if frames:
            duration_s = max(frames) / 30.0  # Approximate from last blink frame
            rate = len(frames) / duration_s * 60 if duration_s > 0 else 0
            print(f"    {subject}: {len(frames)} blinks, ~{rate:.1f}/min")


if __name__ == "__main__":
    main()
