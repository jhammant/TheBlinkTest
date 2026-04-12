#!/usr/bin/env python3
"""Systematic accuracy test for the BlinkCounter blink detection system.

Tests the VideoAnalyzer against multiple videos with known characteristics,
ground truth comparison, cross-video consistency checks, and edge cases.
"""

import os
import sys
import tempfile
import time

import cv2

from blinkcounter.core.video_analyzer import VideoAnalyzer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_DIR = "/tmp/blinkcounter_test"
MAX_DURATION_SECONDS = 180  # trim videos longer than 3 minutes

GROUND_TRUTH_VIDEO = os.path.join(TEST_DIR, "Eye Blink Rate Counter.mp4")
GROUND_TRUTH_BLINKS = 27

TANUJA_VIDEOS = [
    os.path.join(TEST_DIR, "trimmed_Tanuja_-_AWS_AI_Mindset.mp4"),
    os.path.join(TEST_DIR, "trimmed_Tanuja_-_Business_AI.mp4"),
    os.path.join(TEST_DIR, "trimmed_Tanuja_-_Rejection_to_Power.mp4"),
]
TANUJA_LABELS = [
    "Tanuja - AI Mindset (trimmed 2m)",
    "Tanuja - Business AI (trimmed 2m)",
    "Tanuja - Rejection to Power (trimmed 2m)",
]

EDGE_CASES = {
    "Wednesday Addams (expect ~0 blinks)": os.path.join(
        TEST_DIR, "1 Minute NO BLINKING Challenge with Wednesday Addams!.mp4"
    ),
    "Trump 120s (false-positive-prone)": os.path.join(
        TEST_DIR, "trump_120s.mp4"
    ),
    "Jon Hammant (trimmed 3m)": os.path.join(
        TEST_DIR, "Jon Hammant @ AWS - DevOps and Transformation at Amazon.mp4"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trim_video_if_needed(video_path: str, max_seconds: float) -> str:
    """Return video_path unchanged if short enough, else create a trimmed copy."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    if duration <= max_seconds:
        return video_path

    max_frames = int(max_seconds * fps)
    suffix = os.path.splitext(video_path)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    print(f"  [trimmed {duration:.0f}s -> {max_seconds:.0f}s -> {tmp_path}]")
    return tmp_path


def analyze_video(analyzer: VideoAnalyzer, video_path: str, label: str):
    """Run analysis on a single video and return (result, elapsed, info_dict)."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {label}")
    print(f"  File: {video_path}")

    effective_path = trim_video_if_needed(video_path, MAX_DURATION_SECONDS)
    is_trimmed = effective_path != video_path

    cap = cv2.VideoCapture(effective_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"  Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {total_frames}")

    t0 = time.time()
    result = analyzer.analyze(effective_path)
    elapsed = time.time() - t0

    throughput = result.frames_processed / elapsed if elapsed > 0 else 0

    print(f"  Processing time: {elapsed:.1f}s | Throughput: {throughput:.1f} frames/s")
    print(f"  Persons detected: {result.person_count}")

    persons_info = []
    for p in result.persons:
        vis = p.total_visible_duration
        if vis < 5.0:
            print(f"    {p.label}: visible {vis:.1f}s (<5s, skipped)")
            continue
        rate = p.blinks_per_minute
        cls = p.classification.value
        print(
            f"    {p.label}: {p.blink_count} blinks in {vis:.1f}s "
            f"= {rate:.1f} bpm [{cls}]"
        )
        persons_info.append({
            "label": p.label,
            "blink_count": p.blink_count,
            "visible_duration": vis,
            "bpm": rate,
            "classification": cls,
        })

    # Clean up trimmed temp file
    if is_trimmed and os.path.exists(effective_path):
        os.unlink(effective_path)

    return result, elapsed, persons_info


def get_primary_person(result):
    """Return the person with the longest visibility."""
    if not result.persons:
        return None
    return max(result.persons, key=lambda p: p.total_visible_duration)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BlinkCounter Accuracy Test Suite")
    print("=" * 70)

    analyzer = VideoAnalyzer()
    summary_rows = []
    temp_files = []

    # ------------------------------------------------------------------
    # 1. Ground truth test
    # ------------------------------------------------------------------
    print("\n\n### SECTION 1: Ground Truth Comparison ###")
    print(f"Expected blinks: {GROUND_TRUTH_BLINKS}")

    gt_result, gt_elapsed, gt_persons = analyze_video(
        analyzer, GROUND_TRUTH_VIDEO, "Eye Blink Rate Counter (ground truth)"
    )
    primary = get_primary_person(gt_result)
    if primary and primary.total_visible_duration >= 5.0:
        detected = primary.blink_count
        accuracy = (1 - abs(detected - GROUND_TRUTH_BLINKS) / GROUND_TRUTH_BLINKS) * 100
        error = detected - GROUND_TRUTH_BLINKS
        fp_rate = max(0, error) / GROUND_TRUTH_BLINKS * 100 if GROUND_TRUTH_BLINKS > 0 else 0
        print(f"\n  ** Ground Truth Report **")
        print(f"  Expected: {GROUND_TRUTH_BLINKS} | Detected: {detected}")
        print(f"  Error: {error:+d} | Accuracy: {accuracy:.1f}%")
        print(f"  False positive rate: {fp_rate:.1f}%")
        summary_rows.append({
            "video": "Eye Blink Rate Counter",
            "person": primary.label,
            "blinks": detected,
            "bpm": primary.blinks_per_minute,
            "classification": primary.classification.value,
            "duration_s": primary.total_visible_duration,
            "time_s": gt_elapsed,
            "note": f"GT={GROUND_TRUTH_BLINKS}, acc={accuracy:.0f}%",
        })
    else:
        print("  WARNING: No primary person detected with >5s visibility!")
        summary_rows.append({
            "video": "Eye Blink Rate Counter",
            "person": "-",
            "blinks": 0,
            "bpm": 0,
            "classification": "-",
            "duration_s": 0,
            "time_s": gt_elapsed,
            "note": "NO PERSON DETECTED",
        })

    # ------------------------------------------------------------------
    # 2. Cross-video consistency (Tanuja)
    # ------------------------------------------------------------------
    print("\n\n### SECTION 2: Cross-Video Consistency (Tanuja) ###")

    tanuja_rates = []
    tanuja_counts = []

    for label, vpath in zip(TANUJA_LABELS, TANUJA_VIDEOS):
        if not os.path.exists(vpath):
            print(f"\n  SKIP (not found): {vpath}")
            continue

        t_result, t_elapsed, t_persons = analyze_video(analyzer, vpath, label)
        primary = get_primary_person(t_result)
        if primary and primary.total_visible_duration >= 5.0:
            tanuja_rates.append(primary.blinks_per_minute)
            tanuja_counts.append(primary.blink_count)
            summary_rows.append({
                "video": label,
                "person": primary.label,
                "blinks": primary.blink_count,
                "bpm": primary.blinks_per_minute,
                "classification": primary.classification.value,
                "duration_s": primary.total_visible_duration,
                "time_s": t_elapsed,
                "note": "",
            })
        else:
            summary_rows.append({
                "video": label,
                "person": "-",
                "blinks": 0,
                "bpm": 0,
                "classification": "-",
                "duration_s": 0,
                "time_s": t_elapsed,
                "note": "NO PERSON",
            })

    if len(tanuja_rates) >= 2:
        import statistics
        mean_rate = statistics.mean(tanuja_rates)
        stdev_rate = statistics.stdev(tanuja_rates) if len(tanuja_rates) > 1 else 0
        cv = (stdev_rate / mean_rate * 100) if mean_rate > 0 else 0
        print(f"\n  ** Tanuja Cross-Video Consistency **")
        print(f"  Blink rates (bpm): {[f'{r:.1f}' for r in tanuja_rates]}")
        print(f"  Mean: {mean_rate:.1f} bpm | StdDev: {stdev_rate:.1f} | CV: {cv:.1f}%")
        print(f"  Blink counts: {tanuja_counts}")
    else:
        print("\n  Not enough Tanuja videos analyzed for consistency check.")

    # ------------------------------------------------------------------
    # 3. Edge cases
    # ------------------------------------------------------------------
    print("\n\n### SECTION 3: Edge Cases ###")

    for edge_label, vpath in EDGE_CASES.items():
        if not os.path.exists(vpath):
            print(f"\n  SKIP (not found): {vpath}")
            continue

        e_result, e_elapsed, e_persons = analyze_video(analyzer, vpath, edge_label)
        primary = get_primary_person(e_result)
        note = ""
        if "Wednesday" in edge_label:
            if primary:
                note = f"expect ~0, got {primary.blink_count}"
            else:
                note = "no person detected"
        elif "Trump" in edge_label:
            note = "FP-prone subject"
        elif "Jon" in edge_label:
            note = "reference subject"

        if primary and primary.total_visible_duration >= 5.0:
            summary_rows.append({
                "video": edge_label,
                "person": primary.label,
                "blinks": primary.blink_count,
                "bpm": primary.blinks_per_minute,
                "classification": primary.classification.value,
                "duration_s": primary.total_visible_duration,
                "time_s": e_elapsed,
                "note": note,
            })
        else:
            summary_rows.append({
                "video": edge_label,
                "person": "-",
                "blinks": 0,
                "bpm": 0,
                "classification": "-",
                "duration_s": 0,
                "time_s": e_elapsed,
                "note": note or "NO PERSON",
            })

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    header = (
        f"{'Video':<42} {'Person':<10} {'Blinks':>6} {'BPM':>6} "
        f"{'Class':<10} {'Vis(s)':>7} {'Time(s)':>7}  {'Notes'}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['video']:<42} {row['person']:<10} {row['blinks']:>6} "
            f"{row['bpm']:>6.1f} {row['classification']:<10} "
            f"{row['duration_s']:>7.1f} {row['time_s']:>7.1f}  {row['note']}"
        )

    print("\n" + "=" * 70)
    total_time = sum(r["time_s"] for r in summary_rows)
    print(f"Total test time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print("=" * 70)


if __name__ == "__main__":
    main()
