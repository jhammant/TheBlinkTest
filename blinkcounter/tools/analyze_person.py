"""Analyze a specific person across multiple videos.

Usage:
    python -m blinkcounter.tools.analyze_person video1.mp4 video2.mp4 video3.mp4
    python -m blinkcounter.tools.analyze_person --youtube URL1 URL2 URL3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from typing import Optional

import cv2
import numpy as np

from blinkcounter.core.models import AnalysisResult, BlinkClassification, Person
from blinkcounter.core.video_analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_TRIM_SECONDS = 180  # 3 minutes
FACE_MATCH_TOLERANCE = 0.7


def trim_video(video_path: str, max_seconds: float) -> str:
    """Trim a video to the first max_seconds using cv2.

    Returns the path to the trimmed video (or original if already short enough).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    if duration <= max_seconds:
        cap.release()
        return video_path

    max_frames = int(max_seconds * fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    trimmed_path = os.path.join(
        tempfile.gettempdir(),
        f"blinkcounter_trimmed_{os.path.basename(video_path)}",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(trimmed_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    logger.info("Trimmed %s from %.0fs to %.0fs (%s)", video_path, duration, max_seconds, trimmed_path)
    return trimmed_path


def find_common_person(
    video_results: list[AnalysisResult],
    tolerance: float = FACE_MATCH_TOLERANCE,
) -> Optional[dict]:
    """Find the person who appears in the most videos.

    Returns a dict with:
        - person: the Person object from the first video
        - matches: list of (video_index, matched_person) tuples
        - video_count: number of videos they appear in
    """
    import face_recognition

    if not video_results:
        return None

    # Collect all persons with encodings from video 0 as candidates
    candidates = []
    for person in video_results[0].persons:
        if person.face_encoding is not None:
            candidates.append({
                "person": person,
                "matches": [(0, person)],
                "video_count": 1,
                "total_visible": person.total_visible_duration,
            })

    if not candidates:
        return None

    # For each candidate, check if they appear in subsequent videos
    for vid_idx in range(1, len(video_results)):
        vid_persons = video_results[vid_idx].persons
        vid_encodings = []
        vid_persons_with_enc = []
        for p in vid_persons:
            if p.face_encoding is not None:
                vid_encodings.append(p.face_encoding)
                vid_persons_with_enc.append(p)

        if not vid_encodings:
            continue

        for candidate in candidates:
            enc = candidate["person"].face_encoding
            distances = face_recognition.face_distance(vid_encodings, enc)
            best_idx = int(np.argmin(distances))

            if distances[best_idx] < tolerance:
                matched = vid_persons_with_enc[best_idx]
                candidate["matches"].append((vid_idx, matched))
                candidate["video_count"] += 1
                candidate["total_visible"] += matched.total_visible_duration

    # Sort: most videos first, then most total visible time
    candidates.sort(key=lambda c: (c["video_count"], c["total_visible"]), reverse=True)

    return candidates[0] if candidates else None


def _video_label(video_source: str, max_len: int = 35) -> str:
    """Extract a short label from a video path."""
    name = os.path.splitext(os.path.basename(video_source))[0]
    if len(name) > max_len:
        name = name[:max_len - 3] + "..."
    return name


def _progress(progress: float, message: str) -> None:
    """Simple progress callback."""
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {progress:5.1%} {message}", end="", flush=True)
    if progress >= 1.0:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a common person across multiple videos and report their blink rates"
    )
    parser.add_argument("videos", nargs="*", help="Paths to video files")
    parser.add_argument("--youtube", nargs="*", default=[], help="YouTube URLs to download and analyze")
    parser.add_argument("--trim", type=float, default=DEFAULT_TRIM_SECONDS,
                        help=f"Trim videos to first N seconds (default: {DEFAULT_TRIM_SECONDS})")
    parser.add_argument("--output", "-o", default=None,
                        help="Save best face thumbnail to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    video_paths = list(args.videos) if args.videos else []

    # Download YouTube videos if requested
    if args.youtube:
        from blinkcounter.services.youtube import download_video

        for url in args.youtube:
            print(f"Downloading: {url}")
            path = download_video(url)
            video_paths.append(path)

    if len(video_paths) < 2:
        print("Error: Need at least 2 videos for cross-video analysis.", file=sys.stderr)
        sys.exit(1)

    # Analyze each video
    analyzer = VideoAnalyzer()
    results: list[AnalysisResult] = []

    for i, video_path in enumerate(video_paths, 1):
        label = _video_label(video_path)
        print(f"\n--- Video {i}/{len(video_paths)}: {label} ---")

        # Trim long videos
        trimmed = trim_video(video_path, args.trim)
        if trimmed != video_path:
            print(f"  (trimmed to first {args.trim:.0f}s)")

        result = analyzer.analyze(trimmed, progress_callback=_progress)
        results.append(result)

        # Restore original source label
        result.video_source = video_path

        print(f"  Found {result.person_count} person(s), {result.frames_processed} frames processed")
        for p in result.persons:
            print(f"    {p.label}: {p.blinks_per_minute:.1f}/min [{p.classification.value}] "
                  f"({p.total_visible_duration:.0f}s visible)")

    # Cross-video matching
    print("\n" + "=" * 50)
    print("=== Cross-Video Analysis ===")

    common = find_common_person(results)

    if common is None or common["video_count"] < 2:
        print("No common person found across videos.")
        sys.exit(0)

    total_videos = len(results)
    print(f"Common person found in {common['video_count']}/{total_videos} videos\n")

    print("Per-video results:")
    total_duration = 0.0
    total_blinks = 0

    for vid_idx, matched_person in common["matches"]:
        result = results[vid_idx]
        label = _video_label(result.video_source)
        rate = matched_person.blinks_per_minute
        classification = matched_person.classification
        visible = matched_person.total_visible_duration
        total_duration += matched_person.analyzable_duration if matched_person.analyzable_duration >= 1.0 else visible
        total_blinks += matched_person.blink_count

        print(f"  Video {vid_idx + 1} ({label}):  "
              f"{rate:5.1f}/min [{classification.value:9s}]  "
              f"(visible {visible:.0f}s)")

    # Aggregate
    if total_duration > 0:
        aggregate_rate = (total_blinks / total_duration) * 60.0
    else:
        aggregate_rate = 0.0
    aggregate_class = BlinkClassification.from_rate(aggregate_rate)

    print(f"\nAggregate: {aggregate_rate:.1f}/min [{aggregate_class.value}] "
          f"(across {total_duration:.0f}s analyzable time)")

    # Save best thumbnail
    best_person = common["person"]
    if best_person.face_thumbnail is not None:
        output_path = args.output or "common_person.png"
        thumb = cv2.resize(best_person.face_thumbnail, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path, thumb)
        print(f"\nBest face thumbnail saved to: {output_path}")


if __name__ == "__main__":
    main()
