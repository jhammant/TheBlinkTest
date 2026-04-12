"""Search YouTube for a person and analyze their blink rate across multiple videos.

Usage:
    python -m blinkcounter.tools.analyze_by_name "Oprah Winfrey"
    python -m blinkcounter.tools.analyze_by_name "David Attenborough" --videos 3
    python -m blinkcounter.tools.analyze_by_name "Taylor Swift" --max-duration 120
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for videos matching the query.

    Returns list of dicts with 'title', 'url', 'duration' keys.
    """
    import yt_dlp

    search_query = f"ytsearch{max_results}:{query} interview speech talk"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "simulate": True,
        "default_search": f"ytsearch{max_results}",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(search_query, download=False)
        videos = []
        for entry in results.get("entries", []):
            if entry and entry.get("duration", 0) > 30:
                videos.append({
                    "title": entry.get("title", "Unknown"),
                    "url": entry.get("webpage_url", ""),
                    "duration": entry.get("duration", 0),
                })
        return videos


def download_video(url: str, output_dir: str) -> str:
    """Download a YouTube video, return local path."""
    from blinkcounter.services.youtube import download_video as dl
    return dl(url, output_dir=output_dir)


def trim_video(input_path: str, max_seconds: int = 180) -> str:
    """Trim video to max_seconds, return path to trimmed file."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    if duration <= max_seconds:
        cap.release()
        return input_path

    output_path = input_path.replace(".mp4", "_trimmed.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    max_frames = int(max_seconds * fps)
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1

    cap.release()
    out.release()
    return output_path


def find_common_person(all_results: list) -> dict | None:
    """Find the person who appears across the most videos.

    Uses face_recognition encodings to match faces across video results.
    Returns dict with per-video data for the matched person.
    """
    import face_recognition

    if not all_results:
        return None

    # Collect all persons with encodings
    video_persons = []
    for video_label, result in all_results:
        persons_with_enc = [
            p for p in result.persons
            if p.face_encoding is not None and p.total_visible_duration > 5
        ]
        video_persons.append((video_label, persons_with_enc))

    if not video_persons:
        return None

    # For each person in video 1, check how many other videos they appear in
    best_match = None
    best_video_count = 0

    for _, persons in video_persons:
        for candidate in persons:
            if candidate.face_encoding is None:
                continue

            matched_videos = {}
            for video_label, other_persons in video_persons:
                for other in other_persons:
                    if other.face_encoding is None:
                        continue
                    dist = face_recognition.face_distance(
                        [candidate.face_encoding], other.face_encoding
                    )[0]
                    if dist < 0.7:
                        # Match found — keep the one with most visible time
                        if video_label not in matched_videos or other.total_visible_duration > matched_videos[video_label].total_visible_duration:
                            matched_videos[video_label] = other

            if len(matched_videos) > best_video_count:
                best_video_count = len(matched_videos)
                best_match = matched_videos

    if not best_match:
        return None

    # Build result
    per_video_rates = {}
    per_video_persons = {}
    total_blinks = 0
    total_analyzable = 0

    for video_label, person in best_match.items():
        per_video_rates[video_label] = person.blinks_per_minute
        per_video_persons[video_label] = person
        total_blinks += person.blink_count
        analyzable = person.analyzable_duration if person.analyzable_duration >= 1 else person.total_visible_duration
        total_analyzable += analyzable

    aggregate_rate = (total_blinks / total_analyzable * 60) if total_analyzable > 0 else 0

    return {
        "per_video_rates": per_video_rates,
        "per_video_persons": per_video_persons,
        "aggregate_rate": aggregate_rate,
        "total_analyzable": total_analyzable,
        "total_blinks": total_blinks,
        "video_count": len(best_match),
        "face_thumbnail": next(
            (p.face_thumbnail for p in best_match.values() if p.face_thumbnail is not None),
            None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search YouTube for a person and analyze their blink rate"
    )
    parser.add_argument("name", help="Person's name to search for")
    parser.add_argument(
        "--videos", type=int, default=3,
        help="Number of YouTube videos to search for (default: 3)",
    )
    parser.add_argument(
        "--max-duration", type=int, default=180,
        help="Max seconds per video to analyze (default: 180 = 3 min)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save downloaded videos (default: temp dir)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  BLINK RATE ANALYSIS: {args.name}")
    print(f"{'='*60}\n")

    # Step 1: Search YouTube
    print(f"Searching YouTube for '{args.name}'...")
    videos = search_youtube(args.name, max_results=args.videos + 2)  # Extra in case some fail

    if not videos:
        print("No videos found!")
        sys.exit(1)

    print(f"Found {len(videos)} videos:")
    for v in videos:
        mins = v["duration"] // 60
        secs = v["duration"] % 60
        print(f"  [{mins}:{secs:02d}] {v['title']}")
    print()

    # Step 2: Download and analyze
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="blinkcounter_")

    from blinkcounter.core.video_analyzer import VideoAnalyzer
    from blinkcounter.core.assessment import assess_person, format_assessment, format_multi_video_assessment

    all_results = []
    analyzed = 0

    for v in videos:
        if analyzed >= args.videos:
            break

        print(f"{'─'*60}")
        print(f"Video {analyzed + 1}: {v['title'][:50]}...")

        try:
            print("  Downloading...", end=" ", flush=True)
            local_path = download_video(v["url"], output_dir)
            print("done")

            # Trim if needed
            local_path = trim_video(local_path, args.max_duration)

            print("  Analyzing...", end=" ", flush=True)
            start = time.time()
            analyzer = VideoAnalyzer()
            result = analyzer.analyze(local_path, lambda val, msg="": None)
            elapsed = time.time() - start
            print(f"done ({elapsed:.0f}s)")

            if result.persons:
                main_person = max(result.persons, key=lambda p: p.total_visible_duration)
                print(f"  Main person: {main_person.blink_count} blinks, "
                      f"{main_person.blinks_per_minute:.1f}/min [{main_person.classification.value}]")
                all_results.append((v["title"], result))
                analyzed += 1
            else:
                print("  No faces detected, skipping")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n{'='*60}")

    if len(all_results) < 1:
        print("No videos were successfully analyzed!")
        sys.exit(1)

    # Step 3: Find common person across videos
    if len(all_results) > 1:
        print("\nMatching faces across videos...")
        common = find_common_person(all_results)

        if common and common["video_count"] > 1:
            print(format_multi_video_assessment(
                person_label=args.name,
                per_video_rates=common["per_video_rates"],
                aggregate_rate=common["aggregate_rate"],
                total_analyzable=common["total_analyzable"],
            ))
        else:
            print("\nCould not match the same person across videos.")
            print("Showing individual results:\n")
            for title, result in all_results:
                main_person = max(result.persons, key=lambda p: p.total_visible_duration)
                assessment = assess_person(main_person)
                print(f"--- {title[:50]} ---")
                print(format_assessment(assessment))
    else:
        # Single video
        title, result = all_results[0]
        main_person = max(result.persons, key=lambda p: p.total_visible_duration)
        assessment = assess_person(main_person)
        print(format_assessment(assessment))


if __name__ == "__main__":
    main()
