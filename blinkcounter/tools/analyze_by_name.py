"""Search YouTube for a person and analyze their blink rate across multiple videos.

Usage:
    python -m blinkcounter.tools.analyze_by_name "Oprah Winfrey"
    python -m blinkcounter.tools.analyze_by_name "David Attenborough" --videos 3
    python -m blinkcounter.tools.analyze_by_name "Taylor Swift" --max-duration 120
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Suppress noisy warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Suppress yt-dlp error output globally
import logging as _logging
_logging.getLogger("yt_dlp").setLevel(_logging.CRITICAL)

import cv2
import numpy as np


def search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for videos matching the query.

    Returns list of dicts with 'title', 'url', 'duration' keys.
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "simulate": True,
        "cachedir": False,
    }

    full_name_lower = query.lower().strip()

    # Search with multiple queries to maximise coverage
    search_queries = [
        f"ytsearch{max_results * 5}:{query}",  # Plain name (matches YouTube's top results)
        f"ytsearch{max_results * 3}:{query} interview",
        f"ytsearch{max_results * 3}:{query} speech talk",
    ]

    seen_urls = set()
    scored = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for sq in search_queries:
            try:
                # Suppress yt-dlp stderr output (ERROR lines for unavailable videos)
                import io
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    results = ydl.extract_info(sq, download=False)
                finally:
                    sys.stderr = old_stderr
            except Exception:
                continue

            for entry in results.get("entries", []):
                if not entry or entry.get("duration", 0) < 120:
                    continue  # Skip videos under 2 min — too short for reliable rates
                url = entry.get("webpage_url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title_lower = entry.get("title", "").lower()
                desc_lower = entry.get("description", "").lower()

                # Check for full name as a phrase in title or description
                in_title = full_name_lower in title_lower
                in_desc = full_name_lower in desc_lower

                if not in_title and not in_desc:
                    continue

                # Prefer longer videos (more data = more reliable rate)
                dur = entry.get("duration", 0)
                duration_bonus = min(dur / 300, 2.0)  # Up to 2.0 for 10min+
                score = (3 if in_title else 0) + (1 if in_desc else 0) + duration_bonus
                scored.append((score, entry))

    # Sort by score (best matches first) and take top results
    scored.sort(key=lambda x: x[0], reverse=True)
    videos = []
    for _, entry in scored[:max_results]:
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
    parser.add_argument(
        "--high-confidence", action="store_true",
        help="Only analyze high-quality segments (frontal, well-lit, stable). "
             "Discards ambiguous periods for more accurate rates on long videos.",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1,
        help="Process every Nth frame. 1=every frame (accurate), 2=2x faster, 3=3x faster (default: 1)",
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

    output_dir = args.output_dir or tempfile.mkdtemp(prefix="blinkcounter_")

    from blinkcounter.core.video_analyzer import VideoAnalyzer
    from blinkcounter.core.assessment import assess_person, format_assessment, format_multi_video_assessment
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Step 2: Download all videos first (sequential to avoid rate limits)
    downloaded = []  # (title, local_path)
    total_to_dl = min(args.videos, len(videos))
    for i, v in enumerate(videos[:args.videos]):
        mins = v["duration"] // 60
        secs = v["duration"] % 60
        print(f"  [{i+1}/{total_to_dl}] {v['title'][:60]}")
        print(f"         Downloading ({mins}:{secs:02d})...", end=" ", flush=True)

        try:
            from blinkcounter.services.youtube import download_video as dl_video, _get_cache_path
            cached = _get_cache_path(v["url"]).exists()
            local_path = dl_video(v["url"], output_dir=output_dir)
            local_path = trim_video(local_path, args.max_duration)
            print("cached" if cached else "done")
            downloaded.append((v["title"], local_path))
        except Exception as e:
            print(f"failed: {e}")

    if not downloaded:
        print("No videos downloaded successfully!")
        sys.exit(1)

    # Step 3: Analyze all videos in parallel
    n_vids = len(downloaded)
    print(f"\nAnalyzing {n_vids} videos in parallel...")
    all_results = []
    completed = [0]
    import threading as _threading
    _print_lock = _threading.Lock()

    # Track progress per video
    video_progress = {}

    def _print_progress_line():
        """Print a single line showing progress of all videos."""
        parts = []
        for i, (title, _) in enumerate(downloaded):
            short = title[:20]
            pct = video_progress.get(i, 0)
            if pct >= 100:
                parts.append(f"{short}: done")
            else:
                parts.append(f"{short}: {pct}%")
        with _print_lock:
            print(f"\r  {' | '.join(parts)}", end="", flush=True)

    def _analyze_one(idx_title_path):
        idx, title, path = idx_title_path
        video_progress[idx] = 0

        def _on_progress(val, msg=""):
            pct = int(val * 100)
            if pct > video_progress.get(idx, 0) + 5:  # Update every 5%
                video_progress[idx] = pct
                _print_progress_line()

        analyzer = VideoAnalyzer(high_confidence=args.high_confidence, frame_skip=args.frame_skip)
        result = analyzer.analyze(path, _on_progress)
        video_progress[idx] = 100
        _print_progress_line()
        return title, result

    start = time.time()
    indexed = [(i, t, p) for i, (t, p) in enumerate(downloaded)]
    with ThreadPoolExecutor(max_workers=min(n_vids, 4)) as pool:
        futures = {pool.submit(_analyze_one, itp): itp[1] for itp in indexed}
        for future in as_completed(futures):
            title = futures[future]
            completed[0] += 1
            try:
                title, result = future.result()
                elapsed = time.time() - start
                if result.persons:
                    n_persons = len(result.persons)
                    main = max(result.persons, key=lambda p: p.total_visible_duration)
                    analyzable = main.analyzable_duration
                    with _print_lock:
                        print(f"\n  [{completed[0]}/{n_vids}] {title[:50]}")
                        if analyzable < 30:
                            print(f"         skipped — only {analyzable:.0f}s analyzable (need 30s+)")
                        else:
                            print(f"         {n_persons} person(s), {main.blinks_per_minute:.1f} blinks/min "
                                  f"({analyzable:.0f}s analyzable, {elapsed:.0f}s)")
                            all_results.append((title, result))
                else:
                    with _print_lock:
                        print(f"\n  [{completed[0]}/{n_vids}] {title[:45]}... no faces ({elapsed:.0f}s)")
            except Exception as e:
                with _print_lock:
                    print(f"\n  {title[:45]}... error: {e}")

    total_elapsed = time.time() - start
    print(f"\nAll analysis complete ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")

    if len(all_results) < 1:
        print("No videos were successfully analyzed!")
        sys.exit(1)

    # Step 3: Match the TARGET PERSON across all videos
    # The person appearing in the most videos is likely the search subject
    if len(all_results) > 1:
        print(f"\nMatching '{args.name}' across {len(all_results)} videos...")
        common = find_common_person(all_results)

        if common and common["video_count"] > 1:
            # Show the matched person's face
            if common.get("face_thumbnail") is not None:
                face_path = os.path.join(output_dir, "matched_face.png")
                cv2.imwrite(face_path, common["face_thumbnail"])
                print(f"  Matched face saved: {face_path}")
                try:
                    subprocess.Popen(["open", face_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError:
                    pass

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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
