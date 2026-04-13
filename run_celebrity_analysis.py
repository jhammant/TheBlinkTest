"""Analyze blink rates of ~50 public figures across multiple videos.

Run overnight: python run_celebrity_analysis.py
Results saved to: celebrity_results.md
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2


SUBJECTS = [
    # Politicians
    "Barack Obama interview",
    "Joe Biden speech",
    "Vladimir Putin interview",
    "Angela Merkel interview",
    "Boris Johnson interview",
    "Jacinda Ardern interview",
    "Emmanuel Macron interview",
    "Justin Trudeau interview",
    "Narendra Modi interview",
    "Volodymyr Zelenskyy interview",

    # Tech Leaders
    "Mark Zuckerberg interview",
    "Tim Cook interview",
    "Satya Nadella interview",
    "Sundar Pichai interview",
    "Jensen Huang interview",
    "Sam Altman interview",
    "Elon Musk interview",
    "Bill Gates interview",
    "Jeff Bezos interview",
    "Lisa Su interview",

    # Business Leaders
    "Jamie Dimon interview",
    "Warren Buffett interview",
    "Bob Iger interview",
    "Mary Barra interview",
    "Indra Nooyi interview",
    "Reed Hastings interview",
    "Sheryl Sandberg interview",
    "Jack Welch interview",
    "Richard Branson interview",
    "Oprah Winfrey interview",

    # Actors/Entertainers
    "Tom Hanks interview",
    "Keanu Reeves interview",
    "Robert Downey Jr interview",
    "Scarlett Johansson interview",
    "Leonardo DiCaprio interview",
    "Meryl Streep interview",
    "Dwayne Johnson interview",
    "Cate Blanchett interview",
    "Morgan Freeman interview",
    "Anthony Hopkins interview",

    # Athletes/Sports
    "Cristiano Ronaldo interview",
    "Serena Williams interview",
    "Tom Brady interview",
    "Michael Jordan interview",
    "Conor McGregor interview",

    # Other Notable Figures
    "Jordan Peterson interview",
    "Neil deGrasse Tyson interview",
    "David Attenborough interview",
    "Gordon Ramsay interview",
    "Joe Rogan interview",
]


def search_youtube(query: str, max_results: int = 4) -> list[dict]:
    """Search YouTube for videos."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "simulate": True,
        "default_search": f"ytsearch{max_results + 3}",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{max_results + 3}:{query}", download=False)
            videos = []
            for entry in results.get("entries", []):
                if entry and entry.get("duration", 0) > 30:
                    videos.append({
                        "title": entry.get("title", "Unknown"),
                        "url": entry.get("webpage_url", ""),
                        "duration": entry.get("duration", 0),
                    })
            return videos[:max_results]
    except Exception as e:
        print(f"  Search error: {e}")
        return []


# Rate limiting to avoid hammering YouTube
_last_download_time = 0.0
DOWNLOAD_DELAY_SECONDS = 10  # Wait at least 10s between downloads


def download_and_trim(url: str, output_dir: str, max_seconds: int = 180) -> str | None:
    """Download video and trim to max_seconds. Rate-limited to avoid YouTube bans."""
    global _last_download_time

    # Rate limiting
    elapsed = time.time() - _last_download_time
    if elapsed < DOWNLOAD_DELAY_SECONDS:
        wait = DOWNLOAD_DELAY_SECONDS - elapsed
        print(f"    (rate limit: waiting {wait:.0f}s)", flush=True)
        time.sleep(wait)

    try:
        from blinkcounter.services.youtube import download_video
        _last_download_time = time.time()
        local_path = download_video(url, output_dir=output_dir)

        # Trim if needed
        cap = cv2.VideoCapture(local_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps

        if duration <= max_seconds:
            cap.release()
            return local_path

        trimmed = local_path.replace(".mp4", "_trimmed.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(trimmed, fourcc, fps, (w, h))

        count = 0
        max_frames = int(max_seconds * fps)
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            count += 1

        cap.release()
        out.release()
        return trimmed
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def analyze_video(video_path: str, high_confidence: bool = False, frame_skip: int = 3) -> dict | None:
    """Analyze a video and return results."""
    try:
        from blinkcounter.core.video_analyzer import VideoAnalyzer
        analyzer = VideoAnalyzer(high_confidence=high_confidence, frame_skip=frame_skip)
        result = analyzer.analyze(video_path, lambda v, m="": None)

        if not result.persons:
            return None

        main = max(result.persons, key=lambda p: p.total_visible_duration)
        if main.total_visible_duration < 5:
            return None

        analyzable_pct = (main.analyzable_duration / main.total_visible_duration * 100
                          if main.total_visible_duration > 0 else 0)
        return {
            "blink_count": main.blink_count,
            "blinks_per_minute": main.blinks_per_minute,
            "classification": main.classification.value,
            "visible_seconds": main.total_visible_duration,
            "analyzable_seconds": main.analyzable_duration,
            "analyzable_pct": analyzable_pct,
            "person_count": result.person_count,
        }
    except Exception as e:
        print(f"  Analysis error: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Celebrity blink rate analysis")
    parser.add_argument("--high-confidence", action="store_true",
                        help="Only analyze high-quality segments for more accurate rates")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing results and re-analyze all subjects")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="Process every Nth frame (default: 3 for speed)")
    cli_args = parser.parse_args()

    output_dir = tempfile.mkdtemp(prefix="blinkcounter_celeb_")
    suffix = "_hc" if cli_args.high_confidence else ""
    results_file = Path(__file__).parent / f"celebrity_results{suffix}.json"
    report_file = Path(__file__).parent / f"CELEBRITY_RESULTS{suffix}.md"

    # Load existing results to resume (unless --fresh)
    existing = {}
    if results_file.exists() and not cli_args.fresh:
        with open(results_file) as f:
            existing = json.load(f)

    all_results = existing.copy()

    print(f"\n{'='*60}")
    print(f"  CELEBRITY BLINK RATE ANALYSIS")
    print(f"  Analyzing {len(SUBJECTS)} subjects, up to 4 videos each")
    print(f"  Results: {report_file}")
    print(f"{'='*60}\n")

    for i, subject in enumerate(SUBJECTS):
        name = subject.replace(" interview", "").replace(" speech", "")

        if name in all_results:
            print(f"[{i+1}/{len(SUBJECTS)}] {name}: already analyzed, skipping")
            continue

        print(f"\n[{i+1}/{len(SUBJECTS)}] {name}")
        print(f"  Searching YouTube...")

        videos = search_youtube(subject, max_results=4)
        if not videos:
            print(f"  No videos found, skipping")
            all_results[name] = {"error": "No videos found"}
            continue

        video_rates = []
        video_details = []

        for j, video in enumerate(videos):
            print(f"  Video {j+1}: {video['title'][:50]}...")

            local = download_and_trim(video["url"], output_dir, max_seconds=180)
            if not local:
                continue

            print(f"    Analyzing...", end=" ", flush=True)
            start = time.time()
            result = analyze_video(local, high_confidence=cli_args.high_confidence, frame_skip=cli_args.frame_skip)
            elapsed = time.time() - start

            if result:
                print(f"{result['blink_count']} blinks, {result['blinks_per_minute']:.1f}/min [{result['classification']}] ({elapsed:.0f}s)")
                video_rates.append(result["blinks_per_minute"])
                video_details.append({
                    "title": video["title"][:60],
                    "blinks": result["blink_count"],
                    "bpm": result["blinks_per_minute"],
                    "classification": result["classification"],
                    "analyzable_s": result["analyzable_seconds"],
                })
            else:
                print(f"No faces detected ({elapsed:.0f}s)")

            # Clean up trimmed file
            try:
                if "_trimmed" in (local or ""):
                    os.remove(local)
            except OSError:
                pass

        if video_rates:
            avg_rate = sum(video_rates) / len(video_rates)
            from blinkcounter.core.models import BlinkClassification
            classification = BlinkClassification.from_rate(avg_rate).value

            all_results[name] = {
                "avg_bpm": round(avg_rate, 1),
                "classification": classification,
                "videos_analyzed": len(video_rates),
                "per_video": video_details,
            }
            print(f"  >> {name}: {avg_rate:.1f}/min [{classification}] (avg of {len(video_rates)} videos)")
        else:
            all_results[name] = {"error": "No faces detected in any video"}
            print(f"  >> {name}: No usable results")

        # Save progress after each person
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Generate markdown report after each person
        _write_report(all_results, report_file)

    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Results: {report_file}")
    print(f"{'='*60}\n")


def _write_report(all_results: dict, report_file: Path) -> None:
    """Generate markdown report from results."""
    lines = []
    lines.append("# Celebrity Blink Rate Analysis")
    lines.append("")
    lines.append("Blink rates of public figures analyzed across multiple YouTube videos.")
    lines.append("Normal blink rate: 15-20 blinks/min.")
    lines.append("")
    lines.append("| # | Name | Avg BPM | Classification | Videos | Notes |")
    lines.append("|---|------|---------|---------------|--------|-------|")

    # Sort by blink rate (lowest first — most interesting for psychopathy research)
    sorted_results = sorted(
        [(name, data) for name, data in all_results.items() if isinstance(data, dict) and "avg_bpm" in data],
        key=lambda x: x[1]["avg_bpm"],
    )

    for idx, (name, data) in enumerate(sorted_results, 1):
        avg = data["avg_bpm"]
        cls = data["classification"]
        vids = data.get("videos_analyzed", 0)

        # Flag interesting results
        notes = ""
        if avg < 10:
            notes = "Significantly below normal"
        elif avg < 15:
            notes = "Below normal"
        elif avg > 25:
            notes = "Above normal"

        lines.append(f"| {idx} | {name} | {avg} | {cls} | {vids} | {notes} |")

    # Errors
    errors = [(name, data) for name, data in all_results.items() if isinstance(data, dict) and "error" in data]
    if errors:
        lines.append("")
        lines.append("### Subjects with errors")
        lines.append("")
        for name, data in errors:
            lines.append(f"- {name}: {data['error']}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by [TheBlinkTest](https://github.com/jhammant/TheBlinkTest)*")
    lines.append("")
    lines.append("**Disclaimer:** This is experimental research. Blink rate alone cannot diagnose ")
    lines.append("any condition. Many factors affect blink rate. Do not use these results to make ")
    lines.append("judgments about individuals.")

    with open(report_file, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
