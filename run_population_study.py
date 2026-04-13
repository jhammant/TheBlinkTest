"""Population-scale blink rate study.

Analyzes ~1000 random people from YouTube to validate whether the blink
rate distribution matches expected population statistics:
- Mean ~17 blinks/min (Bentivoglio et al., 1997)
- ~1-3% with very low rates (<10/min) — potential psychopathy indicator
- Normal distribution centered around 15-20/min

Uses diverse YouTube searches to find random individuals in interviews,
talks, and conversations.

Usage:
    python run_population_study.py
    python run_population_study.py --target 100  # Smaller sample
    python run_population_study.py --resume       # Continue from last run
"""

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2

# Search queries designed to find diverse random individuals
SEARCH_QUERIES = [
    # Professional interviews (diverse backgrounds)
    "job interview practice real person",
    "employee spotlight video",
    "staff interview corporate",
    "teacher interview school",
    "nurse interview hospital",
    "engineer interview tech",
    "startup founder interview",
    "small business owner interview",
    "real estate agent interview",
    "lawyer interview law firm",
    "doctor interview medical",
    "firefighter interview",
    "police officer interview",
    "chef interview restaurant",
    "musician interview indie",
    "artist interview studio",
    "professor lecture university",
    "PhD student defense presentation",
    "TED talk unknown speaker",
    "TEDx talk local",
    "podcast interview guest",
    "YouTube interview small channel",
    "local news interview",
    "community leader interview",
    "volunteer interview charity",
    "athlete interview amateur",
    "coach interview sports",
    "farmer interview agriculture",
    "pilot interview aviation",
    "scientist interview research",
    # Diverse demographics
    "young professional interview 2024",
    "senior executive interview",
    "college student interview campus",
    "retiree interview life story",
    "immigrant story interview",
    "veteran interview military",
    "parent interview family",
    "teenager interview school project",
    # Geographic diversity
    "interview person London",
    "interview person New York",
    "interview person Tokyo",
    "interview person Sydney",
    "interview person Mumbai",
    "interview person Lagos",
    "interview person Berlin",
    "interview person Toronto",
    "interview person Seoul",
    "interview person Mexico City",
]

# Rate limiting
DOWNLOAD_DELAY = 12  # seconds between YouTube downloads
_last_download = 0.0


def search_youtube(query: str, max_results: int = 3) -> list[dict]:
    """Search YouTube."""
    import yt_dlp
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "simulate": True,
        "default_search": f"ytsearch{max_results + 2}",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{max_results + 2}:{query}", download=False)
            videos = []
            for entry in results.get("entries", []):
                if entry and 30 < entry.get("duration", 0) < 1800:  # 30s - 30min
                    videos.append({
                        "title": entry.get("title", "Unknown"),
                        "url": entry.get("webpage_url", ""),
                        "duration": entry.get("duration", 0),
                    })
            return videos[:max_results]
    except Exception as e:
        return []


def download_and_trim(url: str, output_dir: str, max_seconds: int = 120) -> str | None:
    """Download and trim video with rate limiting."""
    global _last_download
    elapsed = time.time() - _last_download
    if elapsed < DOWNLOAD_DELAY:
        time.sleep(DOWNLOAD_DELAY - elapsed)

    try:
        from blinkcounter.services.youtube import download_video
        _last_download = time.time()
        path = download_video(url, output_dir=output_dir)

        # Trim
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total / fps <= max_seconds:
            cap.release()
            return path

        trimmed = path.replace(".mp4", "_t.mp4")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(trimmed, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(int(max_seconds * fps)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        return trimmed
    except Exception:
        return None


def analyze_video(path: str) -> dict | None:
    """Analyze a video and return results for the main person."""
    try:
        from blinkcounter.core.video_analyzer import VideoAnalyzer
        analyzer = VideoAnalyzer()
        result = analyzer.analyze(path, lambda v, m="": None)
        if not result.persons:
            return None
        main = max(result.persons, key=lambda p: p.total_visible_duration)
        if main.total_visible_duration < 10:
            return None
        analyzable = main.analyzable_duration if main.analyzable_duration >= 1 else main.total_visible_duration
        if analyzable < 10:
            return None
        return {
            "bpm": round(main.blinks_per_minute, 1),
            "blinks": main.blink_count,
            "classification": main.classification.value,
            "analyzable_s": round(analyzable, 1),
            "visible_s": round(main.total_visible_duration, 1),
        }
    except Exception:
        return None


def generate_report(results: list[dict], output_path: Path) -> None:
    """Generate markdown report with statistics."""
    rates = [r["bpm"] for r in results if r.get("bpm") is not None]
    if len(rates) < 10:
        return

    mean = statistics.mean(rates)
    median = statistics.median(rates)
    stdev = statistics.stdev(rates)

    # Filter likely false positives (>35/min)
    clean_rates = [r for r in rates if r <= 35]
    clean_mean = statistics.mean(clean_rates) if clean_rates else 0
    clean_median = statistics.median(clean_rates) if clean_rates else 0

    # Distribution buckets
    buckets = {"<10": 0, "10-15": 0, "15-20": 0, "20-25": 0, "25-30": 0, ">30": 0}
    for r in rates:
        if r < 10: buckets["<10"] += 1
        elif r < 15: buckets["10-15"] += 1
        elif r < 20: buckets["15-20"] += 1
        elif r < 25: buckets["20-25"] += 1
        elif r < 30: buckets["25-30"] += 1
        else: buckets[">30"] += 1

    lines = [
        "# Population Blink Rate Study",
        "",
        f"Analyzed **{len(rates)}** individuals from random YouTube videos.",
        "",
        "## Expected vs Observed",
        "",
        "| Metric | Expected | Observed | Observed (filtered <35) |",
        "|--------|----------|----------|------------------------|",
        f"| Mean | ~17/min | {mean:.1f}/min | {clean_mean:.1f}/min |",
        f"| Median | ~17/min | {median:.1f}/min | {clean_median:.1f}/min |",
        f"| Std Dev | ~5/min | {stdev:.1f}/min | — |",
        f"| Very Low (<10/min) | 1-3% | {buckets['<10']/len(rates)*100:.1f}% ({buckets['<10']}/{len(rates)}) | — |",
        f"| Normal (15-20/min) | ~50% | {buckets['15-20']/len(rates)*100:.1f}% ({buckets['15-20']}/{len(rates)}) | — |",
        "",
        "## Distribution",
        "",
        "| Range | Count | Percentage | |",
        "|-------|-------|------------|---|",
    ]

    for k, v in buckets.items():
        pct = v / len(rates) * 100
        bar = "█" * int(pct / 2)
        lines.append(f"| {k}/min | {v} | {pct:.1f}% | {bar} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Sample size: {len(rates)} individuals",
        f"- {buckets['<10']} individuals ({buckets['<10']/len(rates)*100:.1f}%) showed Very Low blink rates (<10/min)",
        f"- Expected psychopathy prevalence in population: 1-3% (Hare, 2003)",
        f"- {'**Match**' if 0.5 <= buckets['<10']/len(rates)*100 <= 5 else '**Mismatch**'}: "
        f"Our Very Low rate of {buckets['<10']/len(rates)*100:.1f}% {'is' if 0.5 <= buckets['<10']/len(rates)*100 <= 5 else 'is not'} "
        f"consistent with expected psychopathy prevalence",
        "",
        "## Caveats",
        "",
        "- YouTube videos are biased toward public-facing individuals (may have different blink patterns)",
        "- Video quality, lighting, and camera angles affect detection accuracy",
        f"- {buckets['>30']} subjects ({buckets['>30']/len(rates)*100:.1f}%) had rates >30/min which likely include false positives",
        "- Blink rate alone cannot diagnose any condition",
        "",
        "---",
        "",
        "*Generated by [TheBlinkTest](https://github.com/jhammant/TheBlinkTest)*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Population-scale blink rate study")
    parser.add_argument("--target", type=int, default=1000, help="Target number of subjects")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    results_file = project_root / "population_results.json"
    report_file = project_root / "POPULATION_STUDY.md"
    output_dir = tempfile.mkdtemp(prefix="blinkcounter_pop_")

    # Load existing results
    existing_results = []
    existing_urls = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
            existing_urls = {r.get("url", "") for r in existing_results}
        print(f"Resuming: {len(existing_results)} existing results")

    all_results = list(existing_results)
    successful = len([r for r in all_results if r.get("bpm") is not None])

    print(f"\n{'='*60}")
    print(f"  POPULATION BLINK RATE STUDY")
    print(f"  Target: {args.target} subjects")
    print(f"  Current: {successful} successful analyses")
    print(f"{'='*60}\n")

    queries = list(SEARCH_QUERIES)
    query_idx = 0

    while successful < args.target:
        # Cycle through search queries
        query = queries[query_idx % len(queries)]
        query_idx += 1

        # Add randomness to queries
        suffix = random.choice(["2024", "2023", "2022", "real", "authentic", "genuine"])
        search = f"{query} {suffix}"

        print(f"[{successful}/{args.target}] Searching: {search[:50]}...")
        videos = search_youtube(search, max_results=2)

        for video in videos:
            if successful >= args.target:
                break
            if video["url"] in existing_urls:
                continue

            print(f"  Downloading: {video['title'][:45]}...", end=" ", flush=True)
            path = download_and_trim(video["url"], output_dir)
            if not path:
                print("failed")
                continue

            print("analyzing...", end=" ", flush=True)
            start = time.time()
            result = analyze_video(path)
            elapsed = time.time() - start

            entry = {
                "query": query,
                "title": video["title"][:60],
                "url": video["url"],
                "time_s": round(elapsed, 1),
            }

            if result:
                entry.update(result)
                successful += 1
                print(f"{result['bpm']}/min [{result['classification']}] ({elapsed:.0f}s)")
            else:
                entry["bpm"] = None
                print(f"no faces ({elapsed:.0f}s)")

            all_results.append(entry)
            existing_urls.add(video["url"])

            # Save progress
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

            # Update report every 10 successful results
            if successful % 10 == 0:
                valid = [r for r in all_results if r.get("bpm") is not None]
                if len(valid) >= 10:
                    generate_report(valid, report_file)
                    print(f"  >> Report updated: {len(valid)} subjects analyzed")

            # Clean up video file
            try:
                os.remove(path)
            except OSError:
                pass

    # Final report
    valid = [r for r in all_results if r.get("bpm") is not None]
    generate_report(valid, report_file)

    print(f"\n{'='*60}")
    print(f"  STUDY COMPLETE: {successful} subjects analyzed")
    print(f"  Report: {report_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
