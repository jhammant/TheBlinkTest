"""Command-line interface for TheBlinkTest."""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def ensure_models():
    """Download required models if not present."""
    from pathlib import Path

    models_dir = Path(__file__).parent / "models"
    predictor = models_dir / "shape_predictor_68_face_landmarks.dat"

    if not predictor.exists():
        print("First run — downloading required models...")
        import subprocess
        script = models_dir / "download_models.sh"
        if script.exists():
            subprocess.run(["bash", str(script)], check=True)
        else:
            # Direct download of just the dlib predictor
            import urllib.request
            import bz2
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            bz2_path = str(predictor) + ".bz2"
            print(f"  Downloading dlib shape predictor...")
            urllib.request.urlretrieve(url, bz2_path)
            print(f"  Extracting...")
            with open(bz2_path, "rb") as f:
                data = bz2.decompress(f.read())
            with open(str(predictor), "wb") as f:
                f.write(data)
            import os
            os.remove(bz2_path)
            print("  Done!")


def main():
    parser = argparse.ArgumentParser(
        prog="theblinktest",
        description="Analyze blink rates in videos to detect anomalous blinking patterns.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a person by name (searches YouTube)")
    analyze_parser.add_argument("name", help="Person's name to search for")
    analyze_parser.add_argument("--videos", type=int, default=3, help="Number of videos (default: 3)")
    analyze_parser.add_argument("--high-confidence", action="store_true", help="Only use high-quality segments")
    analyze_parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    analyze_parser.add_argument("--max-duration", type=int, default=180, help="Max seconds per video (default: 180)")

    # video command
    video_parser = subparsers.add_parser("video", help="Analyze a local video file or YouTube URL")
    video_parser.add_argument("path", help="Path to video file or YouTube URL")
    video_parser.add_argument("--high-confidence", action="store_true", help="Only use high-quality segments")
    video_parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default: 1)")

    # gui command
    subparsers.add_parser("gui", help="Launch the graphical interface")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    ensure_models()

    if args.command == "analyze":
        from blinkcounter.tools.analyze_by_name import main as analyze_main
        # Inject args into sys.argv for the analyze_by_name script
        sys.argv = ["blink-analyze", args.name,
                     "--videos", str(args.videos),
                     "--max-duration", str(args.max_duration),
                     "--frame-skip", str(args.frame_skip)]
        if args.high_confidence:
            sys.argv.append("--high-confidence")
        analyze_main()

    elif args.command == "video":
        from blinkcounter.core.video_analyzer import VideoAnalyzer
        from blinkcounter.core.assessment import assess_person, format_assessment

        path = args.path
        if path.startswith("http"):
            from blinkcounter.services.youtube import download_video
            print(f"Downloading: {path}")
            path = download_video(path)

        print(f"Analyzing: {path}")
        analyzer = VideoAnalyzer(high_confidence=args.high_confidence, frame_skip=args.frame_skip)
        result = analyzer.analyze(path, lambda v, m="": print(f"\r  {int(v*100)}%", end="", flush=True))
        print()

        if result.persons:
            main_person = max(result.persons, key=lambda p: p.total_visible_duration)
            assessment = assess_person(main_person)
            print(format_assessment(assessment))
        else:
            print("No faces detected in video.")

    elif args.command == "gui":
        try:
            from blinkcounter.main import main as gui_main
            gui_main()
        except ImportError:
            print("GUI requires PyQt6. Install with: pip install theblinktest[gui]")
            sys.exit(1)


if __name__ == "__main__":
    main()
