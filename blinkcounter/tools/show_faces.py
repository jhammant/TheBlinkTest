"""Show detected faces from video analysis with blink rates.

Usage:
    python -m blinkcounter.tools.show_faces video.mp4
    python -m blinkcounter.tools.show_faces video.mp4 --output faces.png
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2
import numpy as np

from blinkcounter.core.models import BlinkClassification, Person
from blinkcounter.core.video_analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)

# Color map for classifications (BGR)
CLASSIFICATION_COLORS = {
    BlinkClassification.VERY_LOW: (0, 0, 200),    # Red
    BlinkClassification.LOW: (0, 140, 255),        # Orange
    BlinkClassification.NORMAL: (0, 180, 0),       # Green
    BlinkClassification.HIGH: (200, 200, 0),       # Cyan
}

CELL_WIDTH = 180
CELL_HEIGHT = 200
THUMB_SIZE = 128
COLUMNS = 4
PADDING = 10
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)


def build_gallery(persons: list[Person]) -> np.ndarray:
    """Build a grid image showing face thumbnails with blink rate info."""
    if not persons:
        # Return a small placeholder
        img = np.full((100, 300, 3), BG_COLOR[0], dtype=np.uint8)
        cv2.putText(img, "No faces detected", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 1)
        return img

    # Sort by visible time (most visible first)
    persons_sorted = sorted(persons, key=lambda p: p.total_visible_duration, reverse=True)

    cols = min(COLUMNS, len(persons_sorted))
    rows = (len(persons_sorted) + cols - 1) // cols

    img_w = cols * CELL_WIDTH + (cols + 1) * PADDING
    img_h = rows * CELL_HEIGHT + (rows + 1) * PADDING

    gallery = np.full((img_h, img_w, 3), BG_COLOR[0], dtype=np.uint8)

    for idx, person in enumerate(persons_sorted):
        row = idx // cols
        col = idx % cols

        x_start = col * CELL_WIDTH + (col + 1) * PADDING
        y_start = row * CELL_HEIGHT + (row + 1) * PADDING

        # Draw face thumbnail
        if person.face_thumbnail is not None:
            thumb = cv2.resize(person.face_thumbnail, (THUMB_SIZE, THUMB_SIZE),
                               interpolation=cv2.INTER_LINEAR)
        else:
            thumb = np.full((THUMB_SIZE, THUMB_SIZE, 3), 60, dtype=np.uint8)
            cv2.putText(thumb, "?", (45, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 2)

        # Center thumbnail horizontally in cell
        thumb_x = x_start + (CELL_WIDTH - THUMB_SIZE) // 2
        gallery[y_start:y_start + THUMB_SIZE, thumb_x:thumb_x + THUMB_SIZE] = thumb

        # Draw border around thumbnail with classification color
        classification = person.classification
        color = CLASSIFICATION_COLORS.get(classification, TEXT_COLOR)
        cv2.rectangle(gallery,
                      (thumb_x - 1, y_start - 1),
                      (thumb_x + THUMB_SIZE, y_start + THUMB_SIZE),
                      color, 2)

        # Text below thumbnail
        text_y = y_start + THUMB_SIZE + 18
        cv2.putText(gallery, person.label, (x_start + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        rate_text = f"{person.blinks_per_minute:.1f}/min"
        text_y += 18
        cv2.putText(gallery, rate_text, (x_start + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        class_text = classification.value
        text_y += 18
        cv2.putText(gallery, class_text, (x_start + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return gallery


def print_summary(persons: list[Person]) -> None:
    """Print a text summary of detected persons."""
    persons_sorted = sorted(persons, key=lambda p: p.total_visible_duration, reverse=True)

    print("\n=== Detected Persons ===")
    if not persons_sorted:
        print("No persons detected.")
        return

    for person in persons_sorted:
        classification = person.classification
        print(
            f"  {person.label:12s}  "
            f"{person.blinks_per_minute:5.1f}/min  "
            f"[{classification.value:9s}]  "
            f"({person.blink_count} blinks in {person.total_visible_duration:.1f}s visible, "
            f"{person.analyzable_duration:.1f}s analyzable)"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show detected faces with blink rates")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="detected_faces.png",
                        help="Output image path (default: detected_faces.png)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't try to display the image with cv2.imshow")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Analyzing video: {args.video}")
    analyzer = VideoAnalyzer()
    result = analyzer.analyze(args.video, progress_callback=_progress)

    print_summary(result.persons)

    gallery = build_gallery(result.persons)
    cv2.imwrite(args.output, gallery)
    print(f"Face gallery saved to: {args.output}")

    if not args.no_display:
        try:
            cv2.imshow("Detected Faces", gallery)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # No display available


def _progress(progress: float, message: str) -> None:
    """Simple progress callback."""
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {progress:5.1%} {message}", end="", flush=True)
    if progress >= 1.0:
        print()


if __name__ == "__main__":
    main()
