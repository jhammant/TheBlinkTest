"""Generate domain-matched training data for eye state CNN classifier.

Uses our EAR detector to auto-label eye crops from real videos.
Only labels high-confidence frames (clear open or clear closed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import dlib
import numpy as np

# Add project root to path so we can import blinkcounter modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blinkcounter.core.blink_detector import calculate_ear, estimate_head_pose, is_face_frontal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EAR_CLOSED_THRESHOLD = 0.18  # Well below threshold -> high-confidence closed
EAR_OPEN_THRESHOLD = 0.28    # Well above threshold -> high-confidence open
EYE_CROP_SIZE = 64
EYE_PAD_RATIO = 0.50         # Expand bounding box by 50% on each side
FRAME_SKIP = 3               # Process every 3rd frame

SHAPE_PREDICTOR_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"
)

# Left eye: landmarks 36-41, Right eye: landmarks 42-47
LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))

VIDEOS = [
    ("/tmp/blinkcounter_test/Eye Blink Rate Counter.mp4", "gt"),
    ("/tmp/blinkcounter_test/1 Minute NO BLINKING Challenge with Wednesday Addams!.mp4", "wed"),
    ("/tmp/blinkcounter_test/Jon Hammant @ AWS - DevOps and Transformation at Amazon.mp4", "jon"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_eye_crop(
    frame: np.ndarray,
    landmarks: np.ndarray,
    eye_indices: list[int],
) -> np.ndarray | None:
    """Extract a padded, resized eye crop from a frame.

    Args:
        frame: Full BGR video frame.
        landmarks: Array of shape (68, 2) with all landmark points.
        eye_indices: List of 6 landmark indices for one eye.

    Returns:
        Resized eye crop (64x64 BGR) or None if the crop is invalid.
    """
    eye_points = landmarks[eye_indices]

    # Bounding box around eye landmarks
    x_min, y_min = eye_points.min(axis=0)
    x_max, y_max = eye_points.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    # Expand by padding ratio on each side
    pad_w = width * EYE_PAD_RATIO
    pad_h = height * EYE_PAD_RATIO

    x_min = int(max(0, x_min - pad_w))
    y_min = int(max(0, y_min - pad_h))
    x_max = int(min(frame.shape[1], x_max + pad_w))
    y_max = int(min(frame.shape[0], y_max + pad_h))

    # Validate crop dimensions
    if x_max - x_min < 4 or y_max - y_min < 4:
        return None

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (EYE_CROP_SIZE, EYE_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    return resized


def process_video(
    video_path: str,
    prefix: str,
    output_dir: Path,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
) -> dict[str, int]:
    """Process a single video and extract labelled eye crops.

    Returns:
        Dict with counts: open, closed, skipped.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video: {video_path}")
        return {"open": 0, "closed": 0, "skipped": 0}

    open_dir = output_dir / "open"
    closed_dir = output_dir / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    stats = {"open": 0, "closed": 0, "skipped": 0}
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Processing {total_frames} frames (every {FRAME_SKIP}rd)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            # Convert shape to numpy array
            landmarks = np.array(
                [(shape.part(i).x, shape.part(i).y) for i in range(68)],
                dtype=np.float64,
            )

            # Check head pose — skip angled faces
            head_pose = estimate_head_pose(landmarks)
            if not is_face_frontal(head_pose):
                stats["skipped"] += 2  # Both eyes skipped
                continue

            # Process each eye
            for eye_name, eye_indices in [("left", LEFT_EYE_INDICES), ("right", RIGHT_EYE_INDICES)]:
                eye_landmarks = landmarks[eye_indices]
                ear = calculate_ear(eye_landmarks)

                # Classify based on EAR confidence thresholds
                if ear < EAR_CLOSED_THRESHOLD:
                    label = "closed"
                elif ear > EAR_OPEN_THRESHOLD:
                    label = "open"
                else:
                    # Ambiguous — skip
                    stats["skipped"] += 1
                    continue

                # Extract eye crop
                crop = extract_eye_crop(frame, landmarks, eye_indices)
                if crop is None:
                    stats["skipped"] += 1
                    continue

                # Save the crop
                filename = f"{prefix}_frame{frame_idx:06d}_{eye_name}.png"
                save_dir = open_dir if label == "open" else closed_dir
                cv2.imwrite(str(save_dir / filename), crop)
                stats[label] += 1

        frame_idx += 1

        # Progress update every 500 processed frames
        if frame_idx % (500 * FRAME_SKIP) == 0:
            print(f"    Frame {frame_idx}/{total_frames} — open: {stats['open']}, closed: {stats['closed']}, skipped: {stats['skipped']}")

    cap.release()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-supervised training data for eye state classifier"
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help="Video file paths (uses built-in list if none provided)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "blinkcounter" / "training_data"),
        help="Output directory for training data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir}")

    # Load dlib models
    if not SHAPE_PREDICTOR_PATH.exists():
        print(f"ERROR: Shape predictor not found at {SHAPE_PREDICTOR_PATH}")
        print("Run: bash blinkcounter/models/download_models.sh")
        sys.exit(1)

    print("Loading dlib face detector and shape predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))

    # Build video list
    if args.videos:
        video_list = [(v, Path(v).stem[:10]) for v in args.videos]
    else:
        video_list = VIDEOS

    # Check which videos exist
    available = []
    for video_path, prefix in video_list:
        if Path(video_path).exists():
            available.append((video_path, prefix))
        else:
            print(f"WARNING: Video not found, skipping: {video_path}")

    if not available:
        print("ERROR: No videos found. Provide video paths as arguments or place test videos in /tmp/blinkcounter_test/")
        sys.exit(1)

    # Process each video
    totals = {"open": 0, "closed": 0, "skipped": 0}

    for video_path, prefix in available:
        print(f"\nProcessing: {Path(video_path).name} (prefix={prefix})")
        stats = process_video(video_path, prefix, output_dir, detector, predictor)
        for key in totals:
            totals[key] += stats[key]
        print(f"  Results: open={stats['open']}, closed={stats['closed']}, skipped={stats['skipped']}")

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Open samples:    {totals['open']}")
    print(f"  Closed samples:  {totals['closed']}")
    print(f"  Skipped:         {totals['skipped']}")
    print(f"  Total saved:     {totals['open'] + totals['closed']}")
    print(f"  Output:          {output_dir}")


if __name__ == "__main__":
    main()
