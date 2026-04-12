"""
Adaptive EAR (Eye Aspect Ratio) blink detection test.

Uses a rolling baseline per person instead of a fixed threshold,
adapting to each individual's eye geometry.
"""

import time
from collections import deque
from statistics import median

import cv2
import dlib
import face_recognition
import numpy as np

# --- Config ---
VIDEO_PATH = "/tmp/blinkcounter_test/aws_120s.mp4"
MODEL_PATH = "blinkcounter/models/shape_predictor_68_face_landmarks.dat"

ROLLING_WINDOW = 30          # EAR samples for baseline
BLINK_OPEN_RATIO = 0.70      # EAR < baseline * 0.70 => eyes closing
BLINK_RECOVER_RATIO = 0.85   # EAR > baseline * 0.85 => eyes reopened
MIN_BLINK_MS = 30            # shortest valid blink
MAX_BLINK_MS = 500           # longest valid blink
FACE_MATCH_TOLERANCE = 0.7   # face_recognition distance threshold

# Landmark index ranges (dlib 68-point model)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def ear(eye_pts):
    """Compute Eye Aspect Ratio for 6 landmark points."""
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_a = np.linalg.norm(p2 - p6)
    vertical_b = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical_a + vertical_b) / (2.0 * horizontal)


class PersonTracker:
    """Per-person state for adaptive blink detection."""

    def __init__(self, person_id, encoding):
        self.person_id = person_id
        self.encoding = encoding
        self.ear_history = deque(maxlen=ROLLING_WINDOW)
        self.blink_count = 0
        self.first_seen_time = None
        self.last_seen_time = None
        self.in_blink = False
        self.blink_start_time = None
        self.baseline = None
        self.frame_count = 0

    def update(self, ear_value, timestamp):
        self.frame_count += 1
        if self.first_seen_time is None:
            self.first_seen_time = timestamp
        self.last_seen_time = timestamp

        self.ear_history.append(ear_value)

        # Need enough samples to establish a baseline
        if len(self.ear_history) < 5:
            return

        self.baseline = median(self.ear_history)
        close_thresh = self.baseline * BLINK_OPEN_RATIO
        open_thresh = self.baseline * BLINK_RECOVER_RATIO

        if not self.in_blink:
            if ear_value < close_thresh:
                self.in_blink = True
                self.blink_start_time = timestamp
        else:
            if ear_value > open_thresh:
                duration_ms = (timestamp - self.blink_start_time) * 1000
                if MIN_BLINK_MS <= duration_ms <= MAX_BLINK_MS:
                    self.blink_count += 1
                self.in_blink = False
                self.blink_start_time = None

    @property
    def visible_seconds(self):
        if self.first_seen_time is None or self.last_seen_time is None:
            return 0.0
        return self.last_seen_time - self.first_seen_time

    @property
    def blinks_per_minute(self):
        secs = self.visible_seconds
        if secs < 1:
            return 0.0
        return self.blink_count / secs * 60.0


def match_or_create_person(encoding, trackers):
    """Match a face encoding to an existing tracker, or create a new one."""
    if not trackers:
        person = PersonTracker(1, encoding)
        trackers.append(person)
        return person

    known_encodings = [t.encoding for t in trackers]
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] < FACE_MATCH_TOLERANCE:
        return trackers[best_idx]

    person = PersonTracker(len(trackers) + 1, encoding)
    trackers.append(person)
    return person


def main():
    print("=" * 60)
    print("  Adaptive EAR Blink Detection Test")
    print("=" * 60)
    print(f"  Video:            {VIDEO_PATH}")
    print(f"  Rolling window:   {ROLLING_WINDOW} samples")
    print(f"  Blink threshold:  baseline * {BLINK_OPEN_RATIO}")
    print(f"  Recovery thresh:  baseline * {BLINK_RECOVER_RATIO}")
    print(f"  Blink duration:   {MIN_BLINK_MS}-{MAX_BLINK_MS} ms")
    print(f"  Face tolerance:   {FACE_MATCH_TOLERANCE}")
    print("=" * 60)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    print(f"  FPS: {fps:.1f}  |  Frames: {total_frames}  |  Duration: {duration_s:.1f}s")
    print("=" * 60)

    trackers = []
    frame_num = 0
    start_wall = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_num / fps if fps > 0 else 0.0
        frame_num += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detector(gray, 0)

        if not faces:
            continue

        # Get face encodings for person tracking
        face_locations = [
            (f.top(), f.right(), f.bottom(), f.left()) for f in faces
        ]
        encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_rect, encoding in zip(faces, encodings):
            shape = predictor(gray, face_rect)
            landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            left_eye_pts = landmarks[LEFT_EYE]
            right_eye_pts = landmarks[RIGHT_EYE]

            left_ear = ear(left_eye_pts)
            right_ear = ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            person = match_or_create_person(encoding, trackers)
            person.update(avg_ear, timestamp)

        # Progress every 5 seconds of video
        if frame_num % int(fps * 5) == 0:
            elapsed = frame_num / fps
            print(f"  [Progress] {elapsed:.0f}s / {duration_s:.0f}s  "
                  f"({frame_num}/{total_frames} frames)  "
                  f"Persons tracked: {len(trackers)}")

    cap.release()
    wall_time = time.time() - start_wall

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Processed {frame_num} / {total_frames} frames in {wall_time:.1f}s")
    print(f"  Persons detected: {len(trackers)}")
    print()

    for t in trackers:
        vis = t.visible_seconds
        bpm = t.blinks_per_minute
        baseline_str = f"{t.baseline:.4f}" if t.baseline else "N/A"
        close_str = f"{t.baseline * BLINK_OPEN_RATIO:.4f}" if t.baseline else "N/A"
        recover_str = f"{t.baseline * BLINK_RECOVER_RATIO:.4f}" if t.baseline else "N/A"

        print(f"  --- Person {t.person_id} ---")
        print(f"    Blinks:              {t.blink_count}")
        print(f"    Visible duration:    {vis:.1f}s")
        print(f"    Blinks/min:          {bpm:.1f}")
        print(f"    Frames seen:         {t.frame_count}")
        print(f"    Final baseline EAR:  {baseline_str}")
        print(f"    Close threshold:     {close_str}  (baseline * {BLINK_OPEN_RATIO})")
        print(f"    Recovery threshold:  {recover_str}  (baseline * {BLINK_RECOVER_RATIO})")
        print()

    print("=" * 60)
    print("  Comparison: fixed threshold @ 0.22 detected ~8 blinks/min")
    print(f"  Adaptive approach detected: "
          + ", ".join(f"Person {t.person_id}: {t.blinks_per_minute:.1f} blinks/min"
                      for t in trackers))
    print("=" * 60)


if __name__ == "__main__":
    main()
