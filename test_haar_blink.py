"""
Blink detection comparison: Haar Cascade eye absence vs Frame Differencing.

Method A: Uses OpenCV's Haar cascade eye detector. Open eyes are detected;
          closed eyes are not. A blink = detected -> not detected -> detected.

Method B: Uses frame differencing on dlib-landmark eye crops. A blink causes
          a spike in pixel difference (eyelid moves down then up).
"""

import time
from collections import deque

import cv2
import dlib
import face_recognition
import numpy as np

# --- Config ---
VIDEO_PATH = "/tmp/blinkcounter_test/aws_120s.mp4"
MODEL_PATH = "blinkcounter/models/shape_predictor_68_face_landmarks.dat"

FACE_MATCH_TOLERANCE = 0.7
MIN_BLINK_MS = 30
MAX_BLINK_MS = 500

# Haar cascade parameters
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_EYE_SIZE = (20, 20)

# Frame diff parameters
DIFF_ROLLING_WINDOW = 30
DIFF_SPIKE_MULTIPLIER = 2.0

# Landmark index ranges (dlib 68-point model)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def get_eye_crop(frame_gray, landmarks, eye_indices, padding=5):
    """Extract a grayscale crop of an eye region from landmarks."""
    pts = landmarks[eye_indices]
    x_min = max(0, pts[:, 0].min() - padding)
    x_max = min(frame_gray.shape[1], pts[:, 0].max() + padding)
    y_min = max(0, pts[:, 1].min() - padding)
    y_max = min(frame_gray.shape[0], pts[:, 1].max() + padding)
    crop = frame_gray[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return crop


class HaarBlinkTracker:
    """Method A: Detect blinks by absence of Haar-detected open eyes."""

    def __init__(self, person_id, encoding):
        self.person_id = person_id
        self.encoding = encoding
        self.blink_count = 0
        self.first_seen_time = None
        self.last_seen_time = None
        self.frame_count = 0
        # State machine: eyes_detected tracks whether eyes were found
        self.eyes_were_detected = True  # assume open initially
        self.eyes_lost_time = None      # timestamp when eyes stopped being detected

    def update(self, eyes_detected, timestamp):
        self.frame_count += 1
        if self.first_seen_time is None:
            self.first_seen_time = timestamp
        self.last_seen_time = timestamp

        if self.eyes_were_detected and not eyes_detected:
            # Transition: open -> closed
            self.eyes_lost_time = timestamp
            self.eyes_were_detected = False
        elif not self.eyes_were_detected and eyes_detected:
            # Transition: closed -> open (blink completed)
            if self.eyes_lost_time is not None:
                duration_ms = (timestamp - self.eyes_lost_time) * 1000
                if MIN_BLINK_MS <= duration_ms <= MAX_BLINK_MS:
                    self.blink_count += 1
            self.eyes_were_detected = True
            self.eyes_lost_time = None
        # If same state as before, do nothing (stay open or stay closed)

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


class FrameDiffBlinkTracker:
    """Method B: Detect blinks via frame differencing on eye region crops."""

    def __init__(self, person_id, encoding):
        self.person_id = person_id
        self.encoding = encoding
        self.blink_count = 0
        self.first_seen_time = None
        self.last_seen_time = None
        self.frame_count = 0
        # Previous eye crops for differencing
        self.prev_left_crop = None
        self.prev_right_crop = None
        # Rolling average of frame diffs
        self.diff_history = deque(maxlen=DIFF_ROLLING_WINDOW)
        # Blink state machine: looking for close-spike then open-spike
        self.state = "idle"  # idle -> closing_detected -> (wait for open spike)
        self.closing_time = None

    def update(self, left_crop, right_crop, timestamp):
        self.frame_count += 1
        if self.first_seen_time is None:
            self.first_seen_time = timestamp
        self.last_seen_time = timestamp

        if left_crop is None or right_crop is None:
            return

        if self.prev_left_crop is None or self.prev_right_crop is None:
            self.prev_left_crop = left_crop.copy()
            self.prev_right_crop = right_crop.copy()
            return

        # Compute mean absolute difference for each eye
        left_diff = self._compute_diff(self.prev_left_crop, left_crop)
        right_diff = self._compute_diff(self.prev_right_crop, right_crop)

        if left_diff is None or right_diff is None:
            self.prev_left_crop = left_crop.copy()
            self.prev_right_crop = right_crop.copy()
            return

        avg_diff = (left_diff + right_diff) / 2.0
        self.diff_history.append(avg_diff)

        # Save crops for next frame
        self.prev_left_crop = left_crop.copy()
        self.prev_right_crop = right_crop.copy()

        # Need enough history for a rolling average
        if len(self.diff_history) < 5:
            return

        rolling_avg = np.mean(self.diff_history)
        threshold = rolling_avg * DIFF_SPIKE_MULTIPLIER

        is_spike = avg_diff > threshold and rolling_avg > 0.5  # noise floor

        if self.state == "idle":
            if is_spike:
                self.state = "closing_detected"
                self.closing_time = timestamp
        elif self.state == "closing_detected":
            elapsed_ms = (timestamp - self.closing_time) * 1000
            if elapsed_ms > MAX_BLINK_MS:
                # Too long, reset
                self.state = "idle"
                self.closing_time = None
            elif is_spike and elapsed_ms >= MIN_BLINK_MS:
                # Second spike = eyes reopening
                self.blink_count += 1
                self.state = "idle"
                self.closing_time = None

    def _compute_diff(self, prev_crop, curr_crop):
        """Compute mean absolute difference between two eye crops."""
        if prev_crop is None or curr_crop is None:
            return None
        if prev_crop.size == 0 or curr_crop.size == 0:
            return None
        # Resize to match dimensions
        h = min(prev_crop.shape[0], curr_crop.shape[0])
        w = min(prev_crop.shape[1], curr_crop.shape[1])
        if h < 3 or w < 3:
            return None
        p = cv2.resize(prev_crop, (w, h))
        c = cv2.resize(curr_crop, (w, h))
        return np.mean(np.abs(p.astype(np.float32) - c.astype(np.float32)))

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


def match_or_create(encoding, haar_trackers, diff_trackers):
    """Match face to existing trackers or create new ones for both methods."""
    if not haar_trackers:
        pid = 1
        haar_t = HaarBlinkTracker(pid, encoding)
        diff_t = FrameDiffBlinkTracker(pid, encoding)
        haar_trackers.append(haar_t)
        diff_trackers.append(diff_t)
        return haar_t, diff_t

    known_encodings = [t.encoding for t in haar_trackers]
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] < FACE_MATCH_TOLERANCE:
        return haar_trackers[best_idx], diff_trackers[best_idx]

    pid = len(haar_trackers) + 1
    haar_t = HaarBlinkTracker(pid, encoding)
    diff_t = FrameDiffBlinkTracker(pid, encoding)
    haar_trackers.append(haar_t)
    diff_trackers.append(diff_t)
    return haar_t, diff_t


def main():
    print("=" * 65)
    print("  Haar Cascade vs Frame Differencing - Blink Detection Comparison")
    print("=" * 65)
    print(f"  Video:              {VIDEO_PATH}")
    print(f"  Face match tol:     {FACE_MATCH_TOLERANCE}")
    print(f"  Blink duration:     {MIN_BLINK_MS}-{MAX_BLINK_MS} ms")
    print(f"  [A] Haar cascade:   haarcascade_eye.xml")
    print(f"  [A] Scale/Neigh:    {HAAR_SCALE_FACTOR}/{HAAR_MIN_NEIGHBORS}")
    print(f"  [B] Diff window:    {DIFF_ROLLING_WINDOW} frames")
    print(f"  [B] Spike mult:     {DIFF_SPIKE_MULTIPLIER}x rolling avg")
    print("=" * 65)

    # Load models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    if eye_cascade.empty():
        print("ERROR: Could not load haarcascade_eye.xml")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    print(f"  FPS: {fps:.1f}  |  Frames: {total_frames}  |  Duration: {duration_s:.1f}s")
    print("=" * 65)

    haar_trackers = []
    diff_trackers = []
    frame_num = 0
    start_wall = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_num / fps if fps > 0 else 0.0
        frame_num += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detector(gray, 0)

        if not faces:
            continue

        # Get face encodings for person tracking
        face_locations = [
            (f.top(), f.right(), f.bottom(), f.left()) for f in faces
        ]
        encodings = face_recognition.face_encodings(rgb, face_locations)

        for face_rect, encoding in zip(faces, encodings):
            shape = predictor(gray, face_rect)
            landmarks = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)]
            )

            haar_t, diff_t = match_or_create(
                encoding, haar_trackers, diff_trackers
            )

            # --- Method A: Haar Cascade eye detection ---
            # Extract the upper half of the face ROI for eye detection
            fx = max(0, face_rect.left())
            fy = max(0, face_rect.top())
            fw = face_rect.width()
            fh = face_rect.height()
            # Clamp to frame
            fx2 = min(gray.shape[1], fx + fw)
            fy2 = min(gray.shape[0], fy + fh)
            face_roi_gray = gray[fy:fy2, fx:fx2]
            # Use upper half of face for eye detection
            upper_half_h = face_roi_gray.shape[0] // 2
            if upper_half_h > 10 and face_roi_gray.shape[1] > 10:
                upper_roi = face_roi_gray[:upper_half_h, :]
                eyes = eye_cascade.detectMultiScale(
                    upper_roi,
                    scaleFactor=HAAR_SCALE_FACTOR,
                    minNeighbors=HAAR_MIN_NEIGHBORS,
                    minSize=HAAR_MIN_EYE_SIZE,
                )
                eyes_detected = len(eyes) > 0
            else:
                eyes_detected = True  # can't tell, assume open

            haar_t.update(eyes_detected, timestamp)

            # --- Method B: Frame differencing on eye crops ---
            left_crop = get_eye_crop(gray, landmarks, LEFT_EYE)
            right_crop = get_eye_crop(gray, landmarks, RIGHT_EYE)
            diff_t.update(left_crop, right_crop, timestamp)

        # Progress every 5 seconds of video
        if fps > 0 and frame_num % int(fps * 5) == 0:
            elapsed = frame_num / fps
            print(
                f"  [Progress] {elapsed:.0f}s / {duration_s:.0f}s  "
                f"({frame_num}/{total_frames} frames)  "
                f"Persons tracked: {len(haar_trackers)}"
            )

    cap.release()
    wall_time = time.time() - start_wall

    # --- Results ---
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Processed {frame_num} / {total_frames} frames in {wall_time:.1f}s")
    print(f"  Persons detected: {len(haar_trackers)}")
    print()

    print("-" * 65)
    print("  Method A: Haar Cascade Eye Detection (absence = blink)")
    print("-" * 65)
    for t in haar_trackers:
        vis = t.visible_seconds
        bpm = t.blinks_per_minute
        print(f"  Person {t.person_id}:")
        print(f"    Blinks:           {t.blink_count}")
        print(f"    Visible duration: {vis:.1f}s")
        print(f"    Blinks/min:       {bpm:.1f}")
        print(f"    Frames seen:      {t.frame_count}")
        print()

    print("-" * 65)
    print("  Method B: Frame Differencing on Eye Crops (spike = blink)")
    print("-" * 65)
    for t in diff_trackers:
        vis = t.visible_seconds
        bpm = t.blinks_per_minute
        print(f"  Person {t.person_id}:")
        print(f"    Blinks:           {t.blink_count}")
        print(f"    Visible duration: {vis:.1f}s")
        print(f"    Blinks/min:       {bpm:.1f}")
        print(f"    Frames seen:      {t.frame_count}")
        print()

    # --- Comparison Summary ---
    print("=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Person':<10} {'Haar Blinks':<14} {'Haar B/min':<13} "
          f"{'Diff Blinks':<14} {'Diff B/min':<13}")
    print(f"  {'-'*10} {'-'*14} {'-'*13} {'-'*14} {'-'*13}")
    for ht, dt in zip(haar_trackers, diff_trackers):
        print(
            f"  {'P' + str(ht.person_id):<10} "
            f"{ht.blink_count:<14} {ht.blinks_per_minute:<13.1f} "
            f"{dt.blink_count:<14} {dt.blinks_per_minute:<13.1f}"
        )
    print()
    print("  Notes:")
    print("  - Typical human blink rate: 15-20 blinks/min")
    print("  - Method A relies on Haar cascade failing to detect closed eyes")
    print("  - Method B relies on pixel-level change spikes in eye region")
    print("=" * 65)


if __name__ == "__main__":
    main()
