"""
Test script comparing two blink detection approaches:

Method 1 - Gradient/Texture Analysis (CNN-inspired heuristic):
    Uses gradient magnitude and pixel intensity variance in eye crops
    to classify eye state. Open eyes have more texture from iris/pupil;
    closed eyes are smoother with lower gradient energy.

Method 2 - Combined EAR + Sclera Openness:
    Combines traditional EAR from landmarks with adaptive-thresholded
    sclera visibility in the eye crop. Both signals must agree for a
    blink to be registered.

Two-pass approach:
    Pass 1 (calibration): Sample frames to collect signal statistics and
        auto-calibrate thresholds per person.
    Pass 2 (detection): Process every frame using calibrated thresholds.
"""

import time
from dataclasses import dataclass, field

import cv2
import dlib
import face_recognition
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_PATH = "/tmp/blinkcounter_test/aws_120s.mp4"
SHAPE_PREDICTOR_PATH = "blinkcounter/models/shape_predictor_68_face_landmarks.dat"

# Eye landmark indices (dlib 68-point model)
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# Eye crop target size
EYE_CROP_W, EYE_CROP_H = 34, 26

# Blink timing constraints (seconds)
MIN_BLINK_DURATION = 0.030   # 30 ms
MAX_BLINK_DURATION = 0.500   # 500 ms

# Face re-identification
FACE_MATCH_TOLERANCE = 0.7
FACE_ENCODING_INTERVAL = 30  # Re-encode every N frames


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _euclidean(a, b) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def compute_ear(eye_landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 eye landmarks."""
    v1 = _euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = _euclidean(eye_landmarks[2], eye_landmarks[4])
    h = _euclidean(eye_landmarks[0], eye_landmarks[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def extract_eye_crop(
    frame: np.ndarray,
    landmarks: np.ndarray,
    eye_indices: list[int],
    padding_factor: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Extract a padded, resized eye crop from the frame."""
    eye_pts = landmarks[eye_indices]
    x_min, y_min = eye_pts.min(axis=0)
    x_max, y_max = eye_pts.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    new_w = w * (1 + padding_factor)
    new_h = h * (1 + padding_factor)

    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(frame.shape[1], cx + new_w / 2))
    y2 = int(min(frame.shape[0], cy + new_h / 2))

    if x2 <= x1 or y2 <= y1:
        return None, eye_pts

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, eye_pts

    crop_resized = cv2.resize(crop, (EYE_CROP_W, EYE_CROP_H))
    return crop_resized, eye_pts


def compute_gradient_features(eye_crop: np.ndarray) -> tuple[float, float]:
    """Compute gradient magnitude mean and pixel variance for an eye crop.

    Returns (mean_gradient, variance).
    """
    if eye_crop is None:
        return 0.0, 0.0

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY) if len(eye_crop.shape) == 3 else eye_crop
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.mean(grad_mag)), float(np.var(gray.astype(np.float64)))


def compute_sclera_openness(eye_crop: np.ndarray) -> float:
    """Compute sclera openness ratio using adaptive thresholding."""
    if eye_crop is None:
        return 1.0

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY) if len(eye_crop.shape) == 3 else eye_crop
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    mask = np.zeros_like(thresh)
    center = (EYE_CROP_W // 2, EYE_CROP_H // 2)
    axes = (EYE_CROP_W // 3, EYE_CROP_H // 3)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    masked = cv2.bitwise_and(thresh, mask)
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 1.0

    return cv2.countNonZero(masked) / total_pixels


# ---------------------------------------------------------------------------
# Per-person tracker
# ---------------------------------------------------------------------------

@dataclass
class PersonState:
    """Tracks blink state for one person across both methods."""
    person_id: str
    face_encoding: np.ndarray | None = None

    # Method 1: gradient-based
    m1_eye_open: bool = True
    m1_close_start: float = 0.0
    m1_blink_count: int = 0
    m1_gradient_threshold: float = 0.0  # auto-calibrated
    m1_variance_threshold: float = 0.0  # auto-calibrated

    # Method 2: EAR + sclera
    m2_eye_open: bool = True
    m2_close_start: float = 0.0
    m2_blink_count: int = 0
    m2_ear_threshold: float = 0.0       # auto-calibrated
    m2_openness_threshold: float = 0.0  # auto-calibrated

    # EAR-only reference (for comparison)
    ear_only_eye_open: bool = True
    ear_only_close_start: float = 0.0
    ear_only_blink_count: int = 0

    first_seen: float = 0.0
    last_seen: float = 0.0

    # Calibration data collectors
    cal_gradients: list = field(default_factory=list)
    cal_variances: list = field(default_factory=list)
    cal_ears: list = field(default_factory=list)
    cal_openness: list = field(default_factory=list)

    @property
    def visible_duration(self) -> float:
        return self.last_seen - self.first_seen

    def m1_blinks_per_min(self) -> float:
        dur = self.visible_duration
        return (self.m1_blink_count / dur * 60.0) if dur > 1.0 else 0.0

    def m2_blinks_per_min(self) -> float:
        dur = self.visible_duration
        return (self.m2_blink_count / dur * 60.0) if dur > 1.0 else 0.0

    def ear_only_blinks_per_min(self) -> float:
        dur = self.visible_duration
        return (self.ear_only_blink_count / dur * 60.0) if dur > 1.0 else 0.0

    def calibrate(self):
        """Set thresholds from collected calibration samples.

        Strategy: most frames have open eyes. Blink frames are brief dips.
        Use percentile-based thresholds: the 10th-15th percentile captures
        the boundary between normal open-eye variation and blink dips.
        """
        if len(self.cal_gradients) < 20:
            # Not enough data; use conservative defaults
            self.m1_gradient_threshold = 20.0
            self.m1_variance_threshold = 200.0
            self.m2_ear_threshold = 0.21
            self.m2_openness_threshold = 0.40
            return

        grads = np.array(self.cal_gradients)
        vars_ = np.array(self.cal_variances)
        ears = np.array(self.cal_ears)
        opens = np.array(self.cal_openness)

        # Method 1: Use 15th percentile as threshold -- values below this
        # are likely closed-eye outliers. Clamp to sensible minimums.
        self.m1_gradient_threshold = max(5.0, float(np.percentile(grads, 15)))
        self.m1_variance_threshold = max(20.0, float(np.percentile(vars_, 15)))

        # Method 2: EAR -- use 15th percentile, clamped to sensible range
        self.m2_ear_threshold = float(np.percentile(ears, 15))
        self.m2_ear_threshold = max(0.15, min(0.28, self.m2_ear_threshold))

        # Sclera openness -- use 20th percentile
        self.m2_openness_threshold = float(np.percentile(opens, 20))
        self.m2_openness_threshold = max(0.2, min(0.6, self.m2_openness_threshold))


def find_or_create_person(
    persons: list[PersonState],
    face_encoding: np.ndarray | None,
    timestamp: float,
    next_id_counter: list[int],
) -> PersonState:
    """Match a face to an existing person or create a new one.

    When no encoding is available (non-encoding frame), match by recency:
    return the most recently seen person if only one face is in view.
    """
    if face_encoding is not None and len(persons) > 0:
        known = [(i, p) for i, p in enumerate(persons) if p.face_encoding is not None]
        if known:
            known_encodings = [p.face_encoding for _, p in known]
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] < FACE_MATCH_TOLERANCE:
                matched = known[best_idx][1]
                matched.last_seen = timestamp
                # Update encoding with latest (moving average would be better,
                # but simple replacement is fine for a test)
                matched.face_encoding = face_encoding
                return matched

    if face_encoding is None and len(persons) > 0:
        # No encoding this frame; match to the person seen most recently
        # (within a short window, assuming face tracking continuity)
        recent = [p for p in persons if (timestamp - p.last_seen) < 0.5]
        if len(recent) == 1:
            recent[0].last_seen = timestamp
            return recent[0]
        # If multiple recent persons, skip -- don't create a new one
        if len(recent) > 1:
            # Return the most recently seen as a best guess
            best = max(recent, key=lambda p: p.last_seen)
            best.last_seen = timestamp
            return best

    # Create new person
    label_idx = next_id_counter[0]
    next_id_counter[0] += 1
    label = chr(ord("A") + label_idx) if label_idx < 26 else f"P{label_idx}"
    person = PersonState(
        person_id=f"Person {label}",
        face_encoding=face_encoding,
        first_seen=timestamp,
        last_seen=timestamp,
    )
    persons.append(person)
    return person


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def get_face_data(frame, gray, detector, predictor):
    """Detect faces and extract landmarks. Returns list of (face_rect, landmarks)."""
    faces = detector(gray, 0)
    results = []
    for face_rect in faces:
        shape = predictor(gray, face_rect)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        results.append((face_rect, landmarks))
    return results


def process_video():
    print(f"Loading shape predictor from: {SHAPE_PREDICTOR_PATH}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video at {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video: {VIDEO_PATH}")
    print(f"  FPS: {fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")

    # ===================================================================
    # PASS 1: Calibration - sample every 3rd frame to collect statistics
    # ===================================================================
    print("\n--- Pass 1: Calibration (sampling every 3rd frame) ---")
    persons: list[PersonState] = []
    next_id_counter = [0]
    frame_num = 0
    cal_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        timestamp = (frame_num - 1) / fps if fps > 0 else 0.0

        # Sample every 3rd frame for calibration
        if frame_num % 3 != 1:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_data = get_face_data(frame, gray, detector, predictor)

        # Compute encodings on EVERY sampled frame during calibration
        # to ensure robust person tracking (we only sample every 3rd frame
        # so the cost is acceptable)
        compute_enc = True

        for face_rect, landmarks in face_data:
            face_encoding = None
            if compute_enc:
                loc = [(face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())]
                encs = face_recognition.face_encodings(rgb, loc)
                if encs:
                    face_encoding = encs[0]

            person = find_or_create_person(persons, face_encoding, timestamp, next_id_counter)
            person.last_seen = timestamp

            # Collect signal values
            left_crop, _ = extract_eye_crop(frame, landmarks, LEFT_EYE_IDX)
            right_crop, _ = extract_eye_crop(frame, landmarks, RIGHT_EYE_IDX)

            lg, lv = compute_gradient_features(left_crop)
            rg, rv = compute_gradient_features(right_crop)
            avg_grad = (lg + rg) / 2.0
            avg_var = (lv + rv) / 2.0

            left_ear = compute_ear(landmarks[LEFT_EYE_IDX])
            right_ear = compute_ear(landmarks[RIGHT_EYE_IDX])
            avg_ear = (left_ear + right_ear) / 2.0

            lo = compute_sclera_openness(left_crop)
            ro = compute_sclera_openness(right_crop)
            avg_open = (lo + ro) / 2.0

            person.cal_gradients.append(avg_grad)
            person.cal_variances.append(avg_var)
            person.cal_ears.append(avg_ear)
            person.cal_openness.append(avg_open)

    cal_elapsed = time.time() - cal_start
    print(f"  Calibration done in {cal_elapsed:.1f}s, found {len(persons)} person(s)")

    # Calibrate thresholds and print diagnostics
    for p in persons:
        p.calibrate()
        if p.visible_duration < 1.0:
            continue
        n = len(p.cal_gradients)
        if n > 0:
            g = np.array(p.cal_gradients)
            v = np.array(p.cal_variances)
            e = np.array(p.cal_ears)
            o = np.array(p.cal_openness)
            print(f"\n  {p.person_id} ({n} samples, {p.visible_duration:.1f}s visible):")
            print(f"    Gradient:  mean={np.mean(g):.1f}, std={np.std(g):.1f}, "
                  f"min={np.min(g):.1f}, p5={np.percentile(g,5):.1f}, "
                  f"p95={np.percentile(g,95):.1f} -> threshold={p.m1_gradient_threshold:.1f}")
            print(f"    Variance:  mean={np.mean(v):.0f}, std={np.std(v):.0f}, "
                  f"min={np.min(v):.0f}, p5={np.percentile(v,5):.0f}, "
                  f"p95={np.percentile(v,95):.0f} -> threshold={p.m1_variance_threshold:.0f}")
            print(f"    EAR:       mean={np.mean(e):.3f}, std={np.std(e):.3f}, "
                  f"min={np.min(e):.3f}, p5={np.percentile(e,5):.3f}, "
                  f"p95={np.percentile(e,95):.3f} -> threshold={p.m2_ear_threshold:.3f}")
            print(f"    Openness:  mean={np.mean(o):.3f}, std={np.std(o):.3f}, "
                  f"min={np.min(o):.3f}, p5={np.percentile(o,5):.3f}, "
                  f"p95={np.percentile(o,95):.3f} -> threshold={p.m2_openness_threshold:.3f}")

    # Merge persons with similar face encodings (same person split across time)
    merged_persons = []
    used = set()
    for i, p1 in enumerate(persons):
        if i in used or p1.face_encoding is None:
            continue
        # Find all persons similar to p1
        group = [p1]
        used.add(i)
        for j, p2 in enumerate(persons):
            if j in used or p2.face_encoding is None:
                continue
            dist = face_recognition.face_distance([p1.face_encoding], p2.face_encoding)[0]
            if dist < FACE_MATCH_TOLERANCE:
                group.append(p2)
                used.add(j)
        # Merge group into p1
        if len(group) > 1:
            print(f"  Merging {[p.person_id for p in group]} into {p1.person_id}")
            for p2 in group[1:]:
                p1.cal_gradients.extend(p2.cal_gradients)
                p1.cal_variances.extend(p2.cal_variances)
                p1.cal_ears.extend(p2.cal_ears)
                p1.cal_openness.extend(p2.cal_openness)
                p1.first_seen = min(p1.first_seen, p2.first_seen)
                p1.last_seen = max(p1.last_seen, p2.last_seen)
            # Re-calibrate with merged data
            p1.calibrate()
        merged_persons.append(p1)
    # Add any persons without encodings
    for i, p in enumerate(persons):
        if i not in used:
            merged_persons.append(p)
    persons = merged_persons

    # Filter out noise persons (visible < 2s)
    persons = [p for p in persons if p.visible_duration >= 2.0]
    print(f"\n  Keeping {len(persons)} person(s) with >= 2s visibility")

    if not persons:
        print("No persons detected with sufficient visibility!")
        return

    # ===================================================================
    # PASS 2: Detection - process every frame
    # ===================================================================
    print("\n--- Pass 2: Blink detection (every frame) ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Reset blink state
    for p in persons:
        p.m1_eye_open = True
        p.m1_close_start = 0.0
        p.m1_blink_count = 0
        p.m2_eye_open = True
        p.m2_close_start = 0.0
        p.m2_blink_count = 0
        p.ear_only_eye_open = True
        p.ear_only_close_start = 0.0
        p.ear_only_blink_count = 0
        p.first_seen = 1e9
        p.last_seen = 0.0

    frame_num = 0
    det_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        timestamp = (frame_num - 1) / fps if fps > 0 else 0.0

        if frame_num % 500 == 0:
            elapsed = time.time() - det_start
            pct = frame_num / total_frames * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_num}/{total_frames} ({pct:.0f}%) [{elapsed:.1f}s]")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_data = get_face_data(frame, gray, detector, predictor)

        compute_enc = (frame_num % FACE_ENCODING_INTERVAL == 1) or frame_num <= 5

        for face_rect, landmarks in face_data:
            # Match to known persons
            face_encoding = None
            if compute_enc:
                loc = [(face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())]
                encs = face_recognition.face_encodings(rgb, loc)
                if encs:
                    face_encoding = encs[0]

            # Try to match to calibrated persons only
            matched_person = None
            if face_encoding is not None:
                known = [(i, p) for i, p in enumerate(persons) if p.face_encoding is not None]
                if known:
                    known_encs = [p.face_encoding for _, p in known]
                    dists = face_recognition.face_distance(known_encs, face_encoding)
                    best_idx = int(np.argmin(dists))
                    if dists[best_idx] < FACE_MATCH_TOLERANCE:
                        matched_person = known[best_idx][1]

            if matched_person is None and face_encoding is None:
                # No encoding; try recency match
                recent = [p for p in persons if abs(timestamp - p.last_seen) < 1.0]
                if len(recent) == 1:
                    matched_person = recent[0]

            if matched_person is None:
                continue  # Skip unknown faces in detection pass

            person = matched_person
            person.first_seen = min(person.first_seen, timestamp)
            person.last_seen = max(person.last_seen, timestamp)

            # Extract eye crops
            left_crop, _ = extract_eye_crop(frame, landmarks, LEFT_EYE_IDX)
            right_crop, _ = extract_eye_crop(frame, landmarks, RIGHT_EYE_IDX)

            # ---------------------------------------------------------------
            # Method 1: Gradient/Texture Analysis
            # ---------------------------------------------------------------
            lg, lv = compute_gradient_features(left_crop)
            rg, rv = compute_gradient_features(right_crop)
            avg_grad = (lg + rg) / 2.0
            avg_var = (lv + rv) / 2.0

            # Eye is closed when EITHER gradient OR variance drops below threshold
            # (a blink causes texture loss even if one signal is noisy)
            eyes_closed_m1 = (avg_grad < person.m1_gradient_threshold or
                              avg_var < person.m1_variance_threshold)

            if person.m1_eye_open and eyes_closed_m1:
                person.m1_eye_open = False
                person.m1_close_start = timestamp
            elif not person.m1_eye_open and not eyes_closed_m1:
                blink_dur = timestamp - person.m1_close_start
                if MIN_BLINK_DURATION <= blink_dur <= MAX_BLINK_DURATION:
                    person.m1_blink_count += 1
                person.m1_eye_open = True

            # ---------------------------------------------------------------
            # Method 2: EAR + Sclera Openness (combined)
            # ---------------------------------------------------------------
            left_ear = compute_ear(landmarks[LEFT_EYE_IDX])
            right_ear = compute_ear(landmarks[RIGHT_EYE_IDX])
            avg_ear = (left_ear + right_ear) / 2.0

            lo = compute_sclera_openness(left_crop)
            ro = compute_sclera_openness(right_crop)
            avg_open = (lo + ro) / 2.0

            # Both signals should agree -- but use EAR as primary, sclera as
            # confirmation. A blink is: EAR below threshold AND sclera below
            # its threshold. Since thresholds are now percentile-based (less
            # conservative), the AND gate works properly.
            eyes_closed_m2 = (avg_ear < person.m2_ear_threshold and
                              avg_open < person.m2_openness_threshold)

            if person.m2_eye_open and eyes_closed_m2:
                person.m2_eye_open = False
                person.m2_close_start = timestamp
            elif not person.m2_eye_open and not eyes_closed_m2:
                blink_dur = timestamp - person.m2_close_start
                if MIN_BLINK_DURATION <= blink_dur <= MAX_BLINK_DURATION:
                    person.m2_blink_count += 1
                person.m2_eye_open = True

            # ---------------------------------------------------------------
            # EAR-only reference (baseline for comparison)
            # ---------------------------------------------------------------
            ear_closed = avg_ear < person.m2_ear_threshold

            if person.ear_only_eye_open and ear_closed:
                person.ear_only_eye_open = False
                person.ear_only_close_start = timestamp
            elif not person.ear_only_eye_open and not ear_closed:
                blink_dur = timestamp - person.ear_only_close_start
                if MIN_BLINK_DURATION <= blink_dur <= MAX_BLINK_DURATION:
                    person.ear_only_blink_count += 1
                person.ear_only_eye_open = True

    cap.release()
    det_elapsed = time.time() - det_start
    total_elapsed = time.time() - cal_start
    print(f"\n  Detection pass: {frame_num} frames in {det_elapsed:.1f}s "
          f"({frame_num / det_elapsed:.1f} fps)")
    print(f"  Total time: {total_elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n{'Person':<12} | {'Visible':<8} | "
          f"{'M1:Gradient':>13} | {'M2:EAR+Sclera':>15} | {'Ref:EAR-only':>14}")
    print(f"{'':12} | {'(sec)':>8} | "
          f"{'Blinks':>6} {'BPM':>6} | {'Blinks':>7} {'BPM':>7} | {'Blinks':>6} {'BPM':>7}")
    print("-" * 90)

    for p in sorted(persons, key=lambda x: x.first_seen):
        dur = p.visible_duration
        if dur < 1.0:
            continue
        print(f"{p.person_id:<12} | {dur:>7.1f}s | "
              f"{p.m1_blink_count:>6} {p.m1_blinks_per_min():>6.1f} | "
              f"{p.m2_blink_count:>7} {p.m2_blinks_per_min():>7.1f} | "
              f"{p.ear_only_blink_count:>6} {p.ear_only_blinks_per_min():>7.1f}")

    print("\n" + "-" * 90)
    print("\nMethod 1 - Gradient/Texture Analysis (CNN-inspired heuristic):")
    print("  Computes Sobel gradient magnitude and pixel intensity variance in")
    print("  the eye crop. Open eyes have more texture (iris/pupil edges);")
    print("  closed eyes are smoother. Thresholds auto-calibrated per person")
    print("  using 15th percentile of calibration samples.")

    print("\nMethod 2 - Combined EAR + Sclera Openness:")
    print("  Combines Eye Aspect Ratio (landmark geometry) with adaptive-")
    print("  thresholded sclera visibility (bright pixel ratio in eye crop).")
    print("  Both signals must drop below their thresholds simultaneously.")

    print("\nReference - EAR-only:")
    print("  Traditional Eye Aspect Ratio thresholding (same adaptive threshold")
    print("  as Method 2, but without requiring sclera confirmation).")

    print("\n" + "=" * 90)
    print("DETAILED PER-PERSON RESULTS")
    print("=" * 90)
    for p in sorted(persons, key=lambda x: x.first_seen):
        dur = p.visible_duration
        if dur < 1.0:
            continue
        print(f"\n  {p.person_id}:")
        print(f"    Visible: {dur:.1f}s (first: {p.first_seen:.1f}s, last: {p.last_seen:.1f}s)")
        print(f"    Calibrated thresholds:")
        print(f"      M1 gradient < {p.m1_gradient_threshold:.1f}, variance < {p.m1_variance_threshold:.0f}")
        print(f"      M2 EAR < {p.m2_ear_threshold:.3f}, openness < {p.m2_openness_threshold:.3f}")
        print(f"    Method 1 (Gradient):     {p.m1_blink_count:>3} blinks, "
              f"{p.m1_blinks_per_min():.1f} blinks/min")
        print(f"    Method 2 (EAR+Sclera):  {p.m2_blink_count:>3} blinks, "
              f"{p.m2_blinks_per_min():.1f} blinks/min")
        print(f"    Reference (EAR-only):   {p.ear_only_blink_count:>3} blinks, "
              f"{p.ear_only_blinks_per_min():.1f} blinks/min")

    print()


if __name__ == "__main__":
    process_video()
