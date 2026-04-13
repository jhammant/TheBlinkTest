"""Face tracking and re-identification using dlib correlation tracking and face_recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import dlib
import face_recognition
import numpy as np

from blinkcounter.constants import (
    FACE_ENCODING_UPDATE_INTERVAL,
    FACE_MATCH_TOLERANCE,
    PERSON_LABELS,
)
from blinkcounter.core.models import Person

# Path to the bundled dlib 68-landmark shape predictor
_PREDICTOR_PATH = str(
    Path(__file__).parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"
)

# dlib 68-landmark eye indices
_LEFT_EYE_INDICES = list(range(36, 42))
_RIGHT_EYE_INDICES = list(range(42, 48))

# Tracking constants
_CORRELATION_TRACKER_CONFIDENCE_THRESHOLD = 6.0  # Below this, re-detect
_MERGE_INTERVAL = 100  # Run person merge every N frames
_MERGE_ENCODING_THRESHOLD = 0.8  # Encoding distance for merging persons
_SPATIAL_WEIGHT = 0.3  # Weight for spatial distance in matching
_ENCODING_WEIGHT = 0.7  # Weight for encoding distance in matching
_WEIGHTED_MATCH_THRESHOLD = 0.8  # Combined score threshold for matching
_ENCODING_RECOMPUTE_INTERVAL = 90  # Re-compute encoding every N frames of visibility (less frequent = faster)
_MAX_SPATIAL_DISTANCE = 300.0  # Normalisation factor for spatial distance (pixels)
_INACTIVE_GRACE_FRAMES = 30  # Keep inactive trackers for this many frames before discarding


@dataclass
class _TrackedFace:
    """Internal state for a face being tracked via correlation tracker."""

    person: Person
    tracker: dlib.correlation_tracker
    last_rect: dlib.rectangle
    last_center: tuple[float, float]  # (cx, cy) of face
    frames_since_encoding: int = 0  # Frames since last encoding computation
    frames_visible: int = 0  # Total frames this face has been visible
    active: bool = True  # Whether the correlation tracker is still valid
    frames_since_inactive: int = 0  # Frames since tracker became inactive
    center_history: list = field(default_factory=list)  # Recent face centers for movement detection


class FaceTracker:
    """Tracks multiple faces across video frames with re-identification.

    Uses dlib correlation trackers for frame-to-frame continuity and
    face_recognition encodings for identity matching. Periodically merges
    fragmented persons that represent the same individual.
    """

    def __init__(self, detect_interval: int = 10) -> None:
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(_PREDICTOR_PATH)
        self._persons: list[Person] = []
        self._tracked_faces: list[_TrackedFace] = []
        self._next_label_index: int = 0
        # Only re-detect faces every N frames (correlation tracker fills the gap)
        self._detect_interval: int = detect_interval
        self._frame_count: int = 0

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> list[tuple[Person, np.ndarray, np.ndarray]]:
        """Process a single BGR frame and return detected persons with eye landmarks.

        Args:
            frame: BGR OpenCV image.
            timestamp: Current timestamp in seconds from video start.

        Returns:
            List of (Person, eye_landmarks, all_landmarks) tuples where:
            - eye_landmarks has shape (2, 6, 2) -- [left_eye, right_eye] pixel coords
            - all_landmarks has shape (68, 2) -- all dlib landmark points (for head pose)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._frame_count += 1
        is_detection_frame = (
            self._frame_count % self._detect_interval == 1
            or not self._tracked_faces
        )

        if is_detection_frame:
            # Full detection + re-identification pass
            output = self._detection_pass(gray, rgb_frame, frame, timestamp)
        else:
            # Correlation tracker update pass (fast)
            output = self._tracking_pass(gray, frame, timestamp)

        # Periodic person merge to reduce fragmentation
        if self._frame_count % _MERGE_INTERVAL == 0:
            self._merge_persons()

        return output

    def get_persons(self) -> list[Person]:
        """Return all tracked persons, with still-image detection."""
        # Check each tracked face for movement — if face barely moved, it's a still image
        for tf in self._tracked_faces:
            if len(tf.center_history) >= 10:
                centers = np.array(tf.center_history)
                # Standard deviation of face center position
                std_x = float(np.std(centers[:, 0]))
                std_y = float(np.std(centers[:, 1]))
                # Real video: face moves at least a few pixels from breathing/micro-movements
                # Still image: std < 1 pixel
                if std_x < 1.0 and std_y < 1.0:
                    tf.person.is_still_image = True
        return list(self._persons)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._persons.clear()
        self._tracked_faces.clear()
        self._next_label_index = 0
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Detection pass: run dlib HOG detector, match/create persons
    # ------------------------------------------------------------------

    def _detection_pass(
        self,
        gray: np.ndarray,
        rgb_frame: np.ndarray,
        frame: np.ndarray,
        timestamp: float,
    ) -> list[tuple[Person, np.ndarray, np.ndarray]]:
        """Run face detection, match to existing tracked faces, create new ones."""
        # Adaptive downscale based on frame width for faster face detection
        width = gray.shape[1]
        if width >= 960:
            s = 0.5
        elif width >= 640:
            s = 0.75
        else:
            s = 1.0
        if s < 1.0:
            small_gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            small_rects = self._detector(small_gray, 0)
            # Scale rects back to original size
            detected_rects = [
                dlib.rectangle(
                    int(r.left() / s), int(r.top() / s),
                    int(r.right() / s), int(r.bottom() / s)
                )
                for r in small_rects
            ]
        else:
            detected_rects = self._detector(gray, 0)

        # Adaptive fallback: if downscaled detection found nothing, retry at
        # full scale with upsample=1 to catch smaller / distant faces.
        if not detected_rects:
            detected_rects = list(self._detector(gray, 1))

        if not detected_rects:
            # No faces found even after fallback -- tick inactive counters
            for tf in self._tracked_faces:
                if tf.active:
                    tf.active = False
                    tf.frames_since_inactive = 0
                else:
                    tf.frames_since_inactive += 1
            # Discard trackers that exceeded the grace period
            self._tracked_faces = [
                tf for tf in self._tracked_faces
                if tf.active or tf.frames_since_inactive <= _INACTIVE_GRACE_FRAMES
            ]
            return []

        output: list[tuple[Person, np.ndarray, np.ndarray]] = []
        matched_tracked_indices: set[int] = set()

        for rect in detected_rects:
            shape = self._predictor(gray, rect)
            all_landmarks = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)],
                dtype=np.float64,
            )
            left_eye = all_landmarks[36:42]
            right_eye = all_landmarks[42:48]
            eye_landmarks = np.array([left_eye, right_eye])

            cx = (rect.left() + rect.right()) / 2.0
            cy = (rect.top() + rect.bottom()) / 2.0

            # Compute encoding (or defer if we can spatially match)
            x_min = max(0, rect.left())
            y_min = max(0, rect.top())
            x_max = min(frame.shape[1], rect.right())
            y_max = min(frame.shape[0], rect.bottom())

            # Padded face crop for better thumbnails
            fw, fh = x_max - x_min, y_max - y_min
            pad_x, pad_y = int(fw * 0.3), int(fh * 0.2)
            crop_x1 = max(0, x_min - pad_x)
            crop_y1 = max(0, y_min - pad_y)
            crop_x2 = min(frame.shape[1], x_max + pad_x)
            crop_y2 = min(frame.shape[0], y_max + pad_y)
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Try to match to existing tracked face using spatial + encoding
            best_tf_idx, needs_encoding = self._find_best_tracked_match(
                rgb_frame, rect, cx, cy, x_min, y_min, x_max, y_max,
                matched_tracked_indices,
            )

            if best_tf_idx is not None:
                matched_tracked_indices.add(best_tf_idx)
                tf = self._tracked_faces[best_tf_idx]
                person = tf.person

                # Restart correlation tracker on the new detection rect
                tf.tracker = dlib.correlation_tracker()
                tf.tracker.start_track(rgb_frame, rect)
                tf.last_rect = rect
                tf.last_center = (cx, cy)
                if len(tf.center_history) < 300:  # Cap at ~10s of history
                    tf.center_history.append((cx, cy))
                tf.active = True
                tf.frames_since_inactive = 0
                tf.frames_visible += 1

                # Re-compute encoding periodically
                tf.frames_since_encoding += 1
                if tf.frames_since_encoding >= _ENCODING_RECOMPUTE_INTERVAL:
                    encoding = self._compute_encoding(rgb_frame, x_min, y_min, x_max, y_max)
                    if encoding is not None:
                        person.face_encoding = encoding
                    tf.frames_since_encoding = 0

                self._update_person_timing(person, timestamp)
                person.face_thumbnail = self._make_thumbnail(face_crop)

                output.append((person, eye_landmarks, all_landmarks))
            else:
                # Before creating a new person, check inactive trackers
                # within grace period for a possible reactivation match
                reactivated_tf = self._find_inactive_match(
                    rgb_frame, cx, cy, x_min, y_min, x_max, y_max,
                    matched_tracked_indices,
                )

                if reactivated_tf is not None:
                    idx = self._tracked_faces.index(reactivated_tf)
                    matched_tracked_indices.add(idx)
                    person = reactivated_tf.person

                    # Restart correlation tracker
                    reactivated_tf.tracker = dlib.correlation_tracker()
                    reactivated_tf.tracker.start_track(rgb_frame, rect)
                    reactivated_tf.last_rect = rect
                    reactivated_tf.last_center = (cx, cy)
                    reactivated_tf.active = True
                    reactivated_tf.frames_since_inactive = 0
                    reactivated_tf.frames_visible += 1

                    self._update_person_timing(person, timestamp)
                    person.face_thumbnail = self._make_thumbnail(face_crop)

                    output.append((person, eye_landmarks, all_landmarks))
                else:
                    # No match -- create new person with encoding
                    encoding = self._compute_encoding(rgb_frame, x_min, y_min, x_max, y_max)
                    person = self._create_person(encoding, face_crop, timestamp)

                    # Start a new correlation tracker
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(rgb_frame, rect)

                    tf = _TrackedFace(
                        person=person,
                        tracker=tracker,
                        last_rect=rect,
                        last_center=(cx, cy),
                        frames_since_encoding=0,
                        frames_visible=1,
                        active=True,
                    )
                    self._tracked_faces.append(tf)

                    output.append((person, eye_landmarks, all_landmarks))

        # Mark unmatched active trackers as inactive; tick inactive counters
        for i, tf in enumerate(self._tracked_faces):
            if i not in matched_tracked_indices:
                if tf.active:
                    tf.active = False
                    tf.frames_since_inactive = 0
                else:
                    tf.frames_since_inactive += 1

        # Discard trackers that exceeded the grace period
        self._tracked_faces = [
            tf for tf in self._tracked_faces
            if tf.active or tf.frames_since_inactive <= _INACTIVE_GRACE_FRAMES
        ]

        return output

    # ------------------------------------------------------------------
    # Tracking pass: use correlation trackers (fast, no detection)
    # ------------------------------------------------------------------

    def _tracking_pass(
        self,
        gray: np.ndarray,
        frame: np.ndarray,
        timestamp: float,
    ) -> list[tuple[Person, np.ndarray, np.ndarray]]:
        """Update correlation trackers and extract landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output: list[tuple[Person, np.ndarray, np.ndarray]] = []

        for tf in self._tracked_faces:
            if not tf.active:
                continue

            # Update correlation tracker
            confidence = tf.tracker.update(rgb_frame)

            if confidence < _CORRELATION_TRACKER_CONFIDENCE_THRESHOLD:
                tf.active = False
                tf.frames_since_inactive = 0
                continue

            # Get tracked position
            pos = tf.tracker.get_position()
            rect = dlib.rectangle(
                int(pos.left()),
                int(pos.top()),
                int(pos.right()),
                int(pos.bottom()),
            )

            # Clamp to frame bounds
            rect_left = max(0, rect.left())
            rect_top = max(0, rect.top())
            rect_right = min(frame.shape[1], rect.right())
            rect_bottom = min(frame.shape[0], rect.bottom())

            if rect_right <= rect_left or rect_bottom <= rect_top:
                tf.active = False
                continue

            clamped_rect = dlib.rectangle(rect_left, rect_top, rect_right, rect_bottom)

            # Extract landmarks from tracked position
            shape = self._predictor(gray, clamped_rect)
            all_landmarks = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)],
                dtype=np.float64,
            )
            left_eye = all_landmarks[36:42]
            right_eye = all_landmarks[42:48]
            eye_landmarks = np.array([left_eye, right_eye])

            tf.last_rect = clamped_rect
            tf.last_center = (
                (clamped_rect.left() + clamped_rect.right()) / 2.0,
                (clamped_rect.top() + clamped_rect.bottom()) / 2.0,
            )
            tf.frames_visible += 1

            self._update_person_timing(tf.person, timestamp)

            # Update thumbnail with padding
            rw, rh = rect_right - rect_left, rect_bottom - rect_top
            rpx, rpy = int(rw * 0.3), int(rh * 0.2)
            ct1 = max(0, rect_top - rpy)
            cb1 = min(frame.shape[0], rect_bottom + rpy)
            cl1 = max(0, rect_left - rpx)
            cr1 = min(frame.shape[1], rect_right + rpx)
            face_crop = frame[ct1:cb1, cl1:cr1]
            tf.person.face_thumbnail = self._make_thumbnail(face_crop)

            output.append((tf.person, eye_landmarks, all_landmarks))

        return output

    # ------------------------------------------------------------------
    # Matching logic: weighted encoding + spatial distance
    # ------------------------------------------------------------------

    def _find_best_tracked_match(
        self,
        rgb_frame: np.ndarray,
        rect: dlib.rectangle,
        cx: float,
        cy: float,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        already_matched: set[int],
    ) -> tuple[Optional[int], bool]:
        """Find the best existing tracked face for a detected rectangle.

        Returns (index into _tracked_faces or None, whether encoding was needed).
        Uses weighted score of encoding distance and spatial proximity.
        """
        if not self._tracked_faces:
            return None, True

        best_idx: Optional[int] = None
        best_score = float("inf")
        encoding: Optional[np.ndarray] = None
        computed_encoding = False

        for i, tf in enumerate(self._tracked_faces):
            if i in already_matched:
                continue

            person = tf.person

            # Spatial distance (always available)
            spatial_dist = np.sqrt(
                (cx - tf.last_center[0]) ** 2 + (cy - tf.last_center[1]) ** 2
            )
            spatial_normalized = min(spatial_dist / _MAX_SPATIAL_DISTANCE, 1.0)

            # If person has no encoding, use spatial only
            if person.face_encoding is None:
                score = spatial_normalized
                if score < best_score and score < _WEIGHTED_MATCH_THRESHOLD:
                    best_score = score
                    best_idx = i
                continue

            # Compute encoding lazily (only once per detection)
            if not computed_encoding:
                encoding = self._compute_encoding(rgb_frame, x_min, y_min, x_max, y_max)
                computed_encoding = True

            if encoding is not None:
                enc_dist = float(
                    face_recognition.face_distance([person.face_encoding], encoding)[0]
                )
                # Weighted combined score
                score = enc_dist * _ENCODING_WEIGHT + spatial_normalized * _SPATIAL_WEIGHT

                if score < best_score and score < _WEIGHTED_MATCH_THRESHOLD:
                    best_score = score
                    best_idx = i
            else:
                # No encoding available, fall back to spatial only
                score = spatial_normalized
                if score < best_score and score < _WEIGHTED_MATCH_THRESHOLD:
                    best_score = score
                    best_idx = i

        return best_idx, computed_encoding

    # ------------------------------------------------------------------
    # Inactive tracker reactivation
    # ------------------------------------------------------------------

    def _find_inactive_match(
        self,
        rgb_frame: np.ndarray,
        cx: float,
        cy: float,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        already_matched: set[int],
    ) -> Optional[_TrackedFace]:
        """Check inactive trackers within grace period for a nearby match.

        Returns the best matching inactive _TrackedFace, or None.
        """
        best_tf: Optional[_TrackedFace] = None
        best_score = float("inf")
        encoding: Optional[np.ndarray] = None
        computed_encoding = False

        for i, tf in enumerate(self._tracked_faces):
            if i in already_matched or tf.active:
                continue
            if tf.frames_since_inactive > _INACTIVE_GRACE_FRAMES:
                continue

            person = tf.person

            spatial_dist = np.sqrt(
                (cx - tf.last_center[0]) ** 2 + (cy - tf.last_center[1]) ** 2
            )
            spatial_normalized = min(spatial_dist / _MAX_SPATIAL_DISTANCE, 1.0)

            if person.face_encoding is None:
                score = spatial_normalized
            else:
                if not computed_encoding:
                    encoding = self._compute_encoding(rgb_frame, x_min, y_min, x_max, y_max)
                    computed_encoding = True

                if encoding is not None:
                    enc_dist = float(
                        face_recognition.face_distance([person.face_encoding], encoding)[0]
                    )
                    score = enc_dist * _ENCODING_WEIGHT + spatial_normalized * _SPATIAL_WEIGHT
                else:
                    score = spatial_normalized

            if score < best_score and score < _WEIGHTED_MATCH_THRESHOLD:
                best_score = score
                best_tf = tf

        return best_tf

    # ------------------------------------------------------------------
    # Person merge: combine fragmented identities
    # ------------------------------------------------------------------

    def _merge_persons(self) -> None:
        """Merge persons with similar face encodings to reduce fragmentation.

        When two persons are merged, blink events from the absorbed person
        are transferred to the surviving person.
        """
        if len(self._persons) < 2:
            return

        merged_away: set[int] = set()

        for i in range(len(self._persons)):
            if i in merged_away:
                continue
            pi = self._persons[i]
            if pi.face_encoding is None:
                continue

            for j in range(i + 1, len(self._persons)):
                if j in merged_away:
                    continue
                pj = self._persons[j]
                if pj.face_encoding is None:
                    continue

                dist = float(
                    face_recognition.face_distance([pi.face_encoding], pj.face_encoding)[0]
                )
                if dist < _MERGE_ENCODING_THRESHOLD:
                    # Merge pj into pi
                    pi.blink_events.extend(pj.blink_events)
                    pi.blink_events.sort(key=lambda e: e.timestamp)
                    pi.first_seen_at = min(pi.first_seen_at, pj.first_seen_at)
                    pi.last_seen_at = max(pi.last_seen_at, pj.last_seen_at)
                    pi.total_visible_duration += pj.total_visible_duration
                    pi.analyzable_duration += pj.analyzable_duration
                    merged_away.add(j)

                    # Update tracked faces referencing the merged person
                    for tf in self._tracked_faces:
                        if tf.person is pj:
                            tf.person = pi

        if merged_away:
            self._persons = [
                p for i, p in enumerate(self._persons) if i not in merged_away
            ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_encoding(
        rgb_frame: np.ndarray,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> Optional[np.ndarray]:
        """Compute a 128-dim face encoding for the given face region."""
        face_location = [(y_min, x_max, y_max, x_min)]
        encodings = face_recognition.face_encodings(rgb_frame, face_location)
        if encodings:
            return encodings[0]
        return None

    def _create_person(
        self,
        encoding: Optional[np.ndarray],
        face_crop: np.ndarray,
        timestamp: float,
    ) -> Person:
        """Create a new tracked person."""
        if self._next_label_index >= len(PERSON_LABELS):
            label_char = f"#{self._next_label_index + 1}"
        else:
            label_char = PERSON_LABELS[self._next_label_index]

        person_id = f"person_{label_char.lower()}"
        label = f"Person {label_char}"

        thumbnail = self._make_thumbnail(face_crop)

        person = Person(
            id=person_id,
            label=label,
            face_encoding=encoding,
            face_thumbnail=thumbnail,
            first_seen_at=timestamp,
            last_seen_at=timestamp,
            total_visible_duration=0.0,
        )
        self._persons.append(person)
        self._next_label_index += 1
        return person

    @staticmethod
    def _update_person_timing(person: Person, timestamp: float) -> None:
        """Update a person's timing fields."""
        if person.last_seen_at > 0:
            person.total_visible_duration += timestamp - person.last_seen_at
        person.last_seen_at = timestamp

    @staticmethod
    @staticmethod
    def _make_thumbnail(face_crop: np.ndarray) -> np.ndarray:
        """Resize face crop to 128x128 thumbnail with padding."""
        if face_crop.size == 0:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        h, w = face_crop.shape[:2]
        # Add 30% padding around the face for a nicer portrait crop
        pad_x, pad_y = int(w * 0.3), int(h * 0.3)
        # We can't pad beyond what was cropped, so just resize what we have
        return cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_AREA)
