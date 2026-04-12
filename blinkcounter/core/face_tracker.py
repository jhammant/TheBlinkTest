"""Face tracking and re-identification using dlib and face_recognition."""

from __future__ import annotations

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
# Left eye: points 36-41, Right eye: points 42-47
# Each eye has 6 points: [outer, upper_outer, upper_inner, inner, lower_inner, lower_outer]
_LEFT_EYE_INDICES = list(range(36, 42))
_RIGHT_EYE_INDICES = list(range(42, 48))


class FaceTracker:
    """Tracks multiple faces across video frames with re-identification."""

    def __init__(self) -> None:
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(_PREDICTOR_PATH)
        self._persons: list[Person] = []
        self._person_frame_counts: dict[str, int] = {}
        self._next_label_index: int = 0
        # Only re-detect faces every N frames, track in between
        self._detect_interval: int = 5
        self._frame_count: int = 0
        self._last_face_rects: list[dlib.rectangle] = []

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> list[tuple[Person, np.ndarray, np.ndarray]]:
        """Process a single BGR frame and return detected persons with eye landmarks.

        Args:
            frame: BGR OpenCV image.
            timestamp: Current timestamp in seconds from video start.

        Returns:
            List of (Person, eye_landmarks, all_landmarks) tuples where:
            - eye_landmarks has shape (2, 6, 2) — [left_eye, right_eye] pixel coords
            - all_landmarks has shape (68, 2) — all dlib landmark points (for head pose)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces periodically (expensive), reuse rects in between
        self._frame_count += 1
        if self._frame_count % self._detect_interval == 1 or not self._last_face_rects:
            self._last_face_rects = self._detector(gray, 0)

        if not self._last_face_rects:
            return []

        output: list[tuple[Person, np.ndarray, np.ndarray]] = []

        for rect in self._last_face_rects:
            # Get 68 landmarks
            shape = self._predictor(gray, rect)

            # Extract ALL 68 landmarks for head pose estimation
            all_landmarks = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)],
                dtype=np.float64,
            )

            # Extract eye landmarks as pixel coordinates
            left_eye = all_landmarks[36:42]
            right_eye = all_landmarks[42:48]
            eye_landmarks = np.array([left_eye, right_eye])

            # Extract face region for identification
            x_min = max(0, rect.left())
            y_min = max(0, rect.top())
            x_max = min(frame.shape[1], rect.right())
            y_max = min(frame.shape[0], rect.bottom())

            face_crop = frame[y_min:y_max, x_min:x_max]
            person = self._match_or_create_person(
                rgb_frame, face_crop, x_min, y_min, x_max, y_max, timestamp
            )

            output.append((person, eye_landmarks, all_landmarks))

        return output

    def get_persons(self) -> list[Person]:
        """Return all tracked persons."""
        return list(self._persons)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._persons.clear()
        self._person_frame_counts.clear()
        self._next_label_index = 0
        self._frame_count = 0
        self._last_face_rects = []

    def _match_or_create_person(
        self,
        rgb_frame: np.ndarray,
        face_crop: np.ndarray,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        timestamp: float,
    ) -> Person:
        """Match a detected face to an existing person or create a new one."""
        face_location = [(y_min, x_max, y_max, x_min)]
        encodings = face_recognition.face_encodings(rgb_frame, face_location)

        encoding: Optional[np.ndarray] = None
        if encodings:
            encoding = encodings[0]

        # Try to match to existing person
        if encoding is not None and self._persons:
            known_encodings = []
            known_indices = []
            for i, person in enumerate(self._persons):
                if person.face_encoding is not None:
                    known_encodings.append(person.face_encoding)
                    known_indices.append(i)

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] < FACE_MATCH_TOLERANCE:
                    person = self._persons[known_indices[best_idx]]
                    self._update_person(person, encoding, face_crop, timestamp)
                    return person

        # No match — create new person
        return self._create_person(encoding, face_crop, timestamp)

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
        self._person_frame_counts[person_id] = 1
        self._next_label_index += 1
        return person

    def _update_person(
        self,
        person: Person,
        encoding: Optional[np.ndarray],
        face_crop: np.ndarray,
        timestamp: float,
    ) -> None:
        """Update an existing person's tracking data."""
        frame_count = self._person_frame_counts.get(person.id, 0) + 1
        self._person_frame_counts[person.id] = frame_count

        # Update timing
        if person.last_seen_at > 0:
            person.total_visible_duration += timestamp - person.last_seen_at
        person.last_seen_at = timestamp

        # Update face encoding periodically
        if (
            encoding is not None
            and frame_count % FACE_ENCODING_UPDATE_INTERVAL == 0
        ):
            person.face_encoding = encoding

        # Update thumbnail
        person.face_thumbnail = self._make_thumbnail(face_crop)

    @staticmethod
    def _make_thumbnail(face_crop: np.ndarray) -> np.ndarray:
        """Resize face crop to 64x64 thumbnail."""
        if face_crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_AREA)
