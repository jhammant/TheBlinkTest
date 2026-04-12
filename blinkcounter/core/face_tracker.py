"""Face tracking and re-identification using MediaPipe and face_recognition."""

from __future__ import annotations

from typing import Optional

import cv2
import face_recognition
import mediapipe as mp
import numpy as np

from blinkcounter.constants import (
    FACE_ENCODING_UPDATE_INTERVAL,
    FACE_MATCH_TOLERANCE,
    LEFT_EYE_INDICES,
    PERSON_LABELS,
    RIGHT_EYE_INDICES,
)
from blinkcounter.core.models import Person


class FaceTracker:
    """Tracks multiple faces across video frames with re-identification."""

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._persons: list[Person] = []
        self._person_frame_counts: dict[str, int] = {}
        self._next_label_index: int = 0

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> list[tuple[Person, np.ndarray]]:
        """Process a single BGR frame and return detected persons with eye landmarks.

        Args:
            frame: BGR OpenCV image.
            timestamp: Current timestamp in seconds from video start.

        Returns:
            List of (Person, eye_landmarks) tuples where eye_landmarks has
            shape (2, 6, 2) — [left_eye, right_eye] each with 6 points.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return []

        output: list[tuple[Person, np.ndarray]] = []

        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks in pixel coordinates
            left_eye_px = self._extract_eye_landmarks(
                face_landmarks, LEFT_EYE_INDICES, w, h
            )
            right_eye_px = self._extract_eye_landmarks(
                face_landmarks, RIGHT_EYE_INDICES, w, h
            )

            # Compute face bounding box from all landmarks
            all_x = [lm.x * w for lm in face_landmarks.landmark]
            all_y = [lm.y * h for lm in face_landmarks.landmark]
            x_min = max(0, int(min(all_x)))
            y_min = max(0, int(min(all_y)))
            x_max = min(w, int(max(all_x)))
            y_max = min(h, int(max(all_y)))

            face_w = max(x_max - x_min, 1)
            face_h = max(y_max - y_min, 1)

            # Normalize eye landmarks to face bounding box
            left_eye_norm = (left_eye_px - np.array([x_min, y_min])) / np.array(
                [face_w, face_h]
            )
            right_eye_norm = (right_eye_px - np.array([x_min, y_min])) / np.array(
                [face_w, face_h]
            )

            eye_landmarks = np.array([left_eye_norm, right_eye_norm])

            # Extract face region for identification
            face_crop = frame[y_min:y_max, x_min:x_max]
            person = self._match_or_create_person(
                rgb_frame, face_crop, x_min, y_min, x_max, y_max, timestamp
            )

            output.append((person, eye_landmarks))

        return output

    def get_persons(self) -> list[Person]:
        """Return all tracked persons."""
        return list(self._persons)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._persons.clear()
        self._person_frame_counts.clear()
        self._next_label_index = 0

    def _extract_eye_landmarks(
        self,
        face_landmarks,
        indices: list[int],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """Extract eye landmark points in pixel coordinates.

        Args:
            face_landmarks: MediaPipe face landmarks.
            indices: List of 6 landmark indices for one eye.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            Array of shape (6, 2) with pixel coordinates.
        """
        points = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            points.append([lm.x * frame_w, lm.y * frame_h])
        return np.array(points, dtype=np.float64)

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
        """Match a detected face to an existing person or create a new one.

        Args:
            rgb_frame: Full RGB frame for face encoding.
            face_crop: BGR cropped face region.
            x_min, y_min, x_max, y_max: Face bounding box in pixels.
            timestamp: Current timestamp in seconds.

        Returns:
            Matched or newly created Person.
        """
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
