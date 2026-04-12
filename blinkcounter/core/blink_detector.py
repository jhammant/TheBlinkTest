"""Blink detection engine using Eye Aspect Ratio (EAR) and state machine."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from blinkcounter.constants import (
    CONSECUTIVE_FRAMES_FOR_BLINK,
    EAR_BLINK_THRESHOLD,
    EAR_HYSTERESIS,
    MAX_BLINK_DURATION_MS,
    MIN_BLINK_DURATION_MS,
)
from blinkcounter.core.models import BlinkEvent, EyeState


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio from 6 landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_landmarks: Array of shape (6, 2) with points:
            p1=outer corner, p2=upper1, p3=upper2,
            p4=inner corner, p5=lower1, p6=lower2

    Returns:
        Eye Aspect Ratio as a float.
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_a = np.linalg.norm(p2 - p6)
    vertical_b = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)


def estimate_head_pose(shape_points: np.ndarray) -> dict:
    """Estimate head pose from dlib 68-landmark points.

    Uses nose bridge and chin to estimate vertical tilt (pitch).
    Uses eye corners to estimate horizontal rotation (yaw).

    Args:
        shape_points: Array of shape (68, 2) with all landmark points.

    Returns:
        Dict with 'pitch_ratio' and 'yaw_ratio' (both ~1.0 when frontal).
    """
    # Nose bridge: point 27 (between eyes) to point 30 (nose tip)
    nose_top = shape_points[27]
    nose_tip = shape_points[30]
    chin = shape_points[8]  # Bottom of chin

    # Pitch: ratio of nose-to-chin distance vs expected
    # When looking down, nose tip moves down relative to chin
    nose_to_chin = np.linalg.norm(nose_tip - chin)
    nose_top_to_chin = np.linalg.norm(nose_top - chin)

    if nose_top_to_chin < 1e-6:
        pitch_ratio = 1.0
    else:
        # When frontal, nose_to_chin / nose_top_to_chin ~ 0.6-0.7
        # When looking down, this ratio decreases
        pitch_ratio = nose_to_chin / nose_top_to_chin

    # Yaw: compare left eye to right eye horizontal distances from nose
    left_eye_corner = shape_points[36]  # Left eye outer corner
    right_eye_corner = shape_points[45]  # Right eye outer corner
    nose_center = shape_points[30]

    left_dist = np.linalg.norm(left_eye_corner - nose_center)
    right_dist = np.linalg.norm(right_eye_corner - nose_center)

    if max(left_dist, right_dist) < 1e-6:
        yaw_ratio = 1.0
    else:
        # When frontal, left_dist / right_dist ~ 1.0
        yaw_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)

    return {"pitch_ratio": pitch_ratio, "yaw_ratio": yaw_ratio}


def is_face_frontal(head_pose: dict, pitch_threshold: float = 0.50, yaw_threshold: float = 0.6) -> bool:
    """Check if face is frontal enough for reliable EAR measurement.

    Args:
        head_pose: Dict from estimate_head_pose().
        pitch_threshold: Minimum pitch_ratio (looking down reduces this).
        yaw_threshold: Minimum yaw_ratio (turning sideways reduces this).
    """
    return head_pose["pitch_ratio"] >= pitch_threshold and head_pose["yaw_ratio"] >= yaw_threshold


QUALITY_THRESHOLD = 0.4  # Below this, skip the frame for blink detection


def calculate_detection_quality(
    left_ear: float,
    right_ear: float,
    head_pose: dict,
    all_landmarks: np.ndarray,
) -> float:
    """Calculate a 0.0-1.0 confidence score for the current frame's detection quality.

    Factors that reduce quality:
    - EAR asymmetry (face turned sideways)
    - Head pose away from frontal (pitch/yaw)
    - Small face size (eye landmarks very close together)

    Args:
        left_ear: Left eye EAR value.
        right_ear: Right eye EAR value.
        head_pose: Dict with 'pitch_ratio' and 'yaw_ratio' from estimate_head_pose().
        all_landmarks: Array of shape (68, 2) with all landmark points.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    quality = 1.0

    # EAR symmetry: large L/R difference means face is turned
    symmetry_factor = min(1.0, 1.0 - abs(left_ear - right_ear) / 0.15)
    quality *= max(0.0, symmetry_factor)

    # Yaw: how frontal the face is horizontally
    yaw_factor = min(1.0, head_pose["yaw_ratio"] / 0.7)
    quality *= max(0.0, yaw_factor)

    # Pitch: how frontal the face is vertically
    pitch_factor = min(1.0, head_pose["pitch_ratio"] / 0.6)
    quality *= max(0.0, pitch_factor)

    # Face size: distance between outer eye corners (landmarks 36 and 45)
    eye_width = float(np.linalg.norm(all_landmarks[36] - all_landmarks[45]))
    size_factor = min(1.0, eye_width / 15.0)
    quality *= max(0.0, size_factor)

    return quality


def check_ear_symmetry(left_ear: float, right_ear: float, max_ratio: float = 3.0) -> bool:
    """Check if left and right EAR are reasonably symmetric.

    Large asymmetry suggests the face is at an angle or landmarks are unreliable.
    """
    if min(left_ear, right_ear) < 0.01:
        return False
    ratio = max(left_ear, right_ear) / max(min(left_ear, right_ear), 0.01)
    return ratio <= max_ratio


class BlinkStateMachine:
    """Per-person state machine for tracking blink events.

    Transitions: OPEN -> CLOSING -> CLOSED -> OPENING -> OPEN

    Includes adaptive baseline tracking and head-pose-aware filtering.
    """

    def __init__(self, person_id: str) -> None:
        self.person_id = person_id
        self.state = EyeState.OPEN
        self._consecutive_below = 0
        self._blink_start_time: Optional[float] = None
        self._min_ear_during_blink: float = 1.0
        # Track recent EAR values for baseline
        self._ear_history: deque[float] = deque(maxlen=90)  # ~3s at 30fps
        self._baseline_ear: float = 0.28  # Default until enough samples
        self._pre_blink_ear: float = 0.28  # EAR just before blink started
        self._pre_blink_nose_tip: Optional[np.ndarray] = None  # Nose position when blink started

    def update(self, ear: float, timestamp: float, head_pose: Optional[dict] = None, nose_tip: Optional[np.ndarray] = None, quality: float = 1.0) -> Optional[BlinkEvent]:
        """Process a new EAR measurement and return a BlinkEvent if a blink completes.

        Args:
            ear: Current Eye Aspect Ratio value.
            timestamp: Current timestamp in seconds from video start.
            head_pose: Optional head pose dict from estimate_head_pose().
            nose_tip: Optional 2D position of landmark 30 (nose tip).
            quality: Detection quality score (0.0-1.0). Frames below
                QUALITY_THRESHOLD are skipped.

        Returns:
            BlinkEvent if a complete valid blink was detected, None otherwise.
        """
        # Skip frame if detection quality is too low
        if quality < QUALITY_THRESHOLD:
            # Reset blink state if quality drops during an active blink detection
            if self.state != EyeState.OPEN:
                self._reset_to_open()
            return None

        # Store latest nose tip for use in _try_complete_blink
        self._current_nose_tip = nose_tip

        # Skip if face is not frontal (head turned or looking down)
        if head_pose is not None and not is_face_frontal(head_pose):
            # Reset state if we lose frontal view during a potential blink
            if self.state != EyeState.OPEN:
                self._reset_to_open()
            return None

        # Update baseline with open-eye EAR values
        if self.state == EyeState.OPEN and ear > EAR_BLINK_THRESHOLD:
            self._ear_history.append(ear)
            if len(self._ear_history) >= 10:
                self._baseline_ear = float(np.median(list(self._ear_history)))

        below_threshold = ear < EAR_BLINK_THRESHOLD
        above_open = ear >= (EAR_BLINK_THRESHOLD + EAR_HYSTERESIS)

        if self.state == EyeState.OPEN:
            if below_threshold:
                self._consecutive_below = 1
                self._blink_start_time = timestamp
                self._min_ear_during_blink = ear
                self._pre_blink_ear = self._baseline_ear
                self._pre_blink_nose_tip = nose_tip.copy() if nose_tip is not None else None
                self.state = EyeState.CLOSING
            return None

        elif self.state == EyeState.CLOSING:
            if below_threshold:
                self._consecutive_below += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
                if self._consecutive_below >= CONSECUTIVE_FRAMES_FOR_BLINK:
                    self.state = EyeState.CLOSED
            else:
                self._reset_to_open()
            return None

        elif self.state == EyeState.CLOSED:
            if below_threshold:
                self._consecutive_below += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
                return None
            else:
                self.state = EyeState.OPENING
                return self._try_complete_blink(timestamp)

        elif self.state == EyeState.OPENING:
            if above_open:
                self.state = EyeState.OPEN
            elif below_threshold:
                self.state = EyeState.CLOSED
                self._consecutive_below += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
            return None

        return None

    def _try_complete_blink(self, timestamp: float) -> Optional[BlinkEvent]:
        """Validate blink duration and EAR drop depth, emit event if valid."""
        if self._blink_start_time is None:
            self._reset_to_open()
            return None

        duration_ms = (timestamp - self._blink_start_time) * 1000.0

        if duration_ms < MIN_BLINK_DURATION_MS or duration_ms > MAX_BLINK_DURATION_MS:
            if duration_ms > MAX_BLINK_DURATION_MS:
                pass  # Long closure: stay in opening, wait for full open
            else:
                self._reset_to_open()
            return None

        # Reject if nose moved significantly — indicates head movement, not a blink
        if (
            self._pre_blink_nose_tip is not None
            and self._current_nose_tip is not None
        ):
            nose_dist = np.linalg.norm(self._current_nose_tip - self._pre_blink_nose_tip)
            if nose_dist > 5.0:
                self._reset_to_open()
                return None

        # Validate that the EAR dropped significantly from baseline
        # A real blink drops EAR by at least 30% from the person's baseline
        if self._pre_blink_ear > 0.1:
            drop_ratio = self._min_ear_during_blink / self._pre_blink_ear
            if drop_ratio > 0.82:
                # EAR didn't drop enough — likely head movement, not a blink
                self._reset_to_open()
                return None

        event = BlinkEvent(
            timestamp=self._blink_start_time,
            person_id=self.person_id,
            ear_value=self._min_ear_during_blink,
        )
        return event

    def _reset_to_open(self) -> None:
        """Reset state machine to OPEN."""
        self.state = EyeState.OPEN
        self._consecutive_below = 0
        self._blink_start_time = None
        self._min_ear_during_blink = 1.0
