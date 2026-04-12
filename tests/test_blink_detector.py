"""Tests for blink detection engine."""

import numpy as np
import pytest

from blinkcounter.constants import (
    CONSECUTIVE_FRAMES_FOR_BLINK,
    EAR_BLINK_THRESHOLD,
    MAX_BLINK_DURATION_MS,
    MIN_BLINK_DURATION_MS,
)
from blinkcounter.core.blink_detector import BlinkStateMachine, calculate_ear
from blinkcounter.core.models import EyeState


def _make_eye_landmarks(ear_target: float) -> np.ndarray:
    """Create synthetic eye landmarks that produce approximately the given EAR.

    Layout: p1=outer corner, p2=upper1, p3=upper2, p4=inner corner, p5=lower1, p6=lower2
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    We fix horizontal distance at 1.0 and set vertical distances to achieve EAR.
    """
    # p1 and p4 on horizontal axis, 1.0 apart
    p1 = np.array([0.0, 0.5])
    p4 = np.array([1.0, 0.5])

    # Each vertical pair contributes ear_target * horizontal to the numerator
    # vertical_a = vertical_b = ear_target * 1.0
    half_v = ear_target * 1.0 / 2.0

    p2 = np.array([0.3, 0.5 + half_v])  # upper1
    p6 = np.array([0.3, 0.5 - half_v])  # lower2
    p3 = np.array([0.7, 0.5 + half_v])  # upper2
    p5 = np.array([0.7, 0.5 - half_v])  # lower1

    return np.array([p1, p2, p3, p4, p5, p6])


class TestCalculateEar:
    def test_calculate_ear_open_eyes(self):
        """Open eyes should produce EAR above the blink threshold."""
        landmarks = _make_eye_landmarks(0.35)
        ear = calculate_ear(landmarks)
        assert ear > EAR_BLINK_THRESHOLD
        assert ear == pytest.approx(0.35, abs=0.01)

    def test_calculate_ear_closed_eyes(self):
        """Closed eyes should produce EAR below the blink threshold."""
        landmarks = _make_eye_landmarks(0.10)
        ear = calculate_ear(landmarks)
        assert ear < EAR_BLINK_THRESHOLD
        assert ear == pytest.approx(0.10, abs=0.01)

    def test_calculate_ear_zero_horizontal(self):
        """Zero horizontal distance should return 0.0 to avoid division by zero."""
        landmarks = np.array([
            [0.5, 0.5],
            [0.5, 0.6],
            [0.5, 0.6],
            [0.5, 0.5],  # same as p1
            [0.5, 0.4],
            [0.5, 0.4],
        ])
        assert calculate_ear(landmarks) == 0.0


class TestBlinkStateMachine:
    def _make_sm(self) -> BlinkStateMachine:
        return BlinkStateMachine(person_id="test_person")

    def _frame_interval(self) -> float:
        """Return a frame interval (~33ms at 30fps) that keeps blink in valid range."""
        return 0.033

    def test_state_machine_detects_blink(self):
        """Simulate EAR sequence: high -> low -> low -> high = blink detected."""
        sm = self._make_sm()
        dt = self._frame_interval()
        t = 0.0

        # Open frames
        assert sm.update(0.30, t) is None
        t += dt

        # Closing frames (at least CONSECUTIVE_FRAMES_FOR_BLINK below threshold)
        for _ in range(CONSECUTIVE_FRAMES_FOR_BLINK):
            assert sm.update(0.15, t) is None
            t += dt

        # Extra closed frame to ensure duration >= MIN_BLINK_DURATION_MS
        assert sm.update(0.15, t) is None
        t += dt

        # Eye opens — blink should be detected
        event = sm.update(0.30, t)
        assert event is not None
        assert event.person_id == "test_person"
        assert event.ear_value == pytest.approx(0.15, abs=0.01)

    def test_state_machine_rejects_noise(self):
        """A single frame dip should not register as a blink."""
        sm = self._make_sm()
        dt = self._frame_interval()
        t = 0.0

        # Open
        assert sm.update(0.30, t) is None
        t += dt

        # Single frame below threshold
        assert sm.update(0.15, t) is None
        t += dt

        # Back to open immediately — state should reset, no blink
        assert sm.update(0.30, t) is None
        assert sm.state == EyeState.OPEN

    def test_state_machine_rejects_long_closure(self):
        """Closure longer than MAX_BLINK_DURATION_MS should be rejected."""
        sm = self._make_sm()
        t = 0.0

        # Open
        sm.update(0.30, t)
        t += 0.033

        # Stay closed for much longer than MAX_BLINK_DURATION_MS
        num_closed_frames = int((MAX_BLINK_DURATION_MS / 1000.0 + 0.5) / 0.033)
        for _ in range(num_closed_frames):
            sm.update(0.15, t)
            t += 0.033

        # Open again — blink should be rejected due to long duration
        event = sm.update(0.30, t)
        assert event is None

    def test_multiple_blinks(self):
        """Detect multiple blinks in sequence."""
        sm = self._make_sm()
        dt = self._frame_interval()
        t = 0.0
        blinks = []

        for _ in range(3):
            # Open phase
            for _ in range(5):
                result = sm.update(0.30, t)
                if result:
                    blinks.append(result)
                t += dt

            # Closed phase
            for _ in range(CONSECUTIVE_FRAMES_FOR_BLINK + 1):
                result = sm.update(0.15, t)
                if result:
                    blinks.append(result)
                t += dt

            # Opening — triggers blink detection
            result = sm.update(0.30, t)
            if result:
                blinks.append(result)
            t += dt

        assert len(blinks) == 3
        for event in blinks:
            assert event.person_id == "test_person"

    def test_initial_state_is_open(self):
        """State machine should start in OPEN state."""
        sm = self._make_sm()
        assert sm.state == EyeState.OPEN
