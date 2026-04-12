"""Blink detection engine using Eye Aspect Ratio (EAR) and state machine."""

from __future__ import annotations

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


class BlinkStateMachine:
    """Per-person state machine for tracking blink events.

    Transitions: OPEN -> CLOSING -> CLOSED -> OPENING -> OPEN
    """

    def __init__(self, person_id: str) -> None:
        self.person_id = person_id
        self.state = EyeState.OPEN
        self._consecutive_below = 0
        self._blink_start_time: Optional[float] = None
        self._min_ear_during_blink: float = 1.0

    def update(self, ear: float, timestamp: float) -> Optional[BlinkEvent]:
        """Process a new EAR measurement and return a BlinkEvent if a blink completes.

        Args:
            ear: Current Eye Aspect Ratio value.
            timestamp: Current timestamp in seconds from video start.

        Returns:
            BlinkEvent if a complete valid blink was detected, None otherwise.
        """
        below_threshold = ear < EAR_BLINK_THRESHOLD
        above_open = ear >= (EAR_BLINK_THRESHOLD + EAR_HYSTERESIS)

        if self.state == EyeState.OPEN:
            if below_threshold:
                self._consecutive_below = 1
                self._blink_start_time = timestamp
                self._min_ear_during_blink = ear
                self.state = EyeState.CLOSING
            return None

        elif self.state == EyeState.CLOSING:
            if below_threshold:
                self._consecutive_below += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
                if self._consecutive_below >= CONSECUTIVE_FRAMES_FOR_BLINK:
                    self.state = EyeState.CLOSED
            else:
                # Went back above threshold before enough consecutive frames
                self._reset_to_open()
            return None

        elif self.state == EyeState.CLOSED:
            if below_threshold:
                self._consecutive_below += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
                return None
            else:
                # EAR rising — transition to opening
                self.state = EyeState.OPENING
                return self._try_complete_blink(timestamp)

        elif self.state == EyeState.OPENING:
            if above_open:
                self.state = EyeState.OPEN
            elif below_threshold:
                # Dropped back down — stay in closed state
                self.state = EyeState.CLOSED
                self._consecutive_below += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
            return None

        return None

    def _try_complete_blink(self, timestamp: float) -> Optional[BlinkEvent]:
        """Validate blink duration and emit event if valid."""
        if self._blink_start_time is None:
            self._reset_to_open()
            return None

        duration_ms = (timestamp - self._blink_start_time) * 1000.0

        if duration_ms < MIN_BLINK_DURATION_MS or duration_ms > MAX_BLINK_DURATION_MS:
            # Invalid duration — reject
            if duration_ms > MAX_BLINK_DURATION_MS:
                # Long closure: stay in opening, wait for full open
                pass
            else:
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
