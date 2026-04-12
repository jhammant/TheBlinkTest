"""Video analysis pipeline for blink detection."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import cv2

from blinkcounter.constants import VIDEO_FRAME_SKIP
from blinkcounter.core.blink_detector import BlinkStateMachine, calculate_ear
from blinkcounter.core.face_tracker import FaceTracker
from blinkcounter.core.models import AnalysisResult

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyzes a video file for blink detection across tracked faces."""

    def __init__(self) -> None:
        self._face_tracker = FaceTracker()

    def analyze(
        self,
        video_path: str,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> AnalysisResult:
        """Analyze a video file and return blink detection results.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback receiving progress as 0.0-1.0.

        Returns:
            AnalysisResult with all detected persons and their blink data.

        Raises:
            ValueError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0.0

        # Calculate frame skip to target ~10 samples/sec
        target_samples_per_sec = 10.0
        frame_skip = max(1, int(round(fps / target_samples_per_sec)))
        if frame_skip < 1:
            frame_skip = VIDEO_FRAME_SKIP

        blink_machines: dict[str, BlinkStateMachine] = {}
        frame_number = 0
        frames_processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue

                timestamp = frame_number / fps if fps > 0 else 0.0

                # Detect faces and get eye landmarks
                tracked_faces = self._face_tracker.process_frame(frame, timestamp)

                for person, eye_landmarks in tracked_faces:
                    # eye_landmarks shape (2, 6, 2): [left_eye, right_eye]
                    left_eye, right_eye = eye_landmarks

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Create state machine for new persons
                    if person.id not in blink_machines:
                        blink_machines[person.id] = BlinkStateMachine(person.id)

                    blink_machines[person.id].update(avg_ear, timestamp)

                frames_processed += 1
                frame_number += 1

                # Report progress
                if progress_callback and total_frames > 0:
                    progress = min(frame_number / total_frames, 1.0)
                    progress_callback(progress, f"Analyzing frame {frame_number}/{total_frames}")
        finally:
            cap.release()

        # Collect persons from face tracker
        persons = self._face_tracker.get_persons()

        if progress_callback:
            progress_callback(1.0)

        return AnalysisResult(
            video_source=video_path,
            duration_seconds=duration_seconds,
            fps=fps,
            frames_processed=frames_processed,
            persons=persons,
        )
