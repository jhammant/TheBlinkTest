"""Data models for blink detection and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from blinkcounter.constants import (
    CLASSIFICATION_LOW_MAX,
    CLASSIFICATION_NORMAL_MAX,
    CLASSIFICATION_VERY_LOW_MAX,
)


class BlinkClassification(Enum):
    """Blink rate classification based on blinks per minute."""

    VERY_LOW = "Very Low"  # < 10/min - potential psychopathy indicator
    LOW = "Low"            # 10-15/min - below normal
    NORMAL = "Normal"      # 15-20/min - typical range
    HIGH = "High"          # > 20/min - above normal (stress/anxiety)

    @staticmethod
    def from_rate(blinks_per_minute: float) -> BlinkClassification:
        if blinks_per_minute < CLASSIFICATION_VERY_LOW_MAX:
            return BlinkClassification.VERY_LOW
        elif blinks_per_minute < CLASSIFICATION_LOW_MAX:
            return BlinkClassification.LOW
        elif blinks_per_minute < CLASSIFICATION_NORMAL_MAX:
            return BlinkClassification.NORMAL
        else:
            return BlinkClassification.HIGH


class EyeState(Enum):
    """State machine states for blink detection."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    OPENING = "opening"


@dataclass
class BlinkEvent:
    """A single detected blink."""

    timestamp: float  # Seconds from video start
    person_id: str
    ear_value: float  # EAR at lowest point of blink


@dataclass
class Person:
    """A tracked individual in a video."""

    id: str
    label: str  # "Person A", "Person B", etc.
    face_encoding: Optional[np.ndarray] = None  # 128-dim face_recognition encoding
    face_thumbnail: Optional[np.ndarray] = None  # Cropped face image (BGR)
    blink_events: list[BlinkEvent] = field(default_factory=list)
    first_seen_at: float = 0.0  # Seconds from video start
    last_seen_at: float = 0.0
    total_visible_duration: float = 0.0  # Total seconds on screen

    @property
    def blink_count(self) -> int:
        return len(self.blink_events)

    @property
    def blinks_per_minute(self) -> float:
        if self.total_visible_duration < 1.0:
            return 0.0
        return (self.blink_count / self.total_visible_duration) * 60.0

    @property
    def classification(self) -> BlinkClassification:
        return BlinkClassification.from_rate(self.blinks_per_minute)


@dataclass
class AnalysisResult:
    """Results from analyzing a single video."""

    video_source: str  # File path or YouTube URL
    duration_seconds: float
    fps: float
    frames_processed: int
    persons: list[Person] = field(default_factory=list)

    @property
    def person_count(self) -> int:
        return len(self.persons)


@dataclass
class VideoSource:
    """A video to be analyzed - either a local file or YouTube URL."""

    path_or_url: str
    label: str = ""  # Optional user-friendly label

    @property
    def is_youtube(self) -> bool:
        return any(
            domain in self.path_or_url
            for domain in ("youtube.com", "youtu.be", "youtube.co")
        )


@dataclass
class BatchResult:
    """Results from analyzing multiple videos with cross-video person matching."""

    video_results: list[AnalysisResult] = field(default_factory=list)
    matched_persons: list[MatchedPerson] = field(default_factory=list)


@dataclass
class MatchedPerson:
    """A person matched across multiple videos."""

    label: str  # "Person A", etc.
    face_thumbnail: Optional[np.ndarray] = None
    face_encoding: Optional[np.ndarray] = None
    per_video_rates: dict[str, float] = field(default_factory=dict)  # video_source -> bpm
    per_video_blink_counts: dict[str, int] = field(default_factory=dict)
    per_video_durations: dict[str, float] = field(default_factory=dict)  # seconds

    @property
    def average_blinks_per_minute(self) -> float:
        if not self.per_video_rates:
            return 0.0
        return sum(self.per_video_rates.values()) / len(self.per_video_rates)

    @property
    def classification(self) -> BlinkClassification:
        return BlinkClassification.from_rate(self.average_blinks_per_minute)
