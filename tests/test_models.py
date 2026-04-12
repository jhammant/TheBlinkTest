"""Tests for data models and classification logic."""

import numpy as np
import pytest

from blinkcounter.core.models import (
    AnalysisResult,
    BlinkClassification,
    BlinkEvent,
    MatchedPerson,
    Person,
    VideoSource,
)


class TestBlinkClassification:
    def test_very_low_rate(self):
        assert BlinkClassification.from_rate(5.0) == BlinkClassification.VERY_LOW
        assert BlinkClassification.from_rate(0.0) == BlinkClassification.VERY_LOW
        assert BlinkClassification.from_rate(9.9) == BlinkClassification.VERY_LOW

    def test_low_rate(self):
        assert BlinkClassification.from_rate(10.0) == BlinkClassification.LOW
        assert BlinkClassification.from_rate(12.0) == BlinkClassification.LOW
        assert BlinkClassification.from_rate(14.9) == BlinkClassification.LOW

    def test_normal_rate(self):
        assert BlinkClassification.from_rate(15.0) == BlinkClassification.NORMAL
        assert BlinkClassification.from_rate(17.0) == BlinkClassification.NORMAL
        assert BlinkClassification.from_rate(19.9) == BlinkClassification.NORMAL

    def test_high_rate(self):
        assert BlinkClassification.from_rate(20.0) == BlinkClassification.HIGH
        assert BlinkClassification.from_rate(25.0) == BlinkClassification.HIGH
        assert BlinkClassification.from_rate(40.0) == BlinkClassification.HIGH


class TestPerson:
    def test_blink_count(self):
        person = Person(id="1", label="Person A")
        assert person.blink_count == 0

        person.blink_events = [
            BlinkEvent(timestamp=1.0, person_id="1", ear_value=0.18),
            BlinkEvent(timestamp=3.0, person_id="1", ear_value=0.19),
        ]
        assert person.blink_count == 2

    def test_blinks_per_minute(self):
        person = Person(
            id="1",
            label="Person A",
            total_visible_duration=60.0,
            blink_events=[
                BlinkEvent(timestamp=float(i), person_id="1", ear_value=0.18)
                for i in range(17)
            ],
        )
        assert person.blinks_per_minute == pytest.approx(17.0)

    def test_blinks_per_minute_short_duration(self):
        person = Person(id="1", label="Person A", total_visible_duration=0.5)
        assert person.blinks_per_minute == 0.0

    def test_blinks_per_minute_scales_correctly(self):
        person = Person(
            id="1",
            label="Person A",
            total_visible_duration=30.0,
            blink_events=[
                BlinkEvent(timestamp=float(i), person_id="1", ear_value=0.18)
                for i in range(8)
            ],
        )
        assert person.blinks_per_minute == pytest.approx(16.0)

    def test_classification(self):
        person = Person(
            id="1",
            label="Person A",
            total_visible_duration=60.0,
            blink_events=[
                BlinkEvent(timestamp=float(i), person_id="1", ear_value=0.18)
                for i in range(5)
            ],
        )
        assert person.classification == BlinkClassification.VERY_LOW


class TestVideoSource:
    def test_youtube_url_detection(self):
        assert VideoSource("https://www.youtube.com/watch?v=abc123").is_youtube
        assert VideoSource("https://youtu.be/abc123").is_youtube
        assert not VideoSource("/path/to/video.mp4").is_youtube
        assert not VideoSource("file.avi").is_youtube


class TestMatchedPerson:
    def test_average_blinks_per_minute(self):
        mp = MatchedPerson(
            label="Person A",
            per_video_rates={"video1.mp4": 12.0, "video2.mp4": 18.0},
        )
        assert mp.average_blinks_per_minute == pytest.approx(15.0)

    def test_average_blinks_empty(self):
        mp = MatchedPerson(label="Person A")
        assert mp.average_blinks_per_minute == 0.0

    def test_classification(self):
        mp = MatchedPerson(
            label="Person A",
            per_video_rates={"v1": 8.0, "v2": 6.0},
        )
        assert mp.classification == BlinkClassification.VERY_LOW


class TestAnalysisResult:
    def test_person_count(self):
        result = AnalysisResult(
            video_source="test.mp4",
            duration_seconds=120.0,
            fps=30.0,
            frames_processed=400,
            persons=[
                Person(id="1", label="Person A"),
                Person(id="2", label="Person B"),
            ],
        )
        assert result.person_count == 2
