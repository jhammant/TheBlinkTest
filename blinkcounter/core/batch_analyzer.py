"""Batch analysis of multiple videos with cross-video person matching."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import face_recognition
import numpy as np

from blinkcounter.constants import FACE_MATCH_TOLERANCE, PERSON_LABELS
from blinkcounter.core.models import (
    AnalysisResult,
    BatchResult,
    MatchedPerson,
    Person,
    VideoSource,
)
from blinkcounter.core.video_analyzer import VideoAnalyzer
from blinkcounter.services.youtube import download_video

logger = logging.getLogger(__name__)


class BatchAnalyzer:
    """Analyzes multiple videos and matches persons across them."""

    def __init__(self, high_confidence: bool = False) -> None:
        self._high_confidence = high_confidence

    def analyze(
        self,
        sources: list[str] | list[VideoSource],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> BatchResult:
        """Analyze a batch of video sources and match persons across videos.

        Args:
            sources: List of video sources to analyze.
            progress_callback: Optional callback receiving (progress 0.0-1.0, status).

        Returns:
            BatchResult with per-video results and cross-video matched persons.
        """
        # Convert strings to VideoSource objects if needed
        video_sources: list[VideoSource] = []
        for s in sources:
            if isinstance(s, str):
                video_sources.append(VideoSource(path_or_url=s))
            else:
                video_sources.append(s)

        video_results: list[AnalysisResult] = []
        total_sources = len(video_sources)

        for idx, source in enumerate(video_sources):
            video_label = source.label or source.path_or_url
            if progress_callback:
                progress_callback(
                    idx / total_sources,
                    f"Processing video {idx + 1}/{total_sources}: {video_label}",
                )

            # Download YouTube videos if needed
            video_path = source.path_or_url
            if source.is_youtube:
                if progress_callback:
                    progress_callback(
                        idx / total_sources,
                        f"Downloading: {video_label}",
                    )
                video_path = download_video(source.path_or_url)

            # Analyze the video
            def _video_progress(p: float) -> None:
                if progress_callback:
                    overall = (idx + p) / total_sources
                    progress_callback(overall, f"Analyzing: {video_label}")

            analyzer = VideoAnalyzer(high_confidence=self._high_confidence)
            result = analyzer.analyze(video_path, _video_progress)
            # Override video_source with the original URL/path
            result.video_source = source.path_or_url
            video_results.append(result)

        if progress_callback:
            progress_callback(0.95, "Matching persons across videos...")

        matched_persons = self._match_persons_across_videos(video_results)

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

        return BatchResult(
            video_results=video_results,
            matched_persons=matched_persons,
        )

    def _match_persons_across_videos(
        self, results: list[AnalysisResult]
    ) -> list[MatchedPerson]:
        """Match persons across multiple video results using face encodings.

        Groups persons with similar face encodings and builds MatchedPerson
        objects with per-video statistics.
        """
        # Collect all persons with their video source
        all_persons: list[tuple[str, Person]] = []
        for result in results:
            for person in result.persons:
                all_persons.append((result.video_source, person))

        if not all_persons:
            return []

        # Build groups of matched persons using union-find style grouping
        groups: list[list[tuple[str, Person]]] = []

        for video_source, person in all_persons:
            matched_group = None

            if person.face_encoding is not None:
                for group in groups:
                    # Compare against the first person in each group that has an encoding
                    for _, group_person in group:
                        if group_person.face_encoding is not None:
                            matches = face_recognition.compare_faces(
                                [group_person.face_encoding],
                                person.face_encoding,
                                tolerance=FACE_MATCH_TOLERANCE,
                            )
                            if matches[0]:
                                matched_group = group
                                break
                    if matched_group is not None:
                        break

            if matched_group is not None:
                matched_group.append((video_source, person))
            else:
                groups.append([(video_source, person)])

        # Build MatchedPerson objects
        matched_persons: list[MatchedPerson] = []
        for group_idx, group in enumerate(groups):
            label = (
                f"Person {PERSON_LABELS[group_idx]}"
                if group_idx < len(PERSON_LABELS)
                else f"Person {group_idx + 1}"
            )

            per_video_rates: dict[str, float] = {}
            per_video_blink_counts: dict[str, int] = {}
            per_video_durations: dict[str, float] = {}
            best_thumbnail = None
            best_thumbnail_size = 0
            best_encoding = None

            for video_source, person in group:
                per_video_rates[video_source] = person.blinks_per_minute
                per_video_blink_counts[video_source] = person.blink_count
                per_video_durations[video_source] = person.total_visible_duration

                # Pick the largest face thumbnail as the best one
                if person.face_thumbnail is not None:
                    size = person.face_thumbnail.shape[0] * person.face_thumbnail.shape[1]
                    if size > best_thumbnail_size:
                        best_thumbnail = person.face_thumbnail
                        best_thumbnail_size = size

                if person.face_encoding is not None and best_encoding is None:
                    best_encoding = person.face_encoding

            matched_persons.append(
                MatchedPerson(
                    label=label,
                    face_thumbnail=best_thumbnail,
                    face_encoding=best_encoding,
                    per_video_rates=per_video_rates,
                    per_video_blink_counts=per_video_blink_counts,
                    per_video_durations=per_video_durations,
                )
            )

        return matched_persons
