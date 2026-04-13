"""Video analysis pipeline for blink detection with parallel processing."""

from __future__ import annotations

import logging
import multiprocessing as mp_proc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import cv2
import numpy as np

from blinkcounter.constants import VIDEO_FRAME_SKIP
from blinkcounter.core.blink_detector import (
    QUALITY_THRESHOLD,
    BlinkStateMachine,
    calculate_detection_quality,
    calculate_ear,
    check_ear_symmetry,
    estimate_head_pose,
)
from blinkcounter.core.models import AnalysisResult, BlinkEvent, Person

logger = logging.getLogger(__name__)


def _extract_eye_crop(frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray | None:
    """Extract a padded eye crop from the frame using landmark points.

    Combines both eyes into one crop for CNN classification.
    """
    # Get bounding box around both eyes
    all_points = np.vstack([left_eye, right_eye])
    x_min = int(all_points[:, 0].min())
    y_min = int(all_points[:, 1].min())
    x_max = int(all_points[:, 0].max())
    y_max = int(all_points[:, 1].max())

    # Pad by 50%
    w = x_max - x_min
    h = y_max - y_min
    pad_x = int(w * 0.5)
    pad_y = int(h * 0.8)  # More vertical padding for eyelids

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(frame.shape[1], x_max + pad_x)
    y_max = min(frame.shape[0], y_max + pad_y)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    return crop


def _analyze_chunk(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    predictor_path: str,
) -> list[dict]:
    """Analyze a chunk of video frames in a separate process.

    Returns a list of per-frame results: [{face_rects, ear_values, timestamps, encodings, thumbnails}]
    """
    import dlib
    import face_recognition

    from blinkcounter.constants import FACE_MATCH_TOLERANCE

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results = []
    frame_num = start_frame
    last_faces = []

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = frame_num / fps

        # Detect faces every 5 frames for speed
        if (frame_num - start_frame) % 5 == 0:
            last_faces = detector(gray, 0)

        frame_data = []
        for face in last_faces:
            shape = predictor(gray, face)

            # Extract ALL 68 landmarks
            all_pts = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)],
                dtype=np.float64,
            )

            left_eye = all_pts[36:42]
            right_eye = all_pts[42:48]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Skip if EAR is wildly asymmetric
            if not check_ear_symmetry(left_ear, right_ear):
                continue

            avg_ear = (left_ear + right_ear) / 2.0

            # Head pose for filtering
            head_pose = estimate_head_pose(all_pts)

            # Face encoding for person matching
            x_min = max(0, face.left())
            y_min = max(0, face.top())
            x_max = min(frame.shape[1], face.right())
            y_max = min(frame.shape[0], face.bottom())

            face_location = [(y_min, x_max, y_max, x_min)]
            encodings = face_recognition.face_encodings(rgb, face_location)
            encoding = encodings[0].tolist() if encodings else None

            # Thumbnail
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size > 0:
                thumbnail = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_AREA)
            else:
                thumbnail = np.zeros((64, 64, 3), dtype=np.uint8)

            frame_data.append({
                "ear": avg_ear,
                "timestamp": timestamp,
                "encoding": encoding,
                "thumbnail": thumbnail.tolist(),
                "face_center": ((x_min + x_max) // 2, (y_min + y_max) // 2),
                "head_pose": head_pose,
            })

        if frame_data:
            results.append(frame_data)

        frame_num += 1

    cap.release()
    return results


class VideoAnalyzer:
    """Analyzes a video file for blink detection across tracked faces."""

    def __init__(self, max_workers: int = 0, use_cnn: bool = False) -> None:
        """Initialize with optional worker count.

        Args:
            max_workers: Number of parallel workers. 0 = auto (CPU count).
            use_cnn: Whether to use CNN eye classifier for blink confirmation.
        """
        if max_workers <= 0:
            max_workers = max(1, mp_proc.cpu_count() or 4)
        self._max_workers = max_workers
        self._use_cnn = use_cnn

    def analyze(
        self,
        video_path: str,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> AnalysisResult:
        """Analyze a video file and return blink detection results.

        Uses parallel processing to maximize CPU utilization.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0.0
        cap.release()

        if total_frames == 0:
            return AnalysisResult(
                video_source=video_path,
                duration_seconds=0.0,
                fps=fps,
                frames_processed=0,
            )

        # Find predictor path
        from pathlib import Path
        predictor_path = str(
            Path(__file__).parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"
        )

        # Always use sequential mode for best accuracy (correlation tracking
        # maintains person identity across frames, which parallel can't do).
        # Sequential is ~4x slower but eliminates person fragmentation.
        return self._analyze_sequential(video_path, fps, total_frames, duration_seconds, predictor_path, progress_callback)

    def _analyze_sequential(
        self,
        video_path: str,
        fps: float,
        total_frames: int,
        duration_seconds: float,
        predictor_path: str,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> AnalysisResult:
        """Pipeline-parallel analysis: read/detect in threads, blink logic in order.

        Uses a producer-consumer pattern:
        - Reader thread reads frames as fast as possible
        - Face tracker processes frames sequentially (maintains correlation tracking)
        - Blink detection runs on tracker output (fast, sequential)
        """
        import queue
        import threading

        from collections import deque as deque_type
        from blinkcounter.core.face_tracker import FaceTracker
        from blinkcounter.core.eye_classifier import EyeStateClassifier

        tracker = FaceTracker(detect_interval=10)
        blink_machines: dict[str, BlinkStateMachine] = {}

        # Temporal blink model — trained on UBFC ground truth (F1=0.717)
        temporal_model = None
        temporal_window = 13
        try:
            import torch
            from pathlib import Path as P2
            from train_temporal_blink_model import TemporalBlinkCNN
            model_path = P2(__file__).parent.parent / "models" / "temporal_blink_model.pth"
            if model_path.exists():
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                ckpt = torch.load(model_path, map_location=device, weights_only=True)
                temporal_window = ckpt.get("window_size", 13)
                temporal_model = TemporalBlinkCNN(window_size=temporal_window).to(device)
                temporal_model.load_state_dict(ckpt["model_state_dict"])
                temporal_model.eval()
                logger.info("Temporal blink model loaded (F1=%.3f)", ckpt.get("val_f1", 0))
        except Exception as e:
            logger.debug("Temporal model not available: %s", e)

        # Per-person EAR buffers for temporal model
        ear_buffers: dict[str, deque_type] = {}
        ear_ts_buffers: dict[str, deque_type] = {}
        temporal_last_blink: dict[str, float] = {}  # Prevent double-counting

        # CNN eye classifier
        cnn = EyeStateClassifier() if self._use_cnn else None
        use_cnn = cnn is not None and cnn.is_available
        if use_cnn:
            logger.info("CNN eye classifier loaded — using ensemble detection")
        else:
            logger.info("Using EAR-only detection")

        # Queue for read frames: (frame_number, frame) or None for end
        frame_queue: queue.Queue = queue.Queue(maxsize=64)
        # Queue for tracked results: (frame_number, tracked_faces) or None for end
        tracked_queue: queue.Queue = queue.Queue(maxsize=64)

        frames_processed = 0
        reader_done = threading.Event()
        tracker_done = threading.Event()

        def reader_worker():
            """Read frames from video into queue."""
            cap = cv2.VideoCapture(video_path)
            fn = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_queue.put((fn, frame))
                    fn += 1
            finally:
                cap.release()
                frame_queue.put(None)  # Sentinel
                reader_done.set()

        def tracker_worker():
            """Process frames through face tracker (sequential, correlation tracking)."""
            while True:
                item = frame_queue.get()
                if item is None:
                    tracked_queue.put(None)  # Sentinel
                    tracker_done.set()
                    break
                fn, frame = item
                timestamp = fn / fps if fps > 0 else 0.0
                tracked_faces = tracker.process_frame(frame, timestamp)
                tracked_queue.put((fn, timestamp, tracked_faces, frame))

        # Start pipeline threads
        reader_thread = threading.Thread(target=reader_worker, daemon=True)
        tracker_thread = threading.Thread(target=tracker_worker, daemon=True)
        reader_thread.start()
        tracker_thread.start()

        # Main thread: consume tracked faces and run blink detection
        frame_number = 0
        prev_timestamp: dict[str, float] = {}  # person_id -> last timestamp
        total_analyzable_frames = 0
        total_quality_frames = 0
        try:
            while True:
                item = tracked_queue.get()
                if item is None:
                    break
                fn, timestamp, tracked_faces, frame = item

                for person, eye_landmarks, all_landmarks in tracked_faces:
                    left_eye, right_eye = eye_landmarks
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)

                    if not check_ear_symmetry(left_ear, right_ear):
                        continue

                    avg_ear = (left_ear + right_ear) / 2.0
                    head_pose = estimate_head_pose(all_landmarks)

                    # Calculate detection quality score
                    quality = calculate_detection_quality(
                        left_ear, right_ear, head_pose, all_landmarks,
                    )

                    # CNN confirmation: when EAR suggests eyes might be closing,
                    # ask the CNN if the eyes actually look closed
                    cnn_closed_prob = None
                    if use_cnn and avg_ear < 0.28:  # Loose pre-filter
                        eye_crop = _extract_eye_crop(frame, left_eye, right_eye)
                        if eye_crop is not None:
                            cnn_closed_prob = cnn.predict(eye_crop)

                    if person.id not in blink_machines:
                        blink_machines[person.id] = BlinkStateMachine(person.id)

                    # Temporal model path: use trained CNN on EAR sequences
                    if temporal_model is not None:
                        import torch as _torch
                        # Buffer EAR values per person
                        if person.id not in ear_buffers:
                            ear_buffers[person.id] = deque_type(maxlen=temporal_window)
                            ear_ts_buffers[person.id] = deque_type(maxlen=temporal_window)
                            temporal_last_blink[person.id] = -1.0

                        ear_buffers[person.id].append(avg_ear)
                        ear_ts_buffers[person.id].append(timestamp)

                        # Once we have a full window, run temporal model
                        if len(ear_buffers[person.id]) == temporal_window:
                            seq = list(ear_buffers[person.id])
                            center_ts = ear_ts_buffers[person.id][temporal_window // 2]
                            inp = _torch.tensor([seq], dtype=_torch.float32).unsqueeze(0)
                            device = next(temporal_model.parameters()).device
                            with _torch.no_grad():
                                logit = temporal_model(inp.to(device)).squeeze()
                                prob = _torch.sigmoid(logit).item()

                            # Blink detected if prob > 0.8 and not too close to last blink
                            if prob > 0.8 and (center_ts - temporal_last_blink[person.id]) > 0.3:
                                from blinkcounter.core.models import BlinkEvent
                                person.blink_events.append(BlinkEvent(
                                    timestamp=center_ts,
                                    person_id=person.id,
                                    ear_value=min(seq),
                                ))
                                temporal_last_blink[person.id] = center_ts
                    else:
                        # Fallback: EAR-only state machine
                        event = blink_machines[person.id].update(
                            avg_ear, timestamp, head_pose,
                            nose_tip=all_landmarks[30], quality=quality,
                            cnn_closed_prob=cnn_closed_prob,
                        )
                        if event is not None:
                            person.blink_events.append(event)

                    # Track analyzable duration per person
                    total_quality_frames += 1
                    if quality >= QUALITY_THRESHOLD:
                        total_analyzable_frames += 1
                        if person.id in prev_timestamp:
                            dt = timestamp - prev_timestamp[person.id]
                            # Only add reasonable intervals (< 1s) to avoid
                            # gaps from re-detection after absence
                            if 0 < dt < 1.0:
                                person.analyzable_duration += dt
                    prev_timestamp[person.id] = timestamp

                frames_processed += 1
                frame_number = fn + 1

                if progress_callback and total_frames > 0 and frames_processed % 200 == 0:
                    progress_callback(
                        min(frames_processed / total_frames, 1.0),
                        f"Analyzing frame {frames_processed}/{total_frames}",
                    )
        finally:
            reader_thread.join(timeout=5)
            tracker_thread.join(timeout=5)

        persons = tracker.get_persons()

        # Log quality stats
        if total_quality_frames > 0:
            analyzable_pct = total_analyzable_frames / total_quality_frames * 100.0
            logger.info(
                "Quality stats: %d/%d frames analyzable (%.1f%%)",
                total_analyzable_frames, total_quality_frames, analyzable_pct,
            )
        for p in persons:
            if p.total_visible_duration > 0:
                logger.info(
                    "%s: analyzable %.1fs / visible %.1fs (%.0f%%)",
                    p.label, p.analyzable_duration, p.total_visible_duration,
                    p.analyzable_duration / p.total_visible_duration * 100.0
                    if p.total_visible_duration > 0 else 0.0,
                )
        if progress_callback:
            progress_callback(1.0, "Analysis complete")

        return AnalysisResult(
            video_source=video_path,
            duration_seconds=duration_seconds,
            fps=fps,
            frames_processed=frames_processed,
            persons=persons,
        )

    def _analyze_parallel(
        self,
        video_path: str,
        fps: float,
        total_frames: int,
        duration_seconds: float,
        predictor_path: str,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> AnalysisResult:
        """Parallel analysis splitting video into chunks across CPU cores."""
        import face_recognition
        from blinkcounter.constants import FACE_MATCH_TOLERANCE

        num_workers = min(self._max_workers, max(1, total_frames // 300))
        chunk_size = total_frames // num_workers
        # Overlap chunks by 30 frames (~1s) to avoid losing blinks at boundaries
        overlap = 30
        chunks = []
        for i in range(num_workers):
            start = max(0, i * chunk_size - overlap) if i > 0 else 0
            end = start + chunk_size + overlap if i < num_workers - 1 else total_frames
            # Track the "valid" range (non-overlapping portion) for blink event filtering
            valid_start = i * chunk_size
            valid_end = (i + 1) * chunk_size if i < num_workers - 1 else total_frames
            chunks.append((start, end, valid_start, valid_end))

        if progress_callback:
            progress_callback(0.0, f"Analyzing with {num_workers} parallel workers...")

        # Process chunks in parallel
        all_chunk_results: list[tuple[list, float, float]] = [None] * num_workers  # (data, valid_start_ts, valid_end_ts)
        completed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for idx, (start, end, valid_start, valid_end) in enumerate(chunks):
                future = executor.submit(
                    _analyze_chunk, video_path, start, end, fps, predictor_path
                )
                futures[future] = (idx, valid_start / fps, valid_end / fps)

            for future in as_completed(futures):
                idx, valid_start_ts, valid_end_ts = futures[future]
                all_chunk_results[idx] = (future.result(), valid_start_ts, valid_end_ts)
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed / num_workers * 0.8,
                        f"Completed chunk {completed}/{num_workers}",
                    )

        if progress_callback:
            progress_callback(0.8, "Merging results and matching persons...")

        # Merge all frame data in order, filtering to valid (non-overlapping) ranges
        all_frame_data = []
        for chunk_entry in all_chunk_results:
            if chunk_entry is None:
                continue
            chunk_data, valid_start_ts, valid_end_ts = chunk_entry
            for frame_faces in chunk_data:
                # Only include frames within this chunk's valid time range
                if frame_faces:
                    ts = frame_faces[0]["timestamp"]
                    if valid_start_ts <= ts < valid_end_ts:
                        all_frame_data.append(frame_faces)

        # Match persons across all frames and detect blinks
        persons, total_processed = self._merge_and_detect(
            all_frame_data, fps, FACE_MATCH_TOLERANCE
        )

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

        return AnalysisResult(
            video_source=video_path,
            duration_seconds=duration_seconds,
            fps=fps,
            frames_processed=total_processed,
            persons=persons,
        )

    def _merge_and_detect(
        self,
        all_frame_data: list[list[dict]],
        fps: float,
        face_match_tolerance: float,
    ) -> tuple[list[Person], int]:
        """Merge frame data from parallel chunks, match persons, detect blinks."""
        import face_recognition
        from blinkcounter.constants import PERSON_LABELS, FACE_ENCODING_UPDATE_INTERVAL

        persons: list[Person] = []
        blink_machines: dict[str, BlinkStateMachine] = {}
        next_label = 0
        total_processed = 0

        for frame_faces in all_frame_data:
            total_processed += 1
            for face_data in frame_faces:
                ear = face_data["ear"]
                timestamp = face_data["timestamp"]
                encoding = np.array(face_data["encoding"]) if face_data["encoding"] else None
                thumbnail = np.array(face_data["thumbnail"], dtype=np.uint8)

                # Match to existing person
                matched_person = None
                if encoding is not None and persons:
                    known = [(i, p) for i, p in enumerate(persons) if p.face_encoding is not None]
                    if known:
                        known_encs = [p.face_encoding for _, p in known]
                        distances = face_recognition.face_distance(known_encs, encoding)
                        best = int(np.argmin(distances))
                        if distances[best] < face_match_tolerance:
                            matched_person = known[best][1]

                if matched_person is None:
                    label_char = PERSON_LABELS[next_label] if next_label < len(PERSON_LABELS) else f"#{next_label+1}"
                    matched_person = Person(
                        id=f"person_{label_char.lower()}",
                        label=f"Person {label_char}",
                        face_encoding=encoding,
                        face_thumbnail=thumbnail,
                        first_seen_at=timestamp,
                        last_seen_at=timestamp,
                    )
                    persons.append(matched_person)
                    next_label += 1
                else:
                    if matched_person.last_seen_at > 0:
                        matched_person.total_visible_duration += timestamp - matched_person.last_seen_at
                    matched_person.last_seen_at = timestamp
                    matched_person.face_thumbnail = thumbnail

                # Blink detection with head pose filtering
                if matched_person.id not in blink_machines:
                    blink_machines[matched_person.id] = BlinkStateMachine(matched_person.id)

                head_pose = face_data.get("head_pose")
                event = blink_machines[matched_person.id].update(ear, timestamp, head_pose)
                if event is not None:
                    matched_person.blink_events.append(event)

        return persons, total_processed
