"""BlinkFormer-based blink detection using a Transformer model.

Integrates the BlinkFormer model (BMVC 2023) for video blink detection.
The model processes sequences of 13 eye-crop frames (48x48 pixels each)
and classifies each sequence as blink or no-blink.

Reference:
    Bo Liu, Yang Xu, Feng Lu. "SynBlink and BlinkFormer: A Synthetic Dataset
    and Transformer-Based Method for Video Blink Detection." BMVC 2023.

Pre-trained weights must be downloaded separately from:
    https://pan.baidu.com/s/1NN_Y5Uiwpxx7-sAA4L0Cyg?pwd=synb

Place the weights file as:
    blinkcounter/models/blinkformer_weights.pth
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default weight location
_DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "blinkformer_weights.pth"
)

# BlinkFormer expects 13-frame sequences of 48x48 eye crops
SEQUENCE_LENGTH = 13
EYE_CROP_SIZE = 48

# ImageNet normalization used during BlinkFormer training
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class BlinkFormerEvent:
    """A blink detected by BlinkFormer."""

    timestamp: float  # Seconds from video start (mid-point of the 13-frame window)
    confidence: float  # Softmax probability of the blink class
    window_start: float  # Start timestamp of the 13-frame window
    window_end: float  # End timestamp of the 13-frame window


def _build_blinkformer(dim: int = 1024, depth: int = 6, heads: int = 16,
                       mlp_dim: int = 2048):
    """Construct the BlinkFormer model architecture.

    Reimplemented here to avoid path/import issues with the original repo.
    Architecture matches the HUST-LEBW configuration from the paper.
    """
    import torch
    import torch.nn as nn
    from einops import rearrange, repeat

    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(self.norm(x), **kwargs)

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Attention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
            super().__init__()
            inner_dim = dim_head * heads
            project_out = not (heads == 1 and dim_head == dim)
            self.heads = heads
            self.scale = dim_head ** -0.5
            self.attend = nn.Softmax(dim=-1)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            ) if project_out else nn.Identity()

        def forward(self, x):
            from einops import rearrange
            from torch import einsum

            b, n, _, h = *x.shape, self.heads
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv
            )
            dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            attn = self.attend(dots)
            out = einsum("b h i j, b h j d -> b h i d", attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)

    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
            super().__init__()
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads,
                                          dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x

    class BlinkFormer(nn.Module):
        def __init__(self, seq_length=13, num_classes=2, dim=1024, depth=6,
                     heads=16, mlp_dim=2048, channels=3, dim_head=64,
                     dropout=0.1, emb_dropout=0.0):
            super().__init__()
            self.image_size = EYE_CROP_SIZE
            patch_dim = channels * self.image_size ** 2

            self.to_patch_embedding = nn.Sequential(
                nn.Flatten(start_dim=2),  # b seq c h w -> b seq (c*h*w)
                nn.Linear(patch_dim, dim),
            )

            self.pos_embedding = nn.Parameter(
                torch.randn(1, seq_length + 1, dim)
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.dropout = nn.Dropout(emb_dropout)
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout
            )
            self.to_latent = nn.Identity()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes),
            )

        def forward(self, img):
            # img shape: (batch, seq, channels, height, width)
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head(x)

    return BlinkFormer(
        num_classes=2, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
    )


class BlinkFormerDetector:
    """Detect blinks in video using the BlinkFormer Transformer model.

    The model processes sliding windows of 13 consecutive eye-crop frames
    (48x48 pixels, single eye) and classifies each window as blink/no-blink.

    Falls back gracefully when weights are missing or dependencies are
    unavailable.

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to the BlinkFormer checkpoint. Defaults to
        ``blinkcounter/models/blinkformer_weights.pth``.
    device : str, optional
        PyTorch device (``"cpu"``, ``"mps"``, ``"cuda"``). Auto-detected.
    confidence_threshold : float
        Minimum softmax probability for the blink class to count as a blink.
    stride : int
        Step size for the sliding window. Lower = more overlap = more
        sensitivity but slower. Default 3 means ~77% overlap between windows.
    dim : int
        Model hidden dimension. Must match the checkpoint.
    depth : int
        Number of transformer layers. Must match the checkpoint.
    heads : int
        Number of attention heads. Must match the checkpoint.
    mlp_dim : int
        MLP hidden dimension. Must match the checkpoint.
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        stride: int = 3,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.stride = stride
        self._model = None
        self._device = None
        self._weights_loaded = False

        resolved_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH

        try:
            import torch

            # Determine device
            if device:
                self._device = torch.device(device)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            # Build model architecture
            model = _build_blinkformer(
                dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
            )

            # Load weights if available
            if resolved_path.exists():
                checkpoint = torch.load(
                    resolved_path, map_location=self._device, weights_only=True,
                )
                # Handle both raw state_dict and wrapped checkpoint formats
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    # Strip 'model.' prefix if present (from pytorch-lightning)
                    state_dict = {
                        k.replace("model.", "", 1) if k.startswith("model.") else k: v
                        for k, v in state_dict.items()
                    }
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
                self._weights_loaded = True
                logger.info(
                    "Loaded BlinkFormer weights from %s on %s",
                    resolved_path, self._device,
                )
            else:
                logger.warning(
                    "BlinkFormer weights not found at %s. "
                    "Model will run with random weights (results will be meaningless). "
                    "Download from: https://pan.baidu.com/s/1NN_Y5Uiwpxx7-sAA4L0Cyg?pwd=synb",
                    resolved_path,
                )

            model.to(self._device)
            model.eval()
            self._model = model

        except ImportError:
            logger.warning(
                "PyTorch or einops not installed. BlinkFormer unavailable. "
                "Install with: pip install torch einops"
            )
        except Exception:
            logger.exception("Failed to initialize BlinkFormer model")

    @property
    def is_available(self) -> bool:
        """Return True if the model was loaded successfully."""
        return self._model is not None

    @property
    def has_trained_weights(self) -> bool:
        """Return True if trained weights were loaded (vs random init)."""
        return self._weights_loaded

    def _preprocess_eye_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalize a single eye crop for BlinkFormer.

        Parameters
        ----------
        crop : np.ndarray
            BGR eye crop of any size.

        Returns
        -------
        np.ndarray
            Float32 array of shape (3, 48, 48), normalized with ImageNet stats.
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Resize to 48x48
        resized = cv2.resize(
            rgb, (EYE_CROP_SIZE, EYE_CROP_SIZE), interpolation=cv2.INTER_CUBIC,
        )
        # Normalize: [0,255] -> [0,1] -> ImageNet normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW
        return normalized.transpose(2, 0, 1)

    def process_frame_sequence(
        self, eye_crops: list[np.ndarray],
    ) -> Optional[tuple[bool, float]]:
        """Classify a sequence of 13 eye crops as blink or no-blink.

        Parameters
        ----------
        eye_crops : list of np.ndarray
            Exactly 13 BGR eye crop images (single eye, any size).

        Returns
        -------
        tuple of (bool, float) or None
            (is_blink, blink_probability) or None if model unavailable.
        """
        if self._model is None:
            return None

        if len(eye_crops) != SEQUENCE_LENGTH:
            logger.warning(
                "BlinkFormer requires exactly %d frames, got %d",
                SEQUENCE_LENGTH, len(eye_crops),
            )
            return None

        import torch

        # Preprocess each frame
        frames = np.stack([
            self._preprocess_eye_crop(crop) for crop in eye_crops
        ])  # Shape: (13, 3, 48, 48)

        # Add batch dimension
        batch = torch.from_numpy(frames).unsqueeze(0).to(self._device)
        # Shape: (1, 13, 3, 48, 48)

        with torch.no_grad():
            logits = self._model(batch)  # Shape: (1, 2)
            probs = torch.softmax(logits, dim=-1)
            blink_prob = probs[0, 1].item()  # Class 1 = blink

        is_blink = blink_prob >= self.confidence_threshold
        return is_blink, blink_prob

    def detect_blinks(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None,
    ) -> list[BlinkFormerEvent]:
        """Detect blinks in a video using sliding-window BlinkFormer inference.

        Extracts eye crops using dlib face/landmark detection, then runs
        BlinkFormer on overlapping 13-frame windows.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        progress_callback : callable, optional
            Called with (progress_fraction, message) during processing.

        Returns
        -------
        list of BlinkFormerEvent
            Detected blink events with timestamps and confidence scores.
        """
        if self._model is None:
            logger.error("BlinkFormer model not available")
            return []

        import torch

        # Extract eye crops from video
        eye_crops, timestamps = self._extract_eye_crops_from_video(
            video_path, progress_callback,
        )

        if len(eye_crops) < SEQUENCE_LENGTH:
            logger.warning(
                "Video too short: only %d eye crops extracted, need %d",
                len(eye_crops), SEQUENCE_LENGTH,
            )
            return []

        if progress_callback:
            progress_callback(0.7, "Running BlinkFormer inference...")

        # Sliding window classification
        events: list[BlinkFormerEvent] = []
        num_windows = (len(eye_crops) - SEQUENCE_LENGTH) // self.stride + 1

        # Batch processing for efficiency
        batch_size = 32
        all_windows = []
        window_indices = []

        for i in range(0, len(eye_crops) - SEQUENCE_LENGTH + 1, self.stride):
            window = eye_crops[i : i + SEQUENCE_LENGTH]
            preprocessed = np.stack([
                self._preprocess_eye_crop(crop) for crop in window
            ])
            all_windows.append(preprocessed)
            window_indices.append(i)

        if not all_windows:
            return []

        # Run inference in batches
        all_probs = []
        for batch_start in range(0, len(all_windows), batch_size):
            batch_end = min(batch_start + batch_size, len(all_windows))
            batch_data = np.stack(all_windows[batch_start:batch_end])
            batch_tensor = torch.from_numpy(batch_data).to(self._device)

            with torch.no_grad():
                logits = self._model(batch_tensor)
                probs = torch.softmax(logits, dim=-1)
                blink_probs = probs[:, 1].cpu().numpy()
                all_probs.extend(blink_probs.tolist())

        # Convert predictions to events with non-maximum suppression
        raw_detections = []
        for idx, (win_idx, prob) in enumerate(zip(window_indices, all_probs)):
            if prob >= self.confidence_threshold:
                mid_frame = win_idx + SEQUENCE_LENGTH // 2
                raw_detections.append(BlinkFormerEvent(
                    timestamp=timestamps[mid_frame],
                    confidence=prob,
                    window_start=timestamps[win_idx],
                    window_end=timestamps[min(win_idx + SEQUENCE_LENGTH - 1,
                                              len(timestamps) - 1)],
                ))

        # Non-maximum suppression: merge overlapping detections
        events = self._nms_blink_events(raw_detections, min_gap_seconds=0.3)

        if progress_callback:
            progress_callback(1.0, f"BlinkFormer detected {len(events)} blinks")

        logger.info(
            "BlinkFormer detected %d blinks in %s", len(events), video_path,
        )
        return events

    def _extract_eye_crops_from_video(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Extract single-eye crops from every frame of a video.

        Uses dlib for face detection and landmark extraction, crops the
        left eye region with padding.

        Returns
        -------
        tuple of (list[np.ndarray], list[float])
            (eye_crops, timestamps) - BGR eye crops and their timestamps.
        """
        import dlib

        predictor_path = str(
            Path(__file__).resolve().parent.parent
            / "models"
            / "shape_predictor_68_face_landmarks.dat"
        )

        if not Path(predictor_path).exists():
            logger.error("dlib shape predictor not found at %s", predictor_path)
            return [], []

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        eye_crops: list[np.ndarray] = []
        timestamps: list[float] = []
        last_face = None
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces every 5 frames for speed
            if frame_num % 5 == 0:
                faces = detector(gray, 0)
                if faces:
                    last_face = faces[0]

            if last_face is not None:
                shape = predictor(gray, last_face)

                # Extract left eye landmarks (indices 36-41)
                left_eye_pts = np.array([
                    [shape.part(i).x, shape.part(i).y] for i in range(36, 42)
                ])

                # Crop left eye with padding
                crop = self._crop_single_eye(frame, left_eye_pts)
                if crop is not None:
                    eye_crops.append(crop)
                    timestamps.append(timestamp)

            frame_num += 1

            if progress_callback and total_frames > 0 and frame_num % 100 == 0:
                progress_callback(
                    0.7 * frame_num / total_frames,
                    f"Extracting eye crops: {frame_num}/{total_frames}",
                )

        cap.release()
        logger.info(
            "Extracted %d eye crops from %d frames (%.1f fps)",
            len(eye_crops), frame_num, fps,
        )
        return eye_crops, timestamps

    @staticmethod
    def _crop_single_eye(
        frame: np.ndarray, eye_points: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Crop a single eye from a frame given 6 landmark points.

        Applies generous padding to capture the full eye region including
        eyelids, then returns a square crop.
        """
        x_min = int(eye_points[:, 0].min())
        y_min = int(eye_points[:, 1].min())
        x_max = int(eye_points[:, 0].max())
        y_max = int(eye_points[:, 1].max())

        w = x_max - x_min
        h = y_max - y_min

        # Generous padding: 80% horizontal, 120% vertical for eyelids
        pad_x = int(w * 0.8)
        pad_y = int(h * 1.2)

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # Make it square using the larger dimension
        half_size = max(w // 2 + pad_x, h // 2 + pad_y)

        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(frame.shape[1], cx + half_size)
        y2 = min(frame.shape[0], cy + half_size)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    @staticmethod
    def _nms_blink_events(
        events: list[BlinkFormerEvent], min_gap_seconds: float = 0.3,
    ) -> list[BlinkFormerEvent]:
        """Non-maximum suppression: keep only the highest-confidence detection
        within each temporal neighborhood.

        Events closer than ``min_gap_seconds`` are merged, keeping the one
        with the highest confidence.
        """
        if not events:
            return []

        # Sort by confidence descending
        sorted_events = sorted(events, key=lambda e: e.confidence, reverse=True)
        kept: list[BlinkFormerEvent] = []

        for event in sorted_events:
            # Check if this event overlaps with any already-kept event
            is_suppressed = False
            for kept_event in kept:
                if abs(event.timestamp - kept_event.timestamp) < min_gap_seconds:
                    is_suppressed = True
                    break
            if not is_suppressed:
                kept.append(event)

        # Return sorted by timestamp
        return sorted(kept, key=lambda e: e.timestamp)
