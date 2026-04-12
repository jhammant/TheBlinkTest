"""CNN-based eye state classifier for blink detection.

Replaces the EAR (Eye Aspect Ratio) heuristic with a learned model that is
more robust to head movement and camera angle changes.

The model is trained by ``train_eye_classifier.py`` on the RT-BENE dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _build_eye_state_cnn():
    """Construct the EyeStateCNN model (must match train_eye_classifier.py).

    Defined here to avoid importing the training script at inference time.
    """
    import torch.nn as nn

    class EyeStateCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    return EyeStateCNN()

# Default model location (relative to this file)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "eye_state_classifier.pth"
)


class EyeStateClassifier:
    """Classify eye crops as open or closed using a trained CNN.

    Falls back gracefully when the model file is missing or PyTorch is not
    installed, returning ``None`` from predict methods so callers can use EAR
    as a fallback.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the saved ``.pth`` checkpoint.  Defaults to
        ``blinkcounter/models/eye_state_classifier.pth``.
    threshold : float
        Probability threshold above which an eye is considered closed.
    device : str, optional
        PyTorch device string (``"cpu"``, ``"mps"``, ``"cuda"``).
        Auto-detected if not provided.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self._model = None
        self._device = None
        self._transform = None

        resolved_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            logger.warning(
                "Eye-state model not found at %s. "
                "CNN predictions will be unavailable; falling back to EAR.",
                resolved_path,
            )
            return

        try:
            import torch
            from torchvision import transforms

            # Determine device
            if device:
                self._device = torch.device(device)
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            # Load checkpoint
            checkpoint = torch.load(resolved_path, map_location=self._device, weights_only=True)

            # Reconstruct model
            model = _build_eye_state_cnn()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self._device)
            model.eval()
            self._model = model

            # Build transform matching training normalisation
            mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
            std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            logger.info(
                "Loaded eye-state CNN (val_acc=%.4f, epoch=%d) on %s",
                checkpoint.get("val_accuracy", -1),
                checkpoint.get("epoch", -1),
                self._device,
            )

        except ImportError:
            logger.warning(
                "PyTorch not installed. CNN eye-state classifier unavailable."
            )
        except Exception:
            logger.exception("Failed to load eye-state classifier model")

    @property
    def is_available(self) -> bool:
        """Return True if the model loaded successfully."""
        return self._model is not None

    def predict(self, eye_crop: np.ndarray) -> Optional[float]:
        """Return the probability that the eye is closed.

        Parameters
        ----------
        eye_crop : np.ndarray
            BGR eye crop from dlib/mediapipe landmarks (any size).

        Returns
        -------
        float or None
            Probability in [0, 1] where 1.0 = certainly closed.
            Returns ``None`` if the model is not available.
        """
        if self._model is None:
            return None

        import torch

        # Preprocess: BGR -> RGB, resize to 64x64
        rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
        tensor = self._transform(resized).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logit = self._model(tensor).squeeze()
            prob = torch.sigmoid(logit).item()

        return prob

    def predict_batch(self, eye_crops: list[np.ndarray]) -> list[Optional[float]]:
        """Return closed-eye probabilities for a batch of eye crops.

        Parameters
        ----------
        eye_crops : list of np.ndarray
            List of BGR eye crops.

        Returns
        -------
        list of float or None
            Per-crop probabilities, or a list of ``None`` if the model is
            unavailable.
        """
        if self._model is None:
            return [None] * len(eye_crops)

        if not eye_crops:
            return []

        import torch

        tensors = []
        for crop in eye_crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
            tensors.append(self._transform(resized))

        batch = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            logits = self._model(batch).squeeze(1)
            probs = torch.sigmoid(logits)

        return probs.cpu().tolist()

    def is_closed(self, eye_crop: np.ndarray) -> Optional[bool]:
        """Convenience: return True if the eye is classified as closed.

        Returns ``None`` if the model is not available.
        """
        prob = self.predict(eye_crop)
        if prob is None:
            return None
        return prob >= self.threshold
