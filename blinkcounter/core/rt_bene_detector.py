"""RT-BENE pre-trained blink detection model integration.

Uses the pre-trained VGG16 blink estimation model from RT-BENE
(Real-Time Blink Estimation in Natural Environments) by Cortacero et al.

The model takes separate left and right eye crops (resized to 60x36),
passes them through twin VGG16 feature extractors, concatenates features,
and outputs a blink probability via sigmoid.

Reference:
    Cortacero, K., Fischer, T., & Demiris, Y. (2019).
    RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation
    in Natural Environments. ICCV Workshops.

License: The RT-BENE model weights are CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model download URLs and checksums from rt_gene/download_tools.py
_MODEL_INFO = {
    "vgg16_1": {
        "url": "https://imperialcollegelondon.box.com/shared/static/wwky1um443vgz9oy90zllv0s7474a5dj.model",
        "md5": "cde99055e3b6dcf9fae6b78191c0fd9b",
        "filename": "rt_bene_vgg16_allsubjects1.model",
    },
    "vgg16_2": {
        "url": "https://imperialcollegelondon.box.com/shared/static/psha8bclz9bv5yd87qetajgovioc03vb3.model",
        "md5": "67339ceefcfec4b3b8b3d7ccb03fadfa",
        "filename": "rt_bene_vgg16_allsubjects2.model",
    },
    "resnet18_1": {
        "url": "https://imperialcollegelondon.box.com/shared/static/p8jmekxhw4k8xtbz6vph3924g6ywnbre.model",
        "md5": "7c228fe7b95ce5960c4c5cae8f2d3a09",
        "filename": "rt_bene_resnet18_allsubjects1.model",
    },
}

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Input size expected by RT-BENE models
_EYE_INPUT_SIZE = (60, 36)  # width, height

# ImageNet normalization used during RT-BENE training
_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD = [0.229, 0.224, 0.225]


def _md5(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_model(model_key: str) -> Optional[Path]:
    """Download a single RT-BENE model if not already present.

    Returns the path to the model file, or None on failure.
    """
    info = _MODEL_INFO[model_key]
    dest = _MODELS_DIR / info["filename"]

    if dest.exists():
        existing_hash = _md5(dest)
        if existing_hash == info["md5"]:
            return dest
        logger.warning(
            "MD5 mismatch for %s (got %s, expected %s). Re-downloading.",
            dest.name, existing_hash, info["md5"],
        )

    try:
        import requests
        logger.info("Downloading RT-BENE model: %s", info["filename"])
        response = requests.get(info["url"], timeout=60, stream=True)
        response.raise_for_status()
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        downloaded_hash = _md5(dest)
        if downloaded_hash != info["md5"]:
            logger.error(
                "MD5 mismatch after download for %s (got %s, expected %s)",
                dest.name, downloaded_hash, info["md5"],
            )
            dest.unlink(missing_ok=True)
            return None
        logger.info("Downloaded %s successfully.", info["filename"])
        return dest
    except Exception:
        logger.exception("Failed to download RT-BENE model: %s", info["filename"])
        return None


def _build_vgg16_blink_model():
    """Construct the RT-BENE VGG16 blink estimation model.

    Architecture matches rt_bene/blink_estimation_models_pytorch.py.
    Twin VGG16 feature extractors (one per eye), concatenated features
    fed through FC layers -> single output (logit).
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    class BlinkEstimationModelVGG16(nn.Module):
        def __init__(self, num_out=1):
            super().__init__()
            _left_model = models.vgg16(weights=None)
            _right_model = models.vgg16(weights=None)

            _left_modules = [module for module in _left_model.features]
            _left_modules.append(_left_model.avgpool)
            self.left_features = nn.Sequential(*_left_modules)

            _right_modules = [module for module in _right_model.features]
            _right_modules.append(_right_model.avgpool)
            self.right_features = nn.Sequential(*_right_modules)

            in_features = (
                _left_model.classifier[0].in_features
                + _right_model.classifier[0].in_features
            )
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.6),
                nn.Linear(512, num_out),
            )

        def forward(self, left_eye, right_eye):
            left_x = self.left_features(left_eye)
            left_x = torch.flatten(left_x, 1)
            right_x = self.right_features(right_eye)
            right_x = torch.flatten(right_x, 1)
            eyes_x = torch.cat((left_x, right_x), dim=1)
            return self.fc(eyes_x)

    return BlinkEstimationModelVGG16()


def _build_resnet18_blink_model():
    """Construct the RT-BENE ResNet18 blink estimation model."""
    import torch
    import torch.nn as nn
    from torchvision import models

    class BlinkEstimationModelResnet18(nn.Module):
        def __init__(self, num_out=1):
            super().__init__()
            _left_model = models.resnet18(weights=None)
            _right_model = models.resnet18(weights=None)

            self.left_features = nn.Sequential(
                _left_model.conv1, _left_model.bn1, _left_model.relu,
                _left_model.maxpool, _left_model.layer1, _left_model.layer2,
                _left_model.layer3, _left_model.layer4, _left_model.avgpool,
            )
            self.right_features = nn.Sequential(
                _right_model.conv1, _right_model.bn1, _right_model.relu,
                _right_model.maxpool, _right_model.layer1, _right_model.layer2,
                _right_model.layer3, _right_model.layer4, _right_model.avgpool,
            )

            in_features = (
                _left_model.fc.in_features + _right_model.fc.in_features
            )
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.6),
                nn.Linear(512, num_out),
            )

        def forward(self, left_eye, right_eye):
            import torch
            left_x = self.left_features(left_eye)
            left_x = torch.flatten(left_x, 1)
            right_x = self.right_features(right_eye)
            right_x = torch.flatten(right_x, 1)
            eyes_x = torch.cat((left_x, right_x), dim=1)
            return self.fc(eyes_x)

    return BlinkEstimationModelResnet18()


_MODEL_BUILDERS = {
    "vgg16": _build_vgg16_blink_model,
    "resnet18": _build_resnet18_blink_model,
}


class RTBeneDetector:
    """RT-BENE pre-trained blink detection model.

    Wraps the RT-BENE PyTorch blink estimation models for use in
    BlinkCounter. Handles model download, loading, and inference.

    The model takes separate left and right eye crops and returns
    the probability that the person is blinking (eyes closed).

    Parameters
    ----------
    model_type : str
        Architecture to use: ``"vgg16"`` (default) or ``"resnet18"``.
    model_key : str, optional
        Which pre-trained checkpoint to load. Defaults to the first
        available model for the chosen architecture.
    device : str, optional
        PyTorch device string. Auto-detected if not provided.
    threshold : float
        Probability threshold for classifying a blink. Default 0.5.
    """

    def __init__(
        self,
        model_type: str = "vgg16",
        model_key: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.model_type = model_type
        self._model = None
        self._device = None
        self._transform = None

        if model_key is None:
            model_key = f"{model_type}_1"

        if model_key not in _MODEL_INFO:
            logger.error(
                "Unknown RT-BENE model key: %s. Available: %s",
                model_key, list(_MODEL_INFO.keys()),
            )
            return

        model_path = _download_model(model_key)
        if model_path is None:
            logger.warning(
                "RT-BENE model not available. Predictions will return None."
            )
            return

        try:
            import torch
            from torchvision import transforms

            # Determine device
            if device:
                self._device = torch.device(device)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            # Build and load model
            if model_type not in _MODEL_BUILDERS:
                logger.error("Unknown model type: %s", model_type)
                return

            model = _MODEL_BUILDERS[model_type]()
            state_dict = torch.load(
                model_path, map_location=self._device, weights_only=True
            )
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            self._model = model

            # Transform matching RT-BENE training pipeline:
            # resize to 60x36 (done in predict), then normalize
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
            ])

            logger.info(
                "Loaded RT-BENE %s model on %s (threshold=%.2f)",
                model_type, self._device, threshold,
            )

        except ImportError:
            logger.warning("PyTorch not installed. RT-BENE detector unavailable.")
        except Exception:
            logger.exception("Failed to load RT-BENE model")

    @property
    def is_available(self) -> bool:
        """Return True if the model loaded successfully."""
        return self._model is not None

    def _preprocess_eye(self, eye_crop: np.ndarray) -> "torch.Tensor":
        """Preprocess an eye crop for model input.

        Args:
            eye_crop: BGR eye crop of any size.

        Returns:
            Preprocessed tensor ready for model input.
        """
        # BGR -> RGB (RT-BENE uses cv2.imread which is BGR, same as our input)
        rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        # Resize to 60x36 as per RT-BENE's transform
        resized = cv2.resize(
            rgb, _EYE_INPUT_SIZE, interpolation=cv2.INTER_CUBIC
        )
        return self._transform(resized)

    def predict_blink(
        self,
        left_eye_crop: np.ndarray,
        right_eye_crop: np.ndarray,
    ) -> Optional[float]:
        """Return the probability that the person is blinking.

        The RT-BENE model requires both left and right eye crops
        for its twin-branch architecture.

        Parameters
        ----------
        left_eye_crop : np.ndarray
            BGR crop of the left eye.
        right_eye_crop : np.ndarray
            BGR crop of the right eye.

        Returns
        -------
        float or None
            Probability in [0.0, 1.0] where 1.0 = certainly blinking.
            Returns None if the model is not available.
        """
        if self._model is None:
            return None

        import torch

        left_tensor = self._preprocess_eye(left_eye_crop).unsqueeze(0).to(self._device)
        right_tensor = self._preprocess_eye(right_eye_crop).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logit = self._model(left_tensor, right_tensor).squeeze()
            prob = torch.sigmoid(logit).item()

        return prob

    def predict_blink_batch(
        self,
        left_eye_crops: list[np.ndarray],
        right_eye_crops: list[np.ndarray],
    ) -> list[Optional[float]]:
        """Return blink probabilities for a batch of eye crop pairs.

        Parameters
        ----------
        left_eye_crops : list of np.ndarray
            BGR crops of left eyes.
        right_eye_crops : list of np.ndarray
            BGR crops of right eyes.

        Returns
        -------
        list of float or None
            Per-pair blink probabilities.
        """
        if self._model is None:
            return [None] * len(left_eye_crops)

        if not left_eye_crops:
            return []

        import torch

        left_tensors = [self._preprocess_eye(c) for c in left_eye_crops]
        right_tensors = [self._preprocess_eye(c) for c in right_eye_crops]

        left_batch = torch.stack(left_tensors).to(self._device)
        right_batch = torch.stack(right_tensors).to(self._device)

        with torch.no_grad():
            logits = self._model(left_batch, right_batch).squeeze(1)
            probs = torch.sigmoid(logits)

        return probs.cpu().tolist()

    def is_blinking(
        self,
        left_eye_crop: np.ndarray,
        right_eye_crop: np.ndarray,
    ) -> Optional[bool]:
        """Convenience: return True if classified as blinking.

        Returns None if the model is not available.
        """
        prob = self.predict_blink(left_eye_crop, right_eye_crop)
        if prob is None:
            return None
        return prob >= self.threshold
