"""Centralized and safe model loading/prediction operations."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

LOGGER = logging.getLogger(__name__)


class ModelService:
    """Loads and serves all ML model predictions."""

    def __init__(self, config: Mapping[str, Any] | Any):
        self.config = config
        self.dyslexia_model = None
        self.handwriting_model = None

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Read config value from flask Config(dict-like) or object attributes."""
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @staticmethod
    def _assert_local_file(path: str) -> Path:
        candidate = Path(path).resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Model file not found: {candidate}")
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_models(self) -> None:
        """Load models once at startup with per-model isolation.

        If one model fails (e.g., missing sklearn for pickled estimator),
        the other model can still load and serve requests.
        """
        loaded_any = False

        dyslexia_path = self._cfg("DYSLEXIA_MODEL_PATH")
        if dyslexia_path:
            try:
                dyslexia_file = self._assert_local_file(dyslexia_path)
                LOGGER.info("Loading dyslexia model from %s", dyslexia_file)
                self.dyslexia_model = joblib.load(dyslexia_file)
                if not hasattr(self.dyslexia_model, "predict"):
                    raise TypeError("Loaded dyslexia model does not expose a predict method.")
                loaded_any = True
                LOGGER.info("Dyslexia model hash=%s", self._sha256(dyslexia_file))
            except Exception:
                self.dyslexia_model = None
                LOGGER.exception("Failed to load dyslexia model. Continuing with partial availability.")

        handwriting_path = self._cfg("HANDWRITING_MODEL_PATH")
        if handwriting_path:
            try:
                handwriting_file = self._assert_local_file(handwriting_path)
                LOGGER.info("Loading handwriting model from %s", handwriting_file)
                self.handwriting_model = tf.keras.models.load_model(handwriting_file, compile=False)
                loaded_any = True
                LOGGER.info("Handwriting model hash=%s", self._sha256(handwriting_file))
            except Exception:
                self.handwriting_model = None
                LOGGER.exception("Failed to load handwriting model. Continuing with partial availability.")

        if not loaded_any:
            raise RuntimeError("No ML models could be loaded at startup.")

    def predict_dyslexia(self, feature_values: list[float]) -> float:
        if self.dyslexia_model is None:
            raise RuntimeError("Dyslexia model is unavailable.")

        input_array = np.asarray([feature_values], dtype=np.float32)
        prediction = self.dyslexia_model.predict(input_array)

        if isinstance(prediction, np.ndarray):
            if prediction.size == 0:
                raise RuntimeError("Dyslexia model returned an empty prediction.")
            return float(prediction.reshape(-1)[0])

        return float(prediction)

    @staticmethod
    def _normalize_01(image_tensor: np.ndarray) -> np.ndarray:
        clipped = np.clip(image_tensor, 0.0, 1.0)
        return clipped.astype(np.float32)

    def _build_handwriting_variants(self, image_tensor: np.ndarray) -> np.ndarray:
        """Create lightweight test-time variants to stabilize small-model output."""
        # image_tensor is expected in [0, 1], shape (H, W, C)
        mean = float(np.mean(image_tensor))
        std = float(np.std(image_tensor))

        # Contrast-normalized variant (prevents low-contrast input collapse).
        contrast = (image_tensor - mean) / max(std, 1e-5)
        contrast = ((contrast * 0.22) + 0.5)

        # Grayscale emphasis variant; repeated across channels to keep shape stable.
        gray = np.mean(image_tensor, axis=2, keepdims=True)
        gray_rgb = np.repeat(gray, image_tensor.shape[2], axis=2)

        # Mild sharpen-like enhancement using center emphasis without external deps.
        sharpen = np.clip((1.25 * image_tensor) - (0.25 * gray_rgb), 0.0, 1.0)

        variants = [
            self._normalize_01(image_tensor),
            self._normalize_01(contrast),
            self._normalize_01(gray_rgb),
            self._normalize_01(sharpen),
        ]
        return np.stack(variants, axis=0)

    def predict_handwriting(self, image_bytes: bytes) -> tuple[float, str, dict[str, Any]]:
        if self.handwriting_model is None:
            raise RuntimeError("Handwriting model is unavailable.")

        image_size = self._cfg("HANDWRITING_IMAGE_SIZE", (128, 128))
        threshold = float(self._cfg("HANDWRITING_THRESHOLD", 0.5))

        image_tensor = keras_image.img_to_array(
            keras_image.load_img(
                BytesIO(image_bytes),
                target_size=image_size,
            )
        )
        image_tensor = image_tensor.astype(np.float32) / 255.0
        image_batch = self._build_handwriting_variants(image_tensor)

        expected_shape = self.handwriting_model.input_shape
        if len(expected_shape) == 4 and tuple(expected_shape[1:3]) != tuple(image_size):
            raise RuntimeError("Configured image size does not match model input shape.")

        prediction = self.handwriting_model.predict(image_batch, verbose=0)
        scores = np.clip(prediction.reshape(-1), 0.0, 1.0).astype(np.float32)
        probability = float(np.mean(scores))
        uncertainty = float(np.std(scores))

        score_means_dyslexic = bool(self._cfg("HANDWRITING_SCORE_MEANS_DYSLEXIC", True))
        if score_means_dyslexic:
            label = "Dyslexic" if probability >= threshold else "Non_Dyslexic"
        else:
            label = "Non_Dyslexic" if probability >= threshold else "Dyslexic"

        quality_score = float(np.clip((float(np.std(image_tensor)) / 0.22), 0.0, 1.0))
        confidence = float(np.clip(1.0 - (uncertainty * 2.2), 0.0, 1.0))
        details = {
            "ensemble_samples": int(scores.shape[0]),
            "score_spread": round(uncertainty, 4),
            "confidence": round(confidence, 4),
            "image_quality_score": round(quality_score, 4),
            "decision_threshold": threshold,
        }

        return probability, label, details
