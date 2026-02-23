"""Centralized and safe model loading/prediction operations."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

LOGGER = logging.getLogger(__name__)


class ModelService:
    """Loads and serves all ML model predictions."""

    def __init__(self, config: Any):
        self.config = config
        self.dyslexia_model = None
        self.handwriting_model = None

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
        """Load models once at startup with sanity checks."""
        dyslexia_path = self._assert_local_file(self.config.DYSLEXIA_MODEL_PATH)
        handwriting_path = self._assert_local_file(self.config.HANDWRITING_MODEL_PATH)

        LOGGER.info("Loading dyslexia model from %s", dyslexia_path)
        self.dyslexia_model = joblib.load(dyslexia_path)
        if not hasattr(self.dyslexia_model, "predict"):
            raise TypeError("Loaded dyslexia model does not expose a predict method.")

        LOGGER.info("Loading handwriting model from %s", handwriting_path)
        self.handwriting_model = tf.keras.models.load_model(handwriting_path, compile=False)

        LOGGER.info(
            "Model hashes: dyslexia=%s handwriting=%s",
            self._sha256(dyslexia_path),
            self._sha256(handwriting_path),
        )

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

    def predict_handwriting(self, image_bytes: bytes) -> tuple[float, str]:
        if self.handwriting_model is None:
            raise RuntimeError("Handwriting model is unavailable.")

        image_tensor = keras_image.img_to_array(
            keras_image.load_img(
                BytesIO(image_bytes),
                target_size=self.config.HANDWRITING_IMAGE_SIZE,
            )
        )
        image_tensor = image_tensor.astype(np.float32) / 255.0
        image_tensor = np.expand_dims(image_tensor, axis=0)

        expected_shape = self.handwriting_model.input_shape
        if len(expected_shape) == 4 and tuple(expected_shape[1:3]) != tuple(self.config.HANDWRITING_IMAGE_SIZE):
            raise RuntimeError("Configured image size does not match model input shape.")

        prediction = self.handwriting_model.predict(image_tensor, verbose=0)
        probability = float(np.clip(prediction.reshape(-1)[0], 0.0, 1.0))

        label = "Non_Dyslexic" if probability >= self.config.HANDWRITING_THRESHOLD else "Dyslexic"
        return probability, label
