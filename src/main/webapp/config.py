"""Application configuration for local and production environments."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class BaseConfig:
    """Default secure configuration shared by all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "replace-in-production")
    DEBUG = False
    TESTING = False

    JSON_SORT_KEYS = False
    JSON_AS_ASCII = False
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 2 * 1024 * 1024))  # 2 MB

    DYSLEXIA_MODEL_PATH = os.getenv(
        "DYSLEXIA_MODEL_PATH",
        str(BASE_DIR / "dyslexia_reg_model.pkl"),
    )
    HANDWRITING_MODEL_PATH = os.getenv(
        "HANDWRITING_MODEL_PATH",
        str(BASE_DIR / "final_model.keras"),
    )

    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
    ALLOWED_IMAGE_MIME_TYPES = {
        "image/png",
        "image/jpeg",
        "image/bmp",
    }

    HANDWRITING_IMAGE_SIZE = (128, 128)
    HANDWRITING_THRESHOLD = float(os.getenv("HANDWRITING_THRESHOLD", "0.5"))
    # Most binary handwriting models output dyslexic-class probability.
    # Set to false only if your model output is non-dyslexic probability.
    HANDWRITING_SCORE_MEANS_DYSLEXIC = os.getenv("HANDWRITING_SCORE_MEANS_DYSLEXIC", "true").lower() == "true"

    EXPECTED_FEATURES = [
        "Reading_Speed",
        "Spelling_Accuracy",
        "Writing_Errors",
        "Cognitive_Score",
        "Phonemic_Awareness_Errors",
        "Attention_Span",
        "Response_Time",
    ]

    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]

    API_TOKENS = {token.strip() for token in os.getenv("API_TOKENS", "").split(",") if token.strip()}

    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    REDIS_URL = os.getenv("REDIS_URL", "")
    USER_DB_PATH = os.getenv("USER_DB_PATH", str(BASE_DIR / "users.db"))


class DevelopmentConfig(BaseConfig):
    """Developer-friendly configuration."""

    DEBUG = True


class ProductionConfig(BaseConfig):
    """Production-ready secure configuration."""

    DEBUG = False


CONFIG_MAPPING = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": ProductionConfig,
}
