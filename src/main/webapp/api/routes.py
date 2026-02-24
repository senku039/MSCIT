"""API routes for prediction and handwriting analysis."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable

import cv2
import numpy as np
import pytesseract
from flask import Blueprint, current_app, jsonify, request

from src.main.webapp.utils.validators import (
    cast_numeric_features,
    validate_feature_payload,
    validate_image_upload,
    validate_json_payload,
)

LOGGER = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _get_client_id() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        if request.method == "OPTIONS":
            return current_app.make_default_options_response()

        tokens = current_app.config.get("API_TOKENS", set())
        if not tokens:
            return func(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token not in tokens:
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)

    return wrapper


def rate_limited(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        if request.method == "OPTIONS":
            return current_app.make_default_options_response()

        client_id = _get_client_id()
        now = time.time()
        interval = 60
        limit = int(current_app.config.get("RATE_LIMIT_PER_MINUTE", 60))
        bucket = _RATE_BUCKETS[client_id]

        while bucket and now - bucket[0] > interval:
            bucket.popleft()

        if len(bucket) >= limit:
            return jsonify({"error": "Rate limit exceeded"}), 429

        bucket.append(now)
        return func(*args, **kwargs)

    return wrapper


@api_bp.route("/", methods=["GET"])
def health() -> Any:
    return jsonify({"message": "Dyslexia Prediction API is running"})


@api_bp.route("/predict", methods=["POST"])
@require_auth
@rate_limited
def predict() -> Any:
    payload = request.get_json(silent=True)
    payload_validation = validate_json_payload(payload)
    if not payload_validation.ok:
        return jsonify({"error": payload_validation.message}), 400

    expected_features = current_app.config["EXPECTED_FEATURES"]
    schema_validation = validate_feature_payload(payload, expected_features)
    if not schema_validation.ok:
        return jsonify({"error": schema_validation.message}), 400

    try:
        feature_values = cast_numeric_features(payload, expected_features)
        model_service = current_app.extensions["model_service"]
        prediction = model_service.predict_dyslexia(feature_values)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        LOGGER.exception("Unhandled dyslexia prediction failure")
        return jsonify({"error": "Prediction service unavailable."}), 500

    if not np.isfinite(prediction):
        LOGGER.warning("Model returned non-finite score; returning safe fallback.")
        return jsonify({"prediction": None, "risk_level": "indeterminate"}), 200

    risk_level = "high_risk" if prediction >= 0.5 else "low_risk"
    return jsonify({"prediction": prediction, "risk_level": risk_level}), 200


@api_bp.route("/handwriting-analysis", methods=["POST"])
@require_auth
@rate_limited
def handwriting_analysis() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    validation = validate_image_upload(
        image_file=image_file,
        allowed_extensions=current_app.config["ALLOWED_IMAGE_EXTENSIONS"],
        allowed_mime_types=current_app.config["ALLOWED_IMAGE_MIME_TYPES"],
    )

    if not validation.ok:
        return jsonify({"error": validation.message}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        model_service = current_app.extensions["model_service"]
        predicted_prob, predicted_class = model_service.predict_handwriting(image_bytes)
    except Exception:
        LOGGER.exception("Unhandled handwriting prediction failure")
        return jsonify({"error": "Handwriting analysis unavailable."}), 500

    return jsonify(
        {
            "predicted_probability": predicted_prob,
            "predicted_class": predicted_class,
            "threshold": current_app.config["HANDWRITING_THRESHOLD"],
        }
    )


@api_bp.route("/upload", methods=["POST"])
@require_auth
@rate_limited
def upload_ocr() -> Any:
    """Extract text from uploaded image and return normalized variants."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file_obj = request.files["file"]
    if not file_obj.filename:
        return jsonify({"error": "No file provided."}), 400

    try:
        file_bytes = np.frombuffer(file_obj.read(), dtype=np.uint8)
        if file_bytes.size == 0:
            return jsonify({"error": "No file provided."}), 400

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file."}), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        extracted_text = pytesseract.image_to_string(denoised).strip()
        corrected_text = " ".join(extracted_text.split())
        simplified_text = corrected_text.lower()

        return jsonify(
            {
                "extracted_text": extracted_text,
                "corrected_text": corrected_text,
                "simplified_text": simplified_text,
            }
        ), 200
    except Exception:
        LOGGER.exception("Unhandled OCR upload failure")
        return jsonify({"error": "OCR processing unavailable."}), 500
