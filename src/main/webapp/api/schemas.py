"""Lightweight schema contracts for API requests and responses."""

from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    pass


PREDICT_FIELDS = [
    "Reading_Speed",
    "Spelling_Accuracy",
    "Writing_Errors",
    "Cognitive_Score",
    "Phonemic_Awareness_Errors",
    "Attention_Span",
    "Response_Time",
]


def parse_predict_request(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise SchemaValidationError("Request body must be a JSON object.")

    missing = [field for field in PREDICT_FIELDS if field not in payload]
    if missing:
        raise SchemaValidationError(f"Missing features: {missing}")

    unexpected = [field for field in payload if field not in PREDICT_FIELDS]
    if unexpected:
        raise SchemaValidationError(f"Unexpected features: {unexpected}")

    casted: dict[str, float] = {}
    for field in PREDICT_FIELDS:
        value = payload[field]
        if isinstance(value, bool):
            raise SchemaValidationError(f"Feature '{field}' must be numeric, boolean is not allowed.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise SchemaValidationError(f"Feature '{field}' must be numeric.") from error
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            raise SchemaValidationError(f"Feature '{field}' must be a finite number.")
        casted[field] = numeric
    return casted


def validate_predict_response(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "prediction",
        "probability_percent",
        "model_probability",
        "model_probability_percent",
        "risk_level",
        "feature_analysis",
        "abnormal_feature_count",
        "observations",
        "summary",
        "recommendations",
        "result_redirect",
    }
    missing = required - set(payload.keys())
    if missing:
        raise SchemaValidationError(f"Invalid predict response; missing keys: {sorted(missing)}")
    return payload


def validate_handwriting_response(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "predicted_probability",
        "probability_percent",
        "predicted_class",
        "risk_level",
        "summary",
        "recommendations",
        "result_redirect",
    }
    missing = required - set(payload.keys())
    if missing:
        raise SchemaValidationError(f"Invalid handwriting response; missing keys: {sorted(missing)}")
    return payload


def validate_ocr_response(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "extracted_text",
        "corrected_text",
        "simplified_text",
        "summary",
        "ocr_quality_score",
        "noise_characters",
        "underscore_artifacts",
        "observations",
        "recommendations",
        "original_text",
        "result_redirect",
    }
    missing = required - set(payload.keys())
    if missing:
        raise SchemaValidationError(f"Invalid OCR response; missing keys: {sorted(missing)}")
    return payload


def validate_image_analysis_response(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"overall_summary", "handwriting", "ocr", "result_redirect"}
    missing = required - set(payload.keys())
    if missing:
        raise SchemaValidationError(f"Invalid image-analysis response; missing keys: {sorted(missing)}")
    if not isinstance(payload["handwriting"], dict) or not isinstance(payload["ocr"], dict):
        raise SchemaValidationError("Image-analysis response sections must be objects.")
    return payload
