"""Validation helpers for payload and upload security."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage


@dataclass
class ValidationResult:
    """Container for validation success/failure."""

    ok: bool
    message: str = ""


def validate_json_payload(payload: Any) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult(False, "Request body must be a JSON object.")
    return ValidationResult(True)


def validate_feature_payload(payload: dict[str, Any], expected_features: list[str]) -> ValidationResult:
    missing = [feature for feature in expected_features if feature not in payload]
    if missing:
        return ValidationResult(False, f"Missing features: {missing}")

    unexpected = [feature for feature in payload if feature not in expected_features]
    if unexpected:
        return ValidationResult(False, f"Unexpected features: {unexpected}")

    return ValidationResult(True)


def cast_numeric_features(payload: dict[str, Any], expected_features: list[str]) -> list[float]:
    values: list[float] = []
    for feature in expected_features:
        value = payload[feature]
        if isinstance(value, bool):
            raise ValueError(f"Feature '{feature}' must be numeric, boolean is not allowed.")
        numeric_value = float(value)
        if numeric_value != numeric_value or numeric_value in (float("inf"), float("-inf")):
            raise ValueError(f"Feature '{feature}' must be a finite number.")
        values.append(numeric_value)
    return values


def allowed_extension(filename: str, allowed_extensions: set[str]) -> bool:
    extension = Path(filename).suffix.lower().lstrip(".")
    return extension in allowed_extensions


def validate_image_upload(
    image_file: FileStorage,
    allowed_extensions: set[str],
    allowed_mime_types: set[str],
) -> ValidationResult:
    if image_file.filename is None or image_file.filename.strip() == "":
        return ValidationResult(False, "Uploaded file must have a filename.")

    if not allowed_extension(image_file.filename, allowed_extensions):
        return ValidationResult(False, "Unsupported file extension.")

    if image_file.mimetype not in allowed_mime_types:
        return ValidationResult(False, "Unsupported MIME type.")

    return ValidationResult(True)
