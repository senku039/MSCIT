"""Validation helpers for payload and upload security."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from werkzeug.datastructures import FileStorage


@dataclass
class ValidationResult:
    """Container for validation success/failure."""

    ok: bool
    message: str = ""


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
