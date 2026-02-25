"""API routes for prediction, handwriting analysis, OCR, and result views."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable

try:
    import cv2
except Exception:  # pragma: no cover - optional native dependency
    cv2 = None
import numpy as np
import pytesseract
from PIL import Image
from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory

from src.main.webapp.api.schemas import (
    SchemaValidationError,
    parse_predict_request,
    validate_handwriting_response,
    validate_image_analysis_response,
    validate_ocr_response,
    validate_predict_response,
)
from src.main.webapp.utils.validators import validate_image_upload

LOGGER = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)
_ALLOWED_PAGE_FILES = {p.name for p in (Path(__file__).resolve().parent.parent).glob("*.html")}
_ALLOWED_ASSET_EXTENSIONS = {".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"}

_FEATURE_RULES: dict[str, dict[str, Any]] = {
    "Attention_Span": {"low": 40, "high": 100, "direction": "higher_better", "display": "Attention Span"},
    "Cognitive_Score": {"low": 40, "high": 100, "direction": "higher_better", "display": "Cognitive Score"},
    "Reading_Speed": {"low": 70, "high": 220, "direction": "higher_better", "display": "Reading Speed"},
    "Spelling_Accuracy": {"low": 60, "high": 100, "direction": "higher_better", "display": "Spelling Accuracy"},
    "Writing_Errors": {"low": 0, "high": 5, "direction": "lower_better", "display": "Writing Errors"},
    "Phonemic_Awareness_Errors": {
        "low": 0,
        "high": 5,
        "direction": "lower_better",
        "display": "Phonemic Awareness Errors",
    },
    "Response_Time": {"low": 0.2, "high": 2.0, "direction": "lower_better", "display": "Response Time"},
}


def _clean_ocr_text(raw_text: str) -> str:
    """Normalize OCR output with conservative cleanup."""
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    joined = "\n".join(lines)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = joined.replace("|", "I")
    return joined.strip()


def _correct_ocr_text(text: str) -> str:
    """Apply lightweight OCR corrections without external NLP dependencies."""
    if not text:
        return ""

    corrected = text
    for old, new in {"_": " ", "|": "I", "0": "o"}.items():
        corrected = corrected.replace(old, new)

    corrected = re.sub(r"[^\x20-\x7E\n]", " ", corrected)
    corrected = re.sub(r"\s+", " ", corrected).strip()

    token_fixes = {
        "w0n't": "won't",
        "y0u": "you",
        "th1s": "this",
        "1": "I",
    }
    tokens = []
    for token in corrected.split(" "):
        lowered = token.lower()
        fixed = token_fixes.get(lowered, token)
        if token[:1].isupper() and fixed and fixed.islower():
            fixed = fixed.capitalize()
        tokens.append(fixed)

    corrected = " ".join(tokens)
    corrected = re.sub(r"\s+([,.;:!?])", r"\1", corrected)
    return corrected


def _simplify_ocr_text(text: str) -> str:
    """Build a simplified text view with reduced punctuation/noise."""
    if not text:
        return ""

    simplified = text.lower()
    simplified = re.sub(r"[^a-z0-9\s]", " ", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()

    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "is", "are",
        "was", "were", "be", "as", "at", "it", "that", "this",
    }
    filtered = [w for w in simplified.split(" ") if w and w not in stop_words]
    return " ".join(filtered)


def _ocr_quality_metrics(raw_text: str, corrected_text: str) -> dict[str, Any]:
    """Generate simple quality indicators for OCR result interpretation."""
    raw_chars = len(raw_text)
    noise_chars = len(re.findall(r"[^\x20-\x7E\n]", raw_text))
    underscore_count = raw_text.count("_")
    correction_delta = abs(len(corrected_text) - len(raw_text))

    quality_score = 100.0
    if raw_chars:
        quality_score -= (noise_chars / raw_chars) * 55
        quality_score -= (underscore_count / raw_chars) * 35
        quality_score -= min(correction_delta / raw_chars, 1.0) * 10
    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    return {
        "quality_score": round(quality_score, 2),
        "noise_characters": noise_chars,
        "underscore_artifacts": underscore_count,
    }


def _classify_risk(probability: float) -> str:
    if probability < 0.33:
        return "Low Risk"
    if probability < 0.66:
        return "Moderate Risk"
    return "High Risk"


def _build_feature_analysis(feature_map: dict[str, float]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    abnormal_observations: list[str] = []

    for key, value in feature_map.items():
        rule = _FEATURE_RULES.get(key)
        if not rule:
            continue

        low = rule["low"]
        high = rule["high"]
        direction = rule["direction"]
        display = rule["display"]

        if direction == "higher_better":
            is_abnormal = value < low
            impact = "Increases risk" if is_abnormal else "Supports lower risk"
            status = "Below expected" if is_abnormal else "Within expected"
        else:
            is_abnormal = value > high
            impact = "Increases risk" if is_abnormal else "Within expected"
            status = "Above expected" if is_abnormal else "Within expected"

        rows.append(
            {
                "feature": display,
                "value": round(value, 3),
                "expected_range": f"{low} - {high}",
                "status": status,
                "impact": impact,
                "abnormal": is_abnormal,
            }
        )

        if is_abnormal:
            abnormal_observations.append(f"{display} is {status.lower()} ({value:.3f}).")

    return rows, abnormal_observations


def _build_prediction_payload(prediction: float, feature_map: dict[str, float]) -> dict[str, Any]:
    model_probability = float(np.clip(prediction, 0.0, 1.0))

    table_rows, abnormal_observations = _build_feature_analysis(feature_map)
    abnormal_count = sum(1 for row in table_rows if row["abnormal"])
    if table_rows:
        severity_sum = 0.0
        for row in table_rows:
            if row["abnormal"]:
                severity_sum += 1.0
            elif "supports" in row["impact"].lower() or "within" in row["impact"].lower():
                severity_sum += 0.15
        feature_signal = float(np.clip(severity_sum / len(table_rows), 0.0, 1.0))
    else:
        feature_signal = 0.0

    screening_probability = float(np.clip((0.8 * model_probability) + (0.2 * feature_signal), 0.0, 1.0))

    probability_percent = round(screening_probability * 100, 2)
    model_probability_percent = round(model_probability * 100, 2)
    risk_level = _classify_risk(screening_probability)

    if risk_level == "High Risk":
        summary = "Screening suggests high dyslexia risk. Multiple indicators need specialist review."
    elif risk_level == "Moderate Risk":
        summary = "Screening suggests moderate dyslexia risk. A follow-up assessment is recommended."
    else:
        summary = "Screening suggests low dyslexia risk. Continue monitoring learning progress."

    if abnormal_count:
        summary += f" ({abnormal_count} feature{'s' if abnormal_count != 1 else ''} flagged outside expected range.)"

    recommendations = [
        "Use this as a screening signal, not a clinical diagnosis.",
        "Review flagged features with an educator/specialist.",
        "If moderate/high risk, schedule a formal dyslexia assessment.",
    ]

    return {
        "prediction": screening_probability,
        "probability_percent": probability_percent,
        "model_probability": model_probability,
        "model_probability_percent": model_probability_percent,
        "risk_level": risk_level,
        "feature_analysis": table_rows,
        "abnormal_feature_count": abnormal_count,
        "observations": abnormal_observations or ["No major abnormal indicators were detected."],
        "summary": summary,
        "recommendations": recommendations,
    }
def _encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")




def _serve_webapp_page(filename: str):
    webapp_dir = Path(__file__).resolve().parent.parent
    return send_from_directory(webapp_dir, filename)

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
        rate_limiter = current_app.extensions["rate_limiter"]
        if not rate_limiter.is_allowed(client_id):
            return jsonify({"error": "Rate limit exceeded"}), 429
        return func(*args, **kwargs)

    return wrapper


@api_bp.route("/", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "service": "dyslexia-prediction-api"})


@api_bp.route("/ready", methods=["GET"])
def readiness() -> Any:
    model_service = current_app.extensions["model_service"]
    model_status = {
        "dyslexia_model_loaded": model_service.dyslexia_model is not None,
        "handwriting_model_loaded": model_service.handwriting_model is not None,
    }
    dependencies = {
        "tesseract_configured": bool(getattr(pytesseract.pytesseract, "tesseract_cmd", "")),
        "redis_enabled": bool(current_app.config.get("REDIS_URL", "")),
    }
    overall = "ready" if all(model_status.values()) else "degraded"
    payload = {"status": overall, "models": model_status, "dependencies": dependencies}
    code = 200 if overall == "ready" else 503
    return jsonify(payload), code


@api_bp.route("/home", methods=["GET"])
def home_page() -> Any:
    return _serve_webapp_page("index.html")


@api_bp.route("/dyslexia-prediction", methods=["GET"])
def dyslexia_prediction_page() -> Any:
    return _serve_webapp_page("dyslexia-prediction.html")


@api_bp.route("/ocr-tool", methods=["GET"])
def ocr_tool_page() -> Any:
    return _serve_webapp_page("image_analysis.html")


@api_bp.route("/handwriting-analysis-page", methods=["GET"])
def handwriting_analysis_page() -> Any:
    return _serve_webapp_page("image_analysis.html")


@api_bp.route("/image-analysis", methods=["GET"])
def image_analysis_page() -> Any:
    return _serve_webapp_page("image_analysis.html")


@api_bp.route("/prediction-result", methods=["GET"])
def prediction_result_page() -> Any:
    return render_template("prediction_result.html")




@api_bp.route("/theme.css", methods=["GET"])
def serve_theme_css():
    webapp_dir = Path(__file__).resolve().parent.parent
    return send_from_directory(webapp_dir, "theme.css")


@api_bp.route("/IMAGES/<path:requested>", methods=["GET"])
def serve_image_assets(requested: str):
    webapp_dir = Path(__file__).resolve().parent.parent
    image_dir = (webapp_dir / "IMAGES").resolve()
    image_path = (image_dir / requested).resolve()

    if not str(image_path).startswith(str(image_dir)):
        return jsonify({"error": "Endpoint not found."}), 404

    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico"} or not image_path.is_file():
        return jsonify({"error": "Endpoint not found."}), 404

    return send_from_directory(image_dir, requested)

@api_bp.route("/assets/<path:requested>", methods=["GET"])
def serve_web_assets(requested: str):
    webapp_dir = Path(__file__).resolve().parent.parent
    asset_path = (webapp_dir / requested).resolve()

    if not str(asset_path).startswith(str(webapp_dir.resolve())):
        return jsonify({"error": "Endpoint not found."}), 404

    if asset_path.suffix.lower() not in _ALLOWED_ASSET_EXTENSIONS or not asset_path.is_file():
        return jsonify({"error": "Endpoint not found."}), 404

    return send_from_directory(webapp_dir, requested)


@api_bp.route("/<string:page>.html", methods=["GET"])
def compatibility_html_route(page: str):
    filename = f"{page}.html"
    if filename not in _ALLOWED_PAGE_FILES:
        return jsonify({"error": "Endpoint not found."}), 404
    return _serve_webapp_page(filename)


@api_bp.route("/src/main/webapp/<path:requested>", methods=["GET"])
def compatibility_legacy_path(requested: str):
    filename = Path(requested).name
    if filename not in _ALLOWED_PAGE_FILES:
        return jsonify({"error": "Endpoint not found."}), 404
    return _serve_webapp_page(filename)


@api_bp.route("/ocr-result", methods=["GET"])
def ocr_result_page() -> Any:
    return render_template("ocr_result.html")


@api_bp.route("/image-analysis-result", methods=["GET"])
def image_analysis_result_page() -> Any:
    return render_template("image_analysis_result.html")


@api_bp.route("/ocr-text-details", methods=["GET"])
def ocr_text_details_page() -> Any:
    return render_template("ocr_text_details.html")


@api_bp.route("/handwriting-result", methods=["GET"])
def handwriting_result_page() -> Any:
    return render_template("handwriting_result.html")


@api_bp.route("/predict", methods=["POST"])
@require_auth
@rate_limited
def predict() -> Any:
    payload = request.get_json(silent=True)

    try:
        feature_map = parse_predict_request(payload)
    except SchemaValidationError as error:
        return jsonify({"error": str(error)}), 400

    expected_features = current_app.config["EXPECTED_FEATURES"]

    try:
        feature_values = [float(feature_map[feature]) for feature in expected_features]
        model_service = current_app.extensions["model_service"]
        prediction = model_service.predict_dyslexia(feature_values)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        LOGGER.exception("Unhandled dyslexia prediction failure")
        return jsonify({"error": "Prediction service unavailable."}), 500

    if not np.isfinite(prediction):
        LOGGER.warning("Model returned non-finite score; returning safe fallback.")
        result_payload = {
            "prediction": 0.0,
            "probability_percent": 0.0,
            "model_probability": 0.0,
            "model_probability_percent": 0.0,
            "risk_level": "Indeterminate",
            "feature_analysis": [],
            "abnormal_feature_count": 0,
            "observations": ["Model output was invalid; result is indeterminate."],
            "summary": "Prediction could not be interpreted safely.",
            "recommendations": ["Retry later and verify model health."],
        }
    else:
        result_payload = _build_prediction_payload(prediction, feature_map)

    token = _encode_payload(result_payload)
    response_payload = validate_predict_response({**result_payload, "result_redirect": f"/prediction-result?data={token}"})
    return jsonify(response_payload), 200


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

    probability_percent = round(predicted_prob * 100, 2)
    risk_level = "Low Risk" if predicted_class == "Non_Dyslexic" else "Moderate/High Risk"
    summary = (
        "Model detected handwriting characteristics requiring follow-up."
        if predicted_class == "Dyslexic"
        else "No major handwriting risk markers detected."
    )
    result_payload = {
        "predicted_probability": predicted_prob,
        "probability_percent": probability_percent,
        "predicted_class": predicted_class,
        "risk_level": risk_level,
        "summary": summary,
        "recommendations": [
            "Use as supportive screening output only.",
            "Combine with cognitive and reading assessments.",
            "Seek specialist review for high-risk outcomes.",
        ],
    }
    token = _encode_payload(result_payload)
    response_payload = validate_handwriting_response({**result_payload, "result_redirect": f"/handwriting-result?data={token}"})
    return jsonify(response_payload)


def _run_ocr_pipeline(file_obj: Any) -> dict[str, Any]:
    raw_file = file_obj.read()
    file_bytes = np.frombuffer(raw_file, dtype=np.uint8)
    if file_bytes.size == 0:
        raise ValueError("No file provided.")

    tesseract_config = "--oem 3 --psm 6"

    if cv2 is not None:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image file.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        upscale = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(upscale, 7, 50, 50)
        thresholded = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        raw_text = pytesseract.image_to_string(thresholded, config=tesseract_config)
    else:
        LOGGER.warning("OpenCV unavailable; using direct OCR fallback pipeline.")
        pil_image = Image.open(BytesIO(raw_file)).convert("L")
        raw_text = pytesseract.image_to_string(pil_image, config=tesseract_config)
    extracted_text = _clean_ocr_text(raw_text)
    corrected_text = _correct_ocr_text(extracted_text)
    simplified_text = _simplify_ocr_text(corrected_text)
    quality = _ocr_quality_metrics(extracted_text, corrected_text)

    observations = [
        "Handwritten/low-light samples may reduce OCR quality.",
        "Corrected text applies lightweight symbol and spacing cleanup.",
        "Simplified text removes punctuation and common stop words for easier scanning.",
    ]
    if quality["quality_score"] < 60:
        observations.append("OCR quality is low; recapture with better lighting for higher accuracy.")

    return {
        "extracted_text": extracted_text,
        "corrected_text": corrected_text,
        "simplified_text": simplified_text,
        "summary": f"OCR completed successfully. Estimated quality: {quality['quality_score']}%.",
        "ocr_quality_score": quality["quality_score"],
        "noise_characters": quality["noise_characters"],
        "underscore_artifacts": quality["underscore_artifacts"],
        "observations": observations,
        "recommendations": [
            "Capture clear, well-lit images.",
            "Keep text horizontal and avoid shadows.",
            "Prefer high-contrast dark text on a plain background.",
        ],
        "original_text": extracted_text,
    }


@api_bp.route("/image-analysis-upload", methods=["POST"])
@require_auth
@rate_limited
def image_analysis_upload() -> Any:
    """Run both handwriting classification and OCR extraction for one uploaded photo."""
    file_obj = request.files.get("file") or request.files.get("image")
    if file_obj is None or not file_obj.filename:
        return jsonify({"error": "No file provided."}), 400

    validation = validate_image_upload(
        image_file=file_obj,
        allowed_extensions=current_app.config["ALLOWED_IMAGE_EXTENSIONS"],
        allowed_mime_types=current_app.config["ALLOWED_IMAGE_MIME_TYPES"],
    )
    if not validation.ok:
        return jsonify({"error": validation.message}), 400

    try:
        image_bytes = file_obj.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        model_service = current_app.extensions["model_service"]
        predicted_prob, predicted_class = model_service.predict_handwriting(image_bytes)

        file_obj.stream.seek(0)
        ocr_payload = _run_ocr_pipeline(file_obj)

        handwriting_payload = {
            "predicted_probability": predicted_prob,
            "probability_percent": round(predicted_prob * 100, 2),
            "predicted_class": predicted_class,
            "risk_level": "Low Risk" if predicted_class == "Non_Dyslexic" else "Moderate/High Risk",
            "summary": (
                "Model detected handwriting characteristics requiring follow-up."
                if predicted_class == "Dyslexic"
                else "No major handwriting risk markers detected."
            ),
            "recommendations": [
                "Use as supportive screening output only.",
                "Combine with cognitive and reading assessments.",
                "Seek specialist review for high-risk outcomes.",
            ],
        }

        unified_payload = {
            "overall_summary": "Combined image analysis completed: handwriting risk + OCR readability insights generated.",
            "handwriting": handwriting_payload,
            "ocr": ocr_payload,
        }
        token = _encode_payload(unified_payload)
        response_payload = validate_image_analysis_response(
            {**unified_payload, "result_redirect": f"/image-analysis-result?data={token}"}
        )
        return jsonify(response_payload), 200
    except Exception:
        LOGGER.exception("Unhandled combined image analysis failure")
        return jsonify({"error": "Combined image analysis unavailable."}), 500


@api_bp.route("/upload", methods=["POST"])
@require_auth
@rate_limited
def upload_ocr() -> Any:
    """Extract text from uploaded image and return normalized variants."""
    file_obj = request.files.get("file") or request.files.get("image")
    if file_obj is None or not file_obj.filename:
        return jsonify({"error": "No file provided."}), 400

    validation = validate_image_upload(
        image_file=file_obj,
        allowed_extensions=current_app.config["ALLOWED_IMAGE_EXTENSIONS"],
        allowed_mime_types=current_app.config["ALLOWED_IMAGE_MIME_TYPES"],
    )
    if not validation.ok:
        return jsonify({"error": validation.message}), 400

    try:
        result_payload = _run_ocr_pipeline(file_obj)
        token = _encode_payload(result_payload)
        response_payload = validate_ocr_response({**result_payload, "result_redirect": f"/ocr-result?data={token}"})
        return jsonify(response_payload), 200
    except Exception:
        LOGGER.exception("Unhandled OCR upload failure")
        return jsonify({"error": "OCR processing unavailable."}), 500
