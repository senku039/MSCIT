# MSCIT Codebase Analysis

## 1) High-level architecture
- The project is a Flask-based dyslexia screening web app with three ML-assisted backend capabilities: structured dyslexia risk scoring (`/predict`), handwriting image classification (`/handwriting-analysis`), and OCR processing (`/upload`).
- The backend uses an application-factory pattern in `app.py`, centralized config in `config.py`, input validation utilities in `utils/validators.py`, route/controller logic in `api/routes.py`, and model loading/inference in `services/model_service.py`.
- The frontend is mostly static HTML pages and JS-driven flows under `src/main/webapp/*.html`, with result dashboards rendered from Jinja templates in `src/main/webapp/templates/*.html`.

## 2) Backend flow
1. `wsgi.py` imports `create_app()` and exposes `app` for Gunicorn.
2. `create_app()` selects config by `FLASK_ENV`, enables CORS, loads models via `ModelService.load_models()`, registers the blueprint, and installs sanitized error handlers.
3. Route decorators in `api/routes.py` optionally enforce bearer-token auth and in-memory per-client rate limiting.
4. Endpoint handlers validate payloads/files, invoke model service methods, enrich outputs with explainability metadata, and return JSON plus a redirect token for result dashboards.

## 3) Security and reliability posture
- Positive practices:
  - Explicit request-size limits (`MAX_CONTENT_LENGTH`), CORS allowlist, optional bearer auth, and basic in-memory throttling.
  - Strict schema validation and numeric casting for `/predict` input.
  - File extension + MIME checks for uploaded images.
  - Path traversal defenses for static asset serving (`resolve()` + prefix checks).
  - Startup model loading wrapped so app can still boot while reporting model failures.
- Tradeoffs / gaps:
  - Rate limit storage is process-local (`_RATE_BUCKETS`), so limits won’t coordinate across workers/instances.
  - Broad `except Exception` handlers intentionally hide details from clients but can obscure failure classes unless logs are monitored.
  - Legacy OCR app (`my_ocr_app/ocr.py`) hardcodes a Windows Tesseract path and diverges from main app conventions.

## 4) ML and scoring logic
- Dyslexia structured prediction:
  - Accepts 7 numeric features in fixed order from config.
  - Combines raw model probability with a handcrafted feature-signal score (80/20 blend) to produce screening probability.
  - Adds risk tier (`Low/Moderate/High`) and a per-feature analysis table with expected ranges and impact notes.
- Handwriting prediction:
  - Loads Keras model, rescales uploaded image to configured size, normalizes to [0,1], checks input shape compatibility.
  - Supports model-orientation switch (`HANDWRITING_SCORE_MEANS_DYSLEXIC`) to interpret score semantics.
- OCR pipeline:
  - OpenCV preprocessing (grayscale, upscale, denoise, adaptive threshold) + Tesseract extraction.
  - Post-processing includes text cleanup, lightweight token substitutions, simplification, and a heuristic quality score.

## 5) Frontend structure
- Primary pages include:
  - `index.html`: landing/navigation shell.
  - `dyslexia-prediction.html`: aggregates cognitive test outputs and submits to `/predict`.
  - Individual tests (`reading_speed_test.html`, `spelling_accuracy_test.html`, `phonemic_awareness_test.html`, `writing_errors_test.html`, `response_time_test.html`, `attention_span_calculator.html`, `cognitive_score_test.html`).
  - `handwriting_analysis.html` for image upload.
  - `ocr.html` for OCR upload.
- Result pages (`templates/prediction_result.html`, `templates/handwriting_result.html`, `templates/ocr_result.html`) decode URL-safe base64 JSON and render a dashboard-style summary.

## 6) Supporting and legacy artifacts
- `run.py` is a standalone webcam gaze-tracking utility writing `attention_score.txt`; it is not integrated through Flask request handling.
- `my_ocr_app/ocr.py` is a separate Flask OCR micro-app using `pyspellchecker` and `TextBlob`; not wired into the primary app factory.
- Extra root files (`fr.html`, `new.html`, `2.ipynb`) appear exploratory/prototype and are outside the packaged Flask app.

## 7) Deployment/runtime expectations
- Dockerfile installs Python deps, copies only `src/main/webapp`, and launches Gunicorn with `wsgi:app`.
- Runtime requires external model artifacts and system-level Tesseract availability.
- `requirements.txt` includes Flask/TensorFlow/OpenCV/Tesseract wrapper stack, consistent with the three core features.

## 8) Suggested next hardening steps
1. Replace in-memory rate limiter with Redis-backed storage for multi-worker correctness.
2. Add request/response schema contracts (e.g., Marshmallow/Pydantic) and endpoint-level unit tests.
3. Add health/readiness probes that explicitly verify model availability status.
4. Consolidate or retire duplicate legacy OCR app and clarify supported execution path.
5. Move large binaries/assets (e.g., installer EXE) out of repo into release artifacts.
