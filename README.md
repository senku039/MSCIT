<div align="center">

# Dyslexia Detection Project

</div>

This repository hosts the Dyslexia Early Detection System, a multi-modal assessment prototype combining cognitive tests, handwriting analysis, eye-tracking, and OCR support.

## Features

- Cognitive tests for reading speed, spelling accuracy, phonemic awareness, response time, and attention.
- Handwriting analysis using a TensorFlow/Keras model.
- Eye-tracking experiment support.
- OCR helper application for readability workflows.
- API endpoints:
  - `POST /predict`
  - `POST /handwriting-analysis`

## Secure backend layout

```text
src/main/webapp/
├── api/
│   └── routes.py
├── services/
│   └── model_service.py
├── utils/
│   └── validators.py
├── app.py
├── config.py
└── wsgi.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Set these before production deployment:

- `FLASK_ENV=production`
- `SECRET_KEY=<long-random-value>`
- `API_TOKENS=<comma-separated-token-list>`
- `CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000` (local dev example)
- `RATE_LIMIT_PER_MINUTE=60`
- `MAX_CONTENT_LENGTH=2097152`
- `DYSLEXIA_MODEL_PATH=/secure/models/dyslexia_reg_model.pkl`
- `HANDWRITING_MODEL_PATH=/secure/models/final_model.keras`
- `HANDWRITING_THRESHOLD=0.5`
- `HANDWRITING_SCORE_MEANS_DYSLEXIC=true`

## Running (development)

```bash
flask --app src.main.webapp.wsgi:app run --debug
```

## Running (production)

Use a WSGI server such as Gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 src.main.webapp.wsgi:app
```

## Notes for model artifacts

Model binaries (`.keras`, `.pkl`, `.h5`) should be stored outside Git (artifact store or Git LFS) for security, size control, and reproducibility.

## OCR runtime dependency

Install the native **Tesseract OCR engine** and ensure `tesseract` is available on your system PATH. `pytesseract` is only the Python wrapper.

## Hardening additions

- Optional Redis-backed rate limiting via `REDIS_URL` (falls back to in-memory limiter if unavailable).
- Request/response schema contracts implemented with lightweight internal validators for key API responses.
- Readiness probe endpoint: `GET /ready` (returns `200` when models are loaded, else `503`).
- Legacy OCR standalone app now proxies to the main Flask application entrypoint for consistency.
- Large native installers are excluded from Git; keep them in external artifact storage.
