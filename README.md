<div align="center">

# Dyslexia Detection Project

</div>

This repository hosts the Dyslexia Early Detection System, a multi-modal assessment prototype combining cognitive tests, handwriting analysis, and eye-tracking support.

## Features

- Cognitive tests for reading speed, spelling accuracy, phonemic awareness, response time, and rapid naming (RAN).
- Handwriting analysis using a TensorFlow/Keras model.
- Eye-tracking experiment support.
- Dedicated handwriting analysis flow for handwriting-risk screening from uploaded handwriting samples.
- API endpoints:
  - `POST /predict`
  - `POST /handwriting-analysis`
  - `POST /image-analysis-upload`

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

## Hardening additions

- Optional Redis-backed rate limiting via `REDIS_URL` (falls back to in-memory limiter if unavailable).
- Request/response schema contracts implemented with lightweight internal validators for key API responses.
- Readiness probe endpoint: `GET /ready` (returns `200` when models are loaded, else `503`).
- Large native installers are excluded from Git; keep them in external artifact storage.


## Reading speed reliability

Reading-speed scoring now enforces basic reliability checks: minimum plausible reading time, minimum focus time, and a quick comprehension question before persisting score.


## Test design updates

- Replaced the earlier attention-span mini task in the level flow with a **Rapid Naming** test, which is generally more specific to reading/dyslexia risk than generic attention metrics.
- Reduced writing-errors prompts from 8 to 5 items to lower fatigue while preserving signal quality.

## UI design tokens and interaction layer

The UI is now standardized with shared tokens and interaction utilities:

- `src/main/webapp/styles/design-tokens.css`: color, spacing, radius, shadow, and motion tokens.
- `src/main/webapp/styles/main.css`: global shell styles for home, cards, modal, and upload interface.
- `src/main/webapp/scripts/ui-interactions.js`: keyboard Enter-to-primary-action behavior, modal focus trap, and reduced-friction accessibility helpers.

### Run UI locally

```bash
flask --app src.main.webapp.wsgi:app run --host 0.0.0.0 --port 5000
```

Open:

- `http://127.0.0.1:5000/home` (home + auth modal)
- `http://127.0.0.1:5000/login` (fallback login route that opens auth modal)
- `http://127.0.0.1:5000/handwriting_analysis.html` (upload screen)
