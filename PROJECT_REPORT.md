# Dyslexia Early Detection System — Technical Documentation (Research + Project Report)

## 1) Abstract
This project is a Flask-based dyslexia early-screening platform that combines (a) a structured multi-test cognitive workflow and (b) a handwriting-image classifier. The system produces **screening-oriented** risk outputs with explicit recommendations and reliability checks. It is intentionally designed as a **decision-support/screening** tool and not as a clinical diagnosis engine.

## 2) Problem Statement and Objectives
The project addresses three practical needs:
1. collect multiple educationally relevant signals (reading speed, spelling, writing errors, phonemic awareness, response time, rapid naming/cognitive proxies),
2. aggregate and validate them consistently via backend API contracts,
3. run an additional handwriting-image classifier for another independent signal path.

Primary goals:
- robust API validation,
- stable model serving,
- low-friction UX for guided test progression,
- clear, interpretable outputs for end users.

## 3) End-to-End Project Workflow

### 3.1 Cognitive screening flow
1. User opens home page and enters the roadmap/game-style test flow.
2. User completes each level page; each page stores score(s) in `sessionStorage`.
3. The roadmap page reads stored values, enforces level progression, and assembles a JSON payload.
4. Frontend submits payload to `POST /predict`.
5. Backend validates schema, calls dyslexia model, builds feature-level interpretation table, computes final screening probability/risk band, and returns `result_redirect`.
6. Frontend navigates to prediction-result page and renders decoded payload.

### 3.2 Handwriting analysis flow
1. User uploads an image in `handwriting_analysis.html`.
2. Frontend posts multipart data to `POST /handwriting-analysis`.
3. Backend validates extension/MIME and file non-empty checks.
4. Backend preprocesses image to model input shape, predicts probability and class.
5. Backend returns response + `result_redirect`; frontend redirects to result template.

## 4) Architecture

## 4.1 Layered view
- **Presentation layer**: static HTML pages + JavaScript for tests and UI flow.
- **API layer**: Flask blueprint routes, security decorators, response shaping.
- **Service layer**: model loading/inference (`ModelService`) and rate limiting (`RateLimiter`).
- **Validation layer**: API schemas + file-upload validators.
- **Configuration layer**: environment-driven config classes.

## 4.2 Runtime composition
`create_app()` initializes config, CORS, model service, rate limiter, blueprint registration, and global error handlers. Extensions are attached in `app.extensions` and consumed by routes through helper access.

## 5) Backend Logic (Detailed)

### 5.1 Application factory
- Loads environment-specific config class.
- Configures CORS origin allowlist.
- Loads models at startup (failure tolerated with degraded behavior).
- Registers route blueprint and standard JSON error handlers.

### 5.2 API routes
Main categories:
- Health/readiness: `/`, `/ready`
- Web pages/static compatibility routes
- Prediction APIs: `/predict`, `/handwriting-analysis`
- Legacy aliases: `/image-analysis`, `/image-analysis-upload` (mapped to handwriting pages/handler)

### 5.3 Security and hardening controls
- Optional bearer token auth via `API_TOKENS`.
- Per-client rate limit via extension (memory or Redis backend).
- Input schema enforcement for prediction payload.
- File extension/MIME checks for image upload.
- NaN/inf guards and clipping for model outputs.
- Standardized error responses for 404/405/500 and payload-too-large.

## 6) Algorithms and Decision Logic

### 6.1 Prediction aggregation (cognitive)
The dyslexia model returns a probability-like score. The route then computes a feature-signal from rule thresholds and combines them:
- `screening_probability = 0.8 * model_probability + 0.2 * feature_signal`

Risk classification is thresholded into Low/Moderate/High bands.

### 6.2 Feature interpretation
Each feature has expected ranges and direction (`higher_better` / `lower_better`). For each feature:
- mark abnormal/normal,
- produce status/impact text,
- include in analysis table and observations.

### 6.3 Handwriting model inference
- Load image bytes with Keras preprocessing.
- Resize to configured shape (default 128x128), normalize to `[0,1]`, add batch dim.
- Predict using TensorFlow/Keras model.
- Clip probability and map to `Dyslexic` / `Non_Dyslexic` using threshold and score semantic config flag.

### 6.4 Rate limiting
Two interchangeable algorithms:
- **In-memory sliding window** with per-client timestamp deque.
- **Redis sorted-set sliding window** (`ZREMRANGEBYSCORE`, `ZCARD`, `ZADD`, `EXPIRE`).

## 7) Data Flow

### 7.1 `POST /predict`
Input JSON -> `parse_predict_request` -> ordered feature vector -> `ModelService.predict_dyslexia` -> payload builder + feature analysis -> response schema validation -> JSON response with redirect token.

### 7.2 `POST /handwriting-analysis`
Multipart file -> upload validator -> bytes read -> `ModelService.predict_handwriting` -> output sanitization + recommendations -> response schema validation -> JSON response with redirect token.

### 7.3 Result rendering flow
Backend returns `result_redirect` with base64-url payload in query parameter. Result templates decode and render summaries/tables/recommendations client-side.

## 8) Training vs Prediction Pipeline

### 8.1 What exists in this repository
- **Inference pipeline only** is implemented in production code.
- Models are loaded from files: `dyslexia_reg_model.pkl` and `final_model.keras`.

### 8.2 Training artifacts/code availability
- Training scripts/pipelines are not present in runtime package.
- Training is assumed external/offline; this repository focuses on serving, validation, and UI workflow.

### 8.3 Reproducibility implications
For research-grade reproducibility, add (in future):
- dataset card and splits,
- training code + hyperparameters,
- evaluation metrics and model card,
- versioned model registry.

## 9) File-by-File Role Map

### 9.1 Core backend files
- `src/main/webapp/app.py` — application factory, extension wiring, global handlers.
- `src/main/webapp/wsgi.py` — WSGI entrypoint.
- `src/main/webapp/config.py` — environment/config classes.
- `src/main/webapp/api/routes.py` — all endpoints, route logic, payload shaping.
- `src/main/webapp/api/schemas.py` — schema-level request/response contracts.
- `src/main/webapp/services/model_service.py` — model loading and inference.
- `src/main/webapp/services/rate_limiter.py` — memory/Redis rate limiter.
- `src/main/webapp/utils/validators.py` — JSON/feature/image upload validation helpers.

### 9.2 Frontend workflow files
- `src/main/webapp/index.html` — landing/home and navigation.
- `src/main/webapp/dyslexia-prediction.html` — guided multi-level flow + `/predict` submission.
- `src/main/webapp/handwriting_analysis.html` — handwriting upload client.
- `src/main/webapp/templates/prediction_result.html` — renders cognitive result payload.
- `src/main/webapp/templates/handwriting_result.html` — renders handwriting result payload.

### 9.3 Test/assessment pages (sessionStorage producers)
- `attention_span_calculator.html`
- `rapid_naming_test.html`
- `spelling_accuracy_test.html`
- `phonemic_awareness_test.html`
- `writing_errors_test.html`
- `cognitive_score_test.html`
- `reading_speed_test.html`
- `response_time_test.html`

### 9.4 Informational/support files
- `README.md` — setup/run notes and hardening guidance.
- `CODEBASE_ANALYSIS.md` — architecture notes/observations.
- `requirements.txt` — Python dependencies.
- `tests/test_api_hardening.py`, `tests/conftest.py` — backend hardening tests + stubs.

### 9.5 Miscellaneous/legacy utilities
- `src/main/webapp/run.py` — standalone OpenCV gaze-tracking utility script (not part of Flask API serving path).
- `src/main/webapp/style.css`, `style1.css`, `style2.css`, `theme.css` — shared and page-level styles.
- `src/main/webapp/info_*.html` — informational content pages.
- `src/main/webapp/IMAGES/*` and `my_ocr_app/assets/*` — static assets.

## 10) Technologies Used

### Backend
- Python 3
- Flask (+ Flask-CORS)
- NumPy
- TensorFlow/Keras
- Joblib
- Optional Redis client

### Frontend
- HTML/CSS/vanilla JavaScript
- Bootstrap 5
- Font Awesome

### Testing
- Pytest

## 11) Research-Paper-Ready Narrative (Reusable)

### Title suggestion
**A Multi-Modal Dyslexia Early-Screening Web System Integrating Cognitive Assessments and Handwriting Classification**

### Method summary
The system integrates structured behavioral tasks and a convolutional handwriting classifier. Behavioral features are passed to a regression/classification model; outputs are combined with interpretable feature-threshold signals for final screening risk stratification. A secondary handwriting branch provides image-based risk support. API contracts and runtime guards ensure reliable serving.

### Suggested evaluation section (for report)
- Functional correctness: endpoint validation/error paths.
- Robustness: readiness/rate-limit/degraded-mode behavior.
- Usability: completion funnel for multi-level test flow.
- Model output sanity: finite output checks and threshold consistency.

### Suggested limitations section
- Clinical diagnosis is out-of-scope.
- Training pipeline and dataset details are not bundled in runtime code.
- Model quality claims require external benchmark data.

### Suggested future work
- Add explicit training pipeline in-repo.
- Add calibration/uncertainty estimation.
- Add audit logging, model version tags, and fairness analysis.
- Add integration/e2e tests for full browser-to-API flow.

## 12) Practical Runbook (Short)
1. Create and activate venv.
2. `pip install -r requirements.txt`
3. Start app via Flask/Werkzeug or `python -m src.main.webapp.app`.
4. Open `/home` for landing page.
5. Use `/ready` to check model readiness.
