# Full Code Analysis Audit — MSCIT Dyslexia System

## Scope
This audit reviews the entire repository from architecture, backend correctness, frontend workflow, ML inference path, security/hardening, testing, and maintainability perspectives.

---

## 1) Repository Structure Assessment

### Core runtime backend
- `src/main/webapp/app.py` — Flask app factory, extension wiring, error handlers.
- `src/main/webapp/config.py` — environment-driven configuration.
- `src/main/webapp/api/routes.py` — endpoint orchestration + response shaping.
- `src/main/webapp/api/schemas.py` — request/response validation contracts.
- `src/main/webapp/services/model_service.py` — model load + inference routines.
- `src/main/webapp/services/rate_limiter.py` — rate limit abstraction (memory + Redis).
- `src/main/webapp/utils/validators.py` — upload/payload validation helpers.

### Frontend workflow layer
- `index.html` — entry page.
- `dyslexia-prediction.html` — level flow + `/predict` submit orchestrator.
- Test pages (`*_test.html`, `attention_span_calculator.html`) — metric producers into `sessionStorage`.
- `handwriting_analysis.html` + result templates — upload + result rendering path.

### Quality/docs layer
- `tests/test_api_hardening.py`, `tests/conftest.py`.
- `README.md`, `CODEBASE_ANALYSIS.md`, `PROJECT_REPORT.md`.

---

## 2) Workflow and Data-Flow Analysis

## 2.1 Cognitive path
1. User completes frontend levels.
2. Individual pages persist scores in `sessionStorage`.
3. Roadmap page aggregates values and posts to `/predict`.
4. Backend schema-validates payload and predicts.
5. Backend enriches with feature interpretation + recommendations.
6. Client receives `result_redirect` and renders final result.

## 2.2 Handwriting path
1. User uploads image.
2. Frontend posts multipart to `/handwriting-analysis`.
3. Backend validates file and MIME/extension.
4. ModelService preprocesses and predicts class/probability.
5. Response is schema-validated and redirected to result page.

---

## 3) Backend Logic Analysis

## 3.1 Strengths
- App factory pattern with centralized extension registration.
- Defensive startup: model-load failure does not kill process.
- Structured health/readiness endpoints.
- Endpoint-level auth/rate-limiting decorators.
- Contract validation for inputs and outputs.
- Finite-value guards and clipping on model outputs.

## 3.2 Important observations
- `routes.py` currently handles both API and many static page routes, making it large and mixed-responsibility.
- Some compatibility aliases remain for legacy URLs (intentional for backward compatibility).
- `run.py` still contains standalone webcam/gaze logic and is not integrated into Flask runtime.

---

## 4) ML/Inference Pipeline Analysis

## 4.1 Dyslexia model path
- Model file loaded via `joblib`.
- Expects numeric vector in fixed feature order.
- Returns a scalar score interpreted as risk probability proxy.

## 4.2 Handwriting model path
- Keras `.keras` model loaded without compile.
- Image bytes -> resize (default 128x128) -> normalize -> batch dimension.
- Thresholding with configurable score semantics (`HANDWRITING_SCORE_MEANS_DYSLEXIC`).

## 4.3 Current limitation
- Training pipeline is not in runtime repository; only inference-serving is present.

---

## 5) Security and Hardening Analysis

## 5.1 Existing hardening
- Optional bearer token auth (`API_TOKENS`).
- Per-client rate limit with optional Redis backend.
- Allowed-extension and MIME checks for uploads.
- Route-level safe static file serving checks for path traversal.
- Standardized 404/405/500 and payload-too-large handlers.

## 5.2 Risks / gaps
1. No strict JSON schema versioning yet (only key-level checks).
2. No centralized structured audit logging per request id.
3. No e2e test covering browser-to-backend flow in CI.
4. No explicit model/version metadata endpoint beyond readiness.
5. Frontend relies on `sessionStorage`; a user can tamper values client-side.

---

## 6) Frontend and UX Analysis

## 6.1 Positive points
- Guided level progression and lock/unlock logic.
- Clear result pages with recommendations.
- Back-button UX improvements in test pages.
- Handwriting upload retries and redirect handling.

## 6.2 Technical debt indicators
- Logic-heavy inline scripts across multiple HTML files.
- Repeated button/nav/sessionStorage patterns (could be componentized/shared JS).
- Hardcoded API base assumptions in roadmap flow (`127.0.0.1`).

---

## 7) Testing Analysis

### Existing tests validate
- health/readiness behavior,
- schema rejection path,
- handwriting endpoint success path,
- missing extension failure path.

### Missing tests (recommended)
1. `/predict` happy-path contract with deterministic mock payload validation.
2. File-upload negative tests (wrong MIME/extension/empty body) for handwriting.
3. Rate limiter behavior tests (memory + redis-fallback simulation).
4. Frontend integration tests for `sessionStorage` to `/predict` mapping.

---

## 8) Maintainability Scorecard (Qualitative)

- **Architecture clarity:** Good (factory + services + routes), but route file is oversized.
- **Reliability:** Good baseline, improved by schema and extension guards.
- **Security posture:** Moderate-good for prototype stage.
- **Test maturity:** Basic backend hardening tests present; integration coverage limited.
- **Docs quality:** Strong (README + project report + this audit).

---

## 9) Prioritized Action Plan

### P0 (high impact)
1. Split `api/routes.py` into `routes_api.py`, `routes_pages.py`, and helpers.
2. Add strict typed schemas (e.g., Pydantic/dataclass validation) with versioned response envelopes.
3. Add CI integration tests for `/predict` + `/handwriting-analysis` with fixture files.

### P1
4. Add request-id logging and structured JSON logs.
5. Expose model metadata endpoint (`/model-info` hash, version, loaded timestamp).
6. Move repeated frontend JS utility logic into shared static JS module.

### P2
7. Add optional server-side session integrity checks to reduce client-side score tampering impact.
8. Add deployment profile docs (Docker + production env variable matrix).

---

## 10) Conclusion
The repository is now a solid inference-serving + assessment platform with meaningful hardening improvements (schemas, readiness, extension-based limiter/model lookup, safer fallbacks). The next improvements should focus on route modularization, stronger integration tests, and stronger traceability (model/version/request metadata).
