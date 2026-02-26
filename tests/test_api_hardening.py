from __future__ import annotations

from src.main.webapp.app import create_app


class StubModelService:
    def predict_dyslexia(self, _):
        return 0.5

    def predict_handwriting(self, _):
        return 0.25, "Non_Dyslexic"


def test_health_endpoint():
    app = create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "dyslexia-prediction-api"


def test_readiness_degraded_without_models():
    app = create_app()
    client = app.test_client()

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "degraded"


def test_predict_rejects_schema_violations():
    app = create_app()
    app.extensions["model_service"] = StubModelService()
    client = app.test_client()

    payload = {
        "Reading_Speed": 100,
        "Spelling_Accuracy": 90,
        "Writing_Errors": 1,
        "Cognitive_Score": 88,
        "Phonemic_Awareness_Errors": 1,
        "Attention_Span": 90,
        "Response_Time": 0.5,
        "Unexpected": 1,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_handwriting_page_exists():
    app = create_app()
    client = app.test_client()

    response = client.get("/handwriting-analysis-page")
    assert response.status_code == 200


def test_handwriting_flow(monkeypatch):
    app = create_app()
    app.extensions["model_service"] = StubModelService()

    client = app.test_client()
    data = {"image": (
        __import__("io").BytesIO(b"fake-bytes"),
        "sample.jpg",
        "image/jpeg",
    )}

    response = client.post("/handwriting-analysis", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["predicted_class"] == "Non_Dyslexic"
    assert payload["result_redirect"].startswith("/handwriting-result?data=")
