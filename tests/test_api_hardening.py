from __future__ import annotations

from src.main.webapp.app import create_app


class StubModelService:
    def predict_dyslexia(self, _):
        return 0.5


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
