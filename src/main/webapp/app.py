"""Flask entrypoint with secure defaults for model-serving endpoints."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from src.main.webapp.auth import init_user_table
from src.main.webapp.api.routes import api_bp
from src.main.webapp.config import CONFIG_MAPPING
from src.main.webapp.services.model_service import ModelService
from src.main.webapp.services.rate_limiter import RateLimiter


def _validate_security_config(app: Flask, env_name: str) -> None:
    """Fail fast on insecure production bootstraps unless explicitly overridden."""
    if env_name != "production":
        return

    if app.config.get("ALLOW_INSECURE_BOOTSTRAP", False):
        logging.getLogger(__name__).warning("ALLOW_INSECURE_BOOTSTRAP=true: skipping strict production checks.")
        return

    secret_key = os.getenv("SECRET_KEY", app.config.get("SECRET_KEY", "replace-in-production"))
    api_tokens = {token.strip() for token in os.getenv("API_TOKENS", "").split(",") if token.strip()}
    if not api_tokens:
        api_tokens = app.config.get("API_TOKENS", set())

    if secret_key == "replace-in-production":
        raise RuntimeError("Refusing to start production with default SECRET_KEY.")

    if not api_tokens:
        raise RuntimeError("Refusing to start production without API_TOKENS.")


def create_app() -> Flask:
    """Application factory for WSGI servers and tests."""
    env_name = os.getenv("FLASK_ENV", "default")
    config_class = CONFIG_MAPPING.get(env_name, CONFIG_MAPPING["default"])

    app = Flask(__name__)
    app.config.from_object(config_class)
    _validate_security_config(app, env_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    CORS(
        app,
        resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    model_service = ModelService(app.config)
    try:
        model_service.load_models()
    except Exception:
        logging.getLogger(__name__).exception(
            "Model loading failed at startup. API will run, but prediction endpoints may return 500 until fixed."
        )
    app.extensions["model_service"] = model_service
    app.extensions["rate_limiter"] = RateLimiter(app.config)

    with app.app_context():
        init_user_table()

    app.register_blueprint(api_bp)

    register_error_handlers(app)
    return app


def register_error_handlers(app: Flask) -> None:
    """Sanitize runtime errors returned by the API."""

    @app.errorhandler(RequestEntityTooLarge)
    def payload_too_large(_: RequestEntityTooLarge):
        return jsonify({"error": "File is too large."}), 413

    @app.errorhandler(404)
    def not_found(_: Exception):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_: Exception):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(Exception)
    def unhandled_error(_: Exception):
        logging.getLogger(__name__).exception("Unhandled server error")
        return jsonify({"error": "Internal server error."}), 500


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
