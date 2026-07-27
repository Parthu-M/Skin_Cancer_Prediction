from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from image_analysis import AnalysisError, inspect_image
from model_runtime import (
    ModelRuntimeError,
    SkinLesionClassifier,
    get_classifier,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("dermalens")


def create_app(
    testing: bool = False,
    classifier: SkinLesionClassifier | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config.update(
        TESTING=testing,
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
    )

    def active_classifier() -> SkinLesionClassifier:
        return classifier or get_classifier()

    @app.after_request
    def set_security_headers(response: Any) -> Any:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self'; "
            "script-src 'self'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'none'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            model_card=active_classifier().card,
        )

    @app.get("/robots.txt")
    def robots() -> Any:
        return app.send_static_file("robots.txt")

    @app.get("/health")
    def health() -> tuple[Any, int]:
        try:
            model = active_classifier()
        except ModelRuntimeError:
            logger.exception("model_health_check_failed")
            return (
                jsonify(
                    {
                        "status": "unavailable",
                        "service": "dermalens",
                        "inference_enabled": False,
                    }
                ),
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        return (
            jsonify(
                {
                    "status": "healthy",
                    "service": "dermalens",
                    "inference_enabled": True,
                    "model_version": model.config["artifact"]["version"],
                }
            ),
            HTTPStatus.OK,
        )

    @app.get("/api/model-card")
    def model_card() -> tuple[Any, int]:
        return jsonify({"model": active_classifier().card}), HTTPStatus.OK

    @app.post("/api/analyze")
    def analyze() -> tuple[Any, int]:
        uploaded = request.files.get("image")
        if uploaded is None:
            return (
                jsonify(
                    {
                        "error": "missing_image",
                        "message": "Choose a JPEG, PNG, or WebP image.",
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )

        try:
            inspected = inspect_image(
                uploaded.stream,
                filename=uploaded.filename or "",
                declared_mime=uploaded.mimetype,
            )
            result = inspected.report
            if result["readiness"] == "ready_for_research_pipeline":
                inference = active_classifier().predict(inspected.image)
            else:
                inference = {
                    "performed": False,
                    "status": "quality_gate_failed",
                    "reason": (
                        "Inference was skipped because one or more image "
                        "quality checks require review."
                    ),
                }
            logger.info(
                (
                    "image_analysis_completed format=%s size=%sx%s "
                    "readiness=%s inference_status=%s"
                ),
                result["file"]["format"],
                result["file"]["width"],
                result["file"]["height"],
                result["readiness"],
                inference["status"],
            )
            return (
                jsonify(
                    {
                        "success": True,
                        "analysis": result,
                        "inference": inference,
                        "disclaimer": (
                            "Non-commercial research demonstration only. "
                            "Model output is not a diagnosis or medical advice."
                        ),
                    }
                ),
                HTTPStatus.OK,
            )
        except AnalysisError as exc:
            return (
                jsonify({"error": "invalid_image", "message": str(exc)}),
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except ModelRuntimeError:
            logger.exception("model_inference_failed")
            return (
                jsonify(
                    {
                        "error": "model_unavailable",
                        "message": ("The research model is temporarily unavailable."),
                    }
                ),
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

    @app.errorhandler(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    def request_too_large(_: Exception) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "error": "request_too_large",
                    "message": "Image exceeds the 8 MB upload limit.",
                }
            ),
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Any:
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "error": "http_error",
                        "message": error.description,
                    }
                ),
                error.code or HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return error

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception) -> tuple[Any, int]:
        logger.exception("unhandled_request_error", exc_info=error)
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "error": "internal_error",
                        "message": "The image could not be analyzed.",
                    }
                ),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return render_template("error.html"), HTTPStatus.INTERNAL_SERVER_ERROR

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
