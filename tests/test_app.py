from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image, ImageDraw

import app as app_module
from app import create_app
from model_runtime import ModelRuntimeError


class FakeClassifier:
    config = {"artifact": {"version": "test-v1"}}
    card = {
        "architecture": "MobileNetV3-Small",
        "status": "validated_research_model",
        "inference_enabled": True,
        "dataset": {"images": 10015},
        "evaluation": {
            "accuracy": 0.72,
            "balanced_accuracy": 0.61,
            "macro_f1": 0.6,
            "test_images": 2004,
        },
        "classes": [
            {"code": "akiec", "name": "Actinic keratoses"},
            {"code": "bcc", "name": "Basal cell carcinoma"},
            {"code": "bkl", "name": "Benign keratosis-like lesions"},
            {"code": "df", "name": "Dermatofibroma"},
            {"code": "mel", "name": "Melanoma"},
            {"code": "nv", "name": "Melanocytic nevi"},
            {"code": "vasc", "name": "Vascular lesions"},
        ],
    }

    def predict(self, _image):
        return {
            "performed": True,
            "status": "research_estimate",
            "top_prediction": {
                "code": "nv",
                "name": "Melanocytic nevi",
                "probability": 0.72,
            },
            "ranked_predictions": [
                {
                    "code": "nv",
                    "name": "Melanocytic nevi",
                    "probability": 0.72,
                }
            ],
            "confidence_threshold": 0.55,
            "normalized_entropy": 0.41,
            "model_version": "test-v1",
            "acquisition_warning": "Dermoscopic images only.",
        }


class BrokenClassifier(FakeClassifier):
    def predict(self, _image):
        raise ModelRuntimeError("test failure")


def image_bytes(
    *,
    size: tuple[int, int] = (512, 512),
    color: tuple[int, int, int] = (150, 90, 80),
    image_format: str = "PNG",
) -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, image_format)
    stream.seek(0)
    return stream


def quality_image_bytes() -> BytesIO:
    image = Image.new("RGB", (512, 512), (190, 125, 110))
    draw = ImageDraw.Draw(image)
    for offset in range(0, 512, 32):
        color = (75, 35, 45) if (offset // 32) % 2 else (235, 180, 150)
        draw.rectangle((offset, 0, offset + 16, 511), fill=color)
        draw.line((0, offset, 511, 511 - offset), fill=(45, 25, 35), width=5)
    stream = BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)
    return stream


@pytest.fixture()
def client():
    return create_app(
        testing=True,
        classifier=FakeClassifier(),
    ).test_client()


def test_unknown_route_preserves_not_found_status(client):
    response = client.get("/missing")

    assert response.status_code == 404


def test_home_is_accessible_and_documents_research_model(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"DERMALENS" in response.data
    assert b"MobileNetV3-Small" in response.data
    assert b"Balanced accuracy" in response.data
    assert b"Not a medical device" in response.data
    assert response.headers["X-Frame-Options"] == "DENY"


def test_health_reports_versioned_inference(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "service": "dermalens",
        "inference_enabled": True,
        "model_version": "test-v1",
    }


def test_health_fails_closed_when_model_cannot_load(monkeypatch):
    def unavailable():
        raise ModelRuntimeError("missing artifact")

    monkeypatch.setattr(app_module, "get_classifier", unavailable)
    client = create_app(testing=True).test_client()

    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json()["inference_enabled"] is False
    assert response.get_json()["status"] == "unavailable"


def test_robots_allows_public_crawlers(client):
    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert b"User-agent: *" in response.data


def test_model_card_documents_validated_research_model(client):
    response = client.get("/api/model-card")
    model = response.get_json()["model"]

    assert response.status_code == 200
    assert model["status"] == "validated_research_model"
    assert model["inference_enabled"] is True
    assert len(model["classes"]) == 7


def test_valid_image_returns_quality_report_and_research_estimate(client):
    response = client.post(
        "/api/analyze",
        data={"image": (quality_image_bytes(), "sample.png", "image/png")},
        content_type="multipart/form-data",
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["analysis"]["file"]["format"] == "PNG"
    assert data["analysis"]["file"]["width"] == 512
    assert len(data["analysis"]["checks"]) == 4
    assert data["analysis"]["readiness"] == "ready_for_research_pipeline"
    assert data["inference"]["performed"] is True
    assert data["inference"]["top_prediction"]["code"] == "nv"
    assert "not a diagnosis" in data["disclaimer"]
    assert response.headers["Cache-Control"] == "no-store"


def test_quality_gate_skips_inference_for_flat_image(client):
    response = client.post(
        "/api/analyze",
        data={"image": (image_bytes(), "flat.png", "image/png")},
        content_type="multipart/form-data",
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["analysis"]["readiness"] == "review_recommended"
    assert data["inference"]["performed"] is False
    assert data["inference"]["status"] == "quality_gate_failed"


def test_runtime_failure_returns_generic_service_unavailable():
    client = create_app(
        testing=True,
        classifier=BrokenClassifier(),
    ).test_client()

    response = client.post(
        "/api/analyze",
        data={"image": (quality_image_bytes(), "sample.png", "image/png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    assert response.get_json() == {
        "error": "model_unavailable",
        "message": "The research model is temporarily unavailable.",
    }


def test_missing_image_is_rejected(client):
    response = client.post("/api/analyze")

    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_image"


def test_fake_image_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"image": (BytesIO(b"not an image"), "fake.png", "image/png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "readable image" in response.get_json()["message"]


def test_wrong_mime_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"image": (image_bytes(), "sample.txt", "text/plain")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "JPEG, PNG, and WebP" in response.get_json()["message"]


def test_path_like_filename_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"image": (image_bytes(), "../sample.png", "image/png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "filename" in response.get_json()["message"]


def test_tiny_image_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={
            "image": (
                image_bytes(size=(64, 64)),
                "tiny.png",
                "image/png",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert "96 × 96" in response.get_json()["message"]


def test_request_size_is_bounded(client):
    response = client.post(
        "/api/analyze",
        data=b"x" * (8 * 1024 * 1024 + 1),
        content_type="application/octet-stream",
    )

    assert response.status_code == 413
    assert response.get_json()["error"] == "request_too_large"
