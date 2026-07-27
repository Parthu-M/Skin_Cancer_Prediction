from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import model_runtime
from model_runtime import ModelRuntimeError, SkinLesionClassifier

CLASSES = [
    {"code": "akiec", "name": "Actinic keratoses"},
    {"code": "bcc", "name": "Basal cell carcinoma"},
    {"code": "bkl", "name": "Benign keratosis-like lesions"},
    {"code": "df", "name": "Dermatofibroma"},
    {"code": "mel", "name": "Melanoma"},
    {"code": "nv", "name": "Melanocytic nevi"},
    {"code": "vasc", "name": "Vascular lesions"},
]


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, inputs):
        assert output_names == ["logits"]
        assert inputs["image"].shape == (1, 3, 224, 224)
        return [
            np.asarray(
                [[-2.0, -1.0, 0.0, 0.5, 3.0, 0.25, -0.5]],
                dtype=np.float32,
            )
        ]


def write_config(tmp_path, artifact, *, sha256=None):
    config = {
        "artifact": {
            "version": "test-v1",
            "sha256": sha256 or hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "standard_deviation": [0.229, 0.224, 0.225],
        },
        "calibration": {
            "temperature": 1.0,
            "abstention_threshold": 0.5,
        },
        "classes": CLASSES,
        "model_card": {"status": "validated_research_model"},
    }
    path = tmp_path / "model-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_classifier_verifies_artifact_and_returns_ranked_estimate(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"versioned-model")
    config = write_config(tmp_path, artifact)
    monkeypatch.setattr(
        model_runtime.ort,
        "InferenceSession",
        lambda *_args, **_kwargs: FakeSession(),
    )

    classifier = SkinLesionClassifier(artifact, config)
    result = classifier.predict(Image.new("RGB", (640, 480), "salmon"))

    assert classifier.card["status"] == "validated_research_model"
    assert result["performed"] is True
    assert result["status"] == "research_estimate"
    assert result["top_prediction"]["code"] == "mel"
    assert len(result["ranked_predictions"]) == 3
    assert 0 <= result["normalized_entropy"] <= 1


def test_classifier_rejects_artifact_checksum_mismatch(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"unexpected-model")
    config = write_config(tmp_path, artifact, sha256="0" * 64)

    with pytest.raises(ModelRuntimeError, match="checksum"):
        SkinLesionClassifier(artifact, config)


def test_versioned_model_artifact_loads_and_runs_inference():
    classifier = SkinLesionClassifier()

    result = classifier.predict(Image.new("RGB", (640, 480), "salmon"))

    assert classifier.card["status"] == "validated_research_model"
    assert classifier.card["evaluation"]["test_images"] == 2004
    assert result["performed"] is True
    assert result["model_version"] == "ham10000-mnv3-v1"
    assert result["status"] in {
        "research_estimate",
        "abstained_low_confidence",
    }
    assert result["top_prediction"]["code"] in {item["code"] for item in CLASSES}
    assert len(result["ranked_predictions"]) == 3
