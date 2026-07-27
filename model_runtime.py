from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "dermalens-ham10000-mobilenetv3.onnx"
DEFAULT_CONFIG_PATH = ROOT / "models" / "model-config.json"


class ModelRuntimeError(RuntimeError):
    """Raised when the versioned research model cannot be loaded safely."""


def _configured_path(variable: str, default: Path) -> Path:
    configured = Path(os.getenv(variable, str(default)))
    return configured if configured.is_absolute() else ROOT / configured


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_and_center_crop(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < height:
        resized_width = 256
        resized_height = int(256 * height / width)
    else:
        resized_height = 256
        resized_width = int(256 * width / height)
    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    left = (resized_width - 224) // 2
    top = (resized_height - 224) // 2
    return resized.crop((left, top, left + 224, top + 224))


def _preprocess(
    image: Image.Image,
    mean: list[float],
    standard_deviation: list[float],
) -> np.ndarray:
    prepared = _resize_and_center_crop(image)
    values = np.asarray(prepared, dtype=np.float32) / 255.0
    values = (values - np.asarray(mean)) / np.asarray(standard_deviation)
    return np.transpose(values, (2, 0, 1))[np.newaxis].astype(np.float32)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= np.max(scaled)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum()


class SkinLesionClassifier:
    """Versioned ONNX classifier with calibrated, abstention-aware output."""

    def __init__(
        self,
        model_path: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or _configured_path(
            "MODEL_PATH",
            DEFAULT_MODEL_PATH,
        )
        self.config_path = config_path or _configured_path(
            "MODEL_CONFIG_PATH",
            DEFAULT_CONFIG_PATH,
        )
        if not self.model_path.is_file():
            raise ModelRuntimeError(f"Model artifact not found: {self.model_path}")
        if not self.config_path.is_file():
            raise ModelRuntimeError(f"Model config not found: {self.config_path}")

        try:
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ModelRuntimeError("Model configuration is unreadable.") from exc

        expected_hash = self.config["artifact"]["sha256"]
        actual_hash = _sha256(self.model_path)
        if actual_hash != expected_hash:
            raise ModelRuntimeError(
                "Model checksum does not match the versioned configuration."
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(
            os.getenv("ONNX_INTRA_OP_THREADS", "2")
        )
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelRuntimeError("ONNX Runtime could not load the model.") from exc
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def card(self) -> dict[str, Any]:
        return self.config["model_card"]

    def predict(self, image: Image.Image) -> dict[str, Any]:
        preprocessing = self.config["preprocessing"]
        tensor = _preprocess(
            image,
            preprocessing["mean"],
            preprocessing["standard_deviation"],
        )
        logits = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0][0]
        temperature = float(self.config["calibration"]["temperature"])
        probabilities = _softmax(logits, temperature)
        ranked_indices = np.argsort(probabilities)[::-1]
        classes = self.config["classes"]
        ranked = [
            {
                "code": classes[int(index)]["code"],
                "name": classes[int(index)]["name"],
                "probability": round(float(probabilities[index]), 6),
            }
            for index in ranked_indices[:3]
        ]
        confidence = ranked[0]["probability"]
        threshold = float(self.config["calibration"]["abstention_threshold"])
        entropy = -float(
            np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
        ) / math.log(len(classes))

        return {
            "performed": True,
            "status": (
                "research_estimate"
                if confidence >= threshold
                else "abstained_low_confidence"
            ),
            "top_prediction": ranked[0],
            "ranked_predictions": ranked,
            "confidence_threshold": threshold,
            "normalized_entropy": round(entropy, 6),
            "model_version": self.config["artifact"]["version"],
            "acquisition_warning": (
                "This model was trained only on retrospective dermoscopic "
                "HAM10000 images. It is not validated for phone photographs, "
                "clinical close-ups, or real-world diagnosis."
            ),
        }


@lru_cache(maxsize=1)
def get_classifier() -> SkinLesionClassifier:
    return SkinLesionClassifier()
