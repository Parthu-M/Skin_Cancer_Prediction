from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_FILENAME = "dermalens-ham10000-mobilenetv3.onnx"
VERSION = "ham10000-mnv3-v1"
MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(metrics: dict[str, Any], model_path: Path) -> dict[str, Any]:
    evaluation = metrics["test"]
    split = metrics["split"]
    limitations = list(metrics["limitations"])
    closed_set_limitation = (
        "The closed seven-class taxonomy omits many skin conditions "
        "and cannot identify an unknown class."
    )
    if closed_set_limitation not in limitations:
        limitations.append(closed_set_limitation)
    classes = metrics["model"]["classes"]
    confidence = metrics["model"]["confidence_policy"]

    return {
        "schema_version": 1,
        "artifact": {
            "version": VERSION,
            "filename": MODEL_FILENAME,
            "format": "ONNX",
            "sha256": sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "preprocessing": {
            "resize_shorter_edge": 256,
            "center_crop": 224,
            "color_mode": "RGB",
            "scale": "0..1",
            "mean": MEAN,
            "standard_deviation": STANDARD_DEVIATION,
        },
        "calibration": {
            "method": "validation temperature scaling",
            "temperature": metrics["model"]["temperature"],
            "abstention_threshold": confidence["threshold"],
            "validation_coverage": confidence["coverage"],
            "validation_selective_accuracy": confidence["selective_accuracy"],
        },
        "classes": classes,
        "model_card": {
            "name": "Dermalens HAM10000 MobileNetV3",
            "architecture": metrics["model"]["architecture"],
            "status": "validated_research_model",
            "inference_enabled": True,
            "input_shape": "224 × 224 RGB",
            "dataset": {
                "name": metrics["dataset"]["name"],
                "doi": metrics["dataset"]["doi"],
                "license": metrics["dataset"]["license"],
                "images": metrics["dataset"]["images"],
                "unique_lesions": metrics["dataset"]["unique_lesions"],
            },
            "split": {
                name: {
                    "images": split[name]["images"],
                    "lesions": split[name]["lesions"],
                }
                for name in ["train", "validation", "test"]
            },
            "evaluation": {
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "weighted_f1": evaluation["weighted_f1"],
                "expected_calibration_error": evaluation["expected_calibration_error"],
                "test_images": split["test"]["images"],
            },
            "classes": classes,
            "intended_use": (
                "Non-commercial portfolio and machine-learning research "
                "demonstration on dermoscopic imagery."
            ),
            "limitations": limitations,
            "not_for": [
                "Diagnosis, screening, triage, or treatment decisions",
                "Clinical deployment or commercial use",
                "Phone photographs or non-dermoscopic clinical images",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    metrics_path = args.models_dir / "metrics.json"
    model_path = args.models_dir / MODEL_FILENAME
    if not metrics_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Training metrics and ONNX model are required.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = build_config(metrics, model_path)
    output_path = args.models_dir / "model-config.json"
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {output_path} with SHA-256 {config['artifact']['sha256']}")


if __name__ == "__main__":
    main()
