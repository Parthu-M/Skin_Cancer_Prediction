from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

SEED = 42
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    logits: np.ndarray
    targets: np.ndarray


class HamDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image_path = self.images_dir / f"{row.image_id}.jpg"
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Dermalens HAM10000 classifier."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("models"), type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--head-epochs", default=3, type=int)
    parser.add_argument("--fine-tune-epochs", default=12, type=int)
    parser.add_argument("--patience", default=4, type=int)
    parser.add_argument(
        "--evaluate-checkpoint",
        action="store_true",
        help="Skip training and evaluate the checkpoint in the output directory.",
    )
    parser.add_argument(
        "--history-log",
        type=Path,
        help=(
            "Training stdout log used to reconstruct history during "
            "checkpoint evaluation."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def sha256_lines(values: Iterable[str]) -> str:
    content = "\n".join(sorted(values)).encode()
    return hashlib.sha256(content).hexdigest()


def load_logged_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Training history log not found: {path}")

    pattern = re.compile(
        r"^(?P<stage>head|fine_tune) "
        r"(?P<epoch>\d+)/\d+ "
        r"train_loss=(?P<train_loss>\d+\.\d+) "
        r"val_bal_acc=(?P<balanced_accuracy>\d+\.\d+) "
        r"val_macro_f1=(?P<macro_f1>\d+\.\d+) "
        r"seconds=(?P<seconds>\d+\.\d+)$"
    )
    history: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        history.append(
            {
                "stage": match["stage"],
                "epoch": int(match["epoch"]),
                "duration_seconds": float(match["seconds"]),
                "train_loss": float(match["train_loss"]),
                "validation_balanced_accuracy": float(match["balanced_accuracy"]),
                "validation_macro_f1": float(match["macro_f1"]),
            }
        )
    if not history:
        raise RuntimeError("No epoch records were found in the training log.")
    return history


def validate_and_load_data(
    images_dir: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    frame = pd.read_csv(metadata_path, sep="\t")
    required = {
        "lesion_id",
        "image_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if len(frame) != 10015:
        raise ValueError(f"Expected 10,015 metadata rows, received {len(frame):,}")
    if set(frame.dx.unique()) != set(CLASS_CODES):
        raise ValueError(f"Unexpected class codes: {sorted(frame.dx.unique())}")
    if frame.image_id.duplicated().any():
        raise ValueError("Metadata contains duplicate image IDs.")

    missing_images = [
        image_id
        for image_id in frame.image_id
        if not (images_dir / f"{image_id}.jpg").is_file()
    ]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images):,} image files are missing. First IDs: {preview}"
        )

    frame = frame.copy()
    frame["target"] = frame.dx.map({code: i for i, code in enumerate(CLASS_CODES)})
    return frame


def build_grouped_splits(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    train_validation_idx, test_idx = next(
        outer.split(frame, frame.target, groups=frame.lesion_id)
    )
    train_validation = frame.iloc[train_validation_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    inner = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED + 1,
    )
    train_idx, validation_idx = next(
        inner.split(
            train_validation,
            train_validation.target,
            groups=train_validation.lesion_id,
        )
    )
    train = train_validation.iloc[train_idx].reset_index(drop=True)
    validation = train_validation.iloc[validation_idx].reset_index(drop=True)

    split_groups = {
        "train": set(train.lesion_id),
        "validation": set(validation.lesion_id),
        "test": set(test.lesion_id),
    }
    if split_groups["train"] & split_groups["validation"]:
        raise RuntimeError("Train and validation lesion groups overlap.")
    if split_groups["train"] & split_groups["test"]:
        raise RuntimeError("Train and test lesion groups overlap.")
    if split_groups["validation"] & split_groups["test"]:
        raise RuntimeError("Validation and test lesion groups overlap.")

    return train, validation, test


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.72, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def build_loaders(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    images_dir: Path,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, evaluation_transform = build_transforms()
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        HamDataset(train, images_dir, train_transform),
        shuffle=True,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        HamDataset(validation, images_dir, evaluation_transform),
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        HamDataset(test, images_dir, evaluation_transform),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, test_loader


def build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_CODES))
    return model


def class_weights(train: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train.target.value_counts().sort_index().to_numpy(dtype=np.float64)
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def summarize_logits(
    loss: float,
    logits: np.ndarray,
    targets: np.ndarray,
) -> EpochResult:
    predictions = logits.argmax(axis=1)
    return EpochResult(
        loss=loss,
        accuracy=float(accuracy_score(targets, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(targets, predictions)),
        macro_f1=float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        logits=logits,
        targets=targets,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()

        total_loss += float(loss.item()) * len(targets)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    logits_array = np.concatenate(all_logits)
    targets_array = np.concatenate(all_targets)
    return summarize_logits(
        total_loss / len(loader.dataset),
        logits_array,
        targets_array,
    )


def train_stage(
    *,
    stage: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    best_path: Path,
    best_score: tuple[float, float],
    history: list[dict[str, Any]],
) -> tuple[tuple[float, float], int]:
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        scheduler.step()
        score = (
            validation_result.balanced_accuracy,
            validation_result.macro_f1,
        )
        duration = time.perf_counter() - started
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "duration_seconds": round(duration, 2),
                "learning_rate": optimizer.param_groups[-1]["lr"],
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "train_balanced_accuracy": train_result.balanced_accuracy,
                "train_macro_f1": train_result.macro_f1,
                "validation_loss": validation_result.loss,
                "validation_accuracy": validation_result.accuracy,
                "validation_balanced_accuracy": (validation_result.balanced_accuracy),
                "validation_macro_f1": validation_result.macro_f1,
            }
        )
        print(
            f"{stage} {epoch:02d}/{epochs:02d} "
            f"train_loss={train_result.loss:.4f} "
            f"val_bal_acc={validation_result.balanced_accuracy:.4f} "
            f"val_macro_f1={validation_result.macro_f1:.4f} "
            f"seconds={duration:.1f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping {stage} after {epoch} epochs.")
                return best_score, epoch

    return best_score, epochs


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            accuracy = (predictions[selected] == targets[selected]).mean()
            confidence = confidences[selected].mean()
            error += selected.mean() * abs(accuracy - confidence)
    return float(error)


def calibrate_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    candidates = np.linspace(0.5, 5.0, 901)
    losses = [
        log_loss(targets, softmax(logits, temperature), labels=range(7))
        for temperature in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def choose_confidence_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    fallback: dict[str, float] | None = None
    for threshold in np.linspace(0.35, 0.9, 56):
        selected = confidences >= threshold
        coverage = float(selected.mean())
        if coverage < 0.25:
            continue
        selective_accuracy = float((predictions[selected] == targets[selected]).mean())
        candidate = {
            "threshold": float(threshold),
            "coverage": coverage,
            "selective_accuracy": selective_accuracy,
        }
        fallback = candidate
        if selective_accuracy >= 0.75:
            return candidate
    return fallback or {
        "threshold": 0.65,
        "coverage": 0.0,
        "selective_accuracy": 0.0,
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, Any]:
    evaluation = frame.reset_index(drop=True).copy()
    evaluation["prediction"] = predictions
    evaluation["age_group"] = pd.cut(
        evaluation.age,
        bins=[0, 39, 59, 200],
        labels=["under_40", "40_to_59", "60_plus"],
    ).astype("string")
    output: dict[str, Any] = {}
    for column in ["sex", "age_group"]:
        output[column] = {}
        for value, group in evaluation.groupby(column, dropna=False):
            if len(group) < 25:
                continue
            output[column][str(value)] = {
                "support": int(len(group)),
                "accuracy": float(accuracy_score(group.target, group.prediction)),
                "balanced_accuracy": float(
                    recall_score(
                        group.target,
                        group.prediction,
                        labels=sorted(group.target.unique()),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
    return output


def plot_artifacts(
    output_dir: Path,
    history: list[dict[str, Any]],
    test_targets: np.ndarray,
    test_predictions: np.ndarray,
) -> None:
    sns.set_theme(style="whitegrid")
    history_frame = pd.DataFrame(history)
    epochs = np.arange(1, len(history_frame) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history_frame.train_loss, label="Train")
    if (
        "validation_loss" in history_frame
        and history_frame.validation_loss.notna().any()
    ):
        axes[0].plot(epochs, history_frame.validation_loss, label="Validation")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        history_frame.validation_balanced_accuracy,
        label="Balanced accuracy",
    )
    axes[1].plot(
        epochs,
        history_frame.validation_macro_f1,
        label="Macro-F1",
    )
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training-history.svg", bbox_inches="tight")
    plt.close(figure)

    matrix = confusion_matrix(
        test_targets,
        test_predictions,
        labels=range(len(CLASS_CODES)),
        normalize="true",
    )
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="crest",
        xticklabels=CLASS_CODES,
        yticklabels=CLASS_CODES,
        vmin=0,
        vmax=1,
        ax=axis,
    )
    axis.set(
        title="Normalized test confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion-matrix.svg", bbox_inches="tight")
    plt.close(figure)


def export_and_verify_onnx(
    model: nn.Module,
    output_path: Path,
    test_loader: DataLoader,
) -> float:
    model.eval()
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images[:4]
    model_cpu = model.cpu().eval()
    with torch.inference_mode():
        torch_output = model_cpu(sample_images).numpy()

    torch.onnx.export(
        model_cpu,
        torch.zeros(1, 3, 224, 224),
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(
        ["logits"],
        {"image": sample_images.cpu().numpy()},
    )[0]
    difference = float(np.max(np.abs(torch_output - onnx_output)))
    if difference > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max difference {difference}")
    return difference


def serializable_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            output[key] = serializable_report(value)
        elif isinstance(value, (np.floating, float)):
            output[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            output[key] = int(value)
        else:
            output[key] = value
    return output


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)

    frame = validate_and_load_data(args.images_dir, args.metadata)
    train, validation, test = build_grouped_splits(frame)
    train_loader, validation_loader, test_loader = build_loaders(
        train,
        validation,
        test,
        args.images_dir,
        args.batch_size,
        args.workers,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train, device),
        label_smoothing=0.05,
    )
    best_path = output_dir / "best-mobilenet-v3-small.pt"
    history = load_logged_history(args.history_log) if args.evaluate_checkpoint else []
    if args.evaluate_checkpoint:
        if not best_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        print(f"Evaluating saved checkpoint: {best_path}", flush=True)
    else:
        best_score = (-math.inf, -math.inf)

        for parameter in model.features.parameters():
            parameter.requires_grad = False
        head_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer,
            T_max=max(args.head_epochs, 1),
        )
        best_score, _ = train_stage(
            stage="head",
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=head_optimizer,
            scheduler=head_scheduler,
            device=device,
            epochs=args.head_epochs,
            patience=args.patience,
            best_path=best_path,
            best_score=best_score,
            history=history,
        )

        for parameter in model.features.parameters():
            parameter.requires_grad = True
        fine_tune_optimizer = torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": 8e-5},
                {"params": model.classifier.parameters(), "lr": 2.5e-4},
            ],
            weight_decay=1e-4,
        )
        fine_tune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            fine_tune_optimizer,
            T_max=max(args.fine_tune_epochs, 1),
        )
        best_score, _ = train_stage(
            stage="fine_tune",
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=fine_tune_optimizer,
            scheduler=fine_tune_scheduler,
            device=device,
            epochs=args.fine_tune_epochs,
            patience=args.patience,
            best_path=best_path,
            best_score=best_score,
            history=history,
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation_result = run_epoch(
        model,
        validation_loader,
        criterion,
        device,
    )
    test_result = run_epoch(model, test_loader, criterion, device)
    temperature = calibrate_temperature(
        validation_result.logits,
        validation_result.targets,
    )
    validation_probabilities = softmax(
        validation_result.logits,
        temperature,
    )
    test_probabilities = softmax(test_result.logits, temperature)
    test_predictions = test_probabilities.argmax(axis=1)
    threshold = choose_confidence_threshold(
        validation_probabilities,
        validation_result.targets,
    )
    test_selected = test_probabilities.max(axis=1) >= threshold["threshold"]
    selective_test = {
        "coverage": float(test_selected.mean()),
        "accuracy": (
            float(
                (
                    test_predictions[test_selected]
                    == test_result.targets[test_selected]
                ).mean()
            )
            if test_selected.any()
            else 0.0
        ),
    }

    onnx_path = output_dir / "dermalens-ham10000-mobilenetv3.onnx"
    onnx_difference = export_and_verify_onnx(
        model,
        onnx_path,
        test_loader,
    )
    plot_artifacts(
        output_dir,
        history,
        test_result.targets,
        test_predictions,
    )

    class_report = classification_report(
        test_result.targets,
        test_predictions,
        labels=range(len(CLASS_CODES)),
        target_names=CLASS_CODES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "HAM10000",
            "doi": "10.7910/DVN/DBW86T",
            "license": "CC BY-NC 4.0",
            "images": len(frame),
            "unique_lesions": int(frame.lesion_id.nunique()),
        },
        "split": {
            name: {
                "images": len(split),
                "lesions": int(split.lesion_id.nunique()),
                "image_ids_sha256": sha256_lines(split.image_id),
                "class_counts": {
                    code: int((split.dx == code).sum()) for code in CLASS_CODES
                },
            }
            for name, split in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        },
        "model": {
            "architecture": "MobileNetV3-Small",
            "initialization": "TorchVision ImageNet-1K weights",
            "input": "224x224 RGB dermoscopic image",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "classes": [
                {"code": code, "name": CLASS_NAMES[code]} for code in CLASS_CODES
            ],
            "temperature": temperature,
            "confidence_policy": threshold,
            "onnx_max_absolute_difference": onnx_difference,
            "onnx_bytes": onnx_path.stat().st_size,
        },
        "validation": {
            "accuracy": validation_result.accuracy,
            "balanced_accuracy": validation_result.balanced_accuracy,
            "macro_f1": validation_result.macro_f1,
            "calibration_error_before": expected_calibration_error(
                softmax(validation_result.logits),
                validation_result.targets,
            ),
            "calibration_error_after": expected_calibration_error(
                validation_probabilities,
                validation_result.targets,
            ),
        },
        "test": {
            "accuracy": float(accuracy_score(test_result.targets, test_predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    test_result.targets,
                    test_predictions,
                )
            ),
            "macro_f1": float(
                f1_score(
                    test_result.targets,
                    test_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    test_result.targets,
                    test_predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "log_loss": float(
                log_loss(
                    test_result.targets,
                    test_probabilities,
                    labels=range(7),
                )
            ),
            "expected_calibration_error": expected_calibration_error(
                test_probabilities,
                test_result.targets,
            ),
            "selective": selective_test,
            "per_class": serializable_report(class_report),
            "subgroups": subgroup_metrics(test, test_predictions),
        },
        "training": {
            "seed": SEED,
            "history": history,
            "device": str(device),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
            "checkpoint_evaluation_only": args.evaluate_checkpoint,
        },
        "limitations": [
            "Research use only; not a medical device or diagnosis.",
            "Validated only on retrospective HAM10000 dermoscopic images.",
            "Clinical photographs and other acquisition types are out of scope.",
            (
                "The closed seven-class taxonomy omits many skin conditions "
                "and cannot identify an unknown class."
            ),
            "Performance may not generalize across devices, sites, or populations.",
            "Rare classes have substantially less training support.",
            "Subgroup slices are descriptive and do not establish fairness.",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    best_path.unlink()

    print(
        "Final test metrics: "
        f"accuracy={metrics['test']['accuracy']:.4f} "
        f"balanced_accuracy={metrics['test']['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['test']['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Model exported to {onnx_path}", flush=True)


if __name__ == "__main__":
    main()
