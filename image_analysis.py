from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
from typing import Any, BinaryIO

from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class AnalysisError(ValueError):
    """Raised when an upload is not safe or useful for analysis."""


@dataclass(frozen=True)
class InspectedImage:
    report: dict[str, Any]
    image: Image.Image


def _score(value: float, minimum: float, maximum: float) -> int:
    if maximum <= minimum:
        return 0
    return int(round(max(0, min(1, (value - minimum) / (maximum - minimum))) * 100))


def inspect_image(
    stream: BinaryIO,
    *,
    filename: str,
    declared_mime: str,
) -> InspectedImage:
    if not filename or PurePath(filename).name != filename:
        raise AnalysisError("Use a simple image filename without folder paths.")
    if declared_mime not in ALLOWED_MIME:
        raise AnalysisError("Only JPEG, PNG, and WebP images are accepted.")

    content = stream.read()
    if not content:
        raise AnalysisError("The selected image is empty.")
    if len(content) > 8 * 1024 * 1024:
        raise AnalysisError("Image exceeds the 8 MB upload limit.")

    try:
        with Image.open(BytesIO(content)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(content)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise AnalysisError("The file content is not JPEG, PNG, or WebP.")
            actual_format = source.format
            width, height = source.size
            if width * height > MAX_PIXELS:
                raise AnalysisError("Image dimensions exceed the 20 megapixel limit.")
            if width < 96 or height < 96:
                raise AnalysisError("Image must be at least 96 × 96 pixels.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise AnalysisError("The file is not a readable image.") from exc
    except Image.DecompressionBombError as exc:
        raise AnalysisError(
            "Image dimensions are too large to process safely."
        ) from exc

    analysis_image = image.copy()
    analysis_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(analysis_image)
    luminance = ImageStat.Stat(gray)
    brightness = float(luminance.mean[0])
    contrast = float(luminance.stddev[0])
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = float(ImageStat.Stat(edges).mean[0])
    rgb_means = ImageStat.Stat(analysis_image).mean

    checks = [
        {
            "key": "resolution",
            "label": "Resolution",
            "passed": min(width, height) >= 224,
            "detail": f"{width} × {height} pixels",
        },
        {
            "key": "exposure",
            "label": "Exposure",
            "passed": 42 <= brightness <= 220,
            "detail": (
                "Within review range"
                if 42 <= brightness <= 220
                else (
                    "Image may be too dark"
                    if brightness < 42
                    else "Image may be too bright"
                )
            ),
        },
        {
            "key": "contrast",
            "label": "Contrast",
            "passed": contrast >= 22,
            "detail": (
                "Sufficient tonal separation"
                if contrast >= 22
                else "Low tonal separation"
            ),
        },
        {
            "key": "sharpness",
            "label": "Edge detail",
            "passed": sharpness >= 6,
            "detail": (
                "Usable edge detail"
                if sharpness >= 6
                else "Image may be soft or blurred"
            ),
        },
    ]
    passed_count = sum(bool(check["passed"]) for check in checks)

    return InspectedImage(
        image=image,
        report={
            "readiness": (
                "ready_for_research_pipeline"
                if passed_count == len(checks)
                else "review_recommended"
            ),
            "passed_checks": passed_count,
            "total_checks": len(checks),
            "file": {
                "name": filename,
                "format": actual_format,
                "mime": declared_mime,
                "size_bytes": len(content),
                "width": width,
                "height": height,
            },
            "metrics": {
                "brightness": _score(brightness, 0, 255),
                "contrast": _score(contrast, 0, 90),
                "edge_detail": _score(sharpness, 0, 35),
                "average_rgb": [round(float(value), 1) for value in rgb_means],
            },
            "checks": checks,
        },
    )


def analyze_image(
    stream: BinaryIO,
    *,
    filename: str,
    declared_mime: str,
) -> dict[str, Any]:
    """Return the quality report without exposing the decoded image."""
    return inspect_image(
        stream,
        filename=filename,
        declared_mime=declared_mime,
    ).report
