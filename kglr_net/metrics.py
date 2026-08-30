"""Segmentation metrics used by KGLR-Net evaluation scripts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


@dataclass(frozen=True)
class SegmentationMetrics:
    """Region and boundary metrics for one binary segmentation mask."""

    dice: float
    iou: float
    recall: float
    precision: float
    hd95: float


def to_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a probability map or mask to a two-dimensional binary array."""

    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask after squeezing, got shape {array.shape}.")
    return (array > threshold).astype(np.uint8)


def confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Return true-positive, false-positive, false-negative, and true-negative counts."""

    pred = to_binary(prediction, threshold)
    truth = to_binary(target, 0.5)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {truth.shape}.")
    tp = float(np.logical_and(pred == 1, truth == 1).sum())
    fp = float(np.logical_and(pred == 1, truth == 0).sum())
    fn = float(np.logical_and(pred == 0, truth == 1).sum())
    tn = float(np.logical_and(pred == 0, truth == 0).sum())
    return tp, fp, fn, tn


def _surface(mask: np.ndarray) -> np.ndarray:
    binary = to_binary(mask).astype(bool)
    if not binary.any():
        return binary
    eroded = binary_erosion(binary, structure=np.ones((3, 3)), border_value=0)
    return np.logical_xor(binary, eroded)


def hd95(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute the symmetric 95th-percentile Hausdorff distance in pixels.

    If exactly one mask is empty, the image diagonal is returned. If both masks
    are empty, the distance is zero.
    """

    pred = to_binary(prediction, threshold).astype(bool)
    truth = to_binary(target, 0.5).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {truth.shape}.")
    if not pred.any() and not truth.any():
        return 0.0

    height, width = truth.shape
    empty_mask_penalty = float(math.hypot(height, width))
    if not pred.any() or not truth.any():
        return empty_mask_penalty

    pred_surface = _surface(pred)
    truth_surface = _surface(truth)
    if not pred_surface.any() or not truth_surface.any():
        return empty_mask_penalty

    distance_to_truth = distance_transform_edt(~truth_surface)[pred_surface]
    distance_to_pred = distance_transform_edt(~pred_surface)[truth_surface]
    distances = np.concatenate((distance_to_truth, distance_to_pred)).astype(np.float64)
    return float(np.percentile(distances, 95))


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> SegmentationMetrics:
    """Compute Dice, IoU, recall, precision, and HD95 for one image."""

    tp, fp, fn, _ = confusion_counts(prediction, target, threshold)
    return SegmentationMetrics(
        dice=float((2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)),
        iou=float((tp + epsilon) / (tp + fp + fn + epsilon)),
        recall=float((tp + epsilon) / (tp + fn + epsilon)),
        precision=float((tp + epsilon) / (tp + fp + epsilon)),
        hd95=hd95(prediction, target, threshold),
    )


def metrics_to_dict(metrics: SegmentationMetrics) -> Dict[str, float]:
    """Convert a metric record to a plain dictionary."""

    return asdict(metrics)


def summarize_rows(
    rows: Sequence[Mapping[str, float]],
    excluded_keys: Iterable[str] = ("file", "name"),
) -> Dict[str, float]:
    """Return sample mean and standard deviation for numeric metric columns."""

    if not rows:
        raise ValueError("Cannot summarize an empty metric collection.")
    excluded = set(excluded_keys)
    keys = [key for key in rows[0] if key not in excluded]
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


__all__ = [
    "SegmentationMetrics",
    "compute_segmentation_metrics",
    "confusion_counts",
    "hd95",
    "metrics_to_dict",
    "summarize_rows",
    "to_binary",
]
