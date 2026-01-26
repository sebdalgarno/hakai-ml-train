"""Core metric computation functions for segmentation evaluation."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import torch


def compute_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> np.ndarray:
    """Compute confusion matrix from predictions and labels.

    Args:
        preds: Predicted class indices (N, H, W)
        labels: Ground truth class indices (N, H, W)
        num_classes: Number of classes
        ignore_index: Label value to ignore

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    mask = labels != ignore_index
    preds_flat = preds[mask].flatten().cpu().numpy()
    labels_flat = labels[mask].flatten().cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(preds_flat, labels_flat, strict=True):
        if 0 <= pred < num_classes and 0 <= label < num_classes:
            cm[label, pred] += 1

    return cm


def compute_pr_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    target_class: int = 1,
    ignore_index: int = -100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute precision-recall curve for a target class.

    Args:
        probs: Probability scores for target class (flattened)
        labels: Ground truth labels (flattened)
        target_class: Class index to compute PR curve for
        ignore_index: Label value to ignore

    Returns:
        Tuple of (precisions, recalls, thresholds, auprc)
    """
    mask = labels != ignore_index
    probs = probs[mask]
    labels = labels[mask]

    binary_labels = (labels == target_class).astype(np.int32)

    # Sort by descending probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = binary_labels[sorted_indices]

    # Compute cumulative TP and FP
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)

    # Total positives
    total_positives = sorted_labels.sum()
    if total_positives == 0:
        return np.array([1.0]), np.array([0.0]), np.array([1.0]), 0.0

    # Precision and recall at each threshold
    precisions = tps / (tps + fps + 1e-10)
    recalls = tps / total_positives

    # Find unique thresholds (sample at most 1000 points for efficiency)
    n_points = min(1000, len(sorted_probs))
    indices = np.linspace(0, len(sorted_probs) - 1, n_points, dtype=int)

    precisions = precisions[indices]
    recalls = recalls[indices]
    thresholds = sorted_probs[indices]

    # Add endpoint (recall=0, precision=1)
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])
    thresholds = np.concatenate([[1.0], thresholds])

    # Compute AUPRC using trapezoidal rule
    auprc = -np.trapz(precisions, recalls)

    return precisions, recalls, thresholds, auprc


def compute_per_sample_metrics(
    pred: np.ndarray, label: np.ndarray, target_class: int = 1, ignore_index: int = -100
) -> dict[str, float]:
    """Compute IoU, precision, recall for a single sample.

    Args:
        pred: Predicted class indices (H, W)
        label: Ground truth class indices (H, W)
        target_class: Class index to compute metrics for
        ignore_index: Label value to ignore

    Returns:
        Dictionary with iou, precision, recall values
    """
    mask = label != ignore_index

    pred_mask = (pred == target_class) & mask
    label_mask = (label == target_class) & mask

    intersection = (pred_mask & label_mask).sum()
    union = (pred_mask | label_mask).sum()
    pred_positive = pred_mask.sum()
    label_positive = label_mask.sum()

    # IoU
    iou = (1.0 if intersection == 0 else 0.0) if union == 0 else intersection / union

    # Precision
    if pred_positive == 0:
        precision = 1.0 if label_positive == 0 else 0.0
    else:
        precision = intersection / pred_positive

    # Recall
    recall = 1.0 if label_positive == 0 else intersection / label_positive

    return {"iou": float(iou), "precision": float(precision), "recall": float(recall)}


def extract_ortho_from_filename(filename: str) -> str:
    """Extract ortho name from chip filename.

    Examples:
        'calmus_u0421_x123_y456.npz' -> 'calmus_u0421'
        'bennett_bay_x50_y100.npz' -> 'bennett_bay'
    """
    # Match pattern: name_xNNN_yNNN.npz
    match = re.match(r"^(.+?)_x\d+_y\d+\.npz$", filename)
    if match:
        return match.group(1)
    return filename.replace(".npz", "")


def compute_global_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    target_class: int = 1,
    ignore_index: int = -100,
) -> dict[str, float]:
    """Compute global/micro-averaged metrics (matches torchmetrics behavior).

    This accumulates TP/FP/FN across all samples before computing metrics,
    which is how torchmetrics computes IoU during training.
    """
    mask = labels != ignore_index

    pred_positive = (preds == target_class) & mask
    label_positive = (labels == target_class) & mask

    tp = (pred_positive & label_positive).sum()
    fp = (pred_positive & ~label_positive).sum()
    fn = (~pred_positive & label_positive).sum()

    # IoU = TP / (TP + FP + FN)
    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0

    # Precision = TP / (TP + FP)
    pred_total = tp + fp
    precision = tp / pred_total if pred_total > 0 else 1.0

    # Recall = TP / (TP + FN)
    label_total = tp + fn
    recall = tp / label_total if label_total > 0 else 1.0

    # F1
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_per_ortho_metrics(
    filenames: list[str],
    metrics_per_sample: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics by ortho.

    Args:
        filenames: List of chip filenames
        metrics_per_sample: List of metric dicts per sample

    Returns:
        Dictionary mapping ortho name to aggregated metrics
    """
    ortho_data = defaultdict(lambda: {"ious": [], "precisions": [], "recalls": []})

    for filename, metrics in zip(filenames, metrics_per_sample, strict=True):
        ortho = extract_ortho_from_filename(filename)
        ortho_data[ortho]["ious"].append(metrics["iou"])
        ortho_data[ortho]["precisions"].append(metrics["precision"])
        ortho_data[ortho]["recalls"].append(metrics["recall"])

    result = {}
    for ortho, data in ortho_data.items():
        result[ortho] = {
            "iou": float(np.mean(data["ious"])),
            "precision": float(np.mean(data["precisions"])),
            "recall": float(np.mean(data["recalls"])),
            "n_chips": len(data["ious"]),
        }

    return result
