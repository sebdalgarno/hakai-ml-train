"""Generate PDF evaluation report for segmentation models."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import NpzSegmentationDataset
from src.evaluate.metrics import (
    compute_confusion_matrix,
    compute_per_ortho_metrics,
    compute_per_sample_metrics,
    compute_pr_curve,
    extract_ortho_from_filename,
)


class EvalDataset(NpzSegmentationDataset):
    """Dataset that also returns filename for each sample."""

    def __getitem__(self, idx):
        chip_path = self.chips[idx]
        data = np.load(chip_path)
        image = data["image"]
        label = data["label"]

        if self.transforms is not None:
            with torch.no_grad():
                augmented = self.transforms(image=image, mask=label)
                image = augmented["image"]
                label = augmented["mask"]

        return image, label, chip_path.name, data["image"]  # Also return raw image


def load_model(config_path: Path, ckpt_path: Path, device: torch.device):
    """Load model from config and checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    class_path = model_config["class_path"]
    init_args = model_config["init_args"]

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    model = model_class(**init_args)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    return model, config


def get_transforms(config: dict) -> A.Compose:
    """Extract test transforms from config."""
    test_transforms_dict = config["data"]["init_args"]["test_transforms"]
    return A.from_dict(test_transforms_dict)


def run_inference(
    model,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run inference on dataset and collect predictions."""
    all_probs = []
    all_preds = []
    all_labels = []
    all_filenames = []
    all_raw_images = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            images, labels, filenames, raw_images = batch
            images = images.to(device)

            logits = model(images)

            # Handle binary vs multiclass
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(1)
                probs_class1 = probs
            else:
                probs = torch.softmax(logits, dim=1)
                probs_class1 = probs[:, 1]  # Probability of target class

            preds = (probs_class1 > threshold).long()

            all_probs.append(probs_class1.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_filenames.extend(filenames)
            all_raw_images.append(raw_images.numpy())

    return {
        "probs": np.concatenate(all_probs),
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "filenames": all_filenames,
        "raw_images": np.concatenate(all_raw_images),
    }


def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    filenames: list[str],
    num_classes: int,
    target_class: int,
    ignore_index: int,
) -> dict:
    """Compute all metrics for evaluation."""
    # Per-sample metrics
    sample_metrics = []
    for i in range(len(preds)):
        metrics = compute_per_sample_metrics(
            preds[i], labels[i], target_class=target_class, ignore_index=ignore_index
        )
        sample_metrics.append(metrics)

    # Aggregate metrics
    ious = [m["iou"] for m in sample_metrics]
    precisions = [m["precision"] for m in sample_metrics]
    recalls = [m["recall"] for m in sample_metrics]

    # Confusion matrix
    preds_tensor = torch.from_numpy(preds)
    labels_tensor = torch.from_numpy(labels)
    cm = compute_confusion_matrix(
        preds_tensor, labels_tensor, num_classes=num_classes, ignore_index=ignore_index
    )

    # PR curve
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()
    pr_precisions, pr_recalls, pr_thresholds, auprc = compute_pr_curve(
        probs_flat, labels_flat, target_class=target_class, ignore_index=ignore_index
    )

    # Per-ortho metrics
    ortho_metrics = compute_per_ortho_metrics(filenames, sample_metrics)

    # F1 score
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-10)

    return {
        "sample_metrics": sample_metrics,
        "mean_iou": np.mean(ious),
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": f1,
        "confusion_matrix": cm,
        "pr_curve": {
            "precisions": pr_precisions,
            "recalls": pr_recalls,
            "thresholds": pr_thresholds,
            "auprc": auprc,
        },
        "ortho_metrics": ortho_metrics,
    }


def create_summary_page(
    config: dict,
    config_path: Path,
    ckpt_path: Path,
    val_metrics: dict,
    test_metrics: dict,
    class_names: list[str],
) -> plt.Figure:
    """Create summary page with model info and metrics table."""
    fig = plt.figure(figsize=(11, 8.5))

    # Title
    fig.suptitle("Model Evaluation Report", fontsize=16, fontweight="bold", y=0.95)

    # Model info section
    ax_info = fig.add_axes([0.1, 0.7, 0.8, 0.2])
    ax_info.axis("off")

    model_config = config["model"]["init_args"]
    info_text = (
        f"Config: {config_path.name}\n"
        f"Checkpoint: {ckpt_path.name}\n"
        f"Architecture: {model_config.get('architecture', 'N/A')}\n"
        f"Backbone: {model_config.get('backbone', 'N/A')}\n"
        f"Loss: {model_config.get('loss', 'N/A')}\n"
        f"Classes: {', '.join(class_names)}"
    )
    ax_info.text(
        0, 1, info_text, fontsize=11, verticalalignment="top", fontfamily="monospace"
    )

    # Metrics table
    ax_table = fig.add_axes([0.1, 0.25, 0.8, 0.4])
    ax_table.axis("off")

    metrics_data = [
        ["Metric", "Validation", "Test"],
        ["IoU", f"{val_metrics['mean_iou']:.4f}", f"{test_metrics['mean_iou']:.4f}"],
        [
            "Precision",
            f"{val_metrics['mean_precision']:.4f}",
            f"{test_metrics['mean_precision']:.4f}",
        ],
        [
            "Recall",
            f"{val_metrics['mean_recall']:.4f}",
            f"{test_metrics['mean_recall']:.4f}",
        ],
        ["F1", f"{val_metrics['mean_f1']:.4f}", f"{test_metrics['mean_f1']:.4f}"],
        [
            "AUPRC",
            f"{val_metrics['pr_curve']['auprc']:.4f}",
            f"{test_metrics['pr_curve']['auprc']:.4f}",
        ],
    ]

    table = ax_table.table(
        cellText=metrics_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.3, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    return fig


def create_confusion_matrix_page(
    cm: np.ndarray, class_names: list[str], title: str = "Test Set"
) -> plt.Figure:
    """Create confusion matrix visualization page."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle(f"Confusion Matrix - {title}", fontsize=14, fontweight="bold")

    # Normalized confusion matrix
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

    for ax, data, subtitle in zip(
        axes, [cm_normalized, cm], ["Normalized (%)", "Raw Counts"], strict=True
    ):
        im = ax.imshow(data, cmap="Blues")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle)

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if subtitle == "Normalized (%)":
                    text = f"{data[i, j]:.1f}%"
                else:
                    text = f"{int(data[i, j]):,}"
                color = "white" if data[i, j] > data.max() / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    return fig


def create_pr_curve_page(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    auprc: float,
    target_class_name: str,
) -> plt.Figure:
    """Create precision-recall curve page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.plot(
        recalls, precisions, "b-", linewidth=2, label=f"PR curve (AUPRC={auprc:.4f})"
    )
    ax.fill_between(recalls, precisions, alpha=0.2)

    # Mark threshold 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax.plot(
        recalls[idx_05],
        precisions[idx_05],
        "ro",
        markersize=10,
        label=f"Threshold=0.5 (P={precisions[idx_05]:.3f}, R={recalls[idx_05]:.3f})",
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        f"Precision-Recall Curve - {target_class_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_per_ortho_page(ortho_metrics: dict[str, dict]) -> plt.Figure:
    """Create per-ortho breakdown page with bar chart and table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Per-Ortho Performance", fontsize=14, fontweight="bold")

    # Sort by IoU
    sorted_orthos = sorted(
        ortho_metrics.keys(), key=lambda x: ortho_metrics[x]["iou"], reverse=True
    )

    # Bar chart
    ax_bar = fig.add_axes([0.1, 0.5, 0.8, 0.35])

    x = range(len(sorted_orthos))
    ious = [ortho_metrics[o]["iou"] for o in sorted_orthos]
    bars = ax_bar.bar(x, ious, color="#4472C4", edgecolor="black")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(sorted_orthos, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("IoU")
    ax_bar.set_ylim([0, 1])
    ax_bar.axhline(
        y=np.mean(ious), color="red", linestyle="--", label=f"Mean: {np.mean(ious):.3f}"
    )
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Table
    ax_table = fig.add_axes([0.1, 0.05, 0.8, 0.38])
    ax_table.axis("off")

    table_data = [["Ortho", "IoU", "Precision", "Recall", "N Chips"]]
    for ortho in sorted_orthos:
        m = ortho_metrics[ortho]
        table_data.append(
            [
                ortho[:20] + "..." if len(ortho) > 20 else ortho,
                f"{m['iou']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                str(m["n_chips"]),
            ]
        )

    # Limit table rows if too many orthos
    if len(table_data) > 20:
        table_data = table_data[:21]  # Header + 20 rows

    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.35, 0.15, 0.15, 0.15, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    return fig


def create_example_page(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    iou: float,
    filename: str,
    rank: int,
    category: str,
) -> plt.Figure:
    """Create example visualization page with 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"{category} #{rank} - IoU: {iou:.4f} - {filename}",
        fontsize=12,
        fontweight="bold",
    )

    # Prepare image for display
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    # Normalize to 0-255 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (
            (image - image.min()) / (image.max() - image.min() + 1e-10) * 255
        ).astype(np.uint8)
    elif image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    # Original image
    if image.ndim == 2:
        axes[0, 0].imshow(image, cmap="gray")
    else:
        axes[0, 0].imshow(image[:, :, :3])  # Take first 3 channels
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Ground truth mask
    axes[0, 1].imshow(label, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    # Prediction mask
    axes[1, 0].imshow(pred, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 0].set_title("Prediction (threshold=0.5)")
    axes[1, 0].axis("off")

    # Probability heatmap
    im = axes[1, 1].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Probability")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

    plt.tight_layout()
    return fig


def generate_report(
    config_path: Path,
    ckpt_path: Path,
    val_dir: Path,
    test_dir: Path,
    output_path: Path,
    class_names: list[str],
    batch_size: int = 16,
    num_workers: int = 4,
    threshold: float = 0.5,
):
    """Generate full evaluation report PDF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, config = load_model(config_path, ckpt_path, device)
    transforms = get_transforms(config)

    num_classes = len(class_names)
    target_class = 1  # Assume binary: background=0, target=1
    ignore_index = config["model"]["init_args"].get("ignore_index", -100)

    # Create datasets
    val_dataset = EvalDataset(str(val_dir), transforms=transforms)
    test_dataset = EvalDataset(str(test_dir), transforms=transforms)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # Run inference
    print("Running inference on validation set...")
    val_results = run_inference(model, val_loader, device, threshold)

    print("Running inference on test set...")
    test_results = run_inference(model, test_loader, device, threshold)

    # Compute metrics
    print("Computing metrics...")
    val_metrics = compute_all_metrics(
        val_results["preds"],
        val_results["labels"],
        val_results["probs"],
        val_results["filenames"],
        num_classes,
        target_class,
        ignore_index,
    )

    test_metrics = compute_all_metrics(
        test_results["preds"],
        test_results["labels"],
        test_results["probs"],
        test_results["filenames"],
        num_classes,
        target_class,
        ignore_index,
    )

    # Find best and worst samples from test set
    test_ious = [m["iou"] for m in test_metrics["sample_metrics"]]
    sorted_indices = np.argsort(test_ious)
    worst_8 = sorted_indices[:8]
    best_8 = sorted_indices[-8:][::-1]

    # Generate PDF
    print(f"Generating PDF report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig = create_summary_page(
            config, config_path, ckpt_path, val_metrics, test_metrics, class_names
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Confusion matrix
        fig = create_confusion_matrix_page(
            test_metrics["confusion_matrix"], class_names
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: PR curve
        fig = create_pr_curve_page(
            test_metrics["pr_curve"]["precisions"],
            test_metrics["pr_curve"]["recalls"],
            test_metrics["pr_curve"]["thresholds"],
            test_metrics["pr_curve"]["auprc"],
            class_names[target_class],
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Per-ortho breakdown
        fig = create_per_ortho_page(test_metrics["ortho_metrics"])
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 5-12: Best predictions
        for rank, idx in enumerate(best_8, 1):
            sample_metrics = test_metrics["sample_metrics"][idx]
            fig = create_example_page(
                image=test_results["raw_images"][idx],
                label=test_results["labels"][idx],
                pred=test_results["preds"][idx],
                prob=test_results["probs"][idx],
                iou=sample_metrics["iou"],
                filename=test_results["filenames"][idx],
                rank=rank,
                category="Best",
            )
            pdf.savefig(fig)
            plt.close(fig)

        # Pages 13-20: Worst predictions
        for rank, idx in enumerate(worst_8, 1):
            sample_metrics = test_metrics["sample_metrics"][idx]
            fig = create_example_page(
                image=test_results["raw_images"][idx],
                label=test_results["labels"][idx],
                pred=test_results["preds"][idx],
                prob=test_results["probs"][idx],
                iou=sample_metrics["iou"],
                filename=test_results["filenames"][idx],
                rank=rank,
                category="Worst",
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation report for segmentation models"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--ckpt", type=Path, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--val-dir", type=Path, required=True, help="Path to validation chip directory"
    )
    parser.add_argument(
        "--test-dir", type=Path, required=True, help="Path to test chip directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output PDF file"
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["bg", "seagrass"],
        help="Class names (default: bg seagrass)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )

    args = parser.parse_args()

    generate_report(
        config_path=args.config,
        ckpt_path=args.ckpt,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_path=args.output,
        class_names=args.class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
