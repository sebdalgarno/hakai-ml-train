"""Generate per-ortho prediction PDFs for diagnosing errors by site."""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter
from tqdm import tqdm

from src.predict.visualize_predictions import (
    get_test_transforms,
    load_model,
    predict_chip,
)


def extract_ortho_name(filename: str) -> str:
    """Extract ortho name from chip filename (e.g. 'koeye_u0388_42.npz' -> 'koeye_u0388')."""
    stem = Path(filename).stem
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def group_chips_by_ortho(chip_dir: Path) -> dict[str, list[Path]]:
    """Group chip files by ortho name."""
    groups = defaultdict(list)
    for f in sorted(chip_dir.glob("*.npz")):
        ortho = extract_ortho_name(f.name)
        groups[ortho].append(f)
    return dict(sorted(groups.items()))


def _add_bookmarks(pdf_path: Path, ortho_page_map: dict[str, int]) -> None:
    """Add sidebar outline bookmarks to the PDF."""
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    for ortho, page_num in ortho_page_map.items():
        writer.add_outline_item(ortho, page_num)

    with open(pdf_path, "wb") as f:
        writer.write(f)

    print(f"Added {len(ortho_page_map)} sidebar bookmarks")


def generate_ortho_pdf(
    chip_dir: Path,
    config_path: Path,
    ckpt_path: Path,
    output_path: Path,
    n_per_ortho: int = 10,
    class_names: list[str] | None = None,
    ignore_index: int = -100,
    seed: int = 42,
    device: str = "cuda",
    threshold: float = 0.5,
) -> None:
    """Generate PDF with predictions grouped by ortho."""
    if class_names is None:
        class_names = ["bg", "seagrass"]

    model, config = load_model(config_path, ckpt_path, device)
    transforms = get_test_transforms(config)
    num_classes = config["model"]["init_args"].get("num_classes", 2)

    n_classes = len(class_names)
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes + 1)

    ortho_groups = group_chips_by_ortho(chip_dir)
    if not ortho_groups:
        print(f"No .npz files found in {chip_dir}")
        return

    # Sample n chips from each ortho
    random.seed(seed)
    sampled = {}
    total = 0
    for ortho, files in ortho_groups.items():
        sampled[ortho] = random.sample(files, min(n_per_ortho, len(files)))
        total += len(sampled[ortho])

    print(
        f"Found {len(ortho_groups)} orthos, sampling {n_per_ortho} per ortho -> {total} chips"
    )

    # Track page index where each ortho's title page lands
    ortho_page_map = {}
    page_idx = 0

    with PdfPages(output_path) as pdf:
        for ortho, files in tqdm(sampled.items(), desc="Orthos"):
            ortho_page_map[ortho] = page_idx

            # Title page per ortho
            fig = plt.figure(figsize=(15, 5))
            fig.text(
                0.5,
                0.5,
                ortho,
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.4,
                f"{len(files)} / {len(ortho_groups[ortho])} chips",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            pdf.savefig(fig)
            plt.close(fig)
            page_idx += 1

            for f in files:
                data = np.load(f)
                image = data["image"]
                label = data["label"]

                pred = predict_chip(
                    model, image, transforms, device, num_classes, threshold
                )

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                ax_img, ax_label, ax_pred = axes

                ax_img.imshow(image)
                ax_img.set_title(f.stem, fontsize=10)
                ax_img.axis("off")

                label_display = label.copy().astype(float)
                label_display[label == ignore_index] = np.nan

                ax_label.imshow(
                    label_display,
                    cmap=cmap,
                    vmin=0,
                    vmax=n_classes,
                    interpolation="nearest",
                )
                ax_label.set_title("Ground Truth", fontsize=10)
                ax_label.axis("off")

                ax_pred.imshow(
                    pred,
                    cmap=cmap,
                    vmin=0,
                    vmax=n_classes,
                    interpolation="nearest",
                )
                ax_pred.set_title(f"Prediction (t={threshold})", fontsize=10)
                ax_pred.axis("off")

                plt.subplots_adjust(
                    wspace=0.05, left=0.02, right=0.98, top=0.92, bottom=0.02
                )
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
                page_idx += 1

    # Post-process: add sidebar bookmarks
    _add_bookmarks(output_path, ortho_page_map)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-ortho prediction visualization PDF"
    )
    parser.add_argument(
        "chip_dir",
        type=Path,
        help="Directory containing .npz chips",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions_by_ortho.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--n-per-ortho",
        type=int,
        default=10,
        help="Number of chips to sample per ortho (default: 10)",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["bg", "seagrass"],
        help="Class names in order",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    generate_ortho_pdf(
        chip_dir=args.chip_dir,
        config_path=args.config,
        ckpt_path=args.ckpt,
        output_path=args.output,
        n_per_ortho=args.n_per_ortho,
        class_names=args.class_names,
        ignore_index=args.ignore_index,
        seed=args.seed,
        device=args.device,
        threshold=args.threshold,
    )
