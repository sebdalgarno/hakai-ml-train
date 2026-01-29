"""Generate PDF with model predictions compared to ground truth labels."""

import argparse
import importlib
import random
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def load_model(config_path: Path, ckpt_path: Path, device: str = "cuda"):
    """Load trained model from checkpoint."""
    with open(config_path, "r") as f:
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


def get_test_transforms(config: dict):
    """Get test transforms from config."""
    transforms_dict = config["data"]["init_args"]["test_transforms"]
    return A.from_dict(transforms_dict)


def predict_chip(
    model,
    image: np.ndarray,
    transforms,
    device: str = "cuda",
    num_classes: int = 2,
    threshold: float = 0.5,
):
    """Run prediction on single chip."""
    augmented = transforms(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)

    if num_classes == 1:
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0)
        pred = (probs > threshold).long()
    else:
        probs = torch.softmax(logits, dim=1).squeeze(0)
        target_probs = probs[1]  # Probability of class 1 (seagrass)
        pred = (target_probs > threshold).long()

    return pred.cpu().numpy()


def generate_prediction_pdf(
    chip_dir: Path,
    config_path: Path,
    ckpt_path: Path,
    output_path: Path,
    n_samples: int = 16,
    class_names: list[str] | None = None,
    ignore_index: int = -100,
    seed: int = 42,
    device: str = "cuda",
    threshold: float = 0.5,
) -> None:
    """Generate PDF with image | label | prediction columns."""
    if class_names is None:
        class_names = ["bg", "class_1"]

    model, config = load_model(config_path, ckpt_path, device)
    transforms = get_test_transforms(config)

    num_classes = config["model"]["init_args"].get("num_classes", 2)

    files = list(chip_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return

    random.seed(seed)
    samples = random.sample(files, min(n_samples, len(files)))

    n_classes = len(class_names)
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes + 1)

    n_pages = len(samples)

    print(f"Generating predictions for {len(samples)} chips (threshold={threshold})...")

    with PdfPages(output_path) as pdf:
        for f in tqdm(samples, desc="Generating pages"):
            data = np.load(f)
            image = data["image"]
            label = data["label"]

            pred = predict_chip(
                model,
                image,
                transforms,
                device,
                num_classes,
                threshold,
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

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PDF visualization of model predictions"
    )
    parser.add_argument(
        "chip_dir",
        type=Path,
        help="Directory containing .npz chips to predict on",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML file",
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
        default=Path("predictions.pdf"),
        help="Output PDF path (default: predictions.pdf)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples to visualize (default: 16)",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["bg", "seagrass"],
        help="Class names in order (default: bg seagrass)",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
        help="Ignore index value (default: -100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for positive class (default: 0.5)",
    )

    args = parser.parse_args()

    generate_prediction_pdf(
        chip_dir=args.chip_dir,
        config_path=args.config,
        ckpt_path=args.ckpt,
        output_path=args.output,
        n_samples=args.n_samples,
        class_names=args.class_names,
        ignore_index=args.ignore_index,
        seed=args.seed,
        device=args.device,
        threshold=args.threshold,
    )
