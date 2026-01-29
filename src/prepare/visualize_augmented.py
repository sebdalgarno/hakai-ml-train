"""Visualize chips with augmentations applied."""

import argparse
import math
import random
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# ImageNet normalization values (used by most configs)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse normalization for display."""
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def load_transforms_from_config(config_path: Path) -> tuple[A.Compose, A.Compose]:
    """Load train and test transforms from config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]["init_args"]
    train_transforms = A.from_dict(data_config["train_transforms"])
    test_transforms = A.from_dict(data_config["test_transforms"])

    return train_transforms, test_transforms


def apply_augmentation(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply augmentation and return image ready for display."""
    augmented = transform(image=image)
    aug_image = augmented["image"]

    # Handle tensor output from ToTensorV2
    if hasattr(aug_image, "numpy"):
        aug_image = aug_image.numpy()

    # Convert CHW to HWC if needed
    if aug_image.ndim == 3 and aug_image.shape[0] in [1, 3, 4]:
        aug_image = np.transpose(aug_image, (1, 2, 0))

    # Unnormalize for display
    if aug_image.dtype in [np.float32, np.float64]:
        aug_image = unnormalize(aug_image, IMAGENET_MEAN, IMAGENET_STD)

    return aug_image


def visualize_augmented_samples(
    chip_dir: Path,
    config_path: Path,
    output_path: Path,
    n_samples: int = 16,
    n_augmentations: int = 8,
    seed: int = 42,
) -> None:
    """Create PDF showing mosaic of augmented images for each tile.

    Each page shows one original tile and a grid of augmented versions.
    No ground truth labels are shown.
    """
    # Load transforms
    train_transforms, _ = load_transforms_from_config(config_path)

    # Get chip files
    files = list(chip_dir.glob("**/*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return

    random.seed(seed)
    samples = random.sample(files, min(n_samples, len(files)))

    print(f"Generating augmented visualizations for {len(samples)} samples...")
    print(f"Config: {config_path.name}")
    print(f"Augmentations per sample: {n_augmentations}")

    # Calculate grid dimensions for augmented images
    n_cols = int(math.ceil(math.sqrt(n_augmentations)))
    n_rows = int(math.ceil(n_augmentations / n_cols))

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(
            0.5,
            0.6,
            "Augmentation Visualization",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.45,
            f"Config: {config_path.name}",
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.38,
            f"Chip directory: {chip_dir}",
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.text(
            0.5,
            0.31,
            f"Samples: {len(samples)} | Augmentations per sample: {n_augmentations}",
            ha="center",
            va="center",
            fontsize=10,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # One page per sample
        for sample_file in tqdm(samples, desc="Processing samples"):
            data = np.load(sample_file)
            orig_image = data["image"]

            # Create figure with mosaic layout
            fig = plt.figure(figsize=(12, 10))

            # Original image at top (spanning full width)
            ax_orig = fig.add_axes([0.3, 0.75, 0.4, 0.22])
            ax_orig.imshow(orig_image)
            ax_orig.set_title(
                f"Original: {sample_file.name}", fontsize=10, fontweight="bold"
            )
            ax_orig.axis("off")

            # Grid of augmented images below
            grid_height = 0.68
            grid_width = 0.95
            cell_width = grid_width / n_cols
            cell_height = grid_height / n_rows
            margin = 0.02

            for aug_idx in range(n_augmentations):
                row = aug_idx // n_cols
                col = aug_idx % n_cols

                # Calculate position
                left = 0.025 + col * cell_width + margin / 2
                bottom = 0.02 + (n_rows - 1 - row) * cell_height + margin / 2
                width = cell_width - margin
                height = cell_height - margin

                ax = fig.add_axes([left, bottom, width, height])

                aug_image = apply_augmentation(orig_image.copy(), train_transforms)

                ax.imshow(aug_image)
                ax.set_title(f"Aug {aug_idx + 1}", fontsize=8)
                ax.axis("off")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved to: {output_path}")


def compare_augmentation_configs(
    chip_dir: Path,
    config_paths: list[Path],
    output_path: Path,
    n_samples: int = 8,
    seed: int = 42,
) -> None:
    """Compare augmentations from multiple configs side by side.

    Each page shows one chip with augmentation from each config.
    """
    # Load all transforms
    all_transforms = []
    config_names = []
    for config_path in config_paths:
        train_transforms, _ = load_transforms_from_config(config_path)
        all_transforms.append(train_transforms)
        config_names.append(config_path.stem)

    # Get chip files
    files = list(chip_dir.glob("**/*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return

    random.seed(seed)
    samples = random.sample(files, min(n_samples, len(files)))

    print(f"Comparing {len(config_paths)} configs on {len(samples)} samples...")

    with PdfPages(output_path) as pdf:
        for sample_file in tqdm(samples, desc="Processing samples"):
            data = np.load(sample_file)
            orig_image = data["image"]

            # Create figure: 1 row x (1 + n_configs) columns
            n_cols = 1 + len(config_paths)
            fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))

            fig.suptitle(sample_file.name, fontsize=10, fontweight="bold")

            # Original
            axes[0].imshow(orig_image)
            axes[0].set_title("Original", fontsize=9)
            axes[0].axis("off")

            # Each config's augmentation
            for cfg_idx, (transform, name) in enumerate(
                zip(all_transforms, config_names, strict=True)
            ):
                aug_image = apply_augmentation(orig_image.copy(), transform)

                col = cfg_idx + 1
                axes[col].imshow(aug_image)
                axes[col].set_title(name, fontsize=8)
                axes[col].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved comparison to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize chips with augmentations applied"
    )
    parser.add_argument(
        "chip_dir",
        type=Path,
        help="Directory containing .npz chips",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to training config YAML with transforms",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/augmented_samples.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples to visualize (default: 16)",
    )
    parser.add_argument(
        "--n-augmentations",
        type=int,
        default=8,
        help="Number of augmentation variants per sample (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        metavar="CONFIG",
        help="Compare multiple configs (provide additional config paths)",
    )

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Compare multiple configs
        all_configs = [args.config] + list(args.compare)
        compare_augmentation_configs(
            args.chip_dir,
            all_configs,
            args.output,
            n_samples=args.n_samples,
            seed=args.seed,
        )
    else:
        # Single config visualization
        visualize_augmented_samples(
            args.chip_dir,
            args.config,
            args.output,
            n_samples=args.n_samples,
            n_augmentations=args.n_augmentations,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
