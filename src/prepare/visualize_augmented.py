"""Visualize chips with augmentations applied."""

import argparse
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


def apply_augmentation(
    image: np.ndarray, label: np.ndarray, transform: A.Compose
) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation and return image ready for display."""
    augmented = transform(image=image, mask=label)
    aug_image = augmented["image"]
    aug_label = augmented["mask"]

    # Handle tensor output from ToTensorV2
    if hasattr(aug_image, "numpy"):
        aug_image = aug_image.numpy()
    if hasattr(aug_label, "numpy"):
        aug_label = aug_label.numpy()

    # Convert CHW to HWC if needed
    if aug_image.ndim == 3 and aug_image.shape[0] in [1, 3, 4]:
        aug_image = np.transpose(aug_image, (1, 2, 0))

    # Unnormalize for display
    if aug_image.dtype in [np.float32, np.float64]:
        aug_image = unnormalize(aug_image, IMAGENET_MEAN, IMAGENET_STD)

    return aug_image, aug_label


def visualize_augmented_samples(
    chip_dir: Path,
    config_path: Path,
    output_path: Path,
    n_samples: int = 16,
    n_augmentations: int = 3,
    class_names: dict[int, str] | None = None,
    ignore_index: int = -100,
    seed: int = 42,
) -> None:
    """Create PDF showing original chips and multiple augmented versions.

    Each page shows one chip with:
    - Original image + label
    - N augmented versions
    """
    if class_names is None:
        class_names = {0: "bg", 1: "seagrass"}

    # Load transforms
    train_transforms, _ = load_transforms_from_config(config_path)

    # Get chip files
    files = list(chip_dir.glob("**/*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return

    random.seed(seed)
    samples = random.sample(files, min(n_samples, len(files)))

    # Color map for labels
    n_classes = max(class_names.keys()) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes + 1)

    print(f"Generating augmented visualizations for {len(samples)} samples...")
    print(f"Config: {config_path.name}")
    print(f"Augmentations per sample: {n_augmentations}")

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(
            0.5, 0.6,
            "Augmentation Visualization",
            ha="center", va="center", fontsize=20, fontweight="bold"
        )
        fig.text(
            0.5, 0.45,
            f"Config: {config_path.name}",
            ha="center", va="center", fontsize=12
        )
        fig.text(
            0.5, 0.38,
            f"Chip directory: {chip_dir}",
            ha="center", va="center", fontsize=10
        )
        fig.text(
            0.5, 0.31,
            f"Samples: {len(samples)} | Augmentations per sample: {n_augmentations}",
            ha="center", va="center", fontsize=10
        )

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0], [0], marker="s", color="w",
                markerfacecolor=cmap(i), markersize=12, label=name
            )
            for i, name in class_names.items()
        ]
        legend_elements.append(
            plt.Line2D(
                [0], [0], marker="s", color="w",
                markerfacecolor="white", markeredgecolor="gray",
                markersize=12, label="ignore"
            )
        )
        fig.legend(
            handles=legend_elements, loc="center",
            bbox_to_anchor=(0.5, 0.18), ncol=len(legend_elements), fontsize=10
        )

        pdf.savefig(fig)
        plt.close(fig)

        # One page per sample
        for sample_file in tqdm(samples, desc="Processing samples"):
            data = np.load(sample_file)
            orig_image = data["image"]
            orig_label = data["label"]

            # Create figure: 2 rows x (1 + n_augmentations) columns
            # Row 1: images, Row 2: labels
            n_cols = 1 + n_augmentations
            fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))

            fig.suptitle(sample_file.name, fontsize=10, fontweight="bold")

            # Original image
            axes[0, 0].imshow(orig_image)
            axes[0, 0].set_title("Original", fontsize=9)
            axes[0, 0].axis("off")

            # Original label
            label_display = orig_label.copy().astype(float)
            label_display[orig_label == ignore_index] = np.nan
            axes[1, 0].imshow(
                label_display, cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest"
            )
            axes[1, 0].set_title("Label", fontsize=9)
            axes[1, 0].axis("off")

            # Augmented versions
            for aug_idx in range(n_augmentations):
                aug_image, aug_label = apply_augmentation(
                    orig_image.copy(), orig_label.copy(), train_transforms
                )

                col = aug_idx + 1

                # Augmented image
                axes[0, col].imshow(aug_image)
                axes[0, col].set_title(f"Aug {aug_idx + 1}", fontsize=9)
                axes[0, col].axis("off")

                # Augmented label
                label_display = aug_label.copy().astype(float)
                label_display[aug_label == ignore_index] = np.nan
                axes[1, col].imshow(
                    label_display, cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest"
                )
                axes[1, col].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved to: {output_path}")


def compare_augmentation_configs(
    chip_dir: Path,
    config_paths: list[Path],
    output_path: Path,
    n_samples: int = 8,
    class_names: dict[int, str] | None = None,
    ignore_index: int = -100,
    seed: int = 42,
) -> None:
    """Compare augmentations from multiple configs side by side.

    Each page shows one chip with augmentation from each config.
    """
    if class_names is None:
        class_names = {0: "bg", 1: "seagrass"}

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

    n_classes = max(class_names.keys()) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes + 1)

    print(f"Comparing {len(config_paths)} configs on {len(samples)} samples...")

    with PdfPages(output_path) as pdf:
        for sample_file in tqdm(samples, desc="Processing samples"):
            data = np.load(sample_file)
            orig_image = data["image"]
            orig_label = data["label"]

            # Create figure: 2 rows x (1 + n_configs) columns
            n_cols = 1 + len(config_paths)
            fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))

            fig.suptitle(sample_file.name, fontsize=10, fontweight="bold")

            # Original
            axes[0, 0].imshow(orig_image)
            axes[0, 0].set_title("Original", fontsize=9)
            axes[0, 0].axis("off")

            label_display = orig_label.copy().astype(float)
            label_display[orig_label == ignore_index] = np.nan
            axes[1, 0].imshow(
                label_display, cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest"
            )
            axes[1, 0].set_title("Label", fontsize=9)
            axes[1, 0].axis("off")

            # Each config's augmentation
            for cfg_idx, (transform, name) in enumerate(zip(all_transforms, config_names, strict=True)):
                aug_image, aug_label = apply_augmentation(
                    orig_image.copy(), orig_label.copy(), transform
                )

                col = cfg_idx + 1

                axes[0, col].imshow(aug_image)
                axes[0, col].set_title(name, fontsize=8)
                axes[0, col].axis("off")

                label_display = aug_label.copy().astype(float)
                label_display[aug_label == ignore_index] = np.nan
                axes[1, col].imshow(
                    label_display, cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest"
                )
                axes[1, col].axis("off")

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
        default=3,
        help="Number of augmentation variants per sample (default: 3)",
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
        "--compare",
        type=Path,
        nargs="+",
        metavar="CONFIG",
        help="Compare multiple configs (provide additional config paths)",
    )

    args = parser.parse_args()

    class_names = {i: name for i, name in enumerate(args.class_names)}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Compare multiple configs
        all_configs = [args.config] + list(args.compare)
        compare_augmentation_configs(
            args.chip_dir,
            all_configs,
            args.output,
            n_samples=args.n_samples,
            class_names=class_names,
            ignore_index=args.ignore_index,
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
            class_names=class_names,
            ignore_index=args.ignore_index,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
