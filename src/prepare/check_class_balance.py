"""Check class distribution across chip dataset to identify class imbalance."""

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def check_label_sanity(
    chip_dir: Path,
    expected_values: set[int] | None = None,
    ignore_index: int = -100,
) -> bool:
    """Check that labels only contain expected values."""
    if expected_values is None:
        expected_values = {0, 1, ignore_index}

    files = list(chip_dir.glob("**/*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return False

    all_values = set()
    problem_files = []

    for f in tqdm(files, desc="Checking labels"):
        label = np.load(f)["label"]
        unique = set(np.unique(label).tolist())
        all_values.update(unique)

        unexpected = unique - expected_values
        if unexpected:
            problem_files.append((f.name, unexpected))

    print(f"\nLabel Sanity Check: {chip_dir}")
    print("-" * 50)
    print(f"Expected values: {sorted(expected_values)}")
    print(f"Found values: {sorted(all_values)}")

    unexpected_global = all_values - expected_values
    if unexpected_global:
        print(f"UNEXPECTED values found: {sorted(unexpected_global)}")
        print(f"Problem files ({len(problem_files)}):")
        for name, vals in problem_files[:10]:
            print(f"  {name}: {vals}")
        if len(problem_files) > 10:
            print(f"  ... and {len(problem_files) - 10} more")
        return False
    else:
        print("OK - All labels contain only expected values")
        return True


def visualize_samples(
    chip_dir: Path,
    output_path: Path,
    n_samples: int = 16,
    class_names: dict[int, str] | None = None,
    ignore_index: int = -100,
    seed: int = 42,
) -> None:
    """Create PDF with random sample chips and their labels."""
    if class_names is None:
        class_names = {0: "bg", 1: "class_1"}

    files = list(chip_dir.glob("**/*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return

    random.seed(seed)
    samples = random.sample(files, min(n_samples, len(files)))

    # Create color map for labels
    n_classes = max(class_names.keys()) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes + 1)

    # 4 samples per page, 2 columns (image + label)
    samples_per_page = 4
    n_pages = (len(samples) + samples_per_page - 1) // samples_per_page

    with PdfPages(output_path) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(samples_per_page, 2, figsize=(10, 12))
            start_idx = page * samples_per_page

            for i in range(samples_per_page):
                sample_idx = start_idx + i
                ax_img = axes[i, 0]
                ax_label = axes[i, 1]

                if sample_idx < len(samples):
                    f = samples[sample_idx]
                    data = np.load(f)
                    img = data["image"]
                    label = data["label"]

                    # Plot image
                    ax_img.imshow(img)
                    ax_img.set_title(f.name, fontsize=8)
                    ax_img.axis("off")

                    # Plot label with color map
                    label_display = label.copy().astype(float)
                    label_display[label == ignore_index] = np.nan

                    ax_label.imshow(
                        label_display,
                        cmap=cmap,
                        vmin=0,
                        vmax=n_classes,
                        interpolation="nearest",
                    )
                    ax_label.set_title("Label", fontsize=8)
                    ax_label.axis("off")
                else:
                    ax_img.axis("off")
                    ax_label.axis("off")

            # Add legend on first page
            if page == 0:
                legend_elements = [
                    plt.Line2D(
                        [0], [0],
                        marker="s",
                        color="w",
                        markerfacecolor=cmap(i),
                        markersize=10,
                        label=name,
                    )
                    for i, name in class_names.items()
                ]
                legend_elements.append(
                    plt.Line2D(
                        [0], [0],
                        marker="s",
                        color="w",
                        markerfacecolor="white",
                        markeredgecolor="gray",
                        markersize=10,
                        label="ignore",
                    )
                )
                fig.legend(
                    handles=legend_elements,
                    loc="upper center",
                    ncol=len(legend_elements),
                    fontsize=8,
                )

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved visualization to: {output_path}")


def check_chip_counts(chip_dir: Path) -> dict[str, int]:
    """Count chips per split and total."""
    splits = ["train", "val", "test"]
    counts = {}

    print(f"\nDataset: {chip_dir}")
    print("-" * 40)
    print(f"{'Split':<10} {'Chips':>15}")
    print("-" * 40)

    total = 0
    for split in splits:
        split_dir = chip_dir / split
        if split_dir.exists():
            count = len(list(split_dir.glob("*.npz")))
            counts[split] = count
            total += count
            print(f"{split:<10} {count:>15,}")
        else:
            counts[split] = 0
            print(f"{split:<10} {'(not found)':>15}")

    print("-" * 40)
    print(f"{'TOTAL':<10} {total:>15,}")
    print("-" * 40)

    counts["total"] = total
    return counts


def check_class_balance(
    chip_dir: Path,
    class_names: dict[int, str] | None = None,
    ignore_index: int = -100,
) -> None:
    if class_names is None:
        class_names = {0: "bg", 1: "class_1", 2: "class_2", 3: "class_3"}

    files = list(chip_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {chip_dir}")
        return

    counts = Counter()
    chips_with_class = Counter()

    for f in tqdm(files, desc="Scanning chips"):
        label = np.load(f)["label"]
        unique, c = np.unique(label, return_counts=True)
        for u, cnt in zip(unique, c):
            counts[int(u)] += cnt
            chips_with_class[int(u)] += 1

    total_pixels = sum(v for k, v in counts.items() if k != ignore_index)
    total_chips = len(files)

    print(f"\nDataset: {chip_dir}")
    print(f"Total chips: {total_chips:,}")
    print(f"Total pixels (excluding ignore): {total_pixels:,}")
    print("\nClass distribution:")
    print("-" * 60)
    print(f"{'Class':<15} {'Pixels':>15} {'%':>8} {'Chips':>10} {'% Chips':>10}")
    print("-" * 60)

    for k in sorted(counts):
        name = class_names.get(k, f"class_{k}")
        if k == ignore_index:
            name = "ignore"
            pct = 0
        else:
            pct = counts[k] / total_pixels * 100

        chip_pct = chips_with_class[k] / total_chips * 100
        print(
            f"{name:<15} {counts[k]:>15,} {pct:>7.1f}% {chips_with_class[k]:>10,} {chip_pct:>9.1f}%"
        )

    print("-" * 60)

    # Calculate and display imbalance ratio
    non_ignore = {k: v for k, v in counts.items() if k != ignore_index}
    if len(non_ignore) >= 2:
        max_class = max(non_ignore.values())
        min_class = min(non_ignore.values())
        ratio = max_class / min_class if min_class > 0 else float("inf")
        print(f"\nImbalance ratio (max/min): {ratio:.1f}:1")
        if ratio > 10:
            print("-> Significant imbalance. Consider Focal Loss or class weighting.")
        elif ratio > 5:
            print("-> Moderate imbalance. Focal Loss may help.")
        else:
            print("-> Relatively balanced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check class distribution in chip dataset"
    )
    parser.add_argument(
        "chip_dir",
        type=Path,
        help="Directory containing .npz chips, or parent directory with train/val/test subdirs",
    )
    parser.add_argument(
        "--class-names",
        "-c",
        nargs="+",
        default=["bg", "seagrass"],
        help="Class names in order (default: bg seagrass)",
    )
    parser.add_argument(
        "--ignore-index",
        "-i",
        type=int,
        default=-100,
        help="Ignore index value (default: -100)",
    )
    parser.add_argument(
        "--all-splits",
        "-a",
        action="store_true",
        help="Check all splits (train/val/test) in the given directory",
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Only show chip counts, skip class balance check",
    )
    parser.add_argument(
        "--sanity-check",
        "-s",
        action="store_true",
        help="Check that labels only contain expected values",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        type=Path,
        metavar="OUTPUT.pdf",
        help="Create PDF with sample chips and labels",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples to visualize (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    args = parser.parse_args()

    class_names = {i: name for i, name in enumerate(args.class_names)}
    expected_values = set(class_names.keys()) | {args.ignore_index}

    if args.counts_only:
        check_chip_counts(args.chip_dir)
    elif args.visualize:
        visualize_samples(
            args.chip_dir,
            args.visualize,
            n_samples=args.n_samples,
            class_names=class_names,
            ignore_index=args.ignore_index,
            seed=args.seed,
        )
    elif args.sanity_check:
        for split in ["train", "val", "test"]:
            split_dir = args.chip_dir / split
            if split_dir.exists():
                check_label_sanity(split_dir, expected_values, args.ignore_index)
    elif args.all_splits:
        print("\n" + "=" * 60)
        print("  CHIP COUNTS")
        print("=" * 60)
        check_chip_counts(args.chip_dir)

        print("\n" + "=" * 60)
        print("  LABEL SANITY CHECK")
        print("=" * 60)
        for split in ["train", "val", "test"]:
            split_dir = args.chip_dir / split
            if split_dir.exists():
                check_label_sanity(split_dir, expected_values, args.ignore_index)

        for split in ["train", "val", "test"]:
            split_dir = args.chip_dir / split
            if split_dir.exists():
                print(f"\n{'='*60}")
                print(f"  {split.upper()} - CLASS BALANCE")
                print(f"{'='*60}")
                check_class_balance(split_dir, class_names, args.ignore_index)
            else:
                print(f"\nSkipping {split} (not found: {split_dir})")
    else:
        check_class_balance(args.chip_dir, class_names, args.ignore_index)
