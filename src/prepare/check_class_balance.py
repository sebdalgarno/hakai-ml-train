"""Check class distribution across chip dataset to identify class imbalance."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


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
    args = parser.parse_args()

    class_names = {i: name for i, name in enumerate(args.class_names)}

    if args.all_splits:
        for split in ["train", "val", "test"]:
            split_dir = args.chip_dir / split
            if split_dir.exists():
                print(f"\n{'='*60}")
                print(f"  {split.upper()}")
                print(f"{'='*60}")
                check_class_balance(split_dir, class_names, args.ignore_index)
            else:
                print(f"\nSkipping {split} (not found: {split_dir})")
    else:
        check_class_balance(args.chip_dir, class_names, args.ignore_index)
