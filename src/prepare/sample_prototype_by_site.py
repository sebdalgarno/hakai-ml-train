"""Sample chips proportionally by site for prototype dataset.

Creates a prototype dataset with equal representation per site,
regardless of how many orthos each site has.
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def extract_site_name(filename: str) -> str:
    """Extract site name from chip filename.

    Handles formats like:
    - koeye_u0388_42.npz -> koeye
    - pruth_bay_u0699_123.npz -> pruth_bay
    - bennett_bay_42.npz -> bennett_bay
    """
    stem = Path(filename).stem

    # Pattern: {site}_u{id}_{chip_idx}
    match = re.match(r"(.+)_u\d+_\d+$", stem)
    if match:
        return match.group(1)

    # Pattern: {site}_{chip_idx} (no u-prefix, like bennett_bay)
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        # Check if the part before _number looks like a site name
        # (doesn't end with _u followed by digits)
        potential_site = match.group(1)
        if not re.search(r"_u\d+$", potential_site):
            return potential_site

    return stem


def group_chips_by_site(chip_dir: Path) -> dict[str, list[Path]]:
    """Group all chip files by site name."""
    chips_by_site = defaultdict(list)

    for chip_file in chip_dir.glob("*.npz"):
        site = extract_site_name(chip_file.name)
        chips_by_site[site].append(chip_file)

    return dict(chips_by_site)


def sample_chips(
    input_dir: Path,
    output_dir: Path,
    chips_per_site: int | None = None,
    fraction: float | None = None,
    seed: int = 42,
    copy: bool = True,
) -> dict[str, int]:
    """Sample chips with equal representation per site.

    Args:
        input_dir: Directory containing chip .npz files
        output_dir: Directory to write sampled chips
        chips_per_site: Fixed number of chips per site (overrides fraction)
        fraction: Fraction of TOTAL chips to sample, distributed equally per site
        seed: Random seed for reproducibility
        copy: If True, copy files; if False, move files

    Returns:
        Dictionary mapping site names to number of chips sampled
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    chips_by_site = group_chips_by_site(input_dir)

    if not chips_by_site:
        print(f"No chips found in {input_dir}")
        return {}

    # Determine chips per site
    site_counts = {site: len(chips) for site, chips in chips_by_site.items()}
    total_chips = sum(site_counts.values())
    min_chips = min(site_counts.values())
    num_sites = len(chips_by_site)

    if chips_per_site is not None:
        target_per_site = chips_per_site
    elif fraction is not None:
        # Calculate target as fraction of total, divided equally among sites
        # e.g., 10% of 100,000 chips with 20 sites = 500 chips per site
        total_target = int(total_chips * fraction)
        target_per_site = max(1, total_target // num_sites)
    else:
        raise ValueError("Must specify either chips_per_site or fraction")

    # Sites with fewer chips than target will contribute all their chips
    expected_total = sum(min(target_per_site, count) for count in site_counts.values())

    print(f"Sites: {num_sites}")
    print(f"Total chips: {total_chips}")
    print(f"Chips per site range: {min_chips} - {max(site_counts.values())}")
    print(f"Target chips per site: {target_per_site} (smaller sites contribute all)")
    print(f"Expected total sampled: {expected_total} ({100 * expected_total / total_chips:.1f}%)")

    sampled_counts = {}
    operation = shutil.copy2 if copy else shutil.move
    op_name = "Copying" if copy else "Moving"

    for site, chips in tqdm(chips_by_site.items(), desc=op_name):
        # Randomly sample from this site's chips
        sampled = random.sample(chips, min(target_per_site, len(chips)))

        for chip_file in sampled:
            operation(chip_file, output_dir / chip_file.name)

        sampled_counts[site] = len(sampled)

    return sampled_counts


def main():
    parser = argparse.ArgumentParser(
        description="Sample chips with equal representation per site."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing chip .npz files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for sampled chips",
    )
    parser.add_argument(
        "--chips-per-site",
        type=int,
        default=None,
        help="Fixed number of chips to sample per site",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of chips to sample (applied to smallest site)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    args = parser.parse_args()

    if args.chips_per_site is None and args.fraction is None:
        parser.error("Must specify either --chips-per-site or --fraction")

    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # Process each split
    splits_found = False
    for split in ["train", "val", "test"]:
        split_input = args.input_dir / split
        if not split_input.exists():
            continue

        splits_found = True
        print(f"\n{'='*50}")
        print(f"{split.upper()} split")
        print(f"{'='*50}")

        split_output = args.output_dir / split

        sampled_counts = sample_chips(
            input_dir=split_input,
            output_dir=split_output,
            chips_per_site=args.chips_per_site,
            fraction=args.fraction,
            seed=args.seed,
            copy=not args.move,
        )

        if sampled_counts:
            total = sum(sampled_counts.values())
            print(f"\nSampled {total} chips from {len(sampled_counts)} sites:")
            for site, count in sorted(sampled_counts.items()):
                print(f"  {site}: {count}")

    # If no splits found, process as flat directory
    if not splits_found:
        print("\nNo train/val/test splits found, processing as flat directory")
        sampled_counts = sample_chips(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chips_per_site=args.chips_per_site,
            fraction=args.fraction,
            seed=args.seed,
            copy=not args.move,
        )

        if sampled_counts:
            total = sum(sampled_counts.values())
            print(f"\nSampled {total} chips from {len(sampled_counts)} sites")


if __name__ == "__main__":
    main()
