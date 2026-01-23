"""Check class and site distribution across chip dataset."""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def parse_site_name(filename: str) -> str:
    """Extract site name from chip filename.

    Handles formats like:
    - sitename_year_123.npz -> sitename
    - sitename_year.npz -> sitename
    """
    stem = Path(filename).stem
    # Remove trailing chip index (e.g., _123)
    match = re.match(r"(.+)_\d+$", stem)
    if match:
        stem = match.group(1)
    # Extract site from sitename_year format
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return stem


def get_site_distribution(chip_dir: Path) -> dict[str, int]:
    """Count chips per site in a directory."""
    files = list(chip_dir.glob("*.npz"))
    site_counts = Counter()
    for f in files:
        site = parse_site_name(f.name)
        site_counts[site] += 1
    return dict(site_counts)


def check_site_balance(chip_dir: Path, all_splits: bool = True) -> None:
    """Check and display site distribution across splits."""
    splits = ["train", "val", "test"] if all_splits else [chip_dir.name]
    base_dir = chip_dir if all_splits else chip_dir.parent

    # Collect site distributions per split
    split_sites: dict[str, dict[str, int]] = {}
    all_sites: set[str] = set()

    for split in splits:
        split_dir = base_dir / split if all_splits else chip_dir
        if split_dir.exists():
            dist = get_site_distribution(split_dir)
            split_sites[split] = dist
            all_sites.update(dist.keys())

    if not split_sites:
        print("No chip files found.")
        return

    # Print header
    print("\n" + "=" * 80)
    print("  SITE DISTRIBUTION")
    print("=" * 80)

    # Calculate totals per split
    split_totals = {split: sum(sites.values()) for split, sites in split_sites.items()}
    grand_total = sum(split_totals.values())

    # Print summary by split
    print(f"\n{'Split':<10} {'Sites':>8} {'Chips':>12} {'% of Total':>12}")
    print("-" * 45)
    for split in splits:
        if split in split_sites:
            n_sites = len(split_sites[split])
            n_chips = split_totals[split]
            pct = 100 * n_chips / grand_total if grand_total > 0 else 0
            print(f"{split:<10} {n_sites:>8} {n_chips:>12,} {pct:>11.1f}%")
    print("-" * 45)
    print(f"{'TOTAL':<10} {len(all_sites):>8} {grand_total:>12,}")

    # Print detailed site breakdown
    print("\n" + "-" * 80)
    header = f"{'Site':<30}"
    for split in splits:
        if split in split_sites:
            header += f" {split:>12}"
    header += f" {'Total':>12}"
    print(header)
    print("-" * 80)

    # Sort sites by total chip count (descending)
    site_totals = {}
    for site in sorted(all_sites):
        total = sum(split_sites.get(split, {}).get(site, 0) for split in splits)
        site_totals[site] = total

    for site in sorted(site_totals.keys(), key=lambda s: site_totals[s], reverse=True):
        row = f"{site:<30}"
        for split in splits:
            if split in split_sites:
                count = split_sites[split].get(site, 0)
                if count > 0:
                    row += f" {count:>12,}"
                else:
                    row += f" {'-':>12}"
        row += f" {site_totals[site]:>12,}"
        print(row)

    print("-" * 80)

    # Check for site leakage (same site in multiple splits)
    sites_per_split = {split: set(sites.keys()) for split, sites in split_sites.items()}
    leakage = []
    checked = set()
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2 or (s1, s2) in checked:
                continue
            checked.add((s1, s2))
            if s1 in sites_per_split and s2 in sites_per_split:
                overlap = sites_per_split[s1] & sites_per_split[s2]
                if overlap:
                    leakage.append((s1, s2, overlap))

    if leakage:
        print("\nWARNING: Site leakage detected!")
        for s1, s2, overlap in leakage:
            print(f"  {s1} & {s2} share sites: {sorted(overlap)}")
    else:
        print("\nNo site leakage detected (each site appears in only one split).")


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
                        [0],
                        [0],
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
                        [0],
                        [0],
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


def get_site_distribution_for_pdf(chip_dir: Path) -> tuple[list, list, bool]:
    """Get site distribution data formatted for PDF table.

    Returns:
        site_data: List of rows for site table
        summary_data: List of rows for summary table
        has_leakage: Whether site leakage was detected
    """
    splits = ["train", "val", "test"]
    split_sites: dict[str, dict[str, int]] = {}
    all_sites: set[str] = set()

    for split in splits:
        split_dir = chip_dir / split
        if split_dir.exists():
            dist = get_site_distribution(split_dir)
            split_sites[split] = dist
            all_sites.update(dist.keys())

    if not split_sites:
        return [], [], False

    # Summary data
    split_totals = {split: sum(sites.values()) for split, sites in split_sites.items()}
    grand_total = sum(split_totals.values())

    summary_data = []
    for split in splits:
        if split in split_sites:
            n_sites = len(split_sites[split])
            n_chips = split_totals[split]
            pct = 100 * n_chips / grand_total if grand_total > 0 else 0
            summary_data.append([split, str(n_sites), f"{n_chips:,}", f"{pct:.1f}%"])
    summary_data.append(["TOTAL", str(len(all_sites)), f"{grand_total:,}", ""])

    # Site data (sorted by total chips descending)
    site_totals = {}
    for site in all_sites:
        total = sum(split_sites.get(split, {}).get(site, 0) for split in splits)
        site_totals[site] = total

    site_data = []
    for site in sorted(site_totals.keys(), key=lambda s: site_totals[s], reverse=True):
        row = [site]
        for split in splits:
            if split in split_sites:
                count = split_sites[split].get(site, 0)
                row.append(f"{count:,}" if count > 0 else "-")
        row.append(f"{site_totals[site]:,}")
        site_data.append(row)

    # Check for leakage
    sites_per_split = {split: set(sites.keys()) for split, sites in split_sites.items()}
    has_leakage = False
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2:
                continue
            if (
                s1 in sites_per_split
                and s2 in sites_per_split
                and sites_per_split[s1] & sites_per_split[s2]
            ):
                has_leakage = True
                break

    return site_data, summary_data, has_leakage


def get_class_balance_stats(
    chip_dir: Path,
    class_names: dict[int, str],
    ignore_index: int = -100,
) -> dict:
    """Get class balance statistics without printing."""
    files = list(chip_dir.glob("*.npz"))
    if not files:
        return {}

    counts = Counter()
    chips_with_class = Counter()

    for f in files:
        label = np.load(f)["label"]
        unique, c = np.unique(label, return_counts=True)
        for u, cnt in zip(unique, c):
            counts[int(u)] += cnt
            chips_with_class[int(u)] += 1

    total_pixels = sum(v for k, v in counts.items() if k != ignore_index)
    total_chips = len(files)

    non_ignore = {k: v for k, v in counts.items() if k != ignore_index}
    if len(non_ignore) >= 2:
        ratio = max(non_ignore.values()) / min(non_ignore.values())
    else:
        ratio = 1.0

    return {
        "total_chips": total_chips,
        "total_pixels": total_pixels,
        "counts": dict(counts),
        "chips_with_class": dict(chips_with_class),
        "imbalance_ratio": ratio,
    }


def generate_stats_pdf(
    chip_dir: Path,
    output_path: Path,
    class_names: dict[int, str] | None = None,
    ignore_index: int = -100,
) -> None:
    """Generate PDF with chip counts and class balance stats, also prints to console."""
    if class_names is None:
        class_names = {0: "bg", 1: "class_1"}

    splits = ["train", "val", "test"]

    # Collect stats once
    all_stats = {}
    counts_data = []
    total_chips = 0

    print("\n" + "=" * 60)
    print("  CHIP COUNTS")
    print("=" * 60)
    print(f"\nDataset: {chip_dir}")
    print("-" * 40)
    print(f"{'Split':<10} {'Chips':>15}")
    print("-" * 40)

    for split in splits:
        split_dir = chip_dir / split
        if split_dir.exists():
            count = len(list(split_dir.glob("*.npz")))
            counts_data.append([split, f"{count:,}"])
            total_chips += count
            print(f"{split:<10} {count:>15,}")
            all_stats[split] = get_class_balance_stats(
                split_dir, class_names, ignore_index
            )
        else:
            print(f"{split:<10} {'(not found)':>15}")

    counts_data.append(["TOTAL", f"{total_chips:,}"])
    print("-" * 40)
    print(f"{'TOTAL':<10} {total_chips:>15,}")
    print("-" * 40)

    # Print class balance to console
    for split in splits:
        if split in all_stats and all_stats[split]:
            stats = all_stats[split]
            print(f"\n{'=' * 60}")
            print(f"  {split.upper()} - CLASS BALANCE")
            print(f"{'=' * 60}")
            print(f"\nTotal chips: {stats['total_chips']:,}")
            print(f"Total pixels (excluding ignore): {stats['total_pixels']:,}")
            print("\nClass distribution:")
            print("-" * 60)
            print(f"{'Class':<15} {'Pixels':>15} {'%':>8}")
            print("-" * 60)
            for k in sorted(stats["counts"].keys()):
                if k == ignore_index:
                    name = "ignore"
                    pct = 0
                else:
                    name = class_names.get(k, f"class_{k}")
                    pct = stats["counts"][k] / stats["total_pixels"] * 100
                print(f"{name:<15} {stats['counts'][k]:>15,} {pct:>7.1f}%")
            print("-" * 60)
            print(f"\nImbalance ratio (max/min): {stats['imbalance_ratio']:.1f}:1")

    # Generate PDF
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Chip counts table
        ax_counts = axes[0]
        ax_counts.axis("off")
        ax_counts.set_title("Chip Counts", fontsize=14, fontweight="bold", loc="left")

        table1 = ax_counts.table(
            cellText=counts_data,
            colLabels=["Split", "Chips"],
            loc="upper left",
            cellLoc="left",
            colWidths=[0.3, 0.3],
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 1.5)

        # Class balance table
        ax_balance = axes[1]
        ax_balance.axis("off")
        ax_balance.set_title(
            "Class Balance", fontsize=14, fontweight="bold", loc="left"
        )

        balance_data = []
        for split in splits:
            if split in all_stats and all_stats[split]:
                stats = all_stats[split]
                for k in sorted(stats["counts"].keys()):
                    if k == ignore_index:
                        continue
                    name = class_names.get(k, f"class_{k}")
                    pct = stats["counts"][k] / stats["total_pixels"] * 100
                    chip_count = stats["chips_with_class"].get(k, 0)
                    chip_pct = chip_count / stats["total_chips"] * 100
                    balance_data.append(
                        [
                            split,
                            name,
                            f"{stats['counts'][k]:,}",
                            f"{pct:.1f}%",
                            f"{chip_count:,}",
                            f"{chip_pct:.1f}%",
                        ]
                    )
                balance_data.append(
                    [
                        split,
                        "Imbalance ratio",
                        f"{stats['imbalance_ratio']:.1f}:1",
                        "",
                        "",
                        "",
                    ]
                )
                balance_data.append(["", "", "", "", "", ""])

        if balance_data:
            table2 = ax_balance.table(
                cellText=balance_data,
                colLabels=["Split", "Class", "Pixels", "%", "Chips", "% Chips"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.15, 0.18, 0.18, 0.12, 0.15, 0.12],
            )
            table2.auto_set_font_size(False)
            table2.set_fontsize(9)
            table2.scale(1, 1.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Site distribution
        site_data, summary_data, has_leakage = get_site_distribution_for_pdf(chip_dir)
        if site_data:
            # Print site distribution to console
            check_site_balance(chip_dir, all_splits=True)

            fig, axes = plt.subplots(2, 1, figsize=(10, 12))

            # Summary table
            ax_summary = axes[0]
            ax_summary.axis("off")
            ax_summary.set_title(
                "Site Distribution Summary", fontsize=14, fontweight="bold", loc="left"
            )

            table_summary = ax_summary.table(
                cellText=summary_data,
                colLabels=["Split", "Sites", "Chips", "% of Total"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.2, 0.15, 0.2, 0.15],
            )
            table_summary.auto_set_font_size(False)
            table_summary.set_fontsize(10)
            table_summary.scale(1, 1.5)

            # Site breakdown table (show top 30 sites to fit on page)
            ax_sites = axes[1]
            ax_sites.axis("off")
            title = "Site Breakdown (by chip count)"
            if has_leakage:
                title += " - WARNING: Site leakage detected!"
            ax_sites.set_title(title, fontsize=14, fontweight="bold", loc="left")

            display_sites = site_data[:30]
            if len(site_data) > 30:
                display_sites.append(["...", "...", "...", "...", "..."])

            table_sites = ax_sites.table(
                cellText=display_sites,
                colLabels=["Site", "train", "val", "test", "Total"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.35, 0.12, 0.12, 0.12, 0.12],
            )
            table_sites.auto_set_font_size(False)
            table_sites.set_fontsize(8)
            table_sites.scale(1, 1.2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nSaved stats to: {output_path}")


def generate_report(
    chip_dir: Path,
    output_path: Path,
    n_samples: int = 48,
    class_names: dict[int, str] | None = None,
    ignore_index: int = -100,
    seed: int = 42,
) -> None:
    """Generate comprehensive PDF report with stats and sample visualizations."""
    if class_names is None:
        class_names = {0: "bg", 1: "class_1"}

    splits = ["train", "val", "test"]
    n_classes = max(class_names.keys()) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes + 1)

    with PdfPages(output_path) as pdf:
        # Page 1: Summary statistics
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Chip counts table
        ax_counts = axes[0]
        ax_counts.axis("off")
        ax_counts.set_title("Chip Counts", fontsize=14, fontweight="bold", loc="left")

        counts_data = []
        total = 0
        for split in splits:
            split_dir = chip_dir / split
            if split_dir.exists():
                count = len(list(split_dir.glob("*.npz")))
                counts_data.append([split, f"{count:,}"])
                total += count
        counts_data.append(["TOTAL", f"{total:,}"])

        table1 = ax_counts.table(
            cellText=counts_data,
            colLabels=["Split", "Chips"],
            loc="upper left",
            cellLoc="left",
            colWidths=[0.3, 0.3],
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 1.5)

        # Class balance table
        ax_balance = axes[1]
        ax_balance.axis("off")
        ax_balance.set_title(
            "Class Balance", fontsize=14, fontweight="bold", loc="left"
        )

        balance_data = []
        for split in splits:
            split_dir = chip_dir / split
            if split_dir.exists():
                stats = get_class_balance_stats(split_dir, class_names, ignore_index)
                if stats:
                    for k in sorted(stats["counts"].keys()):
                        if k == ignore_index:
                            continue
                        name = class_names.get(k, f"class_{k}")
                        pct = stats["counts"][k] / stats["total_pixels"] * 100
                        chip_count = stats["chips_with_class"].get(k, 0)
                        chip_pct = chip_count / stats["total_chips"] * 100
                        balance_data.append(
                            [
                                split,
                                name,
                                f"{stats['counts'][k]:,}",
                                f"{pct:.1f}%",
                                f"{chip_count:,}",
                                f"{chip_pct:.1f}%",
                            ]
                        )
                    balance_data.append(
                        [
                            split,
                            "Imbalance ratio",
                            f"{stats['imbalance_ratio']:.1f}:1",
                            "",
                            "",
                            "",
                        ]
                    )
                    balance_data.append(["", "", "", "", "", ""])

        if balance_data:
            table2 = ax_balance.table(
                cellText=balance_data,
                colLabels=["Split", "Class", "Pixels", "%", "Chips", "% Chips"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.15, 0.18, 0.18, 0.12, 0.15, 0.12],
            )
            table2.auto_set_font_size(False)
            table2.set_fontsize(9)
            table2.scale(1, 1.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Site distribution
        site_data, summary_data, has_leakage = get_site_distribution_for_pdf(chip_dir)
        if site_data:
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))

            ax_summary = axes[0]
            ax_summary.axis("off")
            ax_summary.set_title(
                "Site Distribution Summary", fontsize=14, fontweight="bold", loc="left"
            )

            table_summary = ax_summary.table(
                cellText=summary_data,
                colLabels=["Split", "Sites", "Chips", "% of Total"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.2, 0.15, 0.2, 0.15],
            )
            table_summary.auto_set_font_size(False)
            table_summary.set_fontsize(10)
            table_summary.scale(1, 1.5)

            ax_sites = axes[1]
            ax_sites.axis("off")
            title = "Site Breakdown (by chip count)"
            if has_leakage:
                title += " - WARNING: Site leakage detected!"
            ax_sites.set_title(title, fontsize=14, fontweight="bold", loc="left")

            display_sites = site_data[:30]
            if len(site_data) > 30:
                display_sites.append(["...", "...", "...", "...", "..."])

            table_sites = ax_sites.table(
                cellText=display_sites,
                colLabels=["Site", "train", "val", "test", "Total"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.35, 0.12, 0.12, 0.12, 0.12],
            )
            table_sites.auto_set_font_size(False)
            table_sites.set_fontsize(8)
            table_sites.scale(1, 1.2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Sample visualization pages for each split
        for split in splits:
            split_dir = chip_dir / split
            if not split_dir.exists():
                continue

            files = list(split_dir.glob("*.npz"))
            if not files:
                continue

            random.seed(seed)
            samples = random.sample(files, min(n_samples, len(files)))

            samples_per_page = 4
            n_pages = (len(samples) + samples_per_page - 1) // samples_per_page

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

                        ax_img.imshow(img)
                        ax_img.set_title(f.name, fontsize=8)
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
                        ax_label.set_title("Label", fontsize=8)
                        ax_label.axis("off")
                    else:
                        ax_img.axis("off")
                        ax_label.axis("off")

                # Add split label and legend on first page of each split
                if page == 0:
                    fig.suptitle(
                        f"{split.upper()} samples", fontsize=12, fontweight="bold"
                    )
                    legend_elements = [
                        plt.Line2D(
                            [0],
                            [0],
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
                            [0],
                            [0],
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

    print(f"Saved report to: {output_path}")


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
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        metavar="OUTPUT.pdf",
        help="Generate comprehensive PDF report with stats and samples",
    )
    parser.add_argument(
        "--stats-pdf",
        type=Path,
        metavar="OUTPUT.pdf",
        help="Generate PDF with chip counts and class balance only",
    )
    parser.add_argument(
        "--site-balance",
        action="store_true",
        help="Check site distribution across splits (requires --all-splits or split subdirs)",
    )
    args = parser.parse_args()

    class_names = {i: name for i, name in enumerate(args.class_names)}
    expected_values = set(class_names.keys()) | {args.ignore_index}

    if args.report:
        generate_report(
            args.chip_dir,
            args.report,
            n_samples=args.n_samples,
            class_names=class_names,
            ignore_index=args.ignore_index,
            seed=args.seed,
        )
    elif args.stats_pdf:
        generate_stats_pdf(
            args.chip_dir,
            args.stats_pdf,
            class_names=class_names,
            ignore_index=args.ignore_index,
        )
    elif args.counts_only:
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
    elif args.site_balance:
        check_site_balance(args.chip_dir, all_splits=True)
    elif args.all_splits:
        print("\n" + "=" * 60)
        print("  CHIP COUNTS")
        print("=" * 60)
        check_chip_counts(args.chip_dir)

        for split in ["train", "val", "test"]:
            split_dir = args.chip_dir / split
            if split_dir.exists():
                print(f"\n{'=' * 60}")
                print(f"  {split.upper()} - CLASS BALANCE")
                print(f"{'=' * 60}")
                check_class_balance(split_dir, class_names, args.ignore_index)
            else:
                print(f"\nSkipping {split} (not found: {split_dir})")
    else:
        check_class_balance(args.chip_dir, class_names, args.ignore_index)
