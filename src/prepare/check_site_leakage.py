"""Check for site leakage across train/val/test splits."""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    site_counts = Counter()
    for f in chip_dir.glob("*.npz"):
        site = parse_site_name(f.name)
        site_counts[site] += 1
    return dict(site_counts)


def analyze_site_leakage(
    chip_dir: Path,
) -> tuple[dict[str, dict[str, int]], list[tuple[str, str, set[str]]]]:
    """Analyze site distribution and detect leakage across splits.

    Returns:
        split_sites: Mapping of split name to {site: count} dict
        leakage: List of (split1, split2, overlapping_sites) tuples
    """
    splits = ["train", "val", "test"]
    split_sites: dict[str, dict[str, int]] = {}

    for split in splits:
        split_dir = chip_dir / split
        if split_dir.exists():
            dist = get_site_distribution(split_dir)
            if dist:
                split_sites[split] = dist

    # Detect leakage: check each unique pair of splits
    sites_per_split = {s: set(sites.keys()) for s, sites in split_sites.items()}
    leakage = []
    present = [s for s in splits if s in sites_per_split]
    for i, s1 in enumerate(present):
        for s2 in present[i + 1 :]:
            overlap = sites_per_split[s1] & sites_per_split[s2]
            if overlap:
                leakage.append((s1, s2, overlap))

    return split_sites, leakage


def check_site_leakage(chip_dir: Path) -> None:
    """Check and display site distribution across splits."""
    split_sites, leakage = analyze_site_leakage(chip_dir)

    if not split_sites:
        print("No chip files found.")
        return

    splits = ["train", "val", "test"]
    all_sites: set[str] = set()
    for sites in split_sites.values():
        all_sites.update(sites.keys())

    # Print header
    print("\n" + "=" * 80)
    print("  SITE DISTRIBUTION")
    print("=" * 80)

    # Calculate totals per split
    split_totals = {s: sum(sites.values()) for s, sites in split_sites.items()}
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
    for site in all_sites:
        total = sum(split_sites.get(s, {}).get(site, 0) for s in splits)
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

    # Print leakage results
    if leakage:
        print("\nWARNING: Site leakage detected!")
        for s1, s2, overlap in leakage:
            print(f"  {s1} & {s2} share sites: {sorted(overlap)}")
    else:
        print("\nNo site leakage detected (each site appears in only one split).")


def generate_leakage_pdf(chip_dir: Path, output_path: Path) -> None:
    """Generate PDF with site distribution and leakage analysis."""
    split_sites, leakage = analyze_site_leakage(chip_dir)

    if not split_sites:
        print("No chip files found.")
        return

    splits = ["train", "val", "test"]
    all_sites: set[str] = set()
    for sites in split_sites.values():
        all_sites.update(sites.keys())

    split_totals = {s: sum(sites.values()) for s, sites in split_sites.items()}
    grand_total = sum(split_totals.values())

    # Build summary table data
    summary_data = []
    for split in splits:
        if split in split_sites:
            n_sites = len(split_sites[split])
            n_chips = split_totals[split]
            pct = 100 * n_chips / grand_total if grand_total > 0 else 0
            summary_data.append([split, str(n_sites), f"{n_chips:,}", f"{pct:.1f}%"])
    summary_data.append(["TOTAL", str(len(all_sites)), f"{grand_total:,}", ""])

    # Build site breakdown table data (sorted by total chips descending)
    site_totals = {}
    for site in all_sites:
        total = sum(split_sites.get(s, {}).get(site, 0) for s in splits)
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

    has_leakage = len(leakage) > 0

    # Generate PDF
    with PdfPages(output_path) as pdf:
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

    print(f"Saved site leakage report to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for site leakage across train/val/test splits"
    )
    parser.add_argument(
        "chip_dir",
        type=Path,
        help="Parent directory containing train/val/test subdirs",
    )
    parser.add_argument(
        "--stats-pdf",
        type=Path,
        metavar="OUTPUT.pdf",
        help="Generate PDF with site distribution and leakage analysis",
    )
    args = parser.parse_args()

    # Always print console report
    check_site_leakage(args.chip_dir)

    # Optionally generate PDF
    if args.stats_pdf:
        generate_leakage_pdf(args.chip_dir, args.stats_pdf)
