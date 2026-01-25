"""Generate PDF summary report of chip counts by site and ortho."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
        potential_site = match.group(1)
        if not re.search(r"_u\d+$", potential_site):
            return potential_site

    return stem


def extract_ortho_name(filename: str) -> str:
    """Extract ortho name from chip filename.

    Handles formats like:
    - koeye_u0388_42.npz -> koeye_u0388
    - pruth_bay_u0699_123.npz -> pruth_bay_u0699
    - bennett_bay_42.npz -> bennett_bay
    """
    stem = Path(filename).stem

    # Pattern: {ortho}_{chip_idx}
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        return match.group(1)

    return stem


def count_chips(chip_dir: Path) -> dict:
    """Count chips by site and ortho for each split."""
    results = {}

    for split in ["train", "val", "test"]:
        split_dir = chip_dir / split
        if not split_dir.exists():
            continue

        chips = list(split_dir.glob("*.npz"))

        by_site = defaultdict(int)
        by_ortho = defaultdict(int)

        for chip in chips:
            site = extract_site_name(chip.name)
            ortho = extract_ortho_name(chip.name)
            by_site[site] += 1
            by_ortho[ortho] += 1

        results[split] = {
            "total": len(chips),
            "by_site": dict(sorted(by_site.items())),
            "by_ortho": dict(sorted(by_ortho.items())),
        }

    return results


def create_table_figure(data: dict, title: str, figsize=(10, None)):
    """Create a matplotlib figure with a table."""
    rows = [[k, v] for k, v in data.items()]
    rows.append(["TOTAL", sum(data.values())])

    # Calculate figure height based on number of rows
    n_rows = len(rows)
    fig_height = max(4, 0.4 * n_rows)

    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=["Name", "Count"],
        loc="center",
        cellLoc="left",
        colWidths=[0.6, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style total row
    for j in range(2):
        table[(n_rows, j)].set_facecolor("#D9E2F3")
        table[(n_rows, j)].set_text_props(fontweight="bold")

    plt.tight_layout()
    return fig


def create_bar_chart(data: dict, title: str, figsize=(12, 6)):
    """Create a horizontal bar chart."""
    names = list(data.keys())
    counts = list(data.values())

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, counts, color="#4472C4")
    ax.set_xlabel("Chip Count")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, max(counts) * 1.15)
    plt.tight_layout()
    return fig


def generate_report(chip_dir: Path, output_path: Path):
    """Generate PDF report of chip counts."""
    results = count_chips(chip_dir)

    if not results:
        print(f"No chips found in {chip_dir}")
        return

    with PdfPages(output_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.text(
            0.5,
            0.6,
            "Chip Dataset Summary",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        ax.text(0.5, 0.4, str(chip_dir), ha="center", va="center", fontsize=12)

        # Summary stats
        summary_text = []
        for split, data in results.items():
            summary_text.append(f"{split}: {data['total']:,} chips")
        total_all = sum(d["total"] for d in results.values())
        summary_text.append(f"\nTotal: {total_all:,} chips")

        ax.text(
            0.5, 0.2, "\n".join(summary_text), ha="center", va="center", fontsize=14
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Site breakdown for each split
        for split, data in results.items():
            # Site table
            fig = create_table_figure(
                data["by_site"], f"{split.upper()} - Chips by Site"
            )
            pdf.savefig(fig)
            plt.close(fig)

            # Site bar chart (if not too many sites)
            if len(data["by_site"]) <= 30:
                fig = create_bar_chart(
                    data["by_site"], f"{split.upper()} - Chips by Site"
                )
                pdf.savefig(fig)
                plt.close(fig)

        # Ortho breakdown for each split
        for split, data in results.items():
            # Ortho table
            fig = create_table_figure(
                data["by_ortho"], f"{split.upper()} - Chips by Ortho"
            )
            pdf.savefig(fig)
            plt.close(fig)

            # Ortho bar chart (if not too many orthos)
            if len(data["by_ortho"]) <= 40:
                fig = create_bar_chart(
                    data["by_ortho"], f"{split.upper()} - Chips by Ortho"
                )
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF summary of chip counts by site and ortho."
    )
    parser.add_argument(
        "chip_dir",
        type=Path,
        help="Directory containing chips with structure {train,val,test}/*.npz",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: chip_dir/chip_summary.pdf)",
    )
    args = parser.parse_args()

    output_path = args.output or (args.chip_dir / "chip_summary.pdf")
    generate_report(args.chip_dir, output_path)


if __name__ == "__main__":
    main()
