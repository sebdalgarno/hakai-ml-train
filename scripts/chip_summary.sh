#!/bin/bash
set -e

# Generate PDF summary of chip counts by site and ortho.
#
# Usage:
#   ./scripts/chip_summary.sh /path/to/chips
#   ./scripts/chip_summary.sh /path/to/chips -o /path/to/output.pdf

if [ $# -lt 1 ]; then
    echo "Usage: $0 <chip_dir> [-o output.pdf]"
    echo ""
    echo "Arguments:"
    echo "  chip_dir    Directory containing chips with structure {train,val,test}/*.npz"
    echo "  -o          Output PDF path (default: chip_dir/chip_summary.pdf)"
    exit 1
fi

python -m src.prepare.chip_summary_report "$@"
