#!/bin/bash
# Chip dataset inspection utilities
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"
OUTPUT_DIR="outputs"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"
N_SAMPLES=16

# RUN CHECKS -----
mkdir -p "$OUTPUT_DIR"

python -m src.prepare.check_class_balance "$CHIP_DIR" --all-splits --class-names $CLASS_NAMES

# Generate sample visualizations
echo ""
echo "Generating sample visualizations..."
for split in train val test; do
    if [ -d "$CHIP_DIR/$split" ]; then
        python -m src.prepare.check_class_balance "$CHIP_DIR/$split" \
            --visualize "$OUTPUT_DIR/samples_${split}.pdf" \
            --class-names $CLASS_NAMES \
            --n-samples "$N_SAMPLES"
    fi
done

# Disk usage
echo ""
echo "Disk usage:"
du -sh "$CHIP_DIR"/* 2>/dev/null || echo "Could not calculate disk usage"
