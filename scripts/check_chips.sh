#!/bin/bash
# Chip dataset inspection utilities
set -e

# =============================================================================
# CONFIGURE THESE PATHS
# =============================================================================
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"
OUTPUT_DIR="outputs"

# Class names (space-separated, in order)
CLASS_NAMES="bg seagrass"

# Number of samples per split for visualization
N_SAMPLES=16

# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# CHECK CHIP COUNTS AND CLASS BALANCE
# =============================================================================
python -m src.prepare.check_class_balance "$CHIP_DIR" --all-splits --class-names $CLASS_NAMES

# =============================================================================
# GENERATE SAMPLE VISUALIZATIONS
# =============================================================================
echo ""
echo "============================================================"
echo "  GENERATING SAMPLE VISUALIZATIONS"
echo "============================================================"
for split in train val test; do
    if [ -d "$CHIP_DIR/$split" ]; then
        python -m src.prepare.check_class_balance "$CHIP_DIR/$split" \
            --visualize "$OUTPUT_DIR/samples_${split}.pdf" \
            --class-names $CLASS_NAMES \
            --n-samples "$N_SAMPLES"
    fi
done

# =============================================================================
# DISK USAGE
# =============================================================================
echo ""
echo "============================================================"
echo "  DISK USAGE"
echo "============================================================"
du -sh "$CHIP_DIR"/* 2>/dev/null || echo "Could not calculate disk usage"

# =============================================================================
# ADD MORE CHECKS BELOW
# =============================================================================
