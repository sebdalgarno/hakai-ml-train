#!/bin/bash
# Compute channel statistics for normalization
# Run this after prepare_data.sh
set -e

# PATHS -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"

# Max pixel value (255 for uint8, 65535 for uint16)
MAX_PIXEL_VAL=255.0

# COMPUTE STATS -----
echo "Computing channel statistics from: $CHIP_DIR/train"
echo "Max pixel value: $MAX_PIXEL_VAL"
echo ""

python -m src.prepare.channel_stats "$CHIP_DIR/train" --max_pixel_val "$MAX_PIXEL_VAL"

echo ""
echo "Use these mean/std values in your training config's Normalize transform."
