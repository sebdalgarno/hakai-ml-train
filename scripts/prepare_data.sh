#!/bin/bash
# Data preparation pipeline
# Creates chips from GeoTIFF mosaics for training
set -e  # Exit on error

# DIRECTORIES -----
RAW_DIR="/mnt/class_data/sdalgarno/prototype/raw_data"
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"


# CHIP PARAMETERS -----
CHIP_SIZE=224
CHIP_STRIDE=224
NUM_BANDS=3
DTYPE="uint8"

# Label remapping: index = old value, value = new value
# Example: 0 -100 1 means: 0->0 (bg), 1->-100 (ignore), 2->1 (class 1)
REMAP="0 -100 1"

# RUN PIPELINE -----
echo "Creating chips from: $RAW_DIR"
echo "Output directory: $CHIP_DIR"
echo "Chip size: ${CHIP_SIZE}x${CHIP_SIZE}, stride: $CHIP_STRIDE"
echo "Label remapping: $REMAP"
echo ""

# Create chips from mosaics
python -m src.prepare.make_chip_dataset "$RAW_DIR" "$CHIP_DIR" \
    --size "$CHIP_SIZE" \
    --stride "$CHIP_STRIDE" \
    --num_bands "$NUM_BANDS" \
    --dtype "$DTYPE" \
    --remap $REMAP

# Remove background-only tiles
# echo ""
# echo "Removing background-only tiles..."
# python -m src.prepare.remove_bg_only_tiles "$CHIP_DIR/train"
# python -m src.prepare.remove_bg_only_tiles "$CHIP_DIR/val"
# python -m src.prepare.remove_bg_only_tiles "$CHIP_DIR/test"

# Remove tiles with nodata areas
echo ""
echo "Removing tiles with nodata areas..."
python -m src.prepare.remove_tiles_with_nodata_areas "$CHIP_DIR/train" --num_channels "$NUM_BANDS"
python -m src.prepare.remove_tiles_with_nodata_areas "$CHIP_DIR/val" --num_channels "$NUM_BANDS"
python -m src.prepare.remove_tiles_with_nodata_areas "$CHIP_DIR/test" --num_channels "$NUM_BANDS"

echo ""
echo "Data preparation complete!"
echo "Chips saved to: $CHIP_DIR"
