#!/bin/bash
# Create chips from raw GeoTIFF mosaics and remove nodata tiles
set -e

# DIRECTORIES -----
RAW_DIR="/mnt/class_data/sdalgarno/main/raw_data"
CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024"

# CHIP PARAMETERS -----
CHIP_SIZE=1024
STRIDE=1024  # no overlap
NUM_BANDS=3
DTYPE="uint8"
REMAP="0 -100 1"

# STEP 1: Create chips -----
echo "Step 1: Creating chips from orthos..."
echo "  Raw data: $RAW_DIR"
echo "  Output: $CHIP_DIR"
echo "  Size: ${CHIP_SIZE}px, Stride: ${STRIDE}px"
echo ""

python -m src.prepare.make_chip_dataset "$RAW_DIR" "$CHIP_DIR" \
    --size "$CHIP_SIZE" \
    --stride "$STRIDE" \
    --num_bands "$NUM_BANDS" \
    --dtype "$DTYPE" \
    --remap $REMAP

# STEP 2: Remove nodata tiles -----
echo ""
echo "Step 2: Removing tiles with nodata areas..."
for split in train val test; do
    if [ -d "$CHIP_DIR/$split" ]; then
        echo "  Processing $split..."
        python -m src.prepare.remove_tiles_with_nodata_areas "$CHIP_DIR/$split" --num_channels "$NUM_BANDS"
    fi
done

# SUMMARY -----
echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "Chips: $CHIP_DIR"
echo ""
echo "Chip counts:"
for split in train val test; do
    if [ -d "$CHIP_DIR/$split" ]; then
        count=$(ls -1 "$CHIP_DIR/$split"/*.npz 2>/dev/null | wc -l | tr -d ' ')
        echo "  $split: $count"
    fi
done
