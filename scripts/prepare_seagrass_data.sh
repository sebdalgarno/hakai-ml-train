#!/bin/bash
set -e

# Prepare seagrass chip datasets with site-balanced prototype sampling.
#
# Workflow:
# 1. Create all chips from orthos (no sampling)
# 2. Remove tiles with nodata areas
# 3. Sample prototype dataset with equal representation per site

# DIRECTORIES -----
RAW_DIR="/mnt/class_data/sdalgarno/prototype/raw_data"
MAIN_CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"
PROTOTYPE_CHIP_DIR="/mnt/class_data/sdalgarno/prototype2/chips"

# CHIP PARAMETERS -----
CHIP_SIZE=512
TRAIN_STRIDE=256   # 50% overlap
EVAL_STRIDE=512    # No overlap
NUM_BANDS=3
DTYPE="uint8"
REMAP="0 -100 1"

# PROTOTYPE SAMPLING -----
CHIPS_PER_SITE=500  # Fixed number of chips per site (use --fraction instead if preferred)
SEED=42

# Step 1: Create all chips -----
echo "Step 1: Creating chips from orthos..."
python -m src.prepare.make_chip_dataset_sampled "$RAW_DIR" "$MAIN_CHIP_DIR" \
    --size "$CHIP_SIZE" \
    --train-stride "$TRAIN_STRIDE" \
    --eval-stride "$EVAL_STRIDE" \
    --num-bands "$NUM_BANDS" \
    --dtype "$DTYPE" \
    --remap $REMAP \
    --seed "$SEED"
# Note: No --prototype-output, so no sampling during chip creation

# Step 2: Remove nodata tiles -----
echo ""
echo "Step 2: Removing tiles with nodata areas..."
for split in train val test; do
    if [ -d "$MAIN_CHIP_DIR/$split" ]; then
        echo "  Processing $split..."
        python -m src.prepare.remove_tiles_with_nodata_areas "$MAIN_CHIP_DIR/$split" --num_channels "$NUM_BANDS"
    fi
done

# Step 3: Sample prototype by site -----
echo ""
echo "Step 3: Sampling prototype dataset (equal chips per site)..."
python -m src.prepare.sample_prototype_by_site "$MAIN_CHIP_DIR" "$PROTOTYPE_CHIP_DIR" \
    --chips-per-site "$CHIPS_PER_SITE" \
    --seed "$SEED"

# Summary -----
echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "Main chips: $MAIN_CHIP_DIR"
echo "Prototype chips: $PROTOTYPE_CHIP_DIR"
echo ""
echo "Chip counts:"
for dir in "$MAIN_CHIP_DIR" "$PROTOTYPE_CHIP_DIR"; do
    echo "  $(basename $(dirname $dir))/$(basename $dir):"
    for split in train val test; do
        if [ -d "$dir/$split" ]; then
            count=$(ls -1 "$dir/$split"/*.npz 2>/dev/null | wc -l | tr -d ' ')
            echo "    $split: $count"
        fi
    done
done
