#!/bin/bash
set -e

# Prepare seagrass chip datasets with site-balanced prototype sampling.
#
# Workflow:
# 1. Create all chips from orthos (no sampling)
# 2. Remove tiles with nodata areas
# 3. Sample prototype dataset with equal representation per site

# DIRECTORIES -----
RAW_DIR="/mnt/class_data/sdalgarno/main/raw_data"
MAIN_CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024"
PROTOTYPE_CHIP_DIR="/mnt/class_data/sdalgarno/prototype_frac_15/chips_1024"

# CHIP PARAMETERS -----
CHIP_SIZE=1024
TRAIN_STRIDE=1024  # no overlap
EVAL_STRIDE=1024    # No overlap
NUM_BANDS=3
DTYPE="uint8"
REMAP="0 -100 1"

# PROTOTYPE SAMPLING -----
PROTOTYPE_FRACTION=0.15  # Fraction applied to smallest site, then sampled equally from all
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
    --seed "$SEED" \
    --parallel
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
    --fraction "$PROTOTYPE_FRACTION" \
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
