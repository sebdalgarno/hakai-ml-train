#!/bin/bash
set -e

# DIRECTORIES -----
RAW_DIR="/mnt/class_data/sdalgarno/main/raw_data"
MAIN_CHIP_DIR="/mnt/class_data/sdalgarno/main/chips"
PROTOTYPE_CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"

# CHIP PARAMETERS -----
CHIP_SIZE=512
TRAIN_STRIDE=256   # 50% overlap
EVAL_STRIDE=512    # No overlap
NUM_BANDS=3
DTYPE="uint8"
REMAP="0 -100 1"

# PROTOTYPE -----
PROTOTYPE_FRACTION=0.10
SEED=42

# Create chips -----
python -m src.prepare.make_chip_dataset_sampled "$RAW_DIR" "$MAIN_CHIP_DIR" \
    --prototype-output "$PROTOTYPE_CHIP_DIR" \
    --size "$CHIP_SIZE" \
    --train-stride "$TRAIN_STRIDE" \
    --eval-stride "$EVAL_STRIDE" \
    --num-bands "$NUM_BANDS" \
    --dtype "$DTYPE" \
    --remap $REMAP \
    --prototype-fraction "$PROTOTYPE_FRACTION" \
    --seed "$SEED"

# Remove nodata tiles -----
for dir in "$MAIN_CHIP_DIR" "$PROTOTYPE_CHIP_DIR"; do
    for split in train val test; do
        if [ -d "$dir/$split" ]; then
            python -m src.prepare.remove_tiles_with_nodata_areas "$dir/$split" --num_channels "$NUM_BANDS"
        fi
    done
done
