#!/bin/bash
# Sample prototype dataset with equal representation per site
set -e

# DIRECTORIES -----
MAIN_CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024"
PROTOTYPE_CHIP_DIR="/mnt/class_data/sdalgarno/prototype_frac_50/chips_1024"

# SAMPLING PARAMETERS -----
# Use higher fraction for train to maintain ~73% train ratio
TRAIN_FRACTION=0.58
EVAL_FRACTION=0.50
SEED=42

# MAIN -----
echo "=============================================="
echo "Sampling prototype dataset"
echo "=============================================="
echo "Source: $MAIN_CHIP_DIR"
echo "Output: $PROTOTYPE_CHIP_DIR"
echo "Train fraction: $TRAIN_FRACTION"
echo "Val/Test fraction: $EVAL_FRACTION"
echo "Seed: $SEED"
echo ""

# Sample each split with appropriate fraction
echo "=== TRAIN ==="
python -m src.prepare.sample_prototype_by_site "$MAIN_CHIP_DIR/train" "$PROTOTYPE_CHIP_DIR/train" \
    --fraction "$TRAIN_FRACTION" \
    --seed "$SEED"

echo ""
echo "=== VAL ==="
python -m src.prepare.sample_prototype_by_site "$MAIN_CHIP_DIR/val" "$PROTOTYPE_CHIP_DIR/val" \
    --fraction "$EVAL_FRACTION" \
    --seed "$SEED"

echo ""
echo "=== TEST ==="
python -m src.prepare.sample_prototype_by_site "$MAIN_CHIP_DIR/test" "$PROTOTYPE_CHIP_DIR/test" \
    --fraction "$EVAL_FRACTION" \
    --seed "$SEED"

# SUMMARY -----
echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo ""
echo "Chip counts:"
for dir in "$MAIN_CHIP_DIR" "$PROTOTYPE_CHIP_DIR"; do
    echo "  $(dirname $dir | xargs basename)/$(basename $dir):"
    for split in train val test; do
        if [ -d "$dir/$split" ]; then
            count=$(ls -1 "$dir/$split"/*.npz 2>/dev/null | wc -l | tr -d ' ')
            echo "    $split: $count"
        fi
    done
done
