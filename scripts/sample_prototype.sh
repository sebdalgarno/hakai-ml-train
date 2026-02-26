#!/bin/bash
# Sample prototype dataset with equal representation per site
set -e

# DIRECTORIES -----
MAIN_CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024"
PROTOTYPE_CHIP_DIR="/mnt/class_data/sdalgarno/prototype_frac_50/chips_1024"

# SAMPLING PARAMETERS -----
PROTOTYPE_FRACTION=0.5  # Fraction applied to smallest site, then sampled equally from all
SEED=42

# MAIN -----
echo "=============================================="
echo "Sampling prototype dataset"
echo "=============================================="
echo "Source: $MAIN_CHIP_DIR"
echo "Output: $PROTOTYPE_CHIP_DIR"
echo "Fraction: $PROTOTYPE_FRACTION"
echo "Seed: $SEED"
echo ""

python -m src.prepare.sample_prototype_by_site "$MAIN_CHIP_DIR" "$PROTOTYPE_CHIP_DIR" \
    --fraction "$PROTOTYPE_FRACTION" \
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
