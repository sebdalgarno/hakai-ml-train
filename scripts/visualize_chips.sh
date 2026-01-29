#!/bin/bash
# Generate sample visualization PDFs
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype_frac_25/chips_1024/"
OUTPUT_DIR="outputs/visualize-chips/"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"
N_SAMPLES=48

# RUN -----
mkdir -p "$OUTPUT_DIR"

for split in train val test; do
    if [ -d "$CHIP_DIR/$split" ]; then
        echo "Generating $split samples..."
        python -m src.prepare.check_class_balance "$CHIP_DIR/$split" \
            --visualize "$OUTPUT_DIR/samples_${split}.pdf" \
            --class-names $CLASS_NAMES \
            --n-samples "$N_SAMPLES"
    fi
done

echo ""
echo "PDFs saved to: $OUTPUT_DIR/"
