#!/bin/bash
# Generate prediction visualization PDFs
set -e

# CONFIG -----
CONFIG="configs/seagrass-rgb/unetpp_resnet34_1024.yaml"
CKPT="./seagrass-rgb/6rzxrcho/checkpoints/last.ckpt"

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips_1024"
OUTPUT_DIR="outputs"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"
N_SAMPLES=60
THRESHOLD=0.5

# RUN -----
mkdir -p "$OUTPUT_DIR"

for split in val test; do
    if [ -d "$CHIP_DIR/$split" ]; then
        echo "Generating predictions for $split..."
        python -m src.predict.visualize_predictions "$CHIP_DIR/$split" \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --output "$OUTPUT_DIR/predictions_${split}.pdf" \
            --n-samples "$N_SAMPLES" \
            --class-names $CLASS_NAMES \
            --threshold "$THRESHOLD"
    fi
done

echo ""
echo "PDFs saved to: $OUTPUT_DIR/"
