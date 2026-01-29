#!/bin/bash
# Generate prediction visualization PDFs
set -e

# CONFIG -----
CONFIG="configs/seagrass-rgb/segformer_train50.yaml"
CKPT="/mnt/class_data/sdalgarno/checkpoints/segformer-train50/last.ckpt"

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype_frac_75/chips_1024"
OUTPUT_DIR="outputs/visualize-pred"

# PARAMETERS -----
CLASS_NAMES="Background Eelgrass"
N_SAMPLES=100
THRESHOLD=0.5

# RUN -----
mkdir -p "$OUTPUT_DIR"

echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Output directory: $OUTPUT_DIR"
echo ""

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
