#!/bin/bash
# Generate prediction visualization PDFs
set -e

# CONFIG -----
CONFIG="configs/seagrass-rgb/architecture-experiment/segformer_mitb2_1024.yaml"
CKPT="seagrass-rgb/jhf1t0ih/checkpoints/last.ckpt"

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype_frac_25/chips_1024"
OUTPUT_DIR="outputs/visualize-pred"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"
N_SAMPLES=60
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
