#!/bin/bash
# Generate per-ortho prediction PDFs for diagnosing errors by site
set -e

# CONFIG -----
CONFIG="configs/seagrass-rgb/final/segformer_final.yaml"
CKPT="/mnt/class_data/sdalgarno/checkpoints/final/last-v1.ckpt"

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024/test"
OUTPUT_DIR="outputs/visualize-pred-ortho"

# PARAMETERS -----
N_PER_ORTHO=50
THRESHOLD=0.5

# RUN -----
mkdir -p "$OUTPUT_DIR"

echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Samples per ortho: $N_PER_ORTHO"
echo ""

python -m src.predict.visualize_by_ortho "$CHIP_DIR" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --output "$OUTPUT_DIR/by_ortho.pdf" \
    --n-per-ortho "$N_PER_ORTHO" \
    --class-names bg seagrass \
    --threshold "$THRESHOLD"

echo ""
echo "PDF saved to: $OUTPUT_DIR/by_ortho.pdf"
