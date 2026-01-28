#!/bin/bash
# Generate prediction visualization PDFs
set -e

# CONFIG -----
CONFIG="configs/seagrass-rgb/architecture-experiment/segformer_mitb2_1024.yaml"
CKPT="/mnt/class_data/sdalgarno/checkpoints/architecture-experiment/segformer_mitb2_1024/last.ckpt"

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips_1024"
OUTPUT_BASE="outputs/"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"
N_SAMPLES=60
THRESHOLD=0.5

# EXTRACT MODEL NAME -----
# Extract model name from checkpoint path (e.g., ".../architecture-experiment/segformer_mitb2_1024/last.ckpt" -> "segformer_mitb2_1024")
MODEL_NAME=$(basename "$(dirname "$CKPT")")
OUTPUT_DIR="$OUTPUT_BASE/$MODEL_NAME"

# RUN -----
mkdir -p "$OUTPUT_DIR"

echo "Model: $MODEL_NAME"
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
