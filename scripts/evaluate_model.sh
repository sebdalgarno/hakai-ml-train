#!/bin/bash
set -e

# USAGE -----
# ./scripts/evaluate_model.sh <config_path> <checkpoint_path>
#
# Example:
# ./scripts/evaluate_model.sh configs/seagrass-rgb/unetpp_resnet34_minimal_aug.yaml ./seagrass-rgb/6rzxrcho/checkpoints/last.ckpt

# CONFIG -----
CONFIG="${1:?Error: config path required}"
CKPT="${2:?Error: checkpoint path required}"

VAL_DIR="/mnt/class_data/sdalgarno/prototype/chips_512/val"
TEST_DIR="/mnt/class_data/sdalgarno/prototype/chips_512/test"
CLASS_NAMES="bg seagrass"
BATCH_SIZE=16
NUM_WORKERS=4

# EXTRACT RUN ID -----
# Extract run ID from checkpoint path (e.g., "./seagrass-rgb/6rzxrcho/checkpoints/last.ckpt" -> "6rzxrcho")
RUN_ID=$(echo "$CKPT" | sed -E 's|.*/([^/]+)/checkpoints/.*|\1|')

# OUTPUT -----
OUTPUT_DIR="reports/$RUN_ID"
mkdir -p "$OUTPUT_DIR"
OUTPUT="$OUTPUT_DIR/eval.pdf"

echo "Run ID: $RUN_ID"
echo "Output: $OUTPUT"
echo ""

# RUN -----
python -m src.evaluate.generate_report \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --val-dir "$VAL_DIR" \
    --test-dir "$TEST_DIR" \
    --output "$OUTPUT" \
    --class-names $CLASS_NAMES \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
