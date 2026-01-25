#!/bin/bash
set -e

# USAGE -----
# ./scripts/evaluate_model.sh <config_path> <checkpoint_path>
#
# Example:
# ./scripts/evaluate_model.sh configs/seagrass-rgb/unetpp_resnet34_minimal_aug.yaml checkpoints/best.ckpt

# CONFIG -----
CONFIG="${1:?Error: config path required}"
CKPT="${2:?Error: checkpoint path required}"

VAL_DIR="/mnt/class_data/sdalgarno/prototype/chips_512/val"
TEST_DIR="/mnt/class_data/sdalgarno/prototype/chips_512/test"
CLASS_NAMES="bg seagrass"
BATCH_SIZE=16
NUM_WORKERS=4

# OUTPUT -----
CONFIG_NAME=$(basename "${CONFIG%.yaml}")
OUTPUT="reports/${CONFIG_NAME}_eval.pdf"

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
