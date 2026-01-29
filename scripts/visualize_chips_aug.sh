#!/bin/bash
# Generate visualization PDFs showing chips with augmentations applied
set -e

# USAGE -----
# ./scripts/visualize_chips_aug.sh <config_path>
#
# Examples:
# ./scripts/visualize_chips_aug.sh configs/seagrass-rgb/unetpp_resnet34_light_aug.yaml
# ./scripts/visualize_chips_aug.sh configs/seagrass-rgb/augmentation-experiment/segformer_domain_aug.yaml

# CONFIG -----
CONFIG="${1:?Error: config path required}"

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips_512"
OUTPUT_DIR="outputs/visualize-chips-aug"

# PARAMETERS -----
N_SAMPLES=16
N_AUGMENTATIONS=8

# OUTPUT -----
CONFIG_NAME=$(basename "${CONFIG%.yaml}")
OUTPUT="$OUTPUT_DIR/augmented_${CONFIG_NAME}.pdf"

# RUN -----
mkdir -p "$OUTPUT_DIR"

echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo ""

python -m src.prepare.visualize_augmented "$CHIP_DIR/train" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --n-samples "$N_SAMPLES" \
    --n-augmentations "$N_AUGMENTATIONS"

echo ""
echo "PDF saved to: $OUTPUT"
