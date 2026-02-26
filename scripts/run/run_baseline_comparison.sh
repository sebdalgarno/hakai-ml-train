#!/bin/bash
set -e

# Run baseline augmentation comparison experiments overnight.
# 1. UNet++ ResNet34 with minimal augmentation (D4 only)
# 2. UNet++ ResNet34 with full augmentation

# CONFIG PATHS -----
CONFIG_DIR="configs/seagrass-rgb"
MINIMAL_AUG="$CONFIG_DIR/unetpp_resnet34_minimal_aug.yaml"
FULL_AUG="$CONFIG_DIR/unetpp_resnet34.yaml"

# RUN EXPERIMENTS -----
echo "=============================================="
echo "Starting baseline augmentation comparison"
echo "=============================================="
echo ""

echo "[1/2] Training with MINIMAL augmentation..."
echo "Config: $MINIMAL_AUG"
echo ""
python trainer.py fit --config "$MINIMAL_AUG"

echo ""
echo "[2/2] Training with FULL augmentation..."
echo "Config: $FULL_AUG"
echo ""
python trainer.py fit --config "$FULL_AUG"

echo ""
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo "Check W&B project 'seagrass-rgb' for results."
