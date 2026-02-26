#!/bin/bash
set -e

# Run augmentation comparison experiments on small prototype.
# Expected time: ~3-4 hours per run, ~6-8 hours total

# CONFIG PATHS -----
CONFIG_DIR="configs/seagrass-rgb/small"
MINIMAL_AUG="$CONFIG_DIR/unetpp_resnet34_minimal_aug.yaml"
FULL_AUG="$CONFIG_DIR/unetpp_resnet34.yaml"

# RUN EXPERIMENTS -----
echo "=============================================="
echo "Small prototype augmentation comparison"
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
echo "Check W&B project 'seagrass-rgb' group 'small_prototype' for results."
