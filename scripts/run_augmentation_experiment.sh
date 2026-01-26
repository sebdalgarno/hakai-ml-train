#!/bin/bash
# Run augmentation experiment: systematic comparison of augmentation strategies
set -e

# CONFIG DIR -----
CONFIG_DIR="configs/seagrass-rgb/augmentation-experiment"

# CONFIGS (in order of increasing augmentation) -----
CONFIGS=(
    "$CONFIG_DIR/unetpp_resnet34_no_aug.yaml"
    "$CONFIG_DIR/unetpp_resnet34_geometric_aug.yaml"
    "$CONFIG_DIR/unetpp_resnet34_scale_aug.yaml"
    "$CONFIG_DIR/unetpp_resnet34_full_aug.yaml"
)

# RUN -----
echo "=============================================="
echo "Augmentation Experiment"
echo "=============================================="
echo ""
echo "Progression:"
echo "  1. no_aug       → Normalize only"
echo "  2. geometric    → D4 + Affine(rotate, translate)"
echo "  3. scale        → D4 + Affine(+scale) + RandomResizedCrop"
echo "  4. full         → Scale + color + turbidity + distortion + dropout"
echo ""
echo "Configs to run:"
for config in "${CONFIGS[@]}"; do
    echo "  - $(basename "$config")"
done
echo ""

for config in "${CONFIGS[@]}"; do
    echo "=============================================="
    echo "Running: $(basename "$config")"
    echo "=============================================="
    python trainer.py fit --config "$config"
    echo ""
    echo "Completed: $(basename "$config")"
    echo ""
done

echo "=============================================="
echo "Augmentation experiment complete"
echo "=============================================="
