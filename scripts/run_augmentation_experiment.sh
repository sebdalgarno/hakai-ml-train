#!/bin/bash
# Run augmentation experiment: compare tiered augmentation strategies
set -e

# CONFIG DIR -----
CONFIG_DIR="configs/seagrass-rgb/augmentation-experiment"

# CONFIGS (in order of increasing augmentation) -----
CONFIGS=(
    "$CONFIG_DIR/segformer_default_aug.yaml"
    "$CONFIG_DIR/segformer_scale_aug.yaml"
    "$CONFIG_DIR/segformer_domain_aug.yaml"
)

# DEV VALIDATION -----
echo "=============================================="
echo "Dev Validation (fast_dev_run)"
echo "=============================================="
echo ""
for config in "${CONFIGS[@]}"; do
    echo "Validating: $(basename "$config")"
    python trainer.py fit --config "$config" --trainer.fast_dev_run=true
    echo "  OK"
done
echo ""
echo "All configs validated successfully"
echo ""

# RUN -----
echo "=============================================="
echo "Augmentation Experiment"
echo "=============================================="
echo ""
echo "Comparing augmentation tiers (SegFormer mit-b2, 1024 tiles):"
echo "  1. baseline → D4 only"
echo "  2. default  → Baseline + brightness/contrast + noise + blur"
echo "  3. scale    → Default + affine scale (0.7-1.3)"
echo "  4. domain   → Scale + distortion + CLAHE + HueSaturationValue"
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
