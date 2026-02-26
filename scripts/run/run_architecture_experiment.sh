#!/bin/bash
# Run architecture experiment: compare UNet++ vs SegFormer at 512 vs 1024
set -e

# CONFIG DIR -----
CONFIG_DIR="configs/seagrass-rgb/architecture-experiment"

# ALL CONFIGS (for dev validation) -----
CONFIGS=(
    "$CONFIG_DIR/unetpp_resnet34_512.yaml"
    "$CONFIG_DIR/unetpp_resnet34_1024.yaml"
    "$CONFIG_DIR/segformer_mitb2_512.yaml"
    "$CONFIG_DIR/segformer_mitb2_1024.yaml"
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
echo "Architecture Experiment"
echo "=============================================="
echo ""
echo "Comparing:"
echo "  - UNet++ ResNet34 (512 vs 1024)"
echo "  - SegFormer mit-b2 (512 vs 1024)"
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
echo "Architecture experiment complete"
echo "=============================================="
