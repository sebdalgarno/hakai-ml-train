#!/bin/bash
# Run regional cross-validation experiment
set -e

# CONFIG DIR -----
CONFIG_DIR="configs/seagrass-rgb/regional-cv"

# CONFIGS -----
CONFIGS=(
    "$CONFIG_DIR/segformer_cv_north.yaml"
    "$CONFIG_DIR/segformer_cv_central.yaml"
    "$CONFIG_DIR/segformer_cv_south.yaml"
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
echo "Regional Cross-Validation Experiment"
echo "=============================================="
echo ""
echo "3-fold regional CV:"
echo "  1. cv_north   → North sites held out as test"
echo "  2. cv_central → Central sites held out as test"
echo "  3. cv_south   → South sites held out as test"
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
echo "Regional CV experiment complete"
echo "=============================================="
