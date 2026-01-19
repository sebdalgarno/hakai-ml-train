#!/bin/bash
# Export trained model to ONNX format
set -e

# =============================================================================
# CONFIGURE THESE PATHS
# =============================================================================
# Path to the training config used
CONFIG_PATH="configs/kelp-rgb/segformer_b3.yaml"

# Path to the best checkpoint
CKPT_PATH="checkpoints/best.ckpt"

# Output ONNX file path
OUTPUT_PATH="/mnt/class_data/sdalgarno/prototype/models/model.onnx"

# ONNX opset version (14 for newer models, 11 for broader compatibility)
OPSET=14

# =============================================================================
# EXPORT
# =============================================================================
echo "Exporting model to ONNX..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Opset: $OPSET"
echo ""

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_PATH")"

python -m src.deploy.kom_onnx "$CONFIG_PATH" "$CKPT_PATH" "$OUTPUT_PATH" --opset "$OPSET"

echo ""
echo "Export complete: $OUTPUT_PATH"
