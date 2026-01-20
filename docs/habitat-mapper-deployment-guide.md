# Deploying Models to habitat-mapper

Guide for exporting trained models from hakai-ml-train and deploying them to the habitat-mapper inference pipeline.

## Overview

habitat-mapper uses a config-based model registry. To add a new model:

1. Export model to ONNX format
2. Upload ONNX to HuggingFace
3. Create config JSON in habitat-mapper
4. Submit PR to habitat-mapper repo

## Step 1: Export Model to ONNX

### Export Command

```bash
python -m src.deploy.kom_onnx \
    configs/seagrass-rgb/unetpp_resnet34.yaml \
    path/to/best_checkpoint.ckpt \
    seagrass_rgb_model.onnx \
    --opset 14
```

### Verify Export

```python
import onnxruntime as ort
import numpy as np

# Load and inspect
session = ort.InferenceSession("seagrass_rgb_model.onnx")

# Check input/output names and shapes
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
for out in session.get_outputs():
    print(f"Output: {out.name}, shape: {out.shape}, dtype: {out.type}")

# Test inference
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {"input": dummy_input})
print(f"Output shape: {output[0].shape}")  # Should be (1, num_classes, 224, 224)
```

### Expected ONNX Properties

| Property | Expected Value |
|----------|----------------|
| Input name | `input` |
| Input shape | `(batch, 3, tile_size, tile_size)` - dynamic batch/spatial |
| Output name | `output` |
| Output shape | `(batch, num_classes, tile_size, tile_size)` |
| Output type | Raw logits (no activation applied) |

**Important:** The ONNX model should output raw logits. habitat-mapper applies activation (softmax/sigmoid) based on config.

## Step 2: Upload to HuggingFace

### Create Repository

1. Go to https://huggingface.co/new
2. Create repo under HakaiInstitute organization (or your own for testing)
3. Suggested naming: `HakaiInstitute/seagrass-rgb`

### Upload Model

```bash
# Install huggingface CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload with revision tag (use date-based versioning)
huggingface-cli upload HakaiInstitute/seagrass-rgb \
    seagrass_rgb_model.onnx model.onnx \
    --revision 20260120
```

### Resulting URL

```
https://huggingface.co/HakaiInstitute/seagrass-rgb/resolve/20260120/model.onnx
```

## Step 3: Create habitat-mapper Config

### Config Location

```
habitat-mapper/src/habitat_mapper/configs/seagrass_rgb_YYYYMMDD.json
```

### Config Template

```json
{
  "name": "seagrass-rgb",
  "revision": "20260120",
  "description": "Seagrass segmentation model for RGB drone imagery.",
  "dependencies": [
    "https://huggingface.co/HakaiInstitute/seagrass-rgb/resolve/20260120/model.onnx"
  ],
  "model_filename": "model.onnx",
  "input_channels": 3,
  "normalization": "standard",
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "max_pixel_value": 255.0,
  "activation": "softmax"
}
```

### Config Fields Explained

| Field | Description | Your Value |
|-------|-------------|------------|
| `name` | CLI model identifier (`--model seagrass-rgb`) | `seagrass-rgb` |
| `revision` | Version string (date-based: YYYYMMDD) | `20260120` |
| `description` | Brief model description | Seagrass segmentation... |
| `dependencies` | URLs to download (ONNX file) | HuggingFace URL |
| `model_filename` | Which dependency is the model | `model.onnx` |
| `input_channels` | Number of input bands | `3` (RGB) |
| `normalization` | Preprocessing strategy | `standard` (ImageNet z-score) |
| `mean` | Channel means for normalization | `[0.485, 0.456, 0.406]` |
| `std` | Channel stds for normalization | `[0.229, 0.224, 0.225]` |
| `max_pixel_value` | Pixel scaling divisor | `255.0` |
| `activation` | Output activation function | `softmax` (multiclass) or `sigmoid` (binary) |

### Normalization Must Match Training

Check your training config (`configs/seagrass-rgb/unetpp_resnet34.yaml`):

```yaml
test_transforms:
  transforms:
    - __class_fullname__: Normalize
      max_pixel_value: 255.0
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

These values **must** match the habitat-mapper config exactly.

### Activation Function

| Model Type | num_classes | activation |
|------------|-------------|------------|
| Binary (bg/seagrass) | 2 | `softmax` |
| Binary (single channel) | 1 | `sigmoid` |
| Multiclass | >2 | `softmax` |

Your seagrass model uses `num_classes: 2` with softmax.

## Step 4: Test Locally

### Clone and Install habitat-mapper

```bash
git clone https://github.com/HakaiInstitute/habitat-mapper.git
cd habitat-mapper
pip install -e .
```

### Add Your Config

```bash
cp seagrass_rgb_20260120.json src/habitat_mapper/configs/
```

### Test CLI

```bash
# List models (should show seagrass-rgb)
habitat-mapper models

# Run prediction
habitat-mapper segment test_image.tif output_mask.tif --model seagrass-rgb
```

### Verify Output

- Output should be a GeoTIFF with same CRS/transform as input
- Pixel values: class indices (0=background, 1=seagrass)
- Inspect in QGIS to verify predictions look reasonable

## Step 5: Submit PR to habitat-mapper

### Fork and Branch

```bash
git checkout -b add-seagrass-rgb-model
```

### Add Config File

```bash
git add src/habitat_mapper/configs/seagrass_rgb_20260120.json
git commit -m "Add seagrass-rgb segmentation model"
```

### PR Description Template

```markdown
## Summary
Adds seagrass segmentation model for RGB drone imagery.

## Model Details
- Architecture: UNet++ with ResNet34 encoder
- Input: 3-channel RGB (224x224 tiles)
- Output: 2 classes (background, seagrass)
- Training data: [describe dataset]
- Validation IoU: [your metric]

## Config
- Name: `seagrass-rgb`
- Revision: `20260120`
- HuggingFace: https://huggingface.co/HakaiInstitute/seagrass-rgb

## Testing
- Tested on [X] orthomosaics from [location]
- Visual inspection confirms reasonable predictions
```

## Checklist

Before deployment:

- [ ] Model achieves acceptable validation metrics
- [ ] ONNX export successful and verified
- [ ] ONNX outputs raw logits (no activation baked in)
- [ ] Input/output names are `input` and `output`
- [ ] Dynamic axes enabled for batch and spatial dimensions
- [ ] Normalization parameters match training config
- [ ] Model uploaded to HuggingFace with version tag
- [ ] Config JSON created with correct parameters
- [ ] Local testing with habitat-mapper CLI passes
- [ ] Predictions visually verified in QGIS
- [ ] PR submitted to habitat-mapper

## Updating Models

For new model versions:

1. Train improved model
2. Export to ONNX
3. Upload to HuggingFace with new revision tag (e.g., `20260215`)
4. Add new config file: `seagrass_rgb_20260215.json`
5. habitat-mapper automatically uses latest revision by default

Users can pin to specific versions:
```bash
habitat-mapper segment input.tif output.tif --model seagrass-rgb --revision 20260120
```

## Troubleshooting

### "Model not found" error
- Check config JSON is in `src/habitat_mapper/configs/`
- Verify `name` field matches CLI argument

### Predictions are all zeros/ones
- Check normalization parameters match training
- Verify activation function is correct
- Ensure ONNX outputs logits, not probabilities

### Spatial misalignment
- Verify input image has valid CRS and transform
- Check tile size matches training (224x224)

### Memory errors on large images
- Reduce `--batch-size` (default is often too high)
- Use `--crop-size` matching your tile size (224)
