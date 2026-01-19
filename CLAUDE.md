# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch Lightning-based training repository for semantic segmentation models used in marine ecology research. The models detect kelp forests and mussel beds from aerial/drone imagery. Trained models are exported to ONNX format and deployed to the Kelp-o-Matic inference pipeline.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies and create virtual environment
uv sync

# Install with development tools (Jupyter, pre-commit, etc.)
uv sync --all-groups

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
```

### Training
```bash
# Start training with a config file
python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml

# Resume from checkpoint
python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml --ckpt_path checkpoints/last.ckpt

# Test a trained model
python trainer.py test --config configs/kelp-rgb/segformer_b3.yaml --ckpt_path checkpoints/best.ckpt
```

### Dataset Preparation
```bash
# Create chip dataset from GeoTIFF mosaics
python -m src.prepare.make_chip_dataset <raw_data_dir> <output_dir> \
  --size 224 \
  --stride 224 \
  --num_bands 3 \
  --remap 0 -100 1

# Remove background-only tiles
python -m src.prepare.remove_bg_only_tiles <chip_dir>

# Remove tiles with nodata areas
python -m src.prepare.remove_tiles_with_nodata_areas <chip_dir> --num_channels 3

# Compute channel statistics for normalization
python -m src.prepare.channel_stats <train_chip_dir> --max_pixel_val 255.0
```

### Model Export
```bash
# Export to ONNX (recommended for production)
python -m src.deploy.kom_onnx <config_path> <ckpt_path> <output_path> --opset 14

# Export legacy RGB kelp model
python -m src.deploy.kom_onnx_legacy_kelp_rgb <config_path> <ckpt_path> <output_path>

# Export legacy RGBI kelp model
python -m src.deploy.kom_onnx_legacy_kelp_rgbi <config_path> <ckpt_path> <output_path>

# Export to TorchScript
python -m src.deploy.kom_torchscript <config_path> <ckpt_path> <output_path>
```

### Code Quality
```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Training Pipeline Flow

1. **Data Preparation** (`src/prepare/`): Raw GeoTIFF mosaics → NPZ chips
   - `make_chip_dataset.py` tiles large mosaics into fixed-size chips with label remapping
   - Optional filtering scripts remove unwanted chips (background-only, nodata areas)
   - `channel_stats.py` computes normalization statistics

2. **Data Loading** (`src/data.py`):
   - `NpzSegmentationDataset` loads preprocessed NPZ chips
   - `DataModule` handles train/val/test splits with Albumentations transforms
   - Transforms are serialized in YAML configs and logged to W&B

3. **Model Training** (`src/models/`):
   - `SMPBinarySegmentationModel`: Binary segmentation (background vs target)
   - `SMPMulticlassSegmentationModel`: Multi-class segmentation with per-class metrics
   - Both wrap `segmentation-models-pytorch` architectures with Lightning modules
   - Loss functions from `src/losses.py` (Dice, Lovasz, Focal, etc.)

4. **Training Orchestration** (`trainer.py`):
   - Minimal Lightning CLI wrapper
   - All configuration via YAML files in `configs/`

5. **Model Export** (`src/deploy/`):
   - `kom_onnx.py`: Strips Lightning wrapper, exports raw segmentation model to ONNX
   - Legacy exporters handle backwards compatibility with older Kelp-o-Matic versions
   - Exported models output raw logits (no activation applied)

### Model Architecture Details

**Lightning Module Hierarchy:**
- `SMPBinarySegmentationModel` (base class)
  - Wraps any `segmentation-models-pytorch` model
  - Configurable loss function from `losses.py`
  - Uses OneCycleLR scheduler with warmup
  - Tracks metrics: accuracy, IoU, precision, recall, F1
  - Override `configure_optimizers()` for custom schedulers

- `SMPMulticlassSegmentationModel` (extends binary)
  - Adds per-class metric logging via `class_names` parameter (e.g., `["bg", "macro", "nereo"]`)
  - Metrics averaged across non-background classes for monitoring (excludes index 0)
  - Different activation function (softmax without squeeze)
  - `freeze_backbone: true` option to freeze encoder weights during fine-tuning

**Custom Models:**
- `kom_baseline.py` and `kom_port.py`: Ensemble models that load pretrained TorchScript weights
- These are specialized for kelp detection and use two-stage presence/species inference
- `mae_pretrain.py`: Masked Autoencoder pretraining for self-supervised learning

### Configuration System

All training controlled via PyTorch Lightning CLI YAML configs in `configs/`:

**Directory Organization:**
- `kelp-rgb/`: RGB imagery (3 channels)
- `kelp-rgbi/`: RGBI imagery (4 channels, includes infrared)
- `kelp-ps8b/`: PlanetScope 8-band multispectral
- `mussels-rgb/`: Mussel detection
- `mussels-goosenecks-rgb/`: Multi-class mussel and gooseneck barnacle

**Key Config Sections:**
- `model.class_path`: Python import path to Lightning module
- `model.init_args`: Model hyperparameters (architecture, backbone, loss, learning rate, etc.)
- `data.class_path`: DataModule class
- `data.init_args`: Paths to chip directories, batch size, transforms
- `trainer`: Lightning Trainer settings (GPUs, precision, callbacks, loggers)

**Important Notes:**
- Use YAML anchors (`&ignore_index` and `*ignore_index`) for parameter reuse
- Transforms are serialized Albumentations pipelines (from `A.to_dict()`)
- `ignore_index: -100` masks pixels during training (noise, uncertain labels)
- Update `train_chip_dir`, `val_chip_dir`, `test_chip_dir` paths before training
- `AsyncCheckpointIO` plugin is used for non-blocking checkpoint saves

### Label Remapping System

The `--remap` parameter in `make_chip_dataset.py` uses **positional indexing**:
- Index in the list = old label value
- Value at that index = new label value
- Example: `--remap 0 -100 1 2` means:
  - Old label 0 → new label 0 (background)
  - Old label 1 → new label -100 (ignore during training)
  - Old label 2 → new label 1 (class 1)
  - Old label 3 → new label 2 (class 2)

This is critical for handling noise labels, converting multi-class to binary, or reordering class indices.

## Key Implementation Details

### Data Format
- Input: NPZ files with keys `"image"` (numpy array) and `"label"` (numpy array)
- Images: uint8 (0-255) for RGB or uint16 for multispectral
- Labels: Integer class indices, with -100 for pixels to ignore

### Metrics Logging
- Binary models: Single set of metrics (IoU, precision, recall, F1, accuracy)
- Multiclass models: Per-class metrics + mean across non-background classes
- Epoch-level metrics logged at validation end for checkpoint monitoring
- W&B logging includes transforms, hyperparameters, and model checkpoints

### Creating Transform Configs
To serialize Albumentations transforms for YAML configs:
```python
import albumentations as A
from albumentations import to_dict

transforms = A.Compose([
    A.D4(),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])
print(to_dict(transforms))  # Copy output to YAML config
```

### ONNX Export
- Exports only the wrapped segmentation model (not Lightning module)
- Dynamic axes for batch size and spatial dimensions
- Outputs raw logits (apply softmax/sigmoid during inference)
- Opset 14 recommended for newer models, 11 for broader compatibility

### Loss Functions
Available in `src/losses.py`:
- `DiceLoss`: Good for imbalanced datasets
- `LovaszLoss`: Optimizes IoU directly
- `FocalLoss`: Focuses on hard examples
- `JaccardLoss`: IoU loss
- `TverskyLoss`: Generalization of Dice
- `FocalDiceComboLoss`: Combination loss

## Deployment Flow

1. Train model with Lightning CLI
2. Export checkpoint to ONNX with `src/deploy/kom_onnx.py`
3. Upload ONNX model to S3 bucket (`kelp-o-matic`)
4. Model consumed by [Habitat-Mapper](https://github.com/HakaiInstitute/habitat-mapper) (formerly kelp-o-matic) inference pipeline

## Weights & Biases

- Entity: `hakai`
- Project naming: `kom-{dataset}-{modality}` (e.g., `kom-kelp-rgb`)
- Contact Taylor Denouden for W&B access
- Checkpoints uploaded as artifacts when `log_model: true`

## Coding Conventions

### Workflow Pattern
- **Python modules** (`src/`): Core logic as importable functions with CLI via `argparse`
- **Bash scripts** (`scripts/`): User-facing pipelines that call Python modules with configured parameters
- Scripts should have configurable variables at the top, then call Python modules below

### Bash Script Style
Use minimal, clean comments with section dividers:
```bash
#!/bin/bash
set -e

# SECTION NAME -----
VARIABLE="value"

# Another section -----
python -m src.module.name "$VARIABLE"
```
- Section headers: `# SECTION NAME -----` (use 5 dashes)
- Inline comments only when logic is non-obvious
- No heavy ASCII boxes or excessive decoration
- Group related variables together under a section header

### Python Style
- Follow existing ruff config (isort, flake8-bugbear, flake8-simplify)
- Minimal docstrings - one-liner at top of module/function when helpful
- Type hints for function signatures
- Use `tqdm` for progress bars in CLI tools
- Prefer `pathlib.Path` over string paths

### Comments
- Prefer self-documenting code over comments
- Comment the "why" not the "what"
- No emoji in code or comments

## Technical Stack

- **PyTorch Lightning**: Training orchestration
- **segmentation-models-pytorch**: Model architectures and pretrained encoders
- **albumentationsx**: Data augmentation (note: uses `albumentationsx`, not standard `albumentations`)
- **TorchGeo**: GeoTIFF preprocessing (dev dependency)
- **Weights & Biases**: Experiment tracking
- **uv**: Package management
- **HuggingFace Hub**: Models include `PyTorchModelHubMixin` for potential Hub publishing
