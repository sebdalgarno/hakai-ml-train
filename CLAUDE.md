# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **seagrass/eelgrass segmentation fork** of hakai-ml-train, a PyTorch Lightning-based training repository for semantic segmentation models in marine ecology research. The upstream repo contains code for kelp and mussel detection, but this fork focuses on training binary segmentation models for **seagrass detection from high-resolution drone imagery**.

Trained models are exported to ONNX format and deployed to the [Habitat-Mapper](https://github.com/HakaiInstitute/habitat-mapper) inference pipeline.

### Seagrass-Specific Resources

- **Configs**: `configs/seagrass-rgb/` - training configurations for seagrass models
- **Documentation**: `docs/seagrass-dataset-splits.md` - train/val/test split definitions, regional CV folds
- **Scripts**: `scripts/` - data preparation and training workflows
- **Data & Checkpoints**: `/mnt/class_data/sdalgarno/` (see Data Locations section below)

## Common Development Commands

### Environment Setup
```bash
uv sync                    # Install dependencies
uv sync --all-groups       # Include dev tools (Jupyter, pre-commit)
source .venv/bin/activate  # Activate environment
```

### Training
```bash
# Start training with a config file
python trainer.py fit --config configs/seagrass-rgb/segformer_b2_1024.yaml

# Resume from checkpoint
python trainer.py fit --config configs/seagrass-rgb/segformer_b2_1024.yaml --ckpt_path checkpoints/last.ckpt

# Test a trained model
python trainer.py test --config configs/seagrass-rgb/segformer_b2_1024.yaml --ckpt_path checkpoints/best.ckpt
```

### Seagrass Data Preparation
```bash
# Full pipeline: create chips, remove nodata, sample prototype
./scripts/prepare_seagrass_data.sh

# Create regional CV splits
./scripts/create_cv_splits.sh
```

### Dataset Preparation (General)
```bash
# Create chip dataset from GeoTIFF mosaics
python -m src.prepare.make_chip_dataset <raw_data_dir> <output_dir> \
  --size 1024 \
  --stride 1024 \
  --num_bands 3 \
  --remap 0 -100 1

# Remove tiles with nodata areas
python -m src.prepare.remove_tiles_with_nodata_areas <chip_dir> --num_channels 3

# Compute channel statistics for normalization
python -m src.prepare.channel_stats <train_chip_dir> --max_pixel_val 255.0
```

### Model Export
```bash
# Export to ONNX (recommended for production)
python -m src.deploy.kom_onnx <config_path> <ckpt_path> <output_path> --opset 14
```

### Code Quality
```bash
ruff check .              # Check for issues
ruff check --fix .        # Auto-fix issues
ruff format .             # Format code
pre-commit run --all-files
```

## Architecture

### Training Pipeline Flow

1. **Data Preparation** (`src/prepare/`): Raw GeoTIFF mosaics → NPZ chips
   - `make_chip_dataset.py` tiles large mosaics into fixed-size chips with label remapping
   - Filtering scripts remove unwanted chips (background-only, nodata areas)
   - `sample_prototype_by_site.py` creates balanced subsets for rapid experimentation

2. **Data Loading** (`src/data.py`):
   - `NpzSegmentationDataset` loads preprocessed NPZ chips
   - `DataModule` handles train/val/test splits with Albumentations transforms
   - Transforms are serialized in YAML configs and logged to W&B

3. **Model Training** (`src/models/smp.py`):
   - `SMPBinarySegmentationModel`: Binary segmentation (background vs seagrass)
   - `SMPMulticlassSegmentationModel`: Multi-class segmentation with per-class metrics
   - Both wrap `segmentation-models-pytorch` architectures with Lightning modules

4. **Training Orchestration** (`trainer.py`):
   - Minimal Lightning CLI wrapper
   - All configuration via YAML files in `configs/`

5. **Model Export** (`src/deploy/kom_onnx.py`):
   - Strips Lightning wrapper, exports raw segmentation model to ONNX
   - Exported models output raw logits (no activation applied)

### Model Architecture Details

**Lightning Module Hierarchy:**
- `SMPBinarySegmentationModel` (base class)
  - Wraps any `segmentation-models-pytorch` model
  - Configurable loss function from `losses.py`
  - Uses OneCycleLR scheduler with warmup
  - Tracks metrics: accuracy, IoU, precision, recall, F1

- `SMPMulticlassSegmentationModel` (extends binary)
  - Adds per-class metric logging via `class_names` parameter
  - Metrics averaged across non-background classes for monitoring
  - `freeze_backbone: true` option to freeze encoder weights during fine-tuning

### Configuration System

All training controlled via PyTorch Lightning CLI YAML configs in `configs/`:

**Seagrass Configs** (`configs/seagrass-rgb/`):
- Architecture experiments (SegFormer, UNet++, DeepLabV3+)
- Augmentation experiments
- Regional cross-validation configs

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
- `AsyncCheckpointIO` plugin is used for non-blocking checkpoint saves

### Label Remapping System

The `--remap` parameter in `make_chip_dataset.py` uses **positional indexing**:
- Index in the list = old label value
- Value at that index = new label value
- Example for seagrass: `--remap 0 -100 1` means:
  - Old label 0 → new label 0 (background)
  - Old label 1 → new label -100 (ignore noise/uncertain during training)
  - Old label 2 → new label 1 (seagrass)

## Seagrass Dataset Organization

### Data Locations
All data and checkpoints are stored on the shared mount:
```
/mnt/class_data/sdalgarno/
├── main/
│   ├── raw_data/          # GeoTIFF orthomosaics and labels
│   └── chips_1024/        # Processed NPZ chips (train/val/test)
├── prototype_frac_50/     # Sampled subset for rapid experiments
│   └── chips_1024/
├── cv_north/              # Regional CV: North held out
│   └── chips_1024/
├── cv_central/            # Regional CV: Central held out
│   └── chips_1024/
├── cv_south/              # Regional CV: South held out
│   └── chips_1024/
└── checkpoints/           # Training checkpoints
```

### Data Splits
See `docs/seagrass-dataset-splits.md` for complete split definitions.

**Main Split (Split B)**: 30 sites across 3 regions (South, Central, North)
- Train: 18 sites
- Val: 6 sites
- Test: 6 sites

**Regional Cross-Validation**: 3-fold CV holding out each region
- `cv_north/`: North sites as test
- `cv_central/`: Central sites as test
- `cv_south/`: South sites as test

### Chip Naming Convention
Files follow pattern: `{site}_u{ortho_id}_{chip_idx}.npz`
- Example: `triquet_bay_u0537_42.npz`
- Exception: `bennett_bay_{chip_idx}.npz` (single ortho, no ID)

### Scripts Directory Structure
```
scripts/
├── prepare_seagrass_data.sh   # Main data prep pipeline
├── create_cv_splits.sh        # Regional CV split creation
├── check_balance.sh           # Verify class balance
├── visualize_*.sh             # Visualization utilities
├── run/                       # Experiment-specific run scripts
└── archive/                   # Deprecated one-off scripts
```

## Key Implementation Details

### Data Format
- Input: NPZ files with keys `"image"` (numpy array) and `"label"` (numpy array)
- Images: uint8 (0-255) RGB from drone orthomosaics
- Labels: Integer class indices, with -100 for pixels to ignore

### Loss Functions
Available in `src/losses.py`:
- `DiceLoss`: Good for imbalanced datasets
- `LovaszLoss`: Optimizes IoU directly
- `FocalLoss`: Focuses on hard examples
- `JaccardLoss`: IoU loss
- `TverskyLoss`: Generalization of Dice

### ONNX Export
- Exports only the wrapped segmentation model (not Lightning module)
- Dynamic axes for batch size and spatial dimensions
- Outputs raw logits (apply sigmoid during inference for binary)
- Opset 14 recommended

## Weights & Biases

- Entity: `hakai`
- Project: `kom-seagrass-rgb`
- Contact Taylor Denouden for W&B access
- Checkpoints uploaded as artifacts when `log_model: true`

## Coding Conventions

### Workflow Pattern
- **Python modules** (`src/`): Core logic as importable functions with CLI via `argparse`
- **Bash scripts** (`scripts/`): User-facing pipelines that call Python modules with configured parameters

### Bash Script Style
```bash
#!/bin/bash
set -e

# SECTION NAME -----
VARIABLE="value"

# Another section -----
python -m src.module.name "$VARIABLE"
```
- Section headers: `# SECTION NAME -----` (5 dashes)
- Inline comments only when logic is non-obvious

### Python Style
- Follow existing ruff config (isort, flake8-bugbear, flake8-simplify)
- Minimal docstrings - one-liner at top of module/function when helpful
- Type hints for function signatures
- Prefer `pathlib.Path` over string paths

## Technical Stack

- **PyTorch Lightning**: Training orchestration
- **segmentation-models-pytorch**: Model architectures and pretrained encoders
- **albumentationsx**: Data augmentation (note: uses `albumentationsx`, not standard `albumentations`)
- **Weights & Biases**: Experiment tracking
- **uv**: Package management
