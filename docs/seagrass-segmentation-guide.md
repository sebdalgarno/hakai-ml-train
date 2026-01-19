# Seagrass Segmentation Model Training Guide

A beginner-friendly guide for training deep learning models to detect seagrass (eelgrass) from aerial/drone imagery.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Concepts](#key-concepts)
3. [Recommended Architecture](#recommended-architecture)
4. [Data Preparation](#data-preparation)
5. [Training Configuration](#training-configuration)
6. [Handling Depth Variation](#handling-depth-variation)
7. [Experimental Plan](#experimental-plan)
8. [Glossary](#glossary)

---

## Project Overview

### What We're Building

A **semantic segmentation model** that takes aerial RGB imagery as input and outputs a mask showing where seagrass is present. Think of it like an automated highlighter that colors in all the seagrass pixels.

### Input and Output

```
Input:  RGB image tile (e.g., 512×512 pixels)
Output: Mask of same size where each pixel is labeled:
        - 0 = background (water, sand, rock)
        - 1 = seagrass
```

### The Data

- **70 sites** total (large orthomosaic GeoTIFFs, 1-3 GB each)
- **2-5 cm resolution** (very high detail)
- Each site has corresponding **label masks** (ground truth)

---

## Key Concepts

### What is Semantic Segmentation?

Unlike image classification (which labels an entire image), segmentation labels **every pixel**. The model learns to draw boundaries around objects of interest.

### What is a "Backbone"?

The backbone is the **feature extractor** — a pretrained neural network (usually trained on millions of images) that has already learned to recognize basic visual patterns like edges, textures, and shapes. We reuse this knowledge instead of learning from scratch.

### What is an "Architecture"?

The architecture is the overall model design — how the backbone connects to the final segmentation output. Different architectures are better for different tasks.

### What is a "Tile" or "Chip"?

Your orthomosaics are too large to feed directly into a model. We cut them into smaller, fixed-size pieces called tiles or chips (e.g., 512×512 pixels).

---

## Recommended Architecture

### Primary Recommendation: DeepLabV3+ with EfficientNet-B4

| Component | Choice | Why |
|-----------|--------|-----|
| **Architecture** | DeepLabV3+ | Captures features at multiple scales; handles soft/diffuse boundaries well |
| **Backbone** | EfficientNet-B4 | Good balance of accuracy and efficiency; proven on natural imagery |
| **Tile Size** | 512×512 | ~25m ground coverage at 5cm resolution; enough context for seagrass patches |

### Why This Choice for Seagrass?

Seagrass has **diffuse, gradient boundaries** (unlike sharp-edged objects). DeepLabV3+ uses a technique called **Atrous Spatial Pyramid Pooling (ASPP)** that looks at the image at multiple scales simultaneously — ideal for capturing both fine blade texture and larger meadow patterns.

### Alternative Options

| Architecture | Backbone | When to Use |
|--------------|----------|-------------|
| UNet++ | EfficientNet-B3 | Simpler, faster training; good starting point |
| SegFormer | MiT-B2 | If you want to try transformer-based models |

---

## Data Preparation

### Step 1: Organize Your Raw Data

Create this folder structure with your GeoTIFFs:

```
data/seagrass-prototype/
├── train/
│   ├── images/
│   │   ├── site_01.tif
│   │   ├── site_02.tif
│   │   └── ... (10 sites)
│   └── labels/
│       ├── site_01.tif
│       └── ...
├── val/
│   ├── images/
│   │   └── ... (2 sites)
│   └── labels/
└── test/
    ├── images/
    │   └── ... (3 sites)
    └── labels/
```

### Step 2: Site Selection (15 Sites Recommended)

Choose sites that span environmental variation:

| Split | # Sites | Selection Criteria |
|-------|---------|-------------------|
| Train | 10 | Mix of substrates, depths, seagrass densities |
| Val | 2 | "Average" sites — not your best or worst |
| Test | 3 | Include easy and challenging sites |

**Important**: Split by **site**, not by tile. Tiles from the same orthomosaic are correlated — mixing them across splits causes data leakage and inflated metrics.

### Step 3: Create Tiles

Run the tiling script to cut orthomosaics into 512×512 chips:

```bash
python -m src.prepare.make_chip_dataset \
  data/seagrass-prototype \
  data/seagrass-prototype-512 \
  --size 512 \
  --stride 256 \
  --num_bands 3 \
  --remap 0 1
```

**Parameters explained**:
- `--size 512`: Tile size in pixels
- `--stride 256`: Step size between tiles (256 = 50% overlap, more training samples)
- `--num_bands 3`: RGB (3 channels)
- `--remap 0 1`: Label remapping (adjust based on your label values)

### Step 4: Filter Tiles (Carefully)

Remove tiles that have no-data areas:

```bash
python -m src.prepare.remove_tiles_with_nodata_areas \
  data/seagrass-prototype-512/train \
  --num_channels 3
```

**Important**: Do NOT remove all background-only tiles. Keep 30-50% of them — the model needs to learn what is NOT seagrass.

### Step 5: Compute Normalization Statistics

Calculate mean and standard deviation of your training images:

```bash
python -m src.prepare.channel_stats \
  data/seagrass-prototype-512/train \
  --max_pixel_val 255.0
```

Use these values in your config file's `Normalize` transform.

---

## Training Configuration

### Example Config File

Save as `configs/seagrass-rgb/deeplabv3plus_efficientnet_b4.yaml`:

```yaml
seed_everything: 42

model:
  class_path: "src.models.smp.SMPBinarySegmentationModel"
  init_args:
    architecture: "DeepLabV3Plus"
    backbone: "efficientnet-b4"
    model_opts:
      encoder_weights: imagenet
      in_channels: 3
    num_classes: 1  # Binary: seagrass vs background
    ignore_index: -100
    lr: 1e-4
    wd: 0.01
    b1: 0.9
    b2: 0.95
    loss: "DiceLoss"
    loss_opts:
      mode: "binary"
      ignore_index: -100
      from_logits: true

data:
  class_path: "src.data.DataModule"
  init_args:
    train_chip_dir: "data/seagrass-prototype-512/train"
    val_chip_dir: "data/seagrass-prototype-512/val"
    test_chip_dir: "data/seagrass-prototype-512/test"
    batch_size: 4
    num_workers: 8
    pin_memory: true
    persistent_workers: true
    train_transforms:
      __version__: 2.0.9
      transform:
        __class_fullname__: Compose
        p: 1
        transforms:
          - __class_fullname__: D4
            p: 1.0
          - __class_fullname__: RandomBrightnessContrast
            brightness_limit: [-0.2, 0.2]
            contrast_limit: [-0.2, 0.2]
            p: 0.5
          - __class_fullname__: HueSaturationValue
            hue_shift_limit: [-10, 10]
            sat_shift_limit: [-20, 20]
            val_shift_limit: [-20, 20]
            p: 0.5
          - __class_fullname__: GaussianBlur
            blur_limit: [3, 7]
            p: 0.3
          - __class_fullname__: Normalize
            mean: [0.485, 0.456, 0.406]  # Replace with your computed values
            std: [0.229, 0.224, 0.225]   # Replace with your computed values
            max_pixel_value: 255.0
            p: 1.0
          - __class_fullname__: ToTensorV2
            p: 1.0
    test_transforms:
      __version__: 2.0.9
      transform:
        __class_fullname__: Compose
        p: 1
        transforms:
          - __class_fullname__: Normalize
            mean: [0.485, 0.456, 0.406]  # Same as above
            std: [0.229, 0.224, 0.225]
            max_pixel_value: 255.0
            p: 1.0
          - __class_fullname__: ToTensorV2
            p: 1.0

trainer:
  accelerator: auto
  devices: auto
  precision: bf16-mixed
  log_every_n_steps: 50
  max_epochs: 100
  accumulate_grad_batches: 4  # Effective batch size = 4 × 4 = 16
  gradient_clip_val: 0.5
  default_root_dir: checkpoints
  logger:
    - class_path: lightning.pytorch.loggers.WandbLogger
      init_args:
        entity: hakai
        project: seagrass-rgb
        name: deeplabv3plus_efficientnet_b4
        log_model: true
        tags:
          - seagrass
          - prototype
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: seagrass_epoch-{epoch:02d}_val-iou-{val/iou_epoch:.4f}
        monitor: val/iou_epoch
        mode: max
        save_last: true
        save_top_k: 2
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/iou_epoch
        mode: max
        patience: 20
  plugins:
    - class_path: lightning.pytorch.plugins.io.AsyncCheckpointIO
```

### Key Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `batch_size` | 4 | Images processed together (limited by GPU memory) |
| `accumulate_grad_batches` | 4 | Simulate larger batch by accumulating gradients |
| `lr` | 1e-4 | Learning rate — how big each update step is |
| `max_epochs` | 100 | Maximum training iterations through full dataset |
| `patience` | 20 | Stop early if no improvement for 20 epochs |
| `loss: DiceLoss` | - | Good for imbalanced data (more background than seagrass) |

### Running Training

```bash
# Activate your environment
source .venv/bin/activate

# Start training
python trainer.py fit --config configs/seagrass-rgb/deeplabv3plus_efficientnet_b4.yaml
```

### Monitoring Training

Training progress is logged to Weights & Biases. Key metrics to watch:

- **train/loss**: Should decrease over time
- **val/iou_epoch**: Main metric — higher is better (1.0 = perfect)
- **val/loss**: Should decrease; if it increases while train/loss decreases, you're overfitting

---

## Handling Depth Variation

### The Challenge

Seagrass appearance changes dramatically with water depth:

| Depth | Appearance | Detection Difficulty |
|-------|------------|---------------------|
| Exposed/very shallow | Bright green, sharp texture | Easy |
| 0.5-2m | Green, visible texture | Easy |
| 2-4m | Muted teal, softer texture | Medium |
| >4m | Blue-gray, texture fades | Hard |

Water absorbs red light first, so deeper seagrass loses its green color.

### Your Bathymetry Data

You have **10m resolution bathymetry** — too coarse to use as a direct model input (your imagery is 2-5cm), but valuable for:

#### 1. Site Selection

Use bathymetry to ensure your 15 prototype sites span the depth range:

```
Sites with mostly 0-2m depth:  ~5 sites
Sites with 2-4m depth:         ~5 sites
Sites with >4m depth:          ~5 sites
```

#### 2. Error Analysis (After Training)

Analyze model performance by depth zone:

```python
# Pseudo-code for post-training analysis
for each test tile:
    depth = get_mean_depth_from_bathymetry(tile_location)
    iou = calculate_iou(model_prediction, ground_truth)
    record(depth_zone, iou)

# Result: "Model achieves 0.85 IoU at 0-2m, 0.70 at 2-4m, 0.45 at >4m"
```

#### 3. Inference Confidence Masking

After training, apply a depth-based confidence mask:

```python
# Flag predictions in deep water as low-confidence
predictions[depth > 4.0] = "low_confidence"
```

### Augmentation for Depth Robustness

The config includes color augmentations that help the model handle depth variation:

- **HueSaturationValue**: Simulates color shifts from water column
- **RandomBrightnessContrast**: Simulates variable lighting/depth

---

## Experimental Plan

### Phase 1: Baseline Model

| Step | Action |
|------|--------|
| 1 | Select 15 sites spanning depth/substrate variation |
| 2 | Create tiles at 512×512, 50% overlap |
| 3 | Train DeepLabV3+ EfficientNet-B4 for 100 epochs |
| 4 | Evaluate on test set |

**Success criteria**: val/iou > 0.7

### Phase 2: Architecture Comparison

Train alternative models with same data:

| Experiment | Architecture | Backbone |
|------------|--------------|----------|
| Baseline | DeepLabV3+ | EfficientNet-B4 |
| Alt 1 | UNet++ | EfficientNet-B3 |
| Alt 2 | SegFormer | MiT-B2 |

Compare val/iou to select best architecture.

### Phase 3: Error Analysis

1. Run best model on all test tiles
2. Calculate IoU by depth zone (using bathymetry)
3. Visualize failure cases
4. Document practical detection depth limit

### Phase 4: Scale Up

If prototype succeeds:
1. Add more training sites (30-40)
2. Retrain best architecture
3. Export to ONNX for deployment

---

## Glossary

| Term | Definition |
|------|------------|
| **Backbone** | Pretrained neural network that extracts features from images |
| **Batch size** | Number of images processed together in one training step |
| **Epoch** | One complete pass through all training data |
| **Gradient accumulation** | Technique to simulate larger batches on limited GPU memory |
| **IoU (Intersection over Union)** | Metric measuring overlap between prediction and ground truth (0-1, higher is better) |
| **Learning rate** | Controls how much the model updates on each step |
| **Loss function** | Measures how wrong the model's predictions are |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Semantic segmentation** | Labeling every pixel in an image |
| **Stride** | Step size when creating tiles (smaller stride = more overlap = more tiles) |
| **Tile/Chip** | Small fixed-size piece cut from a large image |
| **Transform/Augmentation** | Random modifications to training images (flips, color changes) to improve generalization |

---

## Quick Reference Commands

```bash
# Create tiles
python -m src.prepare.make_chip_dataset <input_dir> <output_dir> \
  --size 512 --stride 256 --num_bands 3 --remap 0 1

# Remove nodata tiles
python -m src.prepare.remove_tiles_with_nodata_areas <chip_dir> --num_channels 3

# Compute normalization stats
python -m src.prepare.channel_stats <train_chip_dir> --max_pixel_val 255.0

# Train model
python trainer.py fit --config configs/seagrass-rgb/your_config.yaml

# Resume training from checkpoint
python trainer.py fit --config configs/seagrass-rgb/your_config.yaml \
  --ckpt_path checkpoints/last.ckpt

# Test trained model
python trainer.py test --config configs/seagrass-rgb/your_config.yaml \
  --ckpt_path checkpoints/best.ckpt

# Export to ONNX
python -m src.deploy.kom_onnx <config_path> <ckpt_path> <output_path> --opset 14
```

---

## Next Steps

1. **Set up environment**: `uv sync --all-groups`
2. **Select 15 sites** spanning your environmental variation
3. **Organize data** into train/val/test folders
4. **Run preprocessing** (tiling, filtering, stats)
5. **Start training** with the provided config
6. **Monitor on W&B** and iterate

Good luck with your seagrass mapping!
