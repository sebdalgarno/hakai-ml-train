# Model Development Workflow

A three-stage approach for developing generalizable segmentation models from drone orthomosaic imagery.

## Overview

This workflow separates model selection from generalization validation, following best practices for spatial/ecological data where the goal is deployment to unseen sites and regions.

| Stage | Purpose | Data Strategy |
|-------|---------|---------------|
| 1. Architecture Selection | Compare models fairly | Stratified splits (all regions in all buckets) |
| 2. Generalization Validation | Estimate regional transferability | Leave-one-region-out CV |
| 3. Final Model Training | Best model for deployment | Train on all data |

## Key Principles

### Site-Level Splitting
All orthomosaics from a site (including repeat visits across years) must stay in the same train/val/test bucket. This prevents data leakage from spatial autocorrelation—the model cannot "recognize" familiar sites in the test set.

### Maintain Natural Class Distribution
Keep the natural class imbalance (e.g., 15% seagrass / 85% background) rather than balancing to 50/50. Loss functions (Dice, Lovasz, Focal) handle pixel-level imbalance. Balanced sampling would:
- Cause miscalibrated predictions
- Invalidate site-level area estimates
- Not reflect real deployment conditions

### Regional Stratification
With sites from multiple geographic regions, ensure each region is represented appropriately in each stage to avoid confounding architecture performance with regional difficulty.

### Depth as Analysis Covariate
Depth affects visibility and model performance but is used as an **analysis covariate**, not a model input. This means:
- Depth metadata is assigned to each tile during data preparation
- Performance is stratified by depth bin during evaluation (Stage 2)
- Users receive depth-based guidance on where to focus manual review

This approach keeps inference simple (users don't need bathymetry data) while providing actionable guidance on model reliability.

---

## Data Preparation

**Goal:** Create tile datasets with metadata for site, region, and depth.

### Tile Creation

Create chips from orthomosaics using standard pipeline:

```bash
python -m src.prepare.make_chip_dataset <raw_data_dir> <output_dir> \
  --size 512 \
  --stride 512 \
  --num_bands 3 \
  --remap 0 -100 1
```

### Depth Assignment

Assign depth value to each tile from bathymetry raster:

```
1. Extract tile centroid coordinates from orthomosaic georeferencing
2. Sample 10m bathymetry raster at centroid
3. Store depth as tile metadata (CSV or embedded in filename/NPZ)
4. Bin into categories for analysis: shallow (<5m), medium (5-10m), deep (>10m)
```

Example metadata CSV:

```csv
tile_id,site,region,ortho_year,depth_m,depth_bin,has_seagrass
site01_2021_tile_0042,site01,north,2021,3.2,shallow,true
site01_2021_tile_0043,site01,north,2021,7.8,medium,true
site01_2021_tile_0044,site01,north,2021,12.1,deep,false
```

### Site Assignment to Splits

Before any training, assign sites to train/val/test buckets:

```
1. List all unique sites with region and ortho count
2. Stratify by region
3. Assign entire sites (all orthos, all years) to one bucket
4. Document the assignment for reproducibility
```

---

## Stage 1: Architecture Selection (Prototyping)

**Goal:** Identify the best architecture/backbone combination through fair comparison.

### Data Splitting

Assign sites to train/val/test with stratification by region:

```
Example with 15 sites across 3 regions:

North Coast (5 sites):   3 train, 1 val, 1 test
Central Coast (5 sites): 4 train, 1 val, 0 test
South Coast (5 sites):   3 train, 1 val, 1 test
─────────────────────────────────────────────────
Total:                  10 train, 3 val, 2 test
```

All regions appear in training and validation. Test sites span multiple regions.

### Why Stratify (Not Regional Holdout) Here?

- **Fair comparison:** All architectures see all regional conditions during training
- **Differences reflect model quality:** Not regional quirks or difficulty
- **Practical:** With few sites, holding out an entire region leaves too little data for reliable validation

### Depth Role in Stage 1

Depth analysis is **not critical** at this stage:
- All architectures will likely show similar depth-performance relationships
- Keep experiments focused on architecture/backbone comparisons
- Optional: verify depth distribution is similar across train/val/test splits

### Evaluation Metrics

Track at tile level:
- IoU (primary metric for checkpoint selection)
- Precision, Recall, F1
- Per-class metrics if multiclass

Track at site/orthomosaic level:
- Total predicted area vs. manual area (correlation)
- Per-region performance breakdown

### Experiments to Run

| Experiment | Variables | Fixed |
|------------|-----------|-------|
| Architecture | UNet++, DeepLabV3+, SegFormer, UPerNet | Backbone, tile size, loss |
| Backbone | ResNet34, ResNet50, EfficientNet-B1, MiT-B3 | Architecture, tile size, loss |
| Tile Size | 256, 512, 768 | Best architecture/backbone, loss |
| Loss Function | Dice, Lovasz, Focal, Combo | Best architecture/backbone/tile |

### Optional: Depth as Model Input Experiment

If severe depth-related performance issues are suspected, test whether adding depth improves predictions:

| Experiment | Input | Notes |
|------------|-------|-------|
| Baseline | RGB (3 channels) | Standard approach |
| RGB + Depth | RGBD (4 channels) | Depth as 4th channel |

**Considerations:**
- Requires users to have bathymetry data at inference time
- 10m resolution depth is coarse relative to cm-scale imagery
- Adds inference complexity; only pursue if clear benefit

### Output

Selected configuration:
- Architecture (e.g., SegFormer)
- Backbone (e.g., MiT-B3)
- Tile size (e.g., 512)
- Loss function (e.g., LovaszLoss)

---

## Stage 2: Generalization Validation

**Goal:** Estimate how well the selected model generalizes to completely unseen regions, and characterize performance across depth gradients.

### Data Splitting

Leave-one-region-out cross-validation (3-fold for 3 regions):

```
Fold 1: Train on Central + South → Test on North
Fold 2: Train on North + South   → Test on Central
Fold 3: Train on North + Central → Test on South
```

Each fold uses the full data from the included regions (not just prototype sites).

### Why Regional Holdout Here?

- **Tests true generalization:** Model has never seen any sites from the test region
- **Answers the deployment question:** "Will this work for users with data from new areas?"
- **Reveals regional gaps:** If one region performs poorly, investigate why

### Depth-Stratified Analysis (Primary Use of Depth)

For each fold, compute metrics stratified by depth bin:

```
For each held-out region:
  For each depth bin (shallow, medium, deep):
    - Compute IoU, precision, recall
    - Count tiles and seagrass-positive tiles
    - Note any regional × depth interactions
```

This reveals:
- Depth thresholds where performance degrades
- Whether depth effects are consistent across regions
- Specific conditions requiring user caution

### Evaluation Metrics

Per-fold metrics:
- IoU on held-out region (overall)
- IoU by depth bin within held-out region
- Site-level area correlation within held-out region

Aggregate metrics:
- Mean IoU across folds (overall and per depth bin)
- Standard deviation (indicates consistency)
- Worst-case IoU (conservative estimate for user guidance)

### Example Results

```
Leave-one-region-out CV Results (SegFormer-B3):

Overall by Region:
| Held-out Region | Test IoU | Sites | Notes                    |
|-----------------|----------|-------|--------------------------|
| North Coast     | 0.71     | 8     |                          |
| Central Coast   | 0.68     | 12    | More turbid water        |
| South Coast     | 0.74     | 10    |                          |
|-----------------|----------|-------|--------------------------|
| Mean ± Std      | 0.71 ± 0.03 |    | Expected on new regions  |

Stratified by Depth (averaged across regional folds):
| Depth Bin   | Mean IoU | Tile Count | Notes                    |
|-------------|----------|------------|--------------------------|
| < 5m        | 0.77     | 45,000     | High confidence          |
| 5-10m       | 0.70     | 32,000     | Moderate confidence      |
| > 10m       | 0.57     | 18,000     | Manual review recommended|
```

### Decision Points

**Regional variance:** If one region < 0.60 while others > 0.75:
- Investigate what makes that region different
- Consider collecting more diverse training data
- Document limitations for users

**Depth degradation:** If deep water IoU is substantially lower:
- Identify the depth threshold where performance drops
- Document this threshold for users
- Consider whether depth as model input would help (return to Stage 1 experiment)

---

## Stage 3: Final Model Training

**Goal:** Train the best possible model for deployment using all available data.

### Data Strategy

```
Training data: ALL sites from ALL regions
Validation: Small held-out set for learning rate scheduling / early stopping
            (can be random sites, not region-stratified)
```

No data is permanently held out. The CV estimate from Stage 2 serves as the expected generalization performance.

### Training Configuration

Use the selected architecture/backbone/loss from Stage 1:

```yaml
model:
  architecture: "SegFormer"  # from Stage 1
  backbone: "mit_b3"
  loss: "LovaszLoss"

data:
  train_chip_dir: "/path/to/all_regions/train"
  val_chip_dir: "/path/to/all_regions/val"  # small holdout for LR scheduling
```

Train for longer / with more data than prototyping:
- Full dataset (not prototype subset)
- Potentially more epochs
- Same hyperparameters that worked in Stage 1

### Export

Export final model to ONNX for deployment:

```bash
python -m src.deploy.kom_onnx \
  configs/seagrass-rgb/segformer_b3.yaml \
  checkpoints/best.ckpt \
  models/seagrass_segformer_b3.onnx
```

---

## Documentation for Release

When releasing the model, document the development process including depth-stratified performance:

```markdown
## Model Card: Seagrass Segmentation (BC Coast)

### Training Data
- X orthomosaics from Y sites
- Regions: North Coast, Central Coast, South Coast (BC, Canada)
- Total tiles: ~N (512x512 px at Z cm/px resolution)
- Class distribution: ~15% seagrass, ~85% background
- Depth range: 0-20m (from 10m resolution bathymetry)

### Architecture
- Model: SegFormer with MiT-B3 backbone
- Loss: LovaszLoss
- Input: RGB, 512x512 tiles

### Validation
- Architecture selection: Stratified train/val/test across regions
- Generalization: 3-fold leave-one-region-out CV

#### Performance by Region
| Held-out Region | IoU   |
|-----------------|-------|
| North Coast     | 0.71  |
| Central Coast   | 0.68  |
| South Coast     | 0.74  |
| **Mean ± Std**  | **0.71 ± 0.03** |

#### Performance by Depth
| Depth       | IoU   | Recommendation              |
|-------------|-------|-----------------------------|
| < 5m        | 0.77  | High confidence             |
| 5-10m       | 0.70  | Moderate confidence         |
| > 10m       | 0.57  | Manual review recommended   |

### Expected Performance
- New sites within BC coastal regions: ~0.71 IoU
- Performance degrades below ~10m depth
- Performance may vary with water clarity, image resolution, seagrass density

### Limitations
- Trained on BC coast eelgrass; may not generalize to other seagrass species
- Optimal for imagery at X-Y cm/px resolution
- Reduced performance in turbid water (see Central Coast results)
- Reduced performance in deep water (>10m); recommend manual review

### Usage Guidance
1. Run model on orthomosaic to get initial segmentation
2. Overlay bathymetry to identify areas >10m depth
3. Prioritize manual review of deep-water predictions
4. Use predictions in shallow water (<5m) with high confidence
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA PREPARATION                                               │
│  ─────────────────────────────────────────────────────────────  │
│  • Create chips from orthomosaics                               │
│  • Assign depth value to each tile from bathymetry              │
│  • Assign site/region metadata                                  │
│  • Assign sites to train/val/test splits (site-level)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Architecture Selection                                │
│  ─────────────────────────────────────────────────────────────  │
│  Data: Prototype subset (~15 sites)                             │
│  Splits: Stratified by region (train/val/test)                  │
│  Depth role: None (optional RGBD experiment if needed)          │
│  Goal: Select best architecture, backbone, tile size, loss      │
│  Output: Model configuration                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Generalization Validation                             │
│  ─────────────────────────────────────────────────────────────  │
│  Data: Full dataset (all sites)                                 │
│  Splits: Leave-one-region-out CV (3 folds)                      │
│  Depth role: STRATIFY performance analysis by depth bin         │
│  Goal: Estimate performance on unseen regions + depth effects   │
│  Output: Expected IoU ± variance, depth-performance thresholds  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Final Model Training                                  │
│  ─────────────────────────────────────────────────────────────  │
│  Data: Full dataset (ALL regions combined)                      │
│  Splits: Minimal val set for LR scheduling only                 │
│  Depth role: USER GUIDANCE in documentation                     │
│  Goal: Best possible model for deployment                       │
│  Output: ONNX model + model card with depth guidance            │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- Roberts et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*.
- Ploton et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*.
