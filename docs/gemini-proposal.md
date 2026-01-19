# Seagrass Segmentation: Technical Strategy Summary

## 1. Data Splitting Strategy
**Critical Requirement:** Prevent spatial data leakage.
*   **Method:** **GroupKFold Cross-Validation**
*   **Group Variable:** `Site_ID` (or distinct geographic location).
*   **Logic:** Ensure all tiles from a specific site/year appear *only* in Train or *only* in Validation. Do not split randomly by image tile.

## 2. Preprocessing & Tiling
*   **Tile Size:** 512x512 pixels (standard) or 1024x1024 (if seagrass features are massive/connected).
*   **Stride/Overlap:** Use a stride smaller than the tile size (e.g., 512 size with 400 stride) during training to increase dataset size and handle edge artifacts.
*   **Filtering:**
    *   Remove tiles that are 100% deep water (no bottom features) if they dominate the dataset.
    *   Keep tiles with land/sand/rock to force the model to learn negative classes.

## 3. Model Architectures & Backbones

### Primary Baseline (Texture Focus)
*   **Architecture:** **U-Net**
    *   *Why:* Excellent at preserving high-frequency spatial details (exact boundaries of seagrass beds) via skip connections.
*   **Backbone:** **ResNet-34**
    *   *Why:* A strong balance between depth and computational speed. Sufficient feature extraction without overfitting on limited datasets.

### Secondary Experiment (Context Focus)
*   **Architecture:** **SegFormer** (implemented via `segmentation-models-pytorch` or `transformers`)
    *   *Why:* Uses Self-Attention to capture global context. Better at handling "patchy" turbidity and distinguishing seagrass from algae based on surrounding context (e.g., sand channels) rather than just pixel color.
*   **Backbone:** **MiT-B3** (Mix Vision Transformer)
    *   *Why:* Optimized transformer encoder that scales well for segmentation tasks.

## 4. Loss Function
**Goal:** Handle class imbalance (small seagrass patches vs. large ocean) and "hard" examples (deep/faint seagrass).
*   **Formula:** `Loss = DiceLoss + FocalLoss`
    *   **Dice Loss:** Maximizes overlap between prediction and ground truth (handles imbalance).
    *   **Focal Loss:** Down-weights easy examples (clear shallow grass) and focuses gradients on hard examples (deep/submerged grass).

## 5. Augmentations (Albumentations)
**Goal:** Simulate the "Water Column" effect (depth, turbidity, refraction).

| Augmentation Type | Specific Transform | Purpose |
| :--- | :--- | :--- |
| **Color/Depth** | `RandomGamma` | Simulates light attenuation at depth. |
| | `CLAHE` | Contrast Limited Adaptive Histogram Equalization. Restores texture in low-contrast deep water. |
| | `HueSaturationValue` | Shifts green grass to blue/grey to mimic depth. |
| **Lighting/Glint** | `RandomBrightnessContrast` | Robustness to sun angle and exposure. |
| | `RandomSunFlare` | (Use with caution) Simulates surface glint/sparkles. |
| **Geometry** | `ElasticTransform` | Simulates refraction distortions through waves. |
| | `Rotate`, `Flip` | Standard invariance. |

## 6. Training Hyperparameters (Starting Points)
*   **Optimizer:** AdamW
*   **Learning Rate:** `1e-4` with Cosine Annealing scheduler.
*   **Batch Size:** 8 or 16 (depending on GPU VRAM).
*   **Precision:** Mixed Precision (16-bit) is recommended to save memory and speed up training.
