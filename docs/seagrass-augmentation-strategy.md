# Seagrass RGB Augmentation Strategy

This document describes the data augmentation pipeline used for seagrass segmentation model training (`segformer_domain_aug.yaml`), explains the rationale behind each augmentation, and highlights differences from the established kelp-RGB pipeline (`kom_baseline.yaml`).

Both pipelines target the same problem domain (high-resolution drone imagery of coastal marine habitats at 2-5 cm GSD) and share the same overall structure. The seagrass pipeline inherits the kelp-RGB design and adapts it where seagrass-specific conditions warrant different settings.

Config files:
- Seagrass: `configs/seagrass-rgb/augmentation-experiment/segformer_domain_aug.yaml`
- Kelp: `configs/kelp-rgb/kom_baseline.yaml`

## Pipeline Overview

Both pipelines follow a consistent ordering:

| Step | Category | Purpose |
|------|----------|---------|
| 1 | Geometric flips/rotations | Exploit rotational symmetry of nadir imagery |
| 2 | Color/radiometric | Handle lighting and color variation across flights |
| 3 | Sensor noise | Simulate camera sensor noise |
| 4 | Blur | Simulate focus and motion artifacts |
| 5 | Spatial distortion + dropout | Force robustness to occlusion and local deformation |
| 6 | Contrast enhancement | Local contrast adaptation |
| 7 | Color jitter | Subtle global color shifts |
| 8 | Affine transforms | Scale, rotation, and translation variation |
| 9 | Normalize + ToTensor | Preprocessing (always applied) |

## Augmentation Details

### 1. D4 (Dihedral Group of the Square)

Applies all 8 symmetries of a square: 4 rotations (0/90/180/270) and 4 reflections (horizontal, vertical, and both diagonals).

| | Seagrass | Kelp |
|---|---|---|
| p | 1.0 | 1.0 |

**Same in both.** This is the single most important augmentation for nadir/overhead drone imagery. Unlike ground-level photos, overhead images have no canonical "up" direction, so all 8 orientations are equally valid. These transforms are lossless (no interpolation artifacts) and computationally free.

### 2. Color Augmentation (OneOf)

Applies one of either RandomBrightnessContrast or HueSaturationValue. Bundling them in a `OneOf` ensures only one type of color shift is applied per sample, preventing unrealistic stacked distortions.

| | Seagrass | Kelp |
|---|---|---|
| OneOf p | 0.5 | 0.5 |
| **RandomBrightnessContrast** | | |
| brightness_limit | [-0.1, 0.1] | [-0.1, 0.1] |
| contrast_limit | [-0.1, 0.1] | [-0.1, 0.1] |
| **HueSaturationValue** | | |
| hue_shift_limit | [-10, 10] | [-5, 5] |
| sat_shift_limit | [-15, 15] | [-10, 10] |
| val_shift_limit | [-15, 15] | [-15, 15] |

**Difference: Seagrass uses wider hue and saturation ranges.** This is intentional. Seagrass appearance varies more than kelp due to: species color variation (green to brown), depth-dependent blue-green attenuation through the water column, seasonal changes, and variable water turbidity. The wider ranges help the model generalize across these conditions. Brightness/contrast limits are identical.

**Why OneOf?** Literature recommends against stacking multiple independent color augmentations (e.g. applying both brightness shift AND hue shift to the same sample), as this can produce unrealistic color combinations that hurt generalization. The OneOf structure matches kelp-RGB and is the more conservative, well-supported approach.

### 3. Sensor Noise (OneOf)

Simulates camera sensor noise from high ISO settings or low-light conditions.

| | Seagrass | Kelp |
|---|---|---|
| OneOf p | **0.3** | 0.3 |
| **GaussNoise** | | |
| std_range | [0.02, 0.044] | [0.02, 0.044] |
| **ISONoise** | | |
| color_shift | [0.01, 0.05] | [0.01, 0.05] |
| intensity | [0.1, 0.2] | [0.1, 0.2] |

**Same in both.** Literature strongly recommends keeping noise augmentation probability at 0.3 or lower. Excessive noise buries the signal and degrades model performance. The noise parameters themselves simulate realistic sensor behavior without being destructive.

### 4. Blur (OneOf)

Simulates optical defocus, motion during capture, and atmospheric haze.

| | Seagrass | Kelp |
|---|---|---|
| OneOf p | **0.3** | 0.3 |
| **MotionBlur** | | |
| blur_limit | [3, 5] | [3, 7] |
| **MedianBlur** | | |
| blur_limit | [3, 5] | [3, 7] |
| **GaussianBlur** | | |
| sigma_limit | [0.3, 1.5] | [0.5, 3.0] |

**Difference: Seagrass uses milder blur.** Seagrass features are smaller and more subtle than kelp canopy (thin blades vs. large floating mats), so aggressive blur risks destroying the signal. The reduced blur limits preserve fine-grained texture that matters for seagrass detection, while still providing regularization.

### 5. Spatial Distortion + Dropout (OneOf)

Applies one of three spatial augmentations that force the model to be robust to local deformations and partial occlusion.

| | Seagrass | Kelp |
|---|---|---|
| OneOf p | 0.3 | 0.3 |
| **CoarseDropout** | | |
| hole_height_range | [1, 5] | [1, 5] |
| hole_width_range | [1, 5] | [1, 5] |
| num_holes_range | [1, 64] | [1, 64] |
| **GridDistortion** | | |
| distort_limit | [-0.15, 0.15] | [-0.1, 0.1] |
| num_steps | 5 | 10 |
| **ElasticTransform** | present | **absent** |

**Differences:**

- **ElasticTransform** is present in seagrass but not in kelp. This simulates natural deformations in seagrass bed boundaries, which are inherently soft and variable (unlike harder kelp canopy edges). Literature shows the distortion augmentation group (grid + elastic + optical) is one of the highest-performing groups for segmentation tasks.

- **GridDistortion** uses slightly stronger distortion in seagrass (0.15 vs 0.1) with fewer steps (5 vs 10), producing coarser but more pronounced warping.

- **CoarseDropout** is identical. It randomly masks small rectangular regions, forcing the model to make predictions from partial context rather than relying on any single diagnostic feature. This is rated "high impact" in Albumentations documentation.

### 6. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Enhances local contrast by applying histogram equalization in small tiles, particularly useful for images with mixed bright and dark regions.

| | Seagrass | Kelp |
|---|---|---|
| p | **0.2** | 0.1 |
| clip_limit | **[1.0, 4.0]** | [1.0, 2.0] |
| tile_grid_size | [8, 8] | [8, 8] |

**Difference: Seagrass uses stronger and more frequent CLAHE.** Seagrass imagery has more extreme local contrast variation than kelp: bright sandy substrate adjacent to dark seagrass beds, sun glint on shallow water next to submerged vegetation, and shadowed areas near shorelines. The wider clip limit range and higher probability help the model handle these challenging mixed-illumination scenes.

### 7. ColorJitter

Applies subtle, simultaneous shifts to brightness, contrast, saturation, and hue. Unlike the OneOf color augmentation in step 2, these are very small perturbations applied together.

| | Seagrass | Kelp |
|---|---|---|
| p | 0.3 | 0.3 |
| brightness | [0.95, 1.05] | [0.95, 1.05] |
| contrast | [0.95, 1.05] | [0.95, 1.05] |
| hue | [-0.05, 0.05] | [-0.05, 0.05] |
| saturation | [0.95, 1.05] | [0.95, 1.05] |

**Same in both.** These are intentionally subtle (5% variation). The purpose is to simulate minor white balance and exposure differences between adjacent flight lines within a single orthomosaic, complementing the larger color shifts in step 2 that simulate differences between flights.

### 8. Affine Transforms

Applies geometric transformations: scaling, rotation, and translation.

| | Seagrass | Kelp |
|---|---|---|
| p | 0.3 | 0.3 |
| rotate | **[-5, 5]** | [-5, 5] |
| scale | **[0.7, 1.3]** | [0.9, 1.1] |
| translate_percent | **[-0.05, 0.05]** | [-0.05, 0.05] |

**Difference: Seagrass uses a much wider scale range (0.7-1.3 vs 0.9-1.1).** This accounts for greater altitude variation in seagrass survey flights (coastal sites with varying terrain, wind conditions causing altitude drift) and the need to generalize across surveys flown at different heights. At 2-5 cm GSD, small altitude changes produce meaningful scale differences. Literature supports scale ranges up to 0.5-1.4x for drone imagery.

Rotation (+-5 degrees) and translation (+-5%) are identical to kelp-RGB. Small rotation augments D4's discrete 90-degree rotations with slight off-axis variation. Translation simulates the fact that objects of interest are rarely perfectly centered in chips.

## Summary of Key Differences

| Aspect | Seagrass | Kelp | Rationale |
|--------|----------|------|-----------|
| HSV ranges | Wider (hue +-10, sat +-15) | Narrower (hue +-5, sat +-10) | Greater color variability in seagrass |
| Blur strength | Milder (limit 3-5, sigma 0.3-1.5) | Stronger (limit 3-7, sigma 0.5-3.0) | Preserve fine seagrass texture |
| Scale range | Wider (0.7-1.3) | Narrow (0.9-1.1) | Greater altitude variation in surveys |
| CLAHE | Stronger (clip 1-4, p=0.2) | Weaker (clip 1-2, p=0.1) | More extreme local contrast in coastal scenes |
| ElasticTransform | Present | Absent | Soft seagrass bed boundaries |
| GridDistortion | Stronger (0.15) | Weaker (0.1) | More aggressive spatial regularization |

All other augmentations (D4, noise, CoarseDropout, ColorJitter, Affine rotation/translation) use the same or very similar settings.

## Design Principles

1. **Conservative probabilities.** All augmentation groups use p=0.3 or lower (except D4 and the color OneOf). Literature shows noise and blur above p=0.3 can degrade performance.

2. **OneOf grouping.** Mutually exclusive augmentations (e.g., different color shifts, different blur types) are bundled in `OneOf` blocks so only one variant is applied per sample. This prevents unrealistic stacking.

3. **Domain-appropriate parameters.** Settings are adapted from kelp-RGB based on known differences between kelp and seagrass imagery, not arbitrary choices.

4. **Shared foundation.** The overall pipeline structure, ordering, and most parameters are inherited from kelp-RGB, which has been validated in production. Changes are targeted and justified.
