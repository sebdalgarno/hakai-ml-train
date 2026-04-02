# Dataset Split Guidance

Principles for splitting drone imagery datasets into train/validation/test sets for semantic segmentation models. These guidelines were applied to the seagrass model development and are applicable to future marine habitat mapping projects.

## Split Ratio

**70 / 15 / 15** (train / val / test)

This ratio is well-established for aerial imagery segmentation. Sufficient validation and test samples ensure reliable metric estimation, particularly for imbalanced classes like seagrass vs. background.

## No Site Leakage: Site-Level Splitting

**Critical**: Split at the site level, not the tile level.

Each drone survey mosaic is tiled into hundreds or thousands of small image chips for model training. Tiles from the same site share lighting, water conditions, and sensor characteristics. If tiles from one site appear in both training and test sets, the model's apparent accuracy is inflated because it has effectively "seen" the test data.

**Rule: All tiles from a given site must go into the same bucket (train, val, or test).**

### Why This Matters

- Tiles from the same mosaic are spatially correlated — neighbouring tiles share overlapping features
- Random tile-level splitting causes **spatial autocorrelation leakage**, where the model memorizes site-specific patterns rather than learning generalizable features
- Site-level splits ensure the model is evaluated on genuinely unseen locations

This principle extends to multi-visit sites: if a site was surveyed multiple times (e.g., Koeye with 9 visits), all visits stay in the same bucket to prevent temporal leakage from the same location.

## Condition Coverage Across Splits

Each split (train, val, test) should represent the full diversity of conditions the model will encounter in production. When assigning sites to splits, ensure each bucket includes a spread of:

- **Geographic regions** — e.g., South, Central, North coast
- **Lighting conditions** — sun angle, cloud cover, overcast, hazy
- **Water conditions** — turbidity, glare, tannins, calm vs. choppy
- **Target density** — sparse, moderate, dense seagrass
- **Image resolution** — range of ground sampling distances (GSD) in each bucket
- **Image quality** — mix of excellent, good, and moderate quality imagery
- **Edge cases** — shadows, fog, mixed species boundaries, bleaching

This ensures that validation and test metrics reflect real-world performance rather than a narrow subset of conditions. If all the "easy" sites end up in test, the metrics will overstate model quality; if all the "hard" sites end up in test, the metrics will understate it.

### Avoiding Single-Site Dominance

Large sites can dominate a split's metrics. For example, if one site contributes 70% of the validation chips, the validation score mostly reflects performance on that one site. To mitigate this:

- Move very large sites to the training set where their size is an advantage
- Ensure no single site exceeds ~40% of its split's chip count
- Balance the number of sites across val and test so metrics reflect diverse conditions

## Prototype Datasets for Rapid Experimentation

A prototype (reduced) dataset is useful for faster iteration during architecture selection and hyperparameter tuning, where relative performance between experiments matters more than absolute metrics.

The key principle is to **randomly sample from every site** rather than selecting a subset of sites. This preserves the full diversity of conditions and regions in the prototype, so results are more likely to transfer to the full dataset.

### How It Works

A target fraction of chips is chosen (e.g., 50%), and that budget is distributed equally across all sites:

1. Group chips by site within each split (train, val, test)
2. Calculate a per-site target: `(total_chips × fraction) / num_sites`
3. Randomly sample that many chips from each site
4. Sites with fewer chips than the target contribute all their chips

This means every site is represented in the prototype regardless of its size. Large multi-visit sites like Koeye (2,041 chips) are sampled down, while small sites like Arakun (31 chips) contribute everything. The result is a more balanced dataset where small sites have proportionally greater representation than in the full dataset.

Separate fractions can be used for train vs. val/test. For example, a higher train fraction (58%) with a lower eval fraction (50%) maintains the approximate 73/14/13 split ratio from the full dataset.

------------------------------------------------------------------------

## Seagrass Model Splits

The splits used for the seagrass segmentation model, applying the principles above. 30 sites across 3 coastal regions of British Columbia (South, Central, North), totalling 14,001 image chips at 1024x1024 px.

### Split Summary

| Bucket | Sites | Chips | % | South | Central | North |
|--------|-------|-------|---|-------|---------|-------|
| Train | 18 | 10,245 | 73% | 3 | 5 | 10 |
| Val | 6 | 1,995 | 14% | 2 | 2 | 2 |
| Test | 6 | 1,761 | 13% | 1 | 2 | 3 |
| **Total** | **30** | **14,001** | **100%** | **6** | **9** | **15** |

### Test (6 sites)

| Site | Region | Chips | Difficulty | Quality | Key Conditions |
|------|--------|-------|------------|---------|----------------|
| McMullin North | Central | 496 | L | E/G | Clear |
| Triquet Bay | Central | 468 | H/M | E/G/M | Difficult edge, other species |
| Bag Harbour | North | 395 | L | E | Excellent baseline |
| Section Cove | North | 196 | M/L | E/G | Hazy lighting |
| Sedgwick | North | 132 | H/M | G/M | Overcast, cloud reflections |
| Beck | South | 74 | H | M | Shadows, sparse eelgrass |
| **Total** |  | **1,761** |  |  |  |

### Validation (6 sites)

| Site | Region | Chips | Difficulty | Quality | Key Conditions |
|------|--------|-------|------------|---------|----------------|
| Superstition | Central | 914 | H/L | E/G/M | Sparse, fog, algae (4 visits) |
| Kendrick Point | North | 514 | M/L | E | Bleaching |
| Louscoone | North | 214 | M/L | E/G | Overcast, cloud reflections |
| Auseth | South | 134 | L | E | Excellent baseline |
| Triquet | Central | 156 | L | E | Clear baseline |
| Bennett Bay | South | 63 | H | M | Cloudy, glint |
| **Total** |  | **1,995** |  |  |  |

### Training (18 sites)

**South (3 sites):**

| Site | Chips | Difficulty | Quality | Key Conditions |
|------|-------|------------|---------|----------------|
| Grice Bay | 1,875 | H | G | Large sparse, coarse delineation |
| Calmus | 244 | L | E | — |
| Arakun | 31 | L | G | — |
| **Subtotal** | **2,150** | | | |

**Central (5 sites):**

| Site | Chips | Difficulty | Quality | Key Conditions |
|------|-------|------------|---------|----------------|
| Pruth Bay | 2,115 | M/L | E/G/M | Shadows, cloud reflections, subtidal (8 visits) |
| Koeye | 2,041 | H/M/L | E/G/M | Tannins, turbidity, glint (9 visits) |
| Goose SW | 1,340 | H/M/L | E/G | Turbidity, overcast, low density (3 visits) |
| Choked Pass | 589 | L | G | Good all round |
| Goose Grass Bay | 68 | L | G | — |
| **Subtotal** | **6,153** | | | |

**North (10 sites):**

| Site | Chips | Difficulty | Quality | Key Conditions |
|------|-------|------------|---------|----------------|
| Louscoone Head | 571 | M/L | E/G | Shadows |
| Island Bay | 334 | L | G | — |
| Ramsay | 274 | L | G | — |
| Swan Bay | 170 | L | E | Great conditions |
| Takelly Cove | 130 | — | G | — |
| Balcolm Inlet | 121 | M | G | Cloudy |
| Beljay Bay | 120 | L | E | — |
| Louscoone West | 93 | L | E | — |
| Kendrick Point West | 76 | L | E | — |
| Heater Harbour | 53 | M | G | Dark, low light |
| **Subtotal** | **1,942** | | | |

| **Train Total** | **10,245** | | | |

### Condition Coverage

| Condition | Train | Val | Test |
|-----------|-------|-----|------|
| Shadows | Pruth Bay, Louscoone Head | — | Beck |
| Sparse eelgrass | Grice Bay | Superstition | Beck |
| Cloud reflections | Pruth Bay | Louscoone | Sedgwick |
| Cloudy/overcast | Balcolm Inlet, Goose SW | Louscoone | Sedgwick |
| Glint | Koeye | Bennett Bay | — |
| Tannins | Koeye | — | — |
| Turbidity | Goose SW, Koeye | — | — |
| Dark/low light | Heater Harbour | — | — |
| Bleaching | — | Kendrick Point | — |
| Difficult edge | — | — | Triquet Bay |
| Fog/algae | — | Superstition | — |
| Hazy lighting | — | — | Section Cove |
| Clear baseline | Calmus | Auseth, Triquet | McMullin North, Bag Harbour |

### Resolution Coverage

| Bucket | Resolution Range | Sites at Each Resolution |
|--------|------------------|--------------------------|
| Train | 2.0–5.6 cm | 2.0–3.0: Arakun, Calmus, Koeye, Pruth Bay, Ramsay, Swan Bay; 3.0–4.5: Goose SW, Grice Bay, Heater Harbour, Kendrick Point West, Louscoone Head, Louscoone West, Island Bay, Balcolm Inlet; 4.5–5.6: Beljay Bay, Choked Pass, Goose Grass Bay, Takelly Cove |
| Val | 1.6–5.2 cm | 1.6–3.0: Auseth, Superstition, Triquet; 4.0–5.2: Bennett Bay, Kendrick Point, Louscoone |
| Test | 2.3–5.0 cm | 2.3–3.0: McMullin North, Bag Harbour; 3.0–4.5: Triquet Bay, Section Cove, Sedgwick; 4.5–5.0: Beck |

All buckets cover a broad range of ground sampling distances (GSD), ensuring the model sees varied spatial scales during training and is evaluated on representative resolutions.

------------------------------------------------------------------------

## Key Decisions

1. **Koeye (9 visits) in training**: Maximizes diversity for learning tannins, turbidity, and glint conditions. Largest multi-visit site provides temporal variation.

2. **Pruth Bay and Grice Bay in training**: These large sites were moved from val/test to training because they dominated their respective split metrics (70% and 53% of chips). Moving them to training prevents single-site bias while providing diverse training conditions (shadows, cloud reflections, sparse eelgrass).

3. **High-difficulty site distribution**:
    - Train: Grice Bay, Goose SW, Koeye, Pruth Bay
    - Val: Bennett Bay, Superstition
    - Test: Triquet Bay, Sedgwick, Beck

4. **Baseline sites in each bucket**: Clear, excellent-quality sites for metric calibration:
    - Train: Calmus
    - Val: Auseth, Triquet
    - Test: McMullin North, Bag Harbour

5. **Regional balance maintained**: Each bucket has representation from all 3 regions, roughly proportional to the overall distribution.

6. **Challenging conditions in test**: Difficult edges (Triquet Bay), hazy lighting (Section Cove), and shadows/sparse (Beck) appear in test to evaluate generalization to edge cases.

7. **Balanced chip distribution**: No single site exceeds 40% of its split, ensuring metrics reflect model performance across diverse conditions rather than one dominant site.

8. **Balanced difficulty across val/test**: Both val and test contain a mix of easy and challenging sites, ensuring validation metrics are predictive of test performance.
