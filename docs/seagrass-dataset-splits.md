# Seagrass Dataset Splits

Train/val/test splits for Stage 1 (Architecture Selection) experiments. Follows guidance from `dataset-split-guidance.md` and `model-development-workflow.md`.

## Principles

-   **70/15/15 split** (train/val/test)
-   **Site-level splitting**: All visits from a site stay in the same bucket
-   **Regional stratification**: All 3 regions (South, Central, North) represented in all buckets
-   **Condition coverage**: Each split represents diverse lighting, water conditions, target density, and edge cases

------------------------------------------------------------------------

## Split A: Full Dataset (35 sites)

Use this split if all 35 sites are available.

### Sites by Region

| Region  | Sites |
|---------|-------|
| South   | 10    |
| Central | 9     |
| North   | 16    |

### TEST (5 sites, 14%)

| Site | Region | Difficulty | Quality | Key Conditions |
|--------------|--------------|--------------|--------------|------------------|
| Grice Bay | South | H | G | Large sparse area, coarse delineation |
| Superstition | Central | H/M/L | E/G/M | Sparse eelgrass, fog, algae, subtidal |
| Triquet | Central | L | E | Clear baseline |
| Sedgwick | North | H/M | G/M | Overcast, cloud reflections |
| Bag Harbour | North | L | E | Excellent baseline |

### VALIDATION (5 sites, 14%)

| Site | Region | Difficulty | Quality | Key Conditions |
|--------------|--------------|--------------|--------------|------------------|
| Bennett Bay | South | H | M | Cloudy, glint |
| Auseth | South | L | E | Excellent baseline |
| Pruth Bay | Central | M/L | E/G/M | Shadows, cloud reflections, subtidal |
| Kendrick Point | North | M/L | E | Bleaching |
| Louscoone | North | M/L | E/G | Overcast, underwater |

### TRAINING (25 sites, 71%)

**South (7 sites):**

| Site          | Difficulty | Quality | Key Conditions                   |
|---------------|------------|---------|----------------------------------|
| Arakun        | L          | G       | —                                |
| Beck          | H          | M       | Shadows, sparse eelgrass         |
| Calmus        | L          | E       | —                                |
| Jaques Jarvis | L          | G       | —                                |
| Nettle        | L          | G       | Clear subtidal                   |
| Sharp         | H          | G       | Overcast, low density intertidal |
| Turret        | L          | G       | Clear                            |

**Central (6 sites):**

| Site            | Difficulty | Quality | Key Conditions                       |
|-----------------|------------|---------|--------------------------------------|
| Choked Pass     | L          | G/M     | Towed video validation               |
| Goose Grass Bay | L          | G       | —                                    |
| Goose SW        | H/M/L      | E/G     | Low density, turbidity, overcast     |
| Koeye           | H/M/L      | E/G/M   | Tannins, turbidity, glint (9 visits) |
| McMullin North  | L          | E/G     | Clear                                |
| Triquet Bay     | H/M        | E/G/M   | Difficult edge, other species        |

**North (12 sites):**

| Site                | Difficulty | Quality | Key Conditions   |
|---------------------|------------|---------|------------------|
| Balcolm Inlet       | M          | G       | Cloudy           |
| Beljay Bay          | L          | E       | —                |
| Heater Harbour      | M          | G       | Dark, low light  |
| Island Bay          | L          | G       | —                |
| Kendrick Point West | L          | E       | —                |
| Louscoone Head      | M/L        | E/G     | Shadows          |
| Louscoone West      | L          | E       | —                |
| Powrivco            | L          | G       | —                |
| Ramsay              | L          | G       | —                |
| Section Cove        | M/L        | E/G     | Hazy lighting    |
| Swan Bay            | L          | E       | Great conditions |
| Takelly Cove        | —          | G       | —                |

### Summary (Split A)

| Bucket | Sites | South | Central | North | High | Med | Low |
|--------|-------|-------|---------|-------|------|-----|-----|
| Train  | 25    | 7     | 6       | 12    | 5\*  | 7\* | 20+ |
| Val    | 5     | 2     | 1       | 2     | 1    | 3\* | 2   |
| Test   | 5     | 1     | 2       | 2     | 2    | 1\* | 3\* |

\*includes sites with multiple visits spanning difficulty levels

------------------------------------------------------------------------

## Split B: Reduced Dataset (29 sites) — REVISED

Use this split if 9 sites are unavailable (Goose Grass Bay, Jaques Jarvis, Nettle, Powrivco, Sharp, Turret removed from South/Central/North).

**Revision notes**: 1. Original split had pruth_bay (70% of val chips) and grice_bay (53% of test chips) dominating their respective splits. These large sites were moved to training to prevent metrics being skewed by a single site. 2. Superstition swapped from test to val (and McMullin North to test) to balance difficulty across both splits, ensuring validation metrics predict test performance.

### Sites by Region

| Region  | Sites |
|---------|-------|
| South   | 6     |
| Central | 8     |
| North   | 15    |

### TEST (6 sites, 14%)

| Site | Region | Chips | Difficulty | Quality | Key Conditions |
|------------|------------|------------|------------|------------|--------------|
| McMullin North | Central | 4,437 | L | E/G | Clear |
| Triquet Bay | Central | 3,726 | H/M | E/G/M | Difficult edge, other species |
| Bag Harbour | North | 1,852 | L | E | Excellent baseline |
| Section Cove | North | 1,603 | M/L | E/G | Hazy lighting |
| Sedgwick | North | 634 | H/M | G/M | Overcast, cloud reflections |
| Beck | South | 577 | H | M | Shadows, sparse eelgrass |
| **Total** |  | **12,829** |  |  |  |

### VALIDATION (6 sites, 11%)

| Site | Region | Chips | Difficulty | Quality | Key Conditions |
|------------|------------|------------|------------|------------|--------------|
| Superstition | Central | 4,500 | H/L | E/G/M | Sparse, fog, algae (4 visits) |
| Kendrick Point | North | 2,322 | M/L | E | Bleaching |
| Auseth | South | 1,082 | L | E | Excellent baseline |
| Triquet | Central | 724 | L | E | Clear baseline |
| Louscoone | North | 673 | M/L | E/G | Overcast |
| Bennett Bay | South | 310 | H | M | Cloudy, glint |
| **Total** |  | **9,611** |  |  |  |

### TRAINING (17 sites, 75%)

**South (3 sites):**

| Site         | Chips      | Difficulty | Quality | Key Conditions                   |
|--------------|------------|------------|---------|----------------------------------|
| Grice Bay    | 7,870      | H          | G       | Large sparse, coarse delineation |
| Calmus       | 1,961      | L          | E       | —                                |
| Arakun       | 299        | L          | G       | —                                |
| **Subtotal** | **10,130** |            |         |                                  |

**Central (4 sites):**

| Site        | Chips      | Difficulty | Quality | Key Conditions                                   |
|-------------|------------|------------|---------|--------------------------------------------------|
| Koeye       | 17,076     | H/M/L      | E/G/M   | Tannins, turbidity, glint (9 visits)             |
| Goose SW    | 10,642     | H/M/L      | E/G     | Turbidity, overcast, low density (3 visits)      |
| Pruth Bay   | 9,509      | M/L        | E/G/M   | Shadows, cloud reflections, subtidal (8 visits)  |
| Choked Pass | 4,832      | L          | G       | Good all round                                   |
| **Subtotal** | **42,059** |           |         |                                                  |

**North (10 sites):**

| Site                | Chips      | Difficulty | Quality | Key Conditions   |
|---------------------|------------|------------|---------|------------------|
| Louscoone Head      | 4,522      | M/L        | E/G     | Shadows          |
| Island Bay          | 2,614      | L          | G       | —                |
| Ramsay              | 2,125      | L          | G       | —                |
| Swan Bay            | 1,316      | L          | E       | Great conditions |
| Beljay Bay          | 1,116      | L          | E       | —                |
| Takelly Cove        | 1,097      | —          | G       | —                |
| Balcolm Inlet       | 1,010      | M          | G       | Cloudy           |
| Louscoone West      | 867        | L          | E       | —                |
| Kendrick Point West | 691        | L          | E       | —                |
| Heater Harbour      | 461        | M          | G       | Dark, low light  |
| **Subtotal**        | **15,819** |            |         |                  |

| **TRAIN TOTAL** | **68,008** | | | |

### Summary (Split B)

| Bucket | Sites | Chips  | %   | South | Central | North |
|--------|-------|--------|-----|-------|---------|-------|
| Train  | 17    | 68,008 | 75% | 3     | 4       | 10    |
| Val    | 6     | 9,611  | 11% | 2     | 2       | 2     |
| Test   | 6     | 12,829 | 14% | 1     | 2       | 3     |
| **Total** | **29** | **90,448** | **100%** | **6** | **8** | **15** |

Note: Chip counts are after nodata removal. Largest sites (pruth_bay, grice_bay) moved to training to prevent single-site dominance in val/test metrics. Superstition swapped to val to balance difficulty across val/test

### Resolution Coverage (Split B)

| Bucket | Resolution Range | Sites at Each Resolution |
|--------|------------------|--------------------------|
| Train  | 2.0–5.6 cm | 2.0–3.0: Arakun, Calmus, Koeye, Pruth Bay, Ramsay, Swan Bay; 3.0–4.5: Goose SW, Grice Bay, Heater Harbour, Kendrick Point West, Louscoone Head, Louscoone West, Island Bay, Balcolm Inlet; 4.5–5.6: Beljay Bay, Choked Pass, Takelly Cove |
| Val    | 1.6–5.2 cm | 1.6–3.0: Auseth, Superstition, Triquet; 4.0–5.2: Bennett Bay, Kendrick Point, Louscoone |
| Test   | 2.3–5.0 cm | 2.3–3.0: McMullin North, Bag Harbour; 3.0–4.5: Triquet Bay, Section Cove, Sedgwick; 4.5–5.0: Beck |

**Summary:**
- **Train**: 2.0–5.6 cm (full range, 17 sites across 42 orthos)
- **Val**: 1.6–5.2 cm (includes finest resolution via Superstition u0914 at 1.6 cm)
- **Test**: 2.3–5.0 cm (moderate range, 6 sites)

All buckets cover a broad range of ground sampling distances (GSD), ensuring the model sees varied spatial scales during training and is evaluated on representative resolutions.

------------------------------------------------------------------------

## Prototype Dataset (Split B)

A reduced subset of Split B for rapid iteration during architecture selection and hyperparameter tuning. Selects specific orthos (individual survey visits) rather than entire sites to control dataset size while maintaining diversity.

**Design principles:**
- ~18% of full dataset (~17,000 chips) for fast training cycles
- ~74/12/14 split (weighted toward train due to 25% chip overlap in training data)
- One ortho per multi-visit site to maximize site diversity
- All 3 regions represented in each bucket
- Mix of difficulty levels and conditions in each bucket

### TRAIN (9 orthos, ~12,400 chips, 74%)

| Ortho | Chips | Region | Difficulty | Quality | Conditions |
|-------|-------|--------|------------|---------|------------|
| arakun_u0411 | 299 | South | L | G | baseline |
| calmus_u0421 | 1,961 | South | L | E | baseline |
| koeye_u0715 | 1,447 | Central | H/M/L | E/G/M | tannins, turbidity, glint |
| pruth_bay_u0383 | 1,158 | Central | M/L | E/G/M | shadows, cloud reflections |
| goose_sw_u1174 | 2,700 | Central | H/M/L | E/G | turbidity, overcast, low density |
| heater_harbour_u0088 | 461 | North | M | G | dark, low light |
| beljay_bay_u0479 | 1,116 | North | L | E | baseline |
| kendrick_point_west_u0494 | 691 | North | L | E | baseline |
| island_bay_u0486 | 2,614 | North | L | G | baseline |
| **Subtotal** | **12,447** | 2S, 3C, 4N | | | |

### VAL (4 orthos, ~2,000 chips, 12%)

| Ortho | Chips | Region | Difficulty | Quality | Conditions |
|-------|-------|--------|------------|---------|------------|
| bennett_bay | 310 | South | H | M | cloudy, glint |
| triquet_u1160 | 724 | Central | L | E | clear baseline |
| superstition_u1280 | 635 | Central | H/L | E/G/M | fog, sparse, algae |
| louscoone_u0091 | 317 | North | M/L | E/G | overcast |
| **Subtotal** | **1,986** | 1S, 2C, 1N | | | |

### TEST (4 orthos, ~2,300 chips, 14%)

| Ortho | Chips | Region | Difficulty | Quality | Conditions |
|-------|-------|--------|------------|---------|------------|
| beck_u0409 | 577 | South | H | M | shadows, sparse |
| triquet_bay_u0537 | 875 | Central | H/M | E/G/M | difficult edge |
| sedgwick_u0085 | 250 | North | H/M | G/M | overcast, cloud reflections |
| section_cove_u0249 | 562 | North | M/L | E/G | hazy lighting |
| **Subtotal** | **2,264** | 1S, 1C, 2N | | | |

### Summary (Prototype)

| Split | Orthos | Sites | Chips | % | Difficulty | Quality |
|-------|--------|-------|-------|---|------------|---------|
| Train | 9 | 9 | 12,447 | 74% | 3H, 4M, 7L | 5E, 6G, 2M |
| Val | 4 | 4 | 1,986 | 12% | 2H, 2M, 2L | 2E, 2G, 2M |
| Test | 4 | 4 | 2,264 | 14% | 3H, 3M, 1L | 2E, 3G, 2M |
| **Total** | **17** | **17** | **16,697** | 100% | | |

### Prototype Condition Coverage

| Condition | Train | Val | Test |
|-----------|-------|-----|------|
| Shadows | pruth_bay_u0383 | — | beck_u0409 |
| Sparse eelgrass | — | superstition_u1280 | beck_u0409 |
| Cloud reflections | pruth_bay_u0383 | — | sedgwick_u0085 |
| Cloudy/overcast | goose_sw_u1174 | louscoone_u0091 | sedgwick_u0085 |
| Glint | koeye_u0715 | bennett_bay | — |
| Tannins | koeye_u0715 | — | — |
| Turbidity | goose_sw_u1174, koeye_u0715 | — | — |
| Dark/low light | heater_harbour_u0088 | — | — |
| Difficult edge | — | — | triquet_bay_u0537 |
| Fog/algae | — | superstition_u1280 | — |
| Hazy lighting | — | — | section_cove_u0249 |
| Clear baseline | arakun, calmus, beljay_bay, kendrick_point_west, island_bay | triquet_u1160 | — |

Note: Difficulty/quality counts in summary reflect that some orthos span multiple levels (e.g., koeye H/M/L counted as 1H, 1M, 1L). Train weighted toward baseline sites to learn fundamentals; val/test include more challenging conditions.

------------------------------------------------------------------------

## Small Prototype Dataset

A further reduced subset of the Prototype for rapid architecture and hyperparameter comparison. Reduces training to 5 orthos (~6,100 chips) while keeping the same val/test sets. Designed for fast iteration where relative performance matters more than absolute metrics.

**Design principles:**
- ~62% of prototype training data for faster training cycles
- Same val/test as full prototype (evaluation on diverse conditions)
- 5 training orthos covering all 3 regions
- Mix of baseline and challenging conditions
- Sufficient for architecture comparison (UNet++ vs DeepLabV3+ vs SegFormer)
- Sufficient for coarse hyperparameter comparison (augmentation strategies, tile sizes)

**Use cases:**
- Initial architecture screening
- Augmentation strategy comparison
- Tile size comparison (512 vs 1024)
- Learning rate range finding

**Limitations:**
- Absolute IoU will be lower than full training
- Fine-grained hyperparameter tuning may not transfer perfectly
- Validate top candidates on full prototype or Split B

### TRAIN (5 orthos, ~6,100 chips)

| Ortho | Chips | Region | Difficulty | Quality | Conditions |
|-------|-------|--------|------------|---------|------------|
| calmus_u0421 | 1,961 | South | L | E | baseline |
| koeye_u0715 | 1,447 | Central | H/M/L | E/G/M | tannins, turbidity, glint |
| pruth_bay_u0383 | 1,158 | Central | M/L | E/G/M | shadows, cloud reflections |
| beljay_bay_u0479 | 1,116 | North | L | E | baseline |
| heater_harbour_u0088 | 461 | North | M | G | dark, low light |
| **Subtotal** | **~6,143** | 1S, 2C, 2N | | | |

**Removed from full prototype train:** arakun_u0411, goose_sw_u1174, kendrick_point_west_u0494, island_bay_u0486

### VAL (4 orthos, ~2,000 chips) — Same as Prototype

| Ortho | Chips | Region | Difficulty | Quality | Conditions |
|-------|-------|--------|------------|---------|------------|
| bennett_bay | 310 | South | H | M | cloudy, glint |
| triquet_u1160 | 724 | Central | L | E | clear baseline |
| superstition_u1280 | 635 | Central | H/L | E/G/M | fog, sparse, algae |
| louscoone_u0091 | 317 | North | M/L | E/G | overcast |
| **Subtotal** | **1,986** | 1S, 2C, 1N | | | |

### TEST (4 orthos, ~2,300 chips) — Same as Prototype

| Ortho | Chips | Region | Difficulty | Quality | Conditions |
|-------|-------|--------|------------|---------|------------|
| beck_u0409 | 577 | South | H | M | shadows, sparse |
| triquet_bay_u0537 | 875 | Central | H/M | E/G/M | difficult edge |
| sedgwick_u0085 | 250 | North | H/M | G/M | overcast, cloud reflections |
| section_cove_u0249 | 562 | North | M/L | E/G | hazy lighting |
| **Subtotal** | **2,264** | 1S, 1C, 2N | | | |

### Summary (Small Prototype)

| Split | Orthos | Chips | % |
|-------|--------|-------|---|
| Train | 5 | ~6,143 | 59% |
| Val | 4 | 1,986 | 19% |
| Test | 4 | 2,264 | 22% |
| **Total** | **13** | **~10,393** | 100% |

Note: Train % is lower than standard 70% because val/test are held constant from full prototype. This is acceptable for comparison purposes where consistent evaluation matters more than maximum training data.

### Small Prototype Condition Coverage

| Condition | Train | Val | Test |
|-----------|-------|-----|------|
| Shadows | pruth_bay_u0383 | — | beck_u0409 |
| Sparse eelgrass | — | superstition_u1280 | beck_u0409 |
| Cloud reflections | pruth_bay_u0383 | — | sedgwick_u0085 |
| Cloudy/overcast | — | louscoone_u0091 | sedgwick_u0085 |
| Glint | koeye_u0715 | bennett_bay | — |
| Tannins | koeye_u0715 | — | — |
| Turbidity | koeye_u0715 | — | — |
| Dark/low light | heater_harbour_u0088 | — | — |
| Difficult edge | — | — | triquet_bay_u0537 |
| Fog/algae | — | superstition_u1280 | — |
| Hazy lighting | — | — | section_cove_u0249 |
| Clear baseline | calmus_u0421, beljay_bay_u0479 | triquet_u1160 | — |

Note: Removed orthos (goose_sw, island_bay, etc.) were mostly baseline conditions. The 5 retained training orthos still cover key challenging conditions (tannins, shadows, dark lighting) plus baselines from South and North regions.

### Scripts

- `scripts/create_small_prototype_raw.sh`: Copy raw TIF images/labels for selected orthos
- `scripts/create_small_prototype.sh`: Copy pre-made 512px chips for selected orthos

------------------------------------------------------------------------

## Condition Coverage (Full Split B)

Coverage across buckets for the full Split B dataset.

| Condition | Train | Val | Test |
|-------------------------|----------------|----------------|----------------|
| Shadows | Pruth Bay, Louscoone Head | — | Beck |
| Sparse eelgrass | Grice Bay | Superstition | Beck |
| Cloud reflections | Pruth Bay | — | Sedgwick |
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

Note: Split A includes additional condition coverage in training from Sharp (overcast, low density), Nettle (clear subtidal), and other removed sites.

------------------------------------------------------------------------

## Key Decisions

1.  **Koeye (9 visits) in training**: Maximizes diversity for learning tannins, turbidity, and glint conditions. Largest multi-visit site provides temporal variation.

2.  **Pruth Bay and Grice Bay in training (Split B revised)**: Originally in val/test, these large sites dominated metrics (70% and 53% respectively). Moving to training prevents single-site bias while providing diverse training conditions (shadows, cloud reflections, sparse eelgrass).

3.  **High-difficulty site distribution (Split B revised)**:

    -   Train: Grice Bay, Goose SW, Koeye, Pruth Bay
    -   Val: Bennett Bay, Superstition
    -   Test: Triquet Bay, Sedgwick, Beck

4.  **Baseline sites in each bucket**: Clear, excellent-quality sites for metric calibration:

    -   Train: Calmus
    -   Val: Auseth, McMullin North, Triquet
    -   Test: Bag Harbour

5.  **Regional balance maintained**: Each bucket has representation from all 3 regions, roughly proportional to the overall distribution (North has most sites).

6.  **Challenging conditions in test**: Difficult edges (Triquet Bay), hazy lighting (Section Cove), and shadows/sparse (Beck) appear in test to evaluate generalization to edge cases.

7.  **Balanced val/test chip distribution**: No single site exceeds 40% of its split, ensuring metrics reflect model performance across diverse conditions rather than one dominant site.

8.  **Balanced difficulty across val/test**: Superstition (fog, sparse) moved to val so both splits contain challenging conditions. This ensures validation metrics predict test performance and hyperparameter tuning accounts for difficult cases.

------------------------------------------------------------------------

## Regional Cross-Validation Splits (3-Fold)

Three-fold cross-validation holding out each region as test set. Used to evaluate model generalization across geographic regions.

**Design principles:**
- Each fold holds out one region (North, Central, South) as test
- Remaining two regions split into train/val with ~82/18 ratio
- Val set includes mix of difficulty levels and conditions
- Train/val maintain diversity in quality, resolution, and conditions

### CV_NORTH (North held out)

**Test**: All 15 North sites (22,903 chips)

| Site | Chips | Difficulty | Quality | Key Conditions |
|------|-------|------------|---------|----------------|
| Louscoone Head | 4,522 | M/L | E/G | Shadows |
| Island Bay | 2,614 | L | G | — |
| Kendrick Point | 2,322 | M/L | E | Bleaching |
| Ramsay | 2,125 | L | G | — |
| Bag Harbour | 1,852 | L | E | Excellent baseline |
| Section Cove | 1,603 | M/L | E/G | Hazy lighting |
| Swan Bay | 1,316 | L | E | Great conditions |
| Beljay Bay | 1,116 | L | E | — |
| Takelly Cove | 1,097 | — | G | — |
| Balcolm Inlet | 1,010 | M | G | Cloudy |
| Louscoone West | 867 | L | E | — |
| Kendrick Point West | 691 | L | E | — |
| Louscoone | 673 | M/L | E/G | Overcast |
| Sedgwick | 634 | H/M | G/M | Overcast, cloud reflections |
| Heater Harbour | 461 | M | G | Dark, low light |

**Validation** (6 sites, ~11,600 chips, 17%):

| Site | Chips | Region | Difficulty | Quality | Key Conditions |
|------|-------|--------|------------|---------|----------------|
| Superstition | 4,500 | Central | H/L | E/G/M | Sparse, fog, algae |
| McMullin North | 4,437 | Central | L | E/G | Clear |
| Auseth | 1,082 | South | L | E | Excellent baseline |
| Triquet | 724 | Central | L | E | Clear baseline |
| Beck | 577 | South | H | M | Shadows, sparse |
| Bennett Bay | 310 | South | H | M | Cloudy, glint |

**Training** (8 sites, ~55,900 chips, 83%):

| Site | Chips | Region | Difficulty | Quality | Key Conditions |
|------|-------|--------|------------|---------|----------------|
| Koeye | 17,076 | Central | H/M/L | E/G/M | Tannins, turbidity, glint |
| Goose SW | 10,642 | Central | H/M/L | E/G | Turbidity, overcast |
| Pruth Bay | 9,509 | Central | M/L | E/G/M | Shadows, cloud reflections |
| Grice Bay | 7,870 | South | H | G | Large sparse |
| Choked Pass | 4,832 | Central | L | G | Good all round |
| Triquet Bay | 3,726 | Central | H/M | E/G/M | Difficult edge |
| Calmus | 1,961 | South | L | E | — |
| Arakun | 299 | South | L | G | — |

### CV_CENTRAL (Central held out)

**Test**: All 8 Central sites (55,446 chips)

| Site | Chips | Difficulty | Quality | Key Conditions |
|------|-------|------------|---------|----------------|
| Koeye | 17,076 | H/M/L | E/G/M | Tannins, turbidity, glint |
| Goose SW | 10,642 | H/M/L | E/G | Turbidity, overcast |
| Pruth Bay | 9,509 | M/L | E/G/M | Shadows, cloud reflections |
| Choked Pass | 4,832 | L | G | Good all round |
| Superstition | 4,500 | H/L | E/G/M | Sparse, fog, algae |
| McMullin North | 4,437 | L | E/G | Clear |
| Triquet Bay | 3,726 | H/M | E/G/M | Difficult edge |
| Triquet | 724 | L | E | Clear baseline |

**Validation** (6 sites, ~5,600 chips, 16%):

| Site | Chips | Region | Difficulty | Quality | Key Conditions |
|------|-------|--------|------------|---------|----------------|
| Kendrick Point | 2,322 | North | M/L | E | Bleaching |
| Auseth | 1,082 | South | L | E | Excellent baseline |
| Louscoone | 673 | North | M/L | E/G | Overcast |
| Sedgwick | 634 | North | H/M | G/M | Overcast, cloud reflections |
| Beck | 577 | South | H | M | Shadows, sparse |
| Bennett Bay | 310 | South | H | M | Cloudy, glint |

**Training** (15 sites, ~29,400 chips, 84%):

| Site | Chips | Region | Difficulty | Quality | Key Conditions |
|------|-------|--------|------------|---------|----------------|
| Grice Bay | 7,870 | South | H | G | Large sparse |
| Louscoone Head | 4,522 | North | M/L | E/G | Shadows |
| Island Bay | 2,614 | North | L | G | — |
| Ramsay | 2,125 | North | L | G | — |
| Calmus | 1,961 | South | L | E | — |
| Bag Harbour | 1,852 | North | L | E | Excellent baseline |
| Section Cove | 1,603 | North | M/L | E/G | Hazy lighting |
| Swan Bay | 1,316 | North | L | E | Great conditions |
| Beljay Bay | 1,116 | North | L | E | — |
| Takelly Cove | 1,097 | North | — | G | — |
| Balcolm Inlet | 1,010 | North | M | G | Cloudy |
| Louscoone West | 867 | North | L | E | — |
| Kendrick Point West | 691 | North | L | E | — |
| Heater Harbour | 461 | North | M | G | Dark, low light |
| Arakun | 299 | South | L | G | — |

### CV_SOUTH (South held out)

**Test**: All 6 South sites (12,099 chips)

| Site | Chips | Difficulty | Quality | Key Conditions |
|------|-------|------------|---------|----------------|
| Grice Bay | 7,870 | H | G | Large sparse, coarse delineation |
| Calmus | 1,961 | L | E | — |
| Auseth | 1,082 | L | E | Excellent baseline |
| Beck | 577 | H | M | Shadows, sparse |
| Bennett Bay | 310 | H | M | Cloudy, glint |
| Arakun | 299 | L | G | — |

**Validation** (7 sites, ~14,200 chips, 18%):

| Site | Chips | Region | Difficulty | Quality | Key Conditions |
|------|-------|--------|------------|---------|----------------|
| Superstition | 4,500 | Central | H/L | E/G/M | Sparse, fog, algae |
| Triquet Bay | 3,726 | Central | H/M | E/G/M | Difficult edge |
| Kendrick Point | 2,322 | North | M/L | E | Bleaching |
| Section Cove | 1,603 | North | M/L | E/G | Hazy lighting |
| Triquet | 724 | Central | L | E | Clear baseline |
| Louscoone | 673 | North | M/L | E/G | Overcast |
| Sedgwick | 634 | North | H/M | G/M | Overcast, cloud reflections |

**Training** (16 sites, ~64,100 chips, 82%):

| Site | Chips | Region | Difficulty | Quality | Key Conditions |
|------|-------|--------|------------|---------|----------------|
| Koeye | 17,076 | Central | H/M/L | E/G/M | Tannins, turbidity, glint |
| Goose SW | 10,642 | Central | H/M/L | E/G | Turbidity, overcast |
| Pruth Bay | 9,509 | Central | M/L | E/G/M | Shadows, cloud reflections |
| Choked Pass | 4,832 | Central | L | G | Good all round |
| Louscoone Head | 4,522 | North | M/L | E/G | Shadows |
| McMullin North | 4,437 | Central | L | E/G | Clear |
| Island Bay | 2,614 | North | L | G | — |
| Ramsay | 2,125 | North | L | G | — |
| Bag Harbour | 1,852 | North | L | E | Excellent baseline |
| Swan Bay | 1,316 | North | L | E | Great conditions |
| Beljay Bay | 1,116 | North | L | E | — |
| Takelly Cove | 1,097 | North | — | G | — |
| Balcolm Inlet | 1,010 | North | M | G | Cloudy |
| Louscoone West | 867 | North | L | E | — |
| Kendrick Point West | 691 | North | L | E | — |
| Heater Harbour | 461 | North | M | G | Dark, low light |

### Regional CV Summary

| Fold | Test Region | Test Sites | Test Chips | Val Sites | Val Chips | Train Sites | Train Chips |
|------|-------------|------------|------------|-----------|-----------|-------------|-------------|
| CV_NORTH | North | 15 | 22,903 | 6 | ~11,600 | 8 | ~55,900 |
| CV_CENTRAL | Central | 8 | 55,446 | 6 | ~5,600 | 15 | ~29,400 |
| CV_SOUTH | South | 6 | 12,099 | 7 | ~14,200 | 16 | ~64,100 |

**Notes:**
- CV_CENTRAL has largest test set (Central has most chips due to multi-visit sites like Koeye)
- Each val set includes mix of H/M/L difficulty and baseline sites
- Train/val ratios within non-test data: ~82-84% train, 16-18% val