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

### TEST (6 sites, 15%)

| Site | Region | Chips | Difficulty | Quality | Key Conditions |
|------------|------------|------------|------------|------------|--------------|
| McMullin North | Central | 2,888 | L | E/G | Clear |
| Triquet Bay | Central | 2,400 | H/M | E/G/M | Difficult edge, other species |
| Bag Harbour | North | 2,028 | L | E | Excellent baseline |
| Section Cove | North | 1,810 | M/L | E/G | Hazy lighting |
| Sedgwick | North | 1,558 | H/M | G/M | Overcast, cloud reflections |
| Beck | South | 1,034 | H | M | Shadows, sparse eelgrass |
| **Total** |  | **11,718** |  |  |  |

### VALIDATION (6 sites, 10%)

| Site | Region | Chips | Difficulty | Quality | Key Conditions |
|------------|------------|------------|------------|------------|--------------|
| Superstition | Central | 4,062 | H/L | E/G/M | Sparse, fog, algae (4 visits) |
| Kendrick Point | North | 2,270 | M/L | E | Bleaching |
| Auseth | South | 1,494 | L | E | Excellent baseline |
| Triquet | Central | 1,152 | L | E | Clear baseline |
| Louscoone | North | 1,100 | M/L | E/G | Overcast |
| Bennett Bay | South | 644 | H | M | Cloudy, glint |
| **Total** |  | **10,722** |  |  |  |

### TRAINING (16 sites, 75%)

**South (3 sites):**

| Site         | Chips      | Difficulty | Quality | Key Conditions                   |
|--------------|------------|------------|---------|----------------------------------|
| Grice Bay    | 7,870      | H          | G       | Large sparse, coarse delineation |
| Arakun       | 2,064      | L          | G       | —                                |
| Calmus       | 1,314      | L          | E       | —                                |
| **Subtotal** | **11,248** |            |         |                                  |

**Central (4 sites):**

| Site | Chips | Difficulty | Quality | Key Conditions |
|--------------|--------------|--------------|--------------|------------------|
| Pruth Bay | 9,509 | M/L | E/G/M | Shadows, cloud reflections, subtidal (8 visits) |
| Koeye | 8,538 | H/M/L | E/G/M | Tannins, turbidity, glint (9 visits) |
| Goose SW | 6,014 | H/M/L | E/G | Turbidity, overcast, low density (3 visits) |
| Choked Pass | 2,520 | L | G | Good all round |
| **Subtotal** | **26,581** |  |  |  |

**North (9 sites):**

| Site                | Chips      | Difficulty | Quality | Key Conditions   |
|---------------------|------------|------------|---------|------------------|
| Balcolm Inlet       | 7,428      | M          | G       | Cloudy           |
| Swan Bay            | 5,184      | L          | E       | Great conditions |
| Island Bay          | 4,536      | L          | G       | —                |
| Louscoone Head      | 3,776      | M/L        | E/G     | Shadows          |
| Heater Harbour      | 2,864      | M          | G       | Dark, low light  |
| Louscoone West      | 2,166      | L          | E       | —                |
| Kendrick Point West | 1,768      | L          | E       | —                |
| Beljay Bay          | 1,377      | L          | E       | —                |
| Ramsay              | 1,080      | L          | G       | —                |
| **Subtotal**        | **30,179** |            |         |                  |

| **TRAIN TOTAL** \| **68,008** \| \| \| \|

### Summary (Split B)

| Bucket | Sites | Chips  | \%  | South | Central | North |
|--------|-------|--------|-----|-------|---------|-------|
| Train  | 16    | 68,008 | 75% | 3     | 4       | 9     |
| Val    | 6     | 10,722 | 12% | 2     | 2       | 2     |
| Test   | 6     | 11,718 | 13% | 1     | 2       | 3     |

Note: Chip counts are after nodata removal (28 sites total; takelly_cove had no chips after filtering). Largest sites (pruth_bay, grice_bay) moved to training to prevent single-site dominance in val/test metrics. Superstition swapped to val to balance difficulty across val/test

------------------------------------------------------------------------

## Condition Coverage

Coverage across buckets (Split B revised).

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