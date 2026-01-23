# Seagrass Dataset Splits

Train/val/test splits for Stage 1 (Architecture Selection) experiments. Follows guidance from `dataset-split-guidance.md` and `model-development-workflow.md`.

## Principles

- **70/15/15 split** (train/val/test)
- **Site-level splitting**: All visits from a site stay in the same bucket
- **Regional stratification**: All 3 regions (South, Central, North) represented in all buckets
- **Condition coverage**: Each split represents diverse lighting, water conditions, target density, and edge cases

---

## Split A: Full Dataset (35 sites)

Use this split if all 35 sites are available.

### Sites by Region

| Region | Sites |
|--------|-------|
| South | 10 |
| Central | 9 |
| North | 16 |

### TEST (5 sites, 14%)

| Site | Region | Difficulty | Quality | Key Conditions |
|------|--------|------------|---------|----------------|
| Grice Bay | South | H | G | Large sparse area, coarse delineation |
| Superstition | Central | H/M/L | E/G/M | Sparse eelgrass, fog, algae, subtidal |
| Triquet | Central | L | E | Clear baseline |
| Sedgwick | North | H/M | G/M | Overcast, cloud reflections |
| Bag Harbour | North | L | E | Excellent baseline |

### VALIDATION (5 sites, 14%)

| Site | Region | Difficulty | Quality | Key Conditions |
|------|--------|------------|---------|----------------|
| Bennett Bay | South | H | M | Cloudy, glint |
| Auseth | South | L | E | Excellent baseline |
| Pruth Bay | Central | M/L | E/G/M | Shadows, cloud reflections, subtidal |
| Kendrick Point | North | M/L | E | Bleaching |
| Louscoone | North | M/L | E/G | Overcast, underwater |

### TRAINING (25 sites, 71%)

**South (7 sites):**

| Site | Difficulty | Quality | Key Conditions |
|------|------------|---------|----------------|
| Arakun | L | G | — |
| Beck | H | M | Shadows, sparse eelgrass |
| Calmus | L | E | — |
| Jaques Jarvis | L | G | — |
| Nettle | L | G | Clear subtidal |
| Sharp | H | G | Overcast, low density intertidal |
| Turret | L | G | Clear |

**Central (6 sites):**

| Site | Difficulty | Quality | Key Conditions |
|------|------------|---------|----------------|
| Choked Pass | L | G/M | Towed video validation |
| Goose Grass Bay | L | G | — |
| Goose SW | H/M/L | E/G | Low density, turbidity, overcast |
| Koeye | H/M/L | E/G/M | Tannins, turbidity, glint (9 visits) |
| McMullin North | L | E/G | Clear |
| Triquet Bay | H/M | E/G/M | Difficult edge, other species |

**North (12 sites):**

| Site | Difficulty | Quality | Key Conditions |
|------|------------|---------|----------------|
| Balcolm Inlet | M | G | Cloudy |
| Beljay Bay | L | E | — |
| Heater Harbour | M | G | Dark, low light |
| Island Bay | L | G | — |
| Kendrick Point West | L | E | — |
| Louscoone Head | M/L | E/G | Shadows |
| Louscoone West | L | E | — |
| Powrivco | L | G | — |
| Ramsay | L | G | — |
| Section Cove | M/L | E/G | Hazy lighting |
| Swan Bay | L | E | Great conditions |
| Takelly Cove | — | G | — |

### Summary (Split A)

| Bucket | Sites | South | Central | North | High | Med | Low |
|--------|-------|-------|---------|-------|------|-----|-----|
| Train | 25 | 7 | 6 | 12 | 5* | 7* | 20+ |
| Val | 5 | 2 | 1 | 2 | 1 | 3* | 2 |
| Test | 5 | 1 | 2 | 2 | 2 | 1* | 3* |

*includes sites with multiple visits spanning difficulty levels

---

## Split B: Reduced Dataset (29 sites)

Use this split if 9 sites are unavailable (Goose Grass Bay, Jaques Jarvis, Nettle, Powrivco, Sharp, Turret removed from South/Central/North).

### Sites by Region

| Region | Sites |
|--------|-------|
| South | 6 |
| Central | 8 |
| North | 15 |

### TEST (4 sites, 14%)

| Site | Region | Difficulty | Quality | Key Conditions |
|------|--------|------------|---------|----------------|
| Grice Bay | South | H | G | Large sparse, coarse delineation |
| Superstition | Central | H/L | E/G/M | Sparse, fog, algae (4 visits) |
| Sedgwick | North | H/M | G/M | Overcast, cloud reflections |
| Bag Harbour | North | L | E | Excellent baseline |

### VALIDATION (5 sites, 17%)

| Site | Region | Difficulty | Quality | Key Conditions |
|------|--------|------------|---------|----------------|
| Bennett Bay | South | H | M | Cloudy, glint |
| Pruth Bay | Central | M/L | E/G/M | Shadows, cloud reflections, subtidal (8 visits) |
| Triquet | Central | L | E | Clear baseline |
| Kendrick Point | North | M/L | E | Bleaching |
| Louscoone | North | M/L | E/G | Overcast |

### TRAINING (20 sites, 69%)

**South (4 sites):**

| Site | Difficulty | Quality | Key Conditions |
|------|------------|---------|----------------|
| Arakun | L | G | — |
| Auseth | L | E | Excellent baseline |
| Beck | H | M | Shadows, sparse eelgrass |
| Calmus | L | E | — |

**Central (5 sites):**

| Site | Difficulty | Quality | Key Conditions |
|------|------------|---------|----------------|
| Choked Pass | L | G | Good all round |
| Goose SW | H/M/L | E/G | Turbidity, overcast, low density (3 visits) |
| Koeye | H/M/L | E/G/M | Tannins, turbidity, glint (9 visits) |
| McMullin North | L | E/G | Clear |
| Triquet Bay | H/M | E/G/M | Difficult edge, other species |

**North (11 sites):**

| Site | Difficulty | Quality | Key Conditions |
|------|------------|---------|----------------|
| Balcolm Inlet | M | G | Cloudy |
| Beljay Bay | L | E | — |
| Heater Harbour | M | G | Dark, low light |
| Island Bay | L | G | — |
| Kendrick Point West | L | E | — |
| Louscoone Head | M/L | E/G | Shadows |
| Louscoone West | L | E | — |
| Ramsay | L | G | — |
| Section Cove | M/L | E/G | Hazy lighting |
| Swan Bay | L | E | Great |
| Takelly Cove | — | G | — |

### Summary (Split B)

| Bucket | Sites | South | Central | North | High | Med | Low |
|--------|-------|-------|---------|-------|------|-----|-----|
| Train | 20 | 4 | 5 | 11 | 4* | 6* | 15+ |
| Val | 5 | 1 | 2 | 2 | 1 | 4* | 3* |
| Test | 4 | 1 | 1 | 2 | 3* | 1* | 2* |

*includes sites with multiple visits spanning difficulty levels

---

## Condition Coverage

Coverage across buckets for both splits.

| Condition | Train | Val | Test |
|-----------|-------|-----|------|
| Shadows | Beck, Louscoone Head | Pruth Bay | — |
| Sparse eelgrass | Beck | — | Grice Bay, Superstition |
| Cloud reflections | — | Pruth Bay | Sedgwick |
| Cloudy/overcast | Balcolm Inlet, Goose SW | Louscoone | Sedgwick |
| Glint | Koeye | Bennett Bay | — |
| Tannins | Koeye | — | — |
| Turbidity | Goose SW, Koeye | — | — |
| Dark/low light | Heater Harbour | — | — |
| Bleaching | — | Kendrick Point | — |
| Difficult edge | Triquet Bay | — | — |
| Fog/algae | — | — | Superstition |
| Clear baseline | Auseth, Calmus, McMullin North | Triquet | Bag Harbour |

Note: Split A includes additional condition coverage in training from Sharp (overcast, low density), Nettle (clear subtidal), and other removed sites.

---

## Key Decisions

1. **Koeye (9 visits) in training**: Maximizes diversity for learning tannins, turbidity, and glint conditions. Largest multi-visit site provides temporal variation.

2. **Pruth Bay (8 visits) in validation**: Provides robust hyperparameter tuning signal with diverse conditions (shadows, cloud reflections, subtidal).

3. **High-difficulty site distribution**:
   - Train: Beck, Goose SW, Koeye, Triquet Bay (+ Sharp in Split A)
   - Val: Bennett Bay
   - Test: Grice Bay, Superstition, Sedgwick

4. **Baseline sites in each bucket**: Clear, excellent-quality sites for metric calibration:
   - Train: Auseth, Calmus, McMullin North
   - Val: Triquet (+ Auseth in Split A)
   - Test: Bag Harbour

5. **Regional balance maintained**: Each bucket has representation from all 3 regions, roughly proportional to the overall distribution (North has most sites).

6. **Unique conditions isolated to test**: Fog (Superstition) and large-scale sparse mapping (Grice Bay) appear only in test to evaluate generalization to these edge cases.
