# Dataset Split Guidance

Recommendations for tile counts and train/val/test splits for semantic segmentation model development, informed by remote sensing and UAV imagery literature.

## Split Ratio

**70 / 15 / 15** (train / val / test)

This ratio is well-established for aerial imagery segmentation (DeepGlobe, ISPRS, SpaceNet benchmarks). For imbalanced classes like kelp/mussels, adequate validation/test samples ensure reliable metric estimation.

## Benchmark Dataset Sizes (Literature)

| Dataset | Total Images | Train | Val | Test | Resolution |
|---------|-------------|-------|-----|------|------------|
| Semantic Drone | 400 | 320 | 80 | - | 6000×4000 |
| UAVid | 420 | 270 | 150 | - | 3840×2160 |
| AeroScapes | 3,269 | ~2,600 | ~650 | - | 1280×720 |
| VDD | 400 | ~280 | ~80 | ~40 | 4000×3000 |
| IDD (drone) | 811 | 536 | 185 | 40 | varied |
| Flip-n-Slide study | 12,800 tiles | - | - | - | 256×256 |

**Key insight**: UAV/drone segmentation benchmarks typically use 400-3,000 high-resolution images, which become 3,000-15,000 tiles after cropping.

## Tile Counts by Development Phase

Counts are **total tiles after sliding window extraction**.

### 512×512 Tiles

| Phase | Train | Val | Test | Use Case |
|-------|-------|-----|------|----------|
| Architecture search | 8,000-15,000 | 1,500-2,500 | 1,500-2,500 | Compare 5-10 architectures |
| Hyperparameter tuning | 30,000-50,000 | 5,000-8,000 | 5,000-8,000 | Fine-tune top 2-3 models |
| Final training | Full dataset | Full dataset | Full dataset | Production model |

### 1024×1024 Tiles

| Phase | Train | Val | Test | Use Case |
|-------|-------|-----|------|----------|
| Architecture search | 4,000-8,000 | 800-1,500 | 800-1,500 | Compare architectures |
| Hyperparameter tuning | 15,000-25,000 | 2,500-4,000 | 2,500-4,000 | Fine-tune top models |
| Final training | Full dataset | Full dataset | Full dataset | Production model |

## Overlap Strategy

### Training Data: 50% Stride

| Tile Size | Stride | Overlap |
|-----------|--------|---------|
| 512×512 | 256 | 50% |
| 1024×1024 | 512 | 50% |

### Validation/Test Data: No Overlap

Use stride = tile size for unbiased evaluation.

### Caution: Overlapping Tiles Are Not Independent Samples

Literature warns that high overlap can introduce **data redundancy** leading to:
- Biased models exposed to many similar data points
- Inflated apparent dataset size without true diversity
- Potential overfitting to repeated spatial patterns

From [Remote Sensing tile overlap study](https://www.mdpi.com/2072-4292/16/15/2818): "A 12.5% level of overlap ensures information near tile edges can be correctly processed during training and avoids lower data variability that would be introduced by generating tiles with a higher overlap."

### Alternative: Flip-n-Slide Strategy

The [Flip-n-Slide method](https://arxiv.org/html/2404.10927v1) uses varied overlaps (0%, 25%, 50%, 75%) with distinct geometric transformations per overlap level, avoiding redundancy while maintaining context. This improved precision by up to 15.8% for underrepresented classes.

## Mosaic-Level Splitting

**Critical**: Split at the mosaic level, not tile level.

```
66 mosaics total:
├── Train: 46 mosaics (~70%)
├── Val: 10 mosaics (~15%)
└── Test: 10 mosaics (~15%)
```

### Why This Matters

- Tiles from the same mosaic share lighting, water conditions, sensor characteristics
- Random tile-level splitting causes spatial autocorrelation leakage
- Mosaic-level splits test true generalization to unseen locations/dates

From [PMC systematic evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7020775/): CNNs exhibit translational variance where "even the smallest translation in the input can create a difference in the output" - making correlated tiles from the same scene problematic for training/test separation.

## Condition Coverage

When assigning mosaics to splits, ensure each split represents:

- [ ] Geographic locations (different kelp/mussel populations)
- [ ] Seasonal variation (spring/summer/fall)
- [ ] Lighting conditions (sun angle, cloud cover)
- [ ] Water conditions (turbidity, glare, calm vs choppy)
- [ ] Target density (sparse, moderate, dense canopy/beds)
- [ ] Edge cases (mixed species boundaries, shadows, foam)

## Tile Size Considerations

Larger tiles improve performance. From [PMC study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7020775/):
- 128×128 tiles: Dice 0.791
- 640×640 whole image: Dice 0.917

"Larger tile sizes yield more consistent results and mitigate undesirable and unpredictable behavior during inference."

For kelp/mussel detection, 1024×1024 captures sufficient canopy/bed structure while fitting in 20GB VRAM.

## GPU Memory Reference (20GB VRAM)

| Tile Size | Batch Size | Effective Batch (accum=8) | Approx Memory |
|-----------|------------|---------------------------|---------------|
| 512×512 | 8-12 | 64-96 | 12-16 GB |
| 1024×1024 | 4-6 | 32-48 | 14-18 GB |

## References

- [Flip-n-Slide: Concise Tiling Strategy for Earth Observation](https://arxiv.org/html/2404.10927v1) - ICLR 2024 ML4RS Workshop
- [Tile Size and Overlap Effects on Road Segmentation](https://www.mdpi.com/2072-4292/16/15/2818) - Remote Sensing 2024
- [Systematic Evaluation of Image Tiling Effects](https://pmc.ncbi.nlm.nih.gov/articles/PMC7020775/) - PMC 2020
- [UAVid Dataset](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301295) - ISPRS 2020
- [VDD: Varied Drone Dataset](https://arxiv.org/abs/2305.13608) - 2023
- [Semantic Drone Dataset](https://datasetninja.com/semantic-drone)
