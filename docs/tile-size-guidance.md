# Tile/Chip Size Selection for Semantic Segmentation

Guidance for selecting appropriate tile (chip) sizes when preparing high-resolution drone/UAV imagery for deep learning semantic segmentation models.

## Key Principle

> [!important]
> Think in **ground units**, not pixels. The physical area covered by each chip matters more than the pixel dimensions.

## Ground Coverage by Chip Size and GSD

| Chip Size | 2 cm GSD | 3 cm GSD | 5 cm GSD |
|-----------|----------|----------|----------|
| 224 px | 4.5 m | 6.7 m | 11.2 m |
| 256 px | 5.1 m | 7.7 m | 12.8 m |
| 512 px | 10.2 m | 15.4 m | 25.6 m |
| 1024 px | 20.5 m | 30.7 m | 51.2 m |

## Common Chip Sizes in Literature

### 256 x 256 pixels

- **Use case**: Memory-constrained environments, rapid prototyping
- **Pros**: Fast training, fits on smaller GPUs, more gradient updates per epoch
- **Cons**: Limited spatial context, may fragment larger objects
- **Examples**: CCF BDCI 2020 dataset, Potsdam dataset experiments

### 512 x 512 pixels

- **Use case**: General-purpose segmentation of UAV imagery
- **Pros**: Balanced context and detail, widely validated in literature
- **Cons**: May still miss very large features (rivers, large vegetation patches)
- **Examples**: Vaihingen dataset, most UAV semantic segmentation benchmarks
- **Notes**: Often used with 50-75% overlap (stride of 128-256 pixels)

### 1024 x 1024 pixels

- **Use case**: Large contiguous features, species-level discrimination
- **Pros**: Maximum spatial context, captures full object extents
- **Cons**: High GPU memory requirements, fewer chips per mosaic
- **Examples**: Kelp species detection, large-scale land cover mapping

## Factors Influencing Chip Size Selection

### Target Object Size

The chip should be large enough to contain:

- The full extent of target objects
- Sufficient surrounding context for discrimination
- Multiple examples of smaller objects per chip

> [!tip]
> If your targets typically span 5-10 m, aim for chips covering at least 20-30 m to provide context.

### GPU Memory Constraints

Approximate VRAM requirements (batch size 4, mixed precision):

| Chip Size | Approximate VRAM |
|-----------|------------------|
| 256 px | 4-6 GB |
| 512 px | 8-12 GB |
| 1024 px | 16-24 GB |

> [!note]
> Use gradient accumulation to achieve larger effective batch sizes with smaller chips if memory is limited.

### Model Architecture

- **CNNs (U-Net, DeepLabV3+)**: Fixed receptive field; larger chips provide more context
- **Transformers (SegFormer, Swin)**: Global attention can leverage larger chips effectively
- **Hybrid models**: Generally benefit from larger chips but are more memory-intensive

### Training Efficiency

- Smaller chips = more chips per mosaic = more gradient updates per epoch
- Larger chips = fewer chips = potentially faster convergence but slower epochs

## Overlap and Stride

### No Overlap (stride = size)

- Each pixel appears in exactly one chip
- Most memory-efficient
- Risk: Objects at chip boundaries may be poorly segmented

### 50% Overlap (stride = size / 2)

- Each pixel appears in up to 4 chips
- Better boundary handling
- 4x increase in dataset size
- Common choice for production models

### 75% Overlap (stride = size / 4)

- Maximum redundancy
- Best for small or sparse objects
- 16x increase in dataset size
- May lead to overfitting on small datasets

## Recommendations by Application

### Marine Vegetation (Kelp, Seagrass)

| GSD | Recommended Size | Ground Coverage | Rationale |
|--------|------------------|-----------------|-----------|
| 2-3 cm | 1024 px | 20-30 m | Captures canopy structure for species ID |
| 4-5 cm | 512 px | 20-25 m | Balances context with memory |

### Urban and Infrastructure

| GSD | Recommended Size | Ground Coverage | Rationale |
|---------|------------------|-----------------|------------------------------|
| 2-5 cm | 512 px | 10-25 m | Buildings and roads fit well |
| 5-10 cm | 256 px | 12-25 m | Sufficient for most features |

### Agricultural and Vegetation Plots

| GSD | Recommended Size | Ground Coverage | Rationale |
|--------|------------------|-----------------|---------------------------|
| 1-3 cm | 512 px | 5-15 m | Individual plants visible |
| 3-5 cm | 1024 px | 30-50 m | Field-scale patterns |

## Experimental Approach

If uncertain, run a small experiment:

1. Prepare chips at 256, 512, and 1024 pixels
2. Train identical models on each
3. Compare validation IoU and visual results
4. Check boundary artifacts at each scale

## References

- [Semantic segmentation of seagrass habitat from drone imagery based on deep learning](https://www.sciencedirect.com/science/article/abs/pii/S1574954121002211) - Compared tile sizes and normalization for marine vegetation
- [Scale-Adaptive Semantic Segmentation of High-resolution Remote Sensing Imagery](https://arxiv.org/html/2309.15372) - Discusses 512x512 as standard with overlap strategies
- [UAVid: A semantic segmentation dataset for UAV imagery](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301295) - High-resolution UAV benchmark (3840x2160 images)
- [AqUavplant Dataset](https://www.nature.com/articles/s41597-024-04155-6) - Aquatic plant segmentation from 4K UAV imagery
- [Methods and datasets on semantic segmentation for UAV remote sensing images: A review](https://www.sciencedirect.com/science/article/abs/pii/S0924271624000844) - Comprehensive review of UAV segmentation methods
