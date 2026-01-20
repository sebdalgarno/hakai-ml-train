# Seagrass Post-Processing Repo Plan

Plan for a lightweight inference/post-processing repository that consumes ONNX models trained in hakai-ml-train.

## Purpose

Run trained seagrass segmentation models on GeoTIFF orthomosaics and output prediction masks for visualization and polygonization in QGIS.

## Repo Structure

```
seagrass-postprocess/
├── pyproject.toml
├── src/
│   └── seagrass_postprocess/
│       ├── __init__.py
│       ├── predict.py      # Core inference logic
│       ├── polygonize.py   # Raster to vector conversion
│       └── cli.py          # Command-line interface
└── scripts/
    └── run_prediction.sh
```

## Dependencies

Minimal, no PyTorch required:

```toml
[project]
dependencies = [
    "onnxruntime",      # Model inference
    "rasterio",         # GeoTIFF I/O
    "numpy",
    "tqdm",
]

[project.optional-dependencies]
vector = [
    "geopandas",        # Polygonization
    "shapely",
]
```

## Core Functionality

### 1. GeoTIFF Prediction (`predict.py`)

```python
def predict_geotiff(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    tile_size: int = 224,
    batch_size: int = 16,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> None:
    """
    Run inference on GeoTIFF, output prediction mask.

    Steps:
    1. Open input with rasterio, get profile (CRS, transform, etc.)
    2. Pad image to multiple of tile_size
    3. Tile into (tile_size, tile_size) chunks
    4. Batch tiles, normalize, run ONNX inference
    5. Argmax predictions, stitch back to full raster
    6. Write output GeoTIFF with same CRS/transform
    """
```

**Key implementation details:**

- Use `rasterio.windows.Window` for memory-efficient tiling on large images
- Normalize: `(pixel / 255.0 - mean) / std`
- ONNX inference: `ort.InferenceSession(model_path).run(None, {"input": batch})`
- Output: single-band uint8 GeoTIFF (0=background, 1=seagrass)

### 2. Polygonization (`polygonize.py`)

```python
def polygonize_mask(
    mask_path: Path,
    output_path: Path,
    min_area: float = 1.0,  # sq meters
    simplify_tolerance: float = 0.5,
) -> None:
    """
    Convert prediction mask to vector polygons.

    Steps:
    1. Read mask with rasterio
    2. Use rasterio.features.shapes() to vectorize
    3. Filter by class (seagrass = 1)
    4. Convert to GeoDataFrame
    5. Filter small polygons, simplify geometry
    6. Write to GeoPackage/Shapefile
    """
```

## CLI Interface

```bash
# Predict on single GeoTIFF
seagrass-postprocess predict input.tif output_mask.tif \
    --model model.onnx

# Predict on directory of GeoTIFFs
seagrass-postprocess predict-batch ./inputs/ ./outputs/ \
    --model model.onnx

# Polygonize prediction mask
seagrass-postprocess polygonize mask.tif seagrass_polygons.gpkg \
    --min-area 1.0 \
    --simplify 0.5

# Full pipeline: predict + polygonize
seagrass-postprocess pipeline input.tif output_polygons.gpkg \
    --model model.onnx
```

## Implementation Notes

### Tiling Strategy

Simple non-overlapping tiles for development:

```python
for row in range(0, height, tile_size):
    for col in range(0, width, tile_size):
        window = Window(col, row, tile_size, tile_size)
        tile = src.read(window=window)
        # ... predict ...
```

If edge artifacts become problematic later, can add overlap + blending:
- 50% overlap between tiles
- Hann window weighting (center weighted more than edges)
- habitat-mapper has reference implementation in `hann.py`

### Handling Nodata

```python
# Detect nodata regions in input
nodata_mask = (tile == src.nodata).any(axis=0)

# Set prediction to nodata where input was nodata
prediction[nodata_mask] = output_nodata_value
```

### Memory Management

For large orthomosaics that don't fit in memory:
- Process tiles in batches
- Write output incrementally using `rasterio` windowed writing
- Use `np.memmap` if needed for very large images

## Integration with hakai-ml-train

### Export workflow

```bash
# In hakai-ml-train: export trained model to ONNX
python -m src.deploy.kom_onnx \
    configs/seagrass-rgb/unetpp_resnet34.yaml \
    checkpoints/best.ckpt \
    seagrass_rgb.onnx

# Copy ONNX to post-processing repo or reference directly
```

### Normalization parameters

Must match training config:
- mean: `[0.485, 0.456, 0.406]` (ImageNet)
- std: `[0.229, 0.224, 0.225]` (ImageNet)
- max_pixel_value: `255.0`

These could be embedded in model metadata or passed via CLI.

## Future Extensions (not in initial scope)

- Confidence/probability output (not just argmax)
- Batch processing with progress tracking
- Integration with STAC/cloud-optimized GeoTIFFs
- Area statistics and reporting
- Overlap + Hann windowing for cleaner edges
- GPU inference with onnxruntime-gpu

## Quick Start Template

Minimal working example to copy when creating the repo:

```python
# src/seagrass_postprocess/predict.py
import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

def predict_geotiff(input_path, output_path, model_path, tile_size=224):
    session = ort.InferenceSession(model_path)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype='uint8')

        height, width = src.height, src.width
        prediction = np.zeros((height, width), dtype=np.uint8)

        for row in tqdm(range(0, height, tile_size)):
            for col in range(0, width, tile_size):
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)

                window = Window(col, row, w, h)
                tile = src.read(window=window)  # (C, H, W)

                # Pad if needed
                if h < tile_size or w < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=tile.dtype)
                    padded[:, :h, :w] = tile
                    tile = padded

                # Normalize
                tile = (tile.astype(np.float32) / 255.0 - mean) / std
                tile = tile[np.newaxis, ...]  # Add batch dim

                # Predict
                logits = session.run(None, {"input": tile})[0]
                pred = np.argmax(logits, axis=1)[0]  # (H, W)

                # Store (crop if padded)
                prediction[row:row+h, col:col+w] = pred[:h, :w]

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)
```
