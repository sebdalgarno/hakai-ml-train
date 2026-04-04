"""Compute per-site pixel metrics for test sites and regenerate area estimates."""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics
import yaml
from albumentations import Compose, Normalize, ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import NpzSegmentationDataset
from src.models.smp import SMPMulticlassSegmentationModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/eval-per-site")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config and checkpoint
CONFIG_PATH = "configs/seagrass-rgb/final/segformer_final.yaml"
CKPT_PATH = "/mnt/class_data/sdalgarno/checkpoints/final/last-v1.ckpt"
TEST_CHIP_DIR = Path("/mnt/class_data/sdalgarno/main/chips_1024/test")
METADATA_PATH = Path("metadata_subset.csv")


def extract_site_name(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(.+)_u\d+_\d+$", stem)
    if match:
        return match.group(1)
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        potential_site = match.group(1)
        if not re.search(r"_u\d+$", potential_site):
            return potential_site
    return stem


def extract_ortho_name(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def main():
    print(f"Device: {DEVICE}")

    # Load config for transforms
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    model_args = config["model"]["init_args"]
    data_args = config["data"]["init_args"]
    num_classes = model_args["num_classes"]
    ignore_index = model_args.get("ignore_index", -100)

    # Load model
    print("Loading model...")
    model = SMPMulticlassSegmentationModel.load_from_checkpoint(
        CKPT_PATH, map_location=DEVICE
    )
    model.eval()
    model.to(DEVICE)

    # Test transforms (normalize + tensor only)
    test_transforms = Compose(
        [
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # Load metadata for resolution
    metadata_df = pd.read_csv(METADATA_PATH)
    ortho_to_resolution = dict(
        zip(metadata_df["site"], metadata_df["resolution_cm"], strict=False)
    )
    ortho_to_region = dict(
        zip(metadata_df["site"], metadata_df["region"], strict=False)
    )

    # Group test chips by site
    test_chips = sorted(TEST_CHIP_DIR.glob("*.npz"))
    print(f"Found {len(test_chips)} test chips")

    chips_by_site = defaultdict(list)
    for chip_path in test_chips:
        site = extract_site_name(chip_path.name)
        chips_by_site[site].append(chip_path)

    print(f"Sites: {sorted(chips_by_site.keys())}")

    # --- Task 1: Per-site pixel metrics ---
    print("\n=== Per-site pixel metrics ===")
    site_results = []

    for site in sorted(chips_by_site.keys()):
        chips = chips_by_site[site]

        # Create per-site metrics
        iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ).to(DEVICE)
        precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ).to(DEVICE)
        recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ).to(DEVICE)
        f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ).to(DEVICE)

        with torch.no_grad():
            for chip_path in chips:
                data = np.load(chip_path)
                image = data["image"]
                label = data["label"]

                augmented = test_transforms(image=image, mask=label)
                image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
                label_tensor = torch.from_numpy(label).long().unsqueeze(0).to(DEVICE)

                probs = torch.softmax(model(image_tensor), dim=1)
                iou.update(probs, label_tensor)
                precision.update(probs, label_tensor)
                recall.update(probs, label_tensor)
                f1.update(probs, label_tensor)

        site_results.append(
            {
                "Site": site,
                "Chips": len(chips),
                "IoU": iou.compute()[1].item(),
                "Precision": precision.compute()[1].item(),
                "Recall": recall.compute()[1].item(),
                "F1": f1.compute()[1].item(),
            }
        )
        print(
            f"  {site}: {len(chips)} chips, "
            f"IoU={site_results[-1]['IoU']:.3f}, F1={site_results[-1]['F1']:.3f}"
        )

    site_df = pd.DataFrame(site_results)
    site_df.to_csv(OUTPUT_DIR / "per_site_metrics.csv", index=False)
    print(f"\nSaved per-site metrics to {OUTPUT_DIR / 'per_site_metrics.csv'}")
    print(site_df.to_string(index=False))

    # --- Task 2: Per-ortho area estimation ---
    print("\n\n=== Per-ortho area estimation ===")
    area_results = []

    with torch.no_grad():
        for chip_path in tqdm(test_chips, desc="Processing chips"):
            ortho = extract_ortho_name(chip_path.name)
            resolution_cm = ortho_to_resolution.get(ortho)
            region = ortho_to_region.get(ortho)

            if resolution_cm is None:
                continue

            data = np.load(chip_path)
            image = data["image"]
            label = data["label"]

            augmented = test_transforms(image=image, mask=label)
            image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            gt_pixels = int(np.sum((label == 1) & (label != ignore_index)))
            pred_pixels = int(np.sum((pred == 1) & (label != ignore_index)))

            pixel_area_m2 = (resolution_cm / 100) ** 2

            area_results.append(
                {
                    "chip_filename": chip_path.name,
                    "ortho": ortho,
                    "region": region,
                    "resolution_cm": resolution_cm,
                    "gt_pixels": gt_pixels,
                    "pred_pixels": pred_pixels,
                    "gt_area_m2": gt_pixels * pixel_area_m2,
                    "pred_area_m2": pred_pixels * pixel_area_m2,
                }
            )

    chip_df = pd.DataFrame(area_results)

    # Aggregate by ortho
    ortho_df = (
        chip_df.groupby(["ortho", "region"])
        .agg({"gt_pixels": "sum", "pred_pixels": "sum", "resolution_cm": "first"})
        .reset_index()
    )
    ortho_df["gt_area_m2"] = (
        ortho_df["gt_pixels"] * (ortho_df["resolution_cm"] / 100) ** 2
    )
    ortho_df["pred_area_m2"] = (
        ortho_df["pred_pixels"] * (ortho_df["resolution_cm"] / 100) ** 2
    )
    ortho_df["area_error_m2"] = ortho_df["pred_area_m2"] - ortho_df["gt_area_m2"]
    ortho_df["abs_error_m2"] = ortho_df["area_error_m2"].abs()

    ortho_df.to_csv(OUTPUT_DIR / "ortho_areas.csv", index=False)
    print(f"\nSaved ortho areas to {OUTPUT_DIR / 'ortho_areas.csv'}")
    print(ortho_df.to_string(index=False))

    # Summary
    mae = ortho_df["abs_error_m2"].mean()
    rmse = np.sqrt((ortho_df["area_error_m2"] ** 2).mean())
    bias = ortho_df["area_error_m2"].mean()
    pct_errors = (
        ortho_df["area_error_m2"] / ortho_df["gt_area_m2"].replace(0, np.nan) * 100
    )
    mean_pct = pct_errors.mean()

    gt = ortho_df["gt_area_m2"].values
    pred = ortho_df["pred_area_m2"].values
    ss_res = np.sum((pred - gt) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nSummary:")
    print(f"  MAE:  {mae:,.1f} m²")
    print(f"  RMSE: {rmse:,.1f} m²")
    print(f"  Bias: {bias:+,.1f} m²")
    print(f"  Mean % Error: {mean_pct:+.1f}%")
    print(f"  R²: {r2:.3f}")


if __name__ == "__main__":
    main()
