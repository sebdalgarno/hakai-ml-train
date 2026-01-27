"""Create chip datasets with optional prototype sampling.

Creates chips from GeoTIFF mosaics with separate stride options for train vs eval splits,
and optionally creates a prototype dataset with proportional per-site sampling.
"""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm.auto import tqdm


def remap_label(labels, band_remapping, device=None):
    if device is None:
        device = labels.device

    lookup = torch.tensor(band_remapping, dtype=labels.dtype, device=device)
    mask = (labels >= 0) & (labels < len(lookup))
    result = torch.full_like(labels, -100)
    result[mask] = lookup[labels[mask]]

    return result


class RasterMosaicDataset(RasterDataset):
    is_image = True
    separate_files = False

    def __init__(self, *args, img_name, **kwargs):
        self.filename_glob = img_name
        super().__init__(*args, **kwargs)


class KomLabelsDataset(RasterDataset):
    is_image = False

    def __init__(self, *args, img_name, **kwargs):
        self.filename_glob = img_name
        super().__init__(*args, **kwargs)


def load_dataset(data_dir, img_name):
    images = RasterMosaicDataset(paths=[str(data_dir / "images")], img_name=img_name)
    labels = KomLabelsDataset(paths=[str(data_dir / "labels")], img_name=img_name)
    return images & labels


def parse_site_name(filename: str) -> str:
    """Extract site name from filename formatted as 'sitename_year.tif'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def create_chips_with_sampling(
    out_main: Path,
    out_prototype: Path | None,
    split: str,
    dset,
    img_path: Path,
    chip_size: int = 512,
    chip_stride: int = 512,
    num_bands: int = 3,
    band_remapping: tuple[int, ...] = (0, 1),
    dtype=np.uint8,
    prototype_fraction: float = 0.10,
    seed: int = 42,
):
    """Create chips and optionally sample for prototype dataset."""
    main_dir = out_main / split
    main_dir.mkdir(exist_ok=True, parents=True)

    if out_prototype is not None:
        proto_dir = out_prototype / split
        proto_dir.mkdir(exist_ok=True, parents=True)
    else:
        proto_dir = None

    # Per-site seed for reproducible sampling
    site_name = parse_site_name(img_path.name)
    site_seed = seed + hash(site_name) % (2**31)
    rng = random.Random(site_seed)

    sampler = GridGeoSampler(dset, size=chip_size, stride=chip_stride)
    dataloader = DataLoader(
        dset,
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=stack_samples,
    )

    main_count = 0
    proto_count = 0

    for i, batch in enumerate(tqdm(dataloader, desc=img_path.stem, leave=False)):
        img = batch["image"]
        label = batch["mask"]

        height = img.shape[2]
        width = img.shape[3]

        if height < chip_size or width < chip_size:
            continue

        img_array = img[0, :num_bands].numpy()

        assert (
            img_array.max() <= np.iinfo(dtype).max
            and img_array.min() >= np.iinfo(dtype).min
        ), f"Image values should be in [{np.iinfo(dtype).min}, {np.iinfo(dtype).max}]"
        img_array = img_array.astype(dtype)
        label_array = remap_label(label, band_remapping).numpy().astype(np.int64)[0]

        # Skip all black images
        if np.all(img_array == 0):
            continue

        # Set black areas in image to 0 in label
        is_nodata = np.all(img_array == 0, axis=0)
        if np.any(is_nodata):
            label_array[is_nodata] = 0

        chip_data = {
            "image": np.moveaxis(img_array, 0, -1),
            "label": label_array,
        }

        # Always save to main dataset
        np.savez_compressed(main_dir / f"{img_path.stem}_{i}.npz", **chip_data)
        main_count += 1

        # Probabilistically save to prototype dataset
        if proto_dir is not None and rng.random() < prototype_fraction:
            np.savez_compressed(proto_dir / f"{img_path.stem}_{i}.npz", **chip_data)
            proto_count += 1

    return main_count, proto_count


def process_split(
    data_dir: Path,
    split: str,
    output_dir: Path,
    prototype_output: Path | None,
    chip_size: int = 512,
    chip_stride: int = 512,
    num_bands: int = 3,
    band_remapping: tuple[int, ...] = (0, 1),
    dtype=np.uint8,
    prototype_fraction: float = 0.10,
    seed: int = 42,
):
    split_dir = data_dir / split
    imgs = sorted(split_dir.glob("images/*.tif"))

    if not imgs:
        print(f"No TIF files found in {split_dir / 'images'}, skipping {split} split")
        return 0, 0

    print(f"\n{split.upper()} split: {len(imgs)} TIF files")
    for i, x in enumerate(imgs):
        print(f"  {i}: {x.name}")

    total_main = 0
    total_proto = 0

    for img in tqdm(imgs, desc=f"{split.capitalize()} TIFs"):
        ds = load_dataset(split_dir, img_name=img.name)
        main_count, proto_count = create_chips_with_sampling(
            out_main=output_dir,
            out_prototype=prototype_output,
            split=split,
            dset=ds,
            img_path=img,
            chip_size=chip_size,
            chip_stride=chip_stride,
            num_bands=num_bands,
            band_remapping=band_remapping,
            dtype=dtype,
            prototype_fraction=prototype_fraction,
            seed=seed,
        )
        total_main += main_count
        total_proto += proto_count

    return total_main, total_proto


def main():
    parser = argparse.ArgumentParser(
        description="Create chip datasets with optional prototype sampling."
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing train/val/test subdirectories with images/ and labels/.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the main chip dataset.",
    )
    parser.add_argument(
        "--prototype-output",
        type=Path,
        default=None,
        help="Output directory for the prototype chip dataset (optional).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of the square chips (default: 512).",
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=256,
        help="Stride for train split chips (default: 256 for 50%% overlap).",
    )
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=512,
        help="Stride for val/test split chips (default: 512 for no overlap).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=3,
        help="Number of bands to keep from images (default: 3).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        help="Data type for image chips (default: uint8).",
    )
    parser.add_argument(
        "--remap",
        "-r",
        type=int,
        nargs="+",
        default=[0, -100, 1],
        help="Label remapping: index=old value, value=new value (default: 0 -100 1).",
    )
    parser.add_argument(
        "--prototype-fraction",
        type=float,
        default=0.10,
        help="Fraction of chips to include in prototype dataset (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process train/val/test splits in parallel.",
    )
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)

    print(f"Creating chip datasets")
    print(f"  Main output: {args.output_dir}")
    if args.prototype_output:
        print(f"  Prototype output: {args.prototype_output}")
        print(f"  Prototype fraction: {args.prototype_fraction}")
    print(f"  Chip size: {args.size}")
    print(
        f"  Train stride: {args.train_stride} ({100 * (1 - args.train_stride / args.size):.0f}% overlap)"
    )
    print(
        f"  Eval stride: {args.eval_stride} ({100 * (1 - args.eval_stride / args.size):.0f}% overlap)"
    )
    print(f"\nLabel remapping:")
    for i, v in enumerate(args.remap):
        print(f"  {i} -> {v}")
    print("  All other values -> -100")

    summary = {"main": {}, "prototype": {}}

    splits_config = [
        ("train", args.train_stride),
        ("val", args.eval_stride),
        ("test", args.eval_stride),
    ]

    # Common kwargs for process_split
    common_kwargs = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "prototype_output": args.prototype_output,
        "chip_size": args.size,
        "num_bands": args.num_bands,
        "band_remapping": tuple(args.remap),
        "dtype": dtype,
        "prototype_fraction": args.prototype_fraction,
        "seed": args.seed,
    }

    if args.parallel:
        print("\nProcessing splits in parallel...")
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    process_split, split=split, chip_stride=stride, **common_kwargs
                ): split
                for split, stride in splits_config
            }
            for future in futures:
                split_name = futures[future]
                main_count, proto_count = future.result()
                summary["main"][split_name] = main_count
                summary["prototype"][split_name] = proto_count
    else:
        for split, stride in splits_config:
            main_count, proto_count = process_split(
                split=split, chip_stride=stride, **common_kwargs
            )
            summary["main"][split] = main_count
            summary["prototype"][split] = proto_count

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("\nMain dataset:")
    for split, count in summary["main"].items():
        print(f"  {split}: {count} chips")
    print(f"  total: {sum(summary['main'].values())} chips")

    if args.prototype_output:
        print("\nPrototype dataset:")
        for split, count in summary["prototype"].items():
            print(f"  {split}: {count} chips")
        print(f"  total: {sum(summary['prototype'].values())} chips")

        main_total = sum(summary["main"].values())
        proto_total = sum(summary["prototype"].values())
        if main_total > 0:
            print(f"\nPrototype is {100 * proto_total / main_total:.1f}% of main")


if __name__ == "__main__":
    main()
