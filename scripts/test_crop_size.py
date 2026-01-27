# uv run python scripts/test_crop_size.py /path/to/chips_1024 --crop_size 512 --visualize

"""Test that crop_size parameter produces expected tensor shapes."""

import argparse

import albumentations as A
import matplotlib.pyplot as plt

from src.data import DataModule


def main():
    parser = argparse.ArgumentParser(description="Test crop_size parameter in DataModule")
    parser.add_argument("chip_dir", help="Path to chip directory (used for train/val/test)")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop size to test")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--visualize", action="store_true", help="Show sample images")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to visualize")
    args = parser.parse_args()

    # Minimal transforms
    transforms = A.to_dict(
        A.Compose([
            A.Normalize(
                mean=[0.5] * args.num_channels,
                std=[0.5] * args.num_channels,
            ),
            A.ToTensorV2(),
        ])
    )

    dm = DataModule(
        train_chip_dir=args.chip_dir,
        val_chip_dir=args.chip_dir,
        test_chip_dir=args.chip_dir,
        batch_size=args.num_samples,
        num_workers=0,
        crop_size=args.crop_size,
        train_transforms=transforms,
        test_transforms=transforms,
    )
    dm.setup()

    # Check shapes
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    print(f"crop_size: {args.crop_size}")
    print(f"Train image shape: {train_batch[0].shape}")
    print(f"Train label shape: {train_batch[1].shape}")
    print(f"Val image shape: {val_batch[0].shape}")
    print(f"Val label shape: {val_batch[1].shape}")

    expected_size = args.crop_size if args.crop_size else "original"
    print(f"\nExpected spatial size: {expected_size}")

    if args.visualize:
        n = min(args.num_samples, train_batch[0].shape[0])
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            img = train_batch[0][i]
            mask = train_batch[1][i]

            # Undo normalization for display (only show first 3 channels)
            img_display = img[:3].permute(1, 2, 0).numpy() * 0.5 + 0.5
            img_display = img_display.clip(0, 1)

            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f"Image {i} ({img.shape[1]}x{img.shape[2]})")
            axes[0, i].axis("off")

            axes[1, i].imshow(mask.numpy(), cmap="tab10")
            axes[1, i].set_title(f"Label {i}")
            axes[1, i].axis("off")

        plt.suptitle(f"Train samples (crop_size={args.crop_size})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
