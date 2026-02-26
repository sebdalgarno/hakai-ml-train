# Adding New Orthomosaics

This document describes the workflow for adding new orthomosaics to the seagrass dataset.

## Overview

1. Place ortho files in the `add/` directory
2. Configure split assignment in `add_new_orthos.sh`
3. Run chip creation and move to main dataset
4. Update and run CV split script
5. Recreate prototype dataset

## Step 1: Prepare the Add Directory

Place each new orthomosaic in `/mnt/class_data/sdalgarno/add/` with the following structure:

```
add/
└── {ortho_name}/
    ├── images/
    │   └── {ortho_name}.tif
    └── labels/
        └── {ortho_name}.tif
```

The ortho name should follow the convention: `{site}_u{ortho_id}` (e.g., `goose_grass_bay_u0200`).

## Step 2: Configure Split Assignment

Edit `scripts/add_new_orthos.sh` and add entries to the `SPLIT_MAP`:

```bash
SPLIT_MAP["goose_grass_bay_u0200"]="train"
SPLIT_MAP["louscoone_u0253"]="val"
```

**Important:** If a site already exists in the dataset, the new ortho must go to the same split to prevent data leakage. Check `docs/seagrass-dataset-splits.md` for existing site assignments.

## Step 3: Create Chips and Move to Main

```bash
cd /home/instructor/Code/hakai-ml-train
source .venv/bin/activate
./scripts/add_new_orthos.sh
```

This script:
- Creates 1024px and 512px chips from each ortho
- Removes tiles with nodata areas
- Moves raw data to `main/raw_data/{split}/`
- Moves chips to `main/chips_1024/{split}/` and `main/chips_512/{split}/`
- Cleans up temporary files

## Step 4: Update CV Splits

Edit `scripts/create_cv_splits.sh` to add the new site to the appropriate fold lists.

Determine which region the site belongs to (North, Central, or South) and add it to:
- The **test** list for that region's fold
- The **train** or **val** list for the other two folds

Example for a new Central site added to train:
```bash
# cv_central: add to test
copy_sites "$DST/test" \
    koeye goose_sw pruth_bay ... new_site

# cv_north: add to train
copy_sites "$DST/train" \
    ... new_site

# cv_south: add to train
copy_sites "$DST/train" \
    ... new_site
```

Then run the script:
```bash
./scripts/create_cv_splits.sh
```

## Step 5: Recreate Prototype Dataset

```bash
./scripts/sample_prototype.sh
```

This samples chips from `main/chips_1024/` with equal representation per site.

Current sampling fractions:
- Train: 58% per site
- Val/Test: 50% per site

## Step 6: Update Documentation

Update `docs/seagrass-dataset-splits.md` with:
- New site in the appropriate split table
- Updated chip counts
- New ortho in CV fold assignments

## Verification

After completing the workflow, verify chip counts:

```bash
# Main dataset
for split in train val test; do
    echo "$split: $(ls /mnt/class_data/sdalgarno/main/chips_1024/$split/*.npz | wc -l)"
done

# CV splits
for fold in cv_north cv_central cv_south; do
    echo "$fold:"
    for split in train val test; do
        echo "  $split: $(ls /mnt/class_data/sdalgarno/$fold/chips_1024/$split/*.npz | wc -l)"
    done
done

# Prototype
for split in train val test; do
    echo "$split: $(ls /mnt/class_data/sdalgarno/prototype_frac_50/chips_1024/$split/*.npz | wc -l)"
done
```

## Chip Naming Convention

Chips follow the pattern: `{site}_u{ortho_id}_{chip_idx}.npz`

- `site`: Location name (e.g., `triquet_bay`, `goose_grass_bay`)
- `ortho_id`: Unique ortho identifier (e.g., `0200`, `0537`)
- `chip_idx`: Sequential chip number

Exception: `bennett_bay_{chip_idx}.npz` (single ortho, no ID)
