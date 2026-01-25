#!/bin/bash
set -e

# Copy ortho and label files according to Split B REVISED (28 sites) from seagrass-dataset-splits.md
# Structure: clean_subset/{site}_{visit_id}/images/{site}_{visit_id}.tif
#
# Revisions:
# 1. pruth_bay and grice_bay moved to train to prevent single-site dominance
# 2. superstition swapped to val, mcmullin_north to test to balance difficulty

# PATHS -----
SRC_DIR="/Volumes/MFA/eelgrassData/clean_subset2"
DST_DIR="/Volumes/MFA/eelgrassData/model"

# SPLIT B SITE ASSIGNMENTS (REVISED) -----
# Site name prefixes (without visit IDs)

TEST_SITES=(
    # South
    "beck"
    # Central
    "mcmullin_north"
    "triquet_bay"
    # North
    "bag_harbour"
    "section_cove"
    "sedgwick"
)

VAL_SITES=(
    # South
    "auseth"
    "bennett_bay"
    # Central
    "superstition"
    "triquet"
    # North
    "kendrick_point"
    "louscoone"
)

TRAIN_SITES=(
    # South
    "arakun"
    "calmus"
    "grice_bay"
    # Central
    "choked_pass"
    "goose_sw"
    "koeye"
    "pruth_bay"
    # North
    "balcolm_inlet"
    "beljay_bay"
    "heater_harbour"
    "island_bay"
    "kendrick_point_west"
    "louscoone_head"
    "louscoone_west"
    "ramsay"
    "swan_bay"
)

# CREATE DIRECTORIES -----
echo "Creating directory structure..."
mkdir -p "$DST_DIR/train/images"
mkdir -p "$DST_DIR/train/labels"
mkdir -p "$DST_DIR/val/images"
mkdir -p "$DST_DIR/val/labels"
mkdir -p "$DST_DIR/test/images"
mkdir -p "$DST_DIR/test/labels"

# COPY FUNCTION -----
# Finds all visit directories matching site prefix and copies files
copy_site() {
    local site=$1
    local split=$2
    local count=0

    # Find all directories matching the site pattern
    # Use exact prefix match to avoid kendrick_point matching kendrick_point_west
    for dir in "$SRC_DIR"/${site}_u*/; do
        if [[ -d "$dir" ]]; then
            local dirname=$(basename "$dir")
            local src_img="$dir/images/${dirname}.tif"
            local src_lbl="$dir/labels/${dirname}.tif"
            local dst_img="$DST_DIR/$split/images/${dirname}.tif"
            local dst_lbl="$DST_DIR/$split/labels/${dirname}.tif"

            if [[ -f "$src_img" ]]; then
                mv "$src_img" "$dst_img"
                ((count++))
            else
                echo "  [MISSING IMAGE] $src_img"
            fi

            if [[ -f "$src_lbl" ]]; then
                mv "$src_lbl" "$dst_lbl"
            else
                echo "  [MISSING LABEL] $src_lbl"
            fi
        fi
    done

    # Handle special case: bennett_bay has no _u suffix
    if [[ "$site" == "bennett_bay" ]] && [[ -d "$SRC_DIR/bennett_bay" ]]; then
        local src_img="$SRC_DIR/bennett_bay/images/bennett_bay.tif"
        local src_lbl="$SRC_DIR/bennett_bay/labels/bennett_bay.tif"
        if [[ -f "$src_img" ]]; then
            mv "$src_img" "$DST_DIR/$split/images/bennett_bay.tif"
            ((count++))
        fi
        if [[ -f "$src_lbl" ]]; then
            mv "$src_lbl" "$DST_DIR/$split/labels/bennett_bay.tif"
        fi
    fi

    if [[ $count -eq 0 ]]; then
        echo "  [WARNING] No directories found for: $site"
    else
        echo "  [OK] $site: $count visit(s) -> $split"
    fi
}

# COPY FILES -----
echo ""
echo "Copying TEST sites (${#TEST_SITES[@]} sites)..."
for site in "${TEST_SITES[@]}"; do
    copy_site "$site" "test"
done

echo ""
echo "Copying VAL sites (${#VAL_SITES[@]} sites)..."
for site in "${VAL_SITES[@]}"; do
    copy_site "$site" "val"
done

echo ""
echo "Copying TRAIN sites (${#TRAIN_SITES[@]} sites)..."
for site in "${TRAIN_SITES[@]}"; do
    copy_site "$site" "train"
done

# SUMMARY -----
echo ""
echo "Done. Summary:"
echo "  Train images: $(ls -1 "$DST_DIR/train/images" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Train labels: $(ls -1 "$DST_DIR/train/labels" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Val images:   $(ls -1 "$DST_DIR/val/images" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Val labels:   $(ls -1 "$DST_DIR/val/labels" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Test images:  $(ls -1 "$DST_DIR/test/images" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Test labels:  $(ls -1 "$DST_DIR/test/labels" 2>/dev/null | wc -l | tr -d ' ')"
