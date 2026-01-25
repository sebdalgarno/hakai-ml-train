#!/bin/bash
set -e

# Reorganize chips and raw data into revised Split B buckets, then create prototype subset.
#
# Revised Split B moves:
#   - pruth_bay, grice_bay → train (were val/test)
#   - superstition → val (was test)
#   - mcmullin_north, triquet_bay, section_cove, beck → test (were train)
#   - auseth → val (was train)

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_512"
RAW_DIR="/mnt/class_data/sdalgarno/main/raw_data"
PROTOTYPE_DIR="/mnt/class_data/sdalgarno/prototype/chips_512"

# ============================================================================
# STEP 1: Move chips into revised buckets
# ============================================================================
echo "=============================================="
echo "STEP 1: Reorganizing chips into revised split"
echo "=============================================="

# Move chips by ortho pattern (site_u####_*.npz or site_*.npz)
move_chips() {
    local pattern=$1
    local from_split=$2
    local to_split=$3

    local src="$CHIP_DIR/$from_split"
    local dst="$CHIP_DIR/$to_split"

    local count=$(ls -1 "$src"/${pattern}_*.npz 2>/dev/null | wc -l | tr -d ' ')
    if [[ $count -gt 0 ]]; then
        echo "  Moving $count chips: $pattern ($from_split → $to_split)"
        mv "$src"/${pattern}_*.npz "$dst"/
    else
        echo "  [SKIP] No chips found for pattern: $pattern in $from_split"
    fi
}

echo ""
echo "Moving sites to TRAIN..."
# pruth_bay: val → train (8 orthos)
for ortho in pruth_bay_u0383 pruth_bay_u0530 pruth_bay_u0699 pruth_bay_u0785 \
             pruth_bay_u0911 pruth_bay_u1035 pruth_bay_u1158 pruth_bay_u1305; do
    move_chips "$ortho" "val" "train"
done
# grice_bay: test → train
move_chips "grice_bay_u0414" "test" "train"

echo ""
echo "Moving sites to VAL..."
# superstition: test → val (4 orthos)
for ortho in superstition_u0914 superstition_u1040 superstition_u1177 superstition_u1280; do
    move_chips "$ortho" "test" "val"
done
# auseth: train → val
move_chips "auseth_u0413" "train" "val"

echo ""
echo "Moving sites to TEST..."
# mcmullin_north: train → test (2 orthos)
move_chips "mcmullin_north_u0900" "train" "test"
move_chips "mcmullin_north_u1270" "train" "test"
# triquet_bay: train → test (3 orthos)
move_chips "triquet_bay_u0537" "train" "test"
move_chips "triquet_bay_u0709" "train" "test"
move_chips "triquet_bay_u1292" "train" "test"
# section_cove: train → test (2 orthos)
move_chips "section_cove_u0249" "train" "test"
move_chips "section_cove_u0487" "train" "test"
# beck: train → test
move_chips "beck_u0409" "train" "test"

echo ""
echo "Chip reorganization complete."
echo "  Train: $(ls -1 "$CHIP_DIR/train"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo "  Val:   $(ls -1 "$CHIP_DIR/val"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo "  Test:  $(ls -1 "$CHIP_DIR/test"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"

# ============================================================================
# STEP 2: Move raw TIFs into revised buckets
# ============================================================================
echo ""
echo "=============================================="
echo "STEP 2: Reorganizing raw TIFs into revised split"
echo "=============================================="

# Move raw ortho TIFs
move_raw() {
    local ortho=$1
    local from_split=$2
    local to_split=$3

    local src_img="$RAW_DIR/$from_split/images/${ortho}.tif"
    local src_lbl="$RAW_DIR/$from_split/labels/${ortho}.tif"
    local dst_img="$RAW_DIR/$to_split/images/${ortho}.tif"
    local dst_lbl="$RAW_DIR/$to_split/labels/${ortho}.tif"

    if [[ -f "$src_img" ]]; then
        echo "  Moving: $ortho ($from_split → $to_split)"
        mv "$src_img" "$dst_img"
        [[ -f "$src_lbl" ]] && mv "$src_lbl" "$dst_lbl"
    else
        echo "  [SKIP] Not found: $ortho in $from_split"
    fi
}

echo ""
echo "Moving raw TIFs to TRAIN..."
for ortho in pruth_bay_u0383 pruth_bay_u0530 pruth_bay_u0699 pruth_bay_u0785 \
             pruth_bay_u0911 pruth_bay_u1035 pruth_bay_u1158 pruth_bay_u1305; do
    move_raw "$ortho" "val" "train"
done
move_raw "grice_bay_u0414" "test" "train"

echo ""
echo "Moving raw TIFs to VAL..."
for ortho in superstition_u0914 superstition_u1040 superstition_u1177 superstition_u1280; do
    move_raw "$ortho" "test" "val"
done
move_raw "auseth_u0413" "train" "val"

echo ""
echo "Moving raw TIFs to TEST..."
move_raw "mcmullin_north_u0900" "train" "test"
move_raw "mcmullin_north_u1270" "train" "test"
move_raw "triquet_bay_u0537" "train" "test"
move_raw "triquet_bay_u0709" "train" "test"
move_raw "triquet_bay_u1292" "train" "test"
move_raw "section_cove_u0249" "train" "test"
move_raw "section_cove_u0487" "train" "test"
move_raw "beck_u0409" "train" "test"

# Handle special case: bennett_bay has no _u suffix
if [[ -f "$RAW_DIR/val/images/bennett_bay.tif" ]]; then
    echo "  [OK] bennett_bay already in val"
elif [[ -f "$RAW_DIR/train/images/bennett_bay.tif" ]]; then
    echo "  Moving: bennett_bay (train → val)"
    mv "$RAW_DIR/train/images/bennett_bay.tif" "$RAW_DIR/val/images/"
    [[ -f "$RAW_DIR/train/labels/bennett_bay.tif" ]] && mv "$RAW_DIR/train/labels/bennett_bay.tif" "$RAW_DIR/val/labels/"
fi

echo ""
echo "Raw TIF reorganization complete."
echo "  Train: $(ls -1 "$RAW_DIR/train/images"/*.tif 2>/dev/null | wc -l | tr -d ' ') orthos"
echo "  Val:   $(ls -1 "$RAW_DIR/val/images"/*.tif 2>/dev/null | wc -l | tr -d ' ') orthos"
echo "  Test:  $(ls -1 "$RAW_DIR/test/images"/*.tif 2>/dev/null | wc -l | tr -d ' ') orthos"

# ============================================================================
# STEP 3: Copy prototype chips
# ============================================================================
echo ""
echo "=============================================="
echo "STEP 3: Creating prototype chip subset"
echo "=============================================="

# Create prototype directories
mkdir -p "$PROTOTYPE_DIR/train"
mkdir -p "$PROTOTYPE_DIR/val"
mkdir -p "$PROTOTYPE_DIR/test"

# Copy chips for specific orthos
copy_prototype() {
    local ortho=$1
    local split=$2

    local src="$CHIP_DIR/$split"
    local dst="$PROTOTYPE_DIR/$split"

    local count=$(ls -1 "$src"/${ortho}_*.npz 2>/dev/null | wc -l | tr -d ' ')
    if [[ $count -gt 0 ]]; then
        echo "  Copying $count chips: $ortho → prototype/$split"
        cp "$src"/${ortho}_*.npz "$dst"/
    else
        echo "  [SKIP] No chips found for: $ortho in $split"
    fi
}

echo ""
echo "Copying TRAIN prototype orthos (~12,400 chips)..."
# South (2 sites)
copy_prototype "arakun_u0411" "train"
copy_prototype "calmus_u0421" "train"
# Central (3 sites)
copy_prototype "koeye_u0715" "train"
copy_prototype "pruth_bay_u0383" "train"
copy_prototype "goose_sw_u1174" "train"
# North (4 sites)
copy_prototype "heater_harbour_u0088" "train"
copy_prototype "beljay_bay_u0479" "train"
copy_prototype "kendrick_point_west_u0494" "train"
copy_prototype "island_bay_u0486" "train"

echo ""
echo "Copying VAL prototype orthos (~2,000 chips)..."
# South (1 site)
copy_prototype "bennett_bay" "val"
# Central (2 sites)
copy_prototype "triquet_u1160" "val"
copy_prototype "superstition_u1280" "val"
# North (1 site)
copy_prototype "louscoone_u0091" "val"

echo ""
echo "Copying TEST prototype orthos (~2,300 chips)..."
# South (1 site)
copy_prototype "beck_u0409" "test"
# Central (1 site)
copy_prototype "triquet_bay_u0537" "test"
# North (2 sites)
copy_prototype "sedgwick_u0085" "test"
copy_prototype "section_cove_u0249" "test"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo ""
echo "Main dataset (revised split):"
echo "  $CHIP_DIR"
echo "    Train: $(ls -1 "$CHIP_DIR/train"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo "    Val:   $(ls -1 "$CHIP_DIR/val"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo "    Test:  $(ls -1 "$CHIP_DIR/test"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo ""
echo "Prototype dataset:"
echo "  $PROTOTYPE_DIR"
echo "    Train: $(ls -1 "$PROTOTYPE_DIR/train"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo "    Val:   $(ls -1 "$PROTOTYPE_DIR/val"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo "    Test:  $(ls -1 "$PROTOTYPE_DIR/test"/*.npz 2>/dev/null | wc -l | tr -d ' ') chips"
echo ""
echo "Run ./scripts/chip_summary.sh on each directory to verify counts."
