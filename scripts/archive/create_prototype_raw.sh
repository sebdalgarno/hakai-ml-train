#!/bin/bash
set -e

# Create prototype dataset by copying selected ortho TIFs.
# Use this subset to create 512 or 1024 tiles without processing the full dataset.
#
# Copies from: /mnt/class_data/sdalgarno/main/raw_data/{train,val,test}/{images,labels}
# Copies to:   /mnt/class_data/sdalgarno/prototype/raw_data/{train,val,test}/{images,labels}
#
# Full prototype: 17 orthos
#   Train: 9 orthos (~12,400 chips at 512, ~1,500 at 1024)
#   Val: 4 orthos (~2,000 chips at 512, ~377 at 1024)
#   Test: 4 orthos (~2,300 chips at 512, ~301 at 1024)

# DIRECTORIES -----
SRC_DIR="/mnt/class_data/sdalgarno/main/raw_data"
DST_DIR="/mnt/class_data/sdalgarno/prototype/raw_data"

# ORTHO SELECTIONS -----
# Train: 9 orthos with regional balance and condition diversity
TRAIN_ORTHOS=(
    "arakun_u0411"            # South, baseline
    "calmus_u0421"            # South, baseline
    "koeye_u0715"             # Central, tannins/turbidity/glint
    "pruth_bay_u0383"         # Central, shadows/cloud reflections
    "goose_sw_u1174"          # Central, turbidity/overcast/low density
    "heater_harbour_u0088"    # North, dark/low light
    "beljay_bay_u0479"        # North, baseline
    "kendrick_point_west_u0494"  # North, baseline
    "island_bay_u0486"        # North, baseline
)

# Val: 4 orthos
VAL_ORTHOS=(
    "bennett_bay"             # South, cloudy/glint
    "triquet_u1160"           # Central, clear baseline
    "superstition_u1280"      # Central, fog/sparse/algae
    "louscoone_u0091"         # North, overcast
)

# Test: 4 orthos
TEST_ORTHOS=(
    "beck_u0409"              # South, shadows/sparse
    "triquet_bay_u0537"       # Central, difficult edge
    "sedgwick_u0085"          # North, overcast/cloud reflections
    "section_cove_u0249"      # North, hazy lighting
)

# COPY FUNCTION -----
copy_ortho_tif() {
    local ortho=$1
    local split=$2

    local src_img="$SRC_DIR/$split/images"
    local src_lbl="$SRC_DIR/$split/labels"
    local dst_img="$DST_DIR/$split/images"
    local dst_lbl="$DST_DIR/$split/labels"

    # Try exact match first, then with wildcard for extension variations
    local img_file=$(ls "$src_img"/${ortho}.tif 2>/dev/null || ls "$src_img"/${ortho}.tiff 2>/dev/null || echo "")
    local lbl_file=$(ls "$src_lbl"/${ortho}.tif 2>/dev/null || ls "$src_lbl"/${ortho}.tiff 2>/dev/null || echo "")

    if [[ -n "$img_file" && -n "$lbl_file" ]]; then
        echo "  $ortho: copying image + label"
        cp "$img_file" "$dst_img/"
        cp "$lbl_file" "$dst_lbl/"
    else
        # Report what's missing
        if [[ -z "$img_file" ]]; then
            echo "  [WARNING] No image found: $ortho in $split"
        fi
        if [[ -z "$lbl_file" ]]; then
            echo "  [WARNING] No label found: $ortho in $split"
        fi
    fi
}

# CREATE DIRECTORIES -----
echo "=============================================="
echo "Creating prototype raw data"
echo "=============================================="
echo "Source: $SRC_DIR"
echo "Destination: $DST_DIR"
echo ""

mkdir -p "$DST_DIR/train/images"
mkdir -p "$DST_DIR/train/labels"
mkdir -p "$DST_DIR/val/images"
mkdir -p "$DST_DIR/val/labels"
mkdir -p "$DST_DIR/test/images"
mkdir -p "$DST_DIR/test/labels"

# COPY TRAIN -----
echo "TRAIN (9 orthos):"
for ortho in "${TRAIN_ORTHOS[@]}"; do
    copy_ortho_tif "$ortho" "train"
done
train_img=$(ls -1 "$DST_DIR/train/images"/*.tif* 2>/dev/null | wc -l | tr -d ' ')
train_lbl=$(ls -1 "$DST_DIR/train/labels"/*.tif* 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $train_img images, $train_lbl labels"
echo ""

# COPY VAL -----
echo "VAL (4 orthos):"
for ortho in "${VAL_ORTHOS[@]}"; do
    copy_ortho_tif "$ortho" "val"
done
val_img=$(ls -1 "$DST_DIR/val/images"/*.tif* 2>/dev/null | wc -l | tr -d ' ')
val_lbl=$(ls -1 "$DST_DIR/val/labels"/*.tif* 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $val_img images, $val_lbl labels"
echo ""

# COPY TEST -----
echo "TEST (4 orthos):"
for ortho in "${TEST_ORTHOS[@]}"; do
    copy_ortho_tif "$ortho" "test"
done
test_img=$(ls -1 "$DST_DIR/test/images"/*.tif* 2>/dev/null | wc -l | tr -d ' ')
test_lbl=$(ls -1 "$DST_DIR/test/labels"/*.tif* 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $test_img images, $test_lbl labels"
echo ""

# SUMMARY -----
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo "Prototype raw data: $DST_DIR"
echo "  Train: $train_img orthos"
echo "  Val:   $val_img orthos"
echo "  Test:  $test_img orthos"
total=$((train_img + val_img + test_img))
echo "  Total: $total orthos"
echo ""
echo "Next steps:"
echo "  # Create 512 tiles"
echo "  python -m src.prepare.make_chip_dataset $DST_DIR/train $DST_DIR/../chips_512/train --size 512 --stride 384 --num_bands 3 --remap 0 -100 1"
echo ""
echo "  # Create 1024 tiles"
echo "  python -m src.prepare.make_chip_dataset $DST_DIR/train $DST_DIR/../chips_1024/train --size 1024 --stride 768 --num_bands 3 --remap 0 -100 1"
