#!/bin/bash
set -e

# Create a smaller prototype dataset by selecting specific orthos.
# Maintains site diversity and condition coverage while reducing train set.
#
# Small prototype: 10,393 chips (vs 16,697 in full prototype)
#   Train: 5 orthos, 6,143 chips (removed arakun, goose_sw, island_bay, kendrick_point_west)
#   Val: 4 orthos, 1,986 chips (all kept)
#   Test: 4 orthos, 2,264 chips (all kept)

# DIRECTORIES -----
SRC_DIR="/mnt/class_data/sdalgarno/prototype/chips_512"
DST_DIR="/mnt/class_data/sdalgarno/prototype_small/chips_512"

# ORTHO SELECTIONS -----
# Train: Keep 5 orthos with regional balance and condition diversity
TRAIN_ORTHOS=(
    "calmus_u0421"           # South, baseline (1,961 chips)
    "koeye_u0715"            # Central, tannins/turbidity/glint (1,447 chips)
    "pruth_bay_u0383"        # Central, shadows/cloud reflections (1,158 chips)
    "beljay_bay_u0479"       # North, baseline (1,116 chips)
    "heater_harbour_u0088"   # North, dark/low light (461 chips)
)

# Val: Keep all 4 orthos
VAL_ORTHOS=(
    "bennett_bay"            # South, cloudy/glint (310 chips)
    "triquet_u1160"          # Central, baseline (724 chips)
    "superstition_u1280"     # Central, fog/sparse/algae (635 chips)
    "louscoone_u0091"        # North, overcast (317 chips)
)

# Test: Keep all 4 orthos
TEST_ORTHOS=(
    "beck_u0409"             # South, shadows/sparse (577 chips)
    "triquet_bay_u0537"      # Central, difficult edge (875 chips)
    "sedgwick_u0085"         # North, overcast/cloud reflections (250 chips)
    "section_cove_u0249"     # North, hazy lighting (562 chips)
)

# COPY FUNCTION -----
copy_ortho() {
    local ortho=$1
    local split=$2

    local src="$SRC_DIR/$split"
    local dst="$DST_DIR/$split"

    local count=$(ls -1 "$src"/${ortho}_*.npz 2>/dev/null | wc -l | tr -d ' ')
    if [[ $count -gt 0 ]]; then
        echo "  $ortho: $count chips"
        cp "$src"/${ortho}_*.npz "$dst"/
    else
        # Try without underscore suffix (e.g., bennett_bay)
        count=$(ls -1 "$src"/${ortho}[._]*.npz 2>/dev/null | wc -l | tr -d ' ')
        if [[ $count -gt 0 ]]; then
            echo "  $ortho: $count chips"
            cp "$src"/${ortho}[._]*.npz "$dst"/
        else
            echo "  [WARNING] No chips found for: $ortho in $split"
        fi
    fi
}

# CREATE DIRECTORIES -----
echo "=============================================="
echo "Creating small prototype dataset"
echo "=============================================="
echo "Source: $SRC_DIR"
echo "Destination: $DST_DIR"
echo ""

mkdir -p "$DST_DIR/train"
mkdir -p "$DST_DIR/val"
mkdir -p "$DST_DIR/test"

# COPY TRAIN -----
echo "TRAIN (5 orthos):"
for ortho in "${TRAIN_ORTHOS[@]}"; do
    copy_ortho "$ortho" "train"
done
train_total=$(ls -1 "$DST_DIR/train"/*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $train_total chips"
echo ""

# COPY VAL -----
echo "VAL (4 orthos):"
for ortho in "${VAL_ORTHOS[@]}"; do
    copy_ortho "$ortho" "val"
done
val_total=$(ls -1 "$DST_DIR/val"/*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $val_total chips"
echo ""

# COPY TEST -----
echo "TEST (4 orthos):"
for ortho in "${TEST_ORTHOS[@]}"; do
    copy_ortho "$ortho" "test"
done
test_total=$(ls -1 "$DST_DIR/test"/*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $test_total chips"
echo ""

# SUMMARY -----
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo "Small prototype: $DST_DIR"
echo "  Train: $train_total chips"
echo "  Val:   $val_total chips"
echo "  Test:  $test_total chips"
grand_total=$((train_total + val_total + test_total))
echo "  Total: $grand_total chips"
