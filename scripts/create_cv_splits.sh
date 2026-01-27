#!/bin/bash
# Create regional cross-validation splits by copying tiles from main dataset
set -e

# PATHS -----
SRC_DIR="/mnt/class_data/sdalgarno/main/chips_1024"
DST_BASE="/mnt/class_data/sdalgarno"

# FUNCTIONS -----
copy_sites() {
    local dst_dir="$1"
    shift
    local sites=("$@")

    mkdir -p "$dst_dir"
    local count=0

    for site in "${sites[@]}"; do
        # Find all tiles for this site across train/val/test source dirs
        for src_split in train val test; do
            if [ -d "$SRC_DIR/$src_split" ]; then
                for f in "$SRC_DIR/$src_split/${site}_"*.npz; do
                    if [ -f "$f" ]; then
                        cp "$f" "$dst_dir/"
                        ((count++))
                    fi
                done
            fi
        done
    done

    echo "    copied $count tiles"
}

# MAIN -----
echo "=============================================="
echo "Creating Regional CV Splits"
echo "=============================================="
echo "Source: $SRC_DIR"
echo "Destination base: $DST_BASE"

# Check source exists
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory not found: $SRC_DIR"
    exit 1
fi

# CV_NORTH -----
echo ""
echo "Creating cv_north (North held out as test)..."
DST="$DST_BASE/cv_north/chips_1024"
rm -rf "$DST"
mkdir -p "$DST/train" "$DST/val" "$DST/test"

echo "  test (15 North sites):"
copy_sites "$DST/test" \
    louscoone_head island_bay kendrick_point ramsay bag_harbour \
    section_cove swan_bay beljay_bay takelly_cove balcolm_inlet \
    louscoone_west kendrick_point_west louscoone sedgwick heater_harbour

echo "  val (6 sites):"
copy_sites "$DST/val" \
    superstition mcmullin_north auseth triquet beck bennett_bay

echo "  train (8 sites):"
copy_sites "$DST/train" \
    koeye goose_sw pruth_bay grice_bay choked_pass triquet_bay calmus arakun

echo "  Summary: train=$(ls "$DST/train" | wc -l) val=$(ls "$DST/val" | wc -l) test=$(ls "$DST/test" | wc -l)"

# CV_CENTRAL -----
echo ""
echo "Creating cv_central (Central held out as test)..."
DST="$DST_BASE/cv_central/chips_1024"
rm -rf "$DST"
mkdir -p "$DST/train" "$DST/val" "$DST/test"

echo "  test (8 Central sites):"
copy_sites "$DST/test" \
    koeye goose_sw pruth_bay choked_pass superstition mcmullin_north triquet_bay triquet

echo "  val (6 sites):"
copy_sites "$DST/val" \
    kendrick_point auseth louscoone sedgwick beck bennett_bay

echo "  train (15 sites):"
copy_sites "$DST/train" \
    grice_bay louscoone_head island_bay ramsay calmus bag_harbour \
    section_cove swan_bay beljay_bay takelly_cove balcolm_inlet \
    louscoone_west kendrick_point_west heater_harbour arakun

echo "  Summary: train=$(ls "$DST/train" | wc -l) val=$(ls "$DST/val" | wc -l) test=$(ls "$DST/test" | wc -l)"

# CV_SOUTH -----
echo ""
echo "Creating cv_south (South held out as test)..."
DST="$DST_BASE/cv_south/chips_1024"
rm -rf "$DST"
mkdir -p "$DST/train" "$DST/val" "$DST/test"

echo "  test (6 South sites):"
copy_sites "$DST/test" \
    grice_bay calmus auseth beck bennett_bay arakun

echo "  val (7 sites):"
copy_sites "$DST/val" \
    superstition triquet_bay kendrick_point section_cove triquet louscoone sedgwick

echo "  train (16 sites):"
copy_sites "$DST/train" \
    koeye goose_sw pruth_bay choked_pass louscoone_head mcmullin_north \
    island_bay ramsay bag_harbour swan_bay beljay_bay takelly_cove \
    balcolm_inlet louscoone_west kendrick_point_west heater_harbour

echo "  Summary: train=$(ls "$DST/train" | wc -l) val=$(ls "$DST/val" | wc -l) test=$(ls "$DST/test" | wc -l)"

echo ""
echo "=============================================="
echo "Regional CV splits complete"
echo "=============================================="
