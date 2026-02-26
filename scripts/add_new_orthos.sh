#!/bin/bash
# Process new orthos from add/ directory: create chips and move to main dataset
set -e

# PATHS -----
ADD_DIR="/mnt/class_data/sdalgarno/add"
MAIN_DIR="/mnt/class_data/sdalgarno/main"

# CHIP PARAMETERS -----
NUM_BANDS=3
DTYPE="uint8"
REMAP="0 -100 1"
SEED=42

# SPLIT ASSIGNMENTS -----
# Format: "ortho_name:split" (based on docs/seagrass-dataset-splits.md)
declare -A SPLIT_MAP
SPLIT_MAP["goose_grass_bay_u0200"]="train"
SPLIT_MAP["louscoone_u0253"]="val"

# FUNCTIONS -----
create_chips() {
    local raw_dir="$1"
    local chip_dir="$2"
    local chip_size="$3"

    echo "  Creating ${chip_size}px chips..."
    python -m src.prepare.make_chip_dataset "$raw_dir" "$chip_dir" \
        --size "$chip_size" \
        --stride "$chip_size" \
        --num_bands "$NUM_BANDS" \
        --dtype "$DTYPE" \
        --remap $REMAP

    # Remove nodata tiles
    echo "  Removing nodata tiles..."
    for split_dir in "$chip_dir"/*; do
        if [ -d "$split_dir" ]; then
            python -m src.prepare.remove_tiles_with_nodata_areas "$split_dir" --num_channels "$NUM_BANDS"
        fi
    done
}

# MAIN -----
echo "=============================================="
echo "Processing new orthos from: $ADD_DIR"
echo "=============================================="

# Check add directory exists
if [ ! -d "$ADD_DIR" ]; then
    echo "ERROR: Add directory not found: $ADD_DIR"
    exit 1
fi

# Get list of orthos to process
ORTHOS=($(ls -d "$ADD_DIR"/*/ 2>/dev/null | xargs -n1 basename))

if [ ${#ORTHOS[@]} -eq 0 ]; then
    echo "No orthos found in $ADD_DIR"
    exit 0
fi

echo "Found orthos: ${ORTHOS[*]}"
echo ""

for ortho in "${ORTHOS[@]}"; do
    echo "----------------------------------------------"
    echo "Processing: $ortho"
    echo "----------------------------------------------"

    # Get split assignment
    split="${SPLIT_MAP[$ortho]}"
    if [ -z "$split" ]; then
        echo "WARNING: No split assignment for $ortho - skipping"
        echo "Add entry to SPLIT_MAP in this script"
        continue
    fi
    echo "  Assigned to: $split"

    ortho_dir="$ADD_DIR/$ortho"

    # Verify images and labels exist
    if [ ! -d "$ortho_dir/images" ] || [ ! -d "$ortho_dir/labels" ]; then
        echo "ERROR: Missing images/ or labels/ in $ortho_dir"
        continue
    fi

    # Create temporary raw_data structure for chip creation
    temp_raw="$ADD_DIR/temp_raw_$ortho"
    mkdir -p "$temp_raw/$split/images" "$temp_raw/$split/labels"

    # Copy files to temp structure
    cp "$ortho_dir/images/"* "$temp_raw/$split/images/"
    cp "$ortho_dir/labels/"* "$temp_raw/$split/labels/"

    # Create chips at both sizes
    chip_dir_1024="$ADD_DIR/chips_1024_$ortho"
    chip_dir_512="$ADD_DIR/chips_512_$ortho"

    create_chips "$temp_raw" "$chip_dir_1024" 1024
    create_chips "$temp_raw" "$chip_dir_512" 512

    # Count chips created
    count_1024=$(ls "$chip_dir_1024/$split"/*.npz 2>/dev/null | wc -l)
    count_512=$(ls "$chip_dir_512/$split"/*.npz 2>/dev/null | wc -l)
    echo "  Created: $count_1024 chips (1024px), $count_512 chips (512px)"

    # Move raw data to main
    echo "  Moving raw data to main/raw_data/$split/..."
    mv "$ortho_dir/images/"* "$MAIN_DIR/raw_data/$split/images/"
    mv "$ortho_dir/labels/"* "$MAIN_DIR/raw_data/$split/labels/"

    # Move chips to main
    echo "  Moving chips to main/..."
    mkdir -p "$MAIN_DIR/chips_1024/$split" "$MAIN_DIR/chips_512/$split"
    mv "$chip_dir_1024/$split/"*.npz "$MAIN_DIR/chips_1024/$split/"
    mv "$chip_dir_512/$split/"*.npz "$MAIN_DIR/chips_512/$split/"

    # Cleanup temp directories
    rm -rf "$temp_raw" "$chip_dir_1024" "$chip_dir_512" "$ortho_dir"

    echo "  Done"
    echo ""
done

# Cleanup empty add directory
rmdir "$ADD_DIR" 2>/dev/null || true

echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo ""
echo "Chip counts in main:"
for size in 1024 512; do
    echo "  chips_$size:"
    for split in train val test; do
        if [ -d "$MAIN_DIR/chips_$size/$split" ]; then
            count=$(ls -1 "$MAIN_DIR/chips_$size/$split"/*.npz 2>/dev/null | wc -l)
            echo "    $split: $count"
        fi
    done
done
