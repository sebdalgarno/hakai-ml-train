#!/bin/bash
# Check chip counts and class balance
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024/"
OUTPUT_DIR="outputs"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"
DATASET_TAG=$(echo "$CHIP_DIR" | sed 's:/*$::' | rev | cut -d/ -f1-2 | rev | tr '/' '_')

# RUN -----
mkdir -p "$OUTPUT_DIR"

python -m src.prepare.check_class_balance "$CHIP_DIR" \
    --stats-pdf "$OUTPUT_DIR/${DATASET_TAG}-class_balance.pdf" \
    --class-names $CLASS_NAMES
