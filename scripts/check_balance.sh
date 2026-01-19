#!/bin/bash
# Check chip counts and class balance
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"
OUTPUT_DIR="outputs"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"

# RUN -----
mkdir -p "$OUTPUT_DIR"

python -m src.prepare.check_class_balance "$CHIP_DIR" --all-splits --class-names $CLASS_NAMES

python -m src.prepare.check_class_balance "$CHIP_DIR" \
    --stats-pdf "$OUTPUT_DIR/class_balance.pdf" \
    --class-names $CLASS_NAMES
