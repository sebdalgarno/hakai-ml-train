#!/bin/bash
# Check for site leakage across train/val/test splits
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/cv_central/chips_1024/"
OUTPUT_DIR="outputs"

DATASET_TAG=$(echo "$CHIP_DIR" | sed 's:/*$::' | rev | cut -d/ -f1-2 | rev | tr '/' '_')

# RUN -----
mkdir -p "$OUTPUT_DIR"

python -m src.prepare.check_site_leakage "$CHIP_DIR" \
    --stats-pdf "$OUTPUT_DIR/${DATASET_TAG}-site_leakage.pdf"
