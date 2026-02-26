#!/bin/bash
# Check for site leakage across train/val/test splits
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/main/chips_1024/"
OUTPUT_DIR="outputs"

# RUN -----
mkdir -p "$OUTPUT_DIR"

python -m src.prepare.check_site_leakage "$CHIP_DIR" \
    --stats-pdf "$OUTPUT_DIR/site_leakage.pdf"
