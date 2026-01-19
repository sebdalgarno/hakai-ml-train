#!/bin/bash
# Check that labels contain only expected values
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"

# RUN -----
python -m src.prepare.check_class_balance "$CHIP_DIR" --sanity-check --class-names $CLASS_NAMES
