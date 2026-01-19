#!/bin/bash
# Check chip counts and class balance
set -e

# DIRECTORIES -----
CHIP_DIR="/mnt/class_data/sdalgarno/prototype/chips"

# PARAMETERS -----
CLASS_NAMES="bg seagrass"

# RUN -----
python -m src.prepare.check_class_balance "$CHIP_DIR" --all-splits --class-names $CLASS_NAMES
