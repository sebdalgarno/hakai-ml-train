#!/bin/bash
# Sample from CV splits to create balanced train/val sets
# Proportional sampling by site
set -e

# CONFIG -----
SRC_BASE="/mnt/class_data/sdalgarno"
TARGET_TRAIN=2000
TARGET_VAL=500
SEED=42

# CV SPLITS -----
CV_SPLITS=("cv_north" "cv_central" "cv_south")

# FUNCTIONS -----
sample_proportional() {
    local src_dir="$1"
    local dst_dir="$2"
    local target="$3"
    local seed="$4"

    # Use Python for proportional sampling
    python3 << EOF
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import re

def extract_site(filename):
    stem = Path(filename).stem
    match = re.match(r"(.+)_u\d+_\d+$", stem)
    if match:
        return match.group(1)
    match = re.match(r"(.+)_(\d+)$", stem)
    if match:
        potential_site = match.group(1)
        if not re.search(r"_u\d+$", potential_site):
            return potential_site
    return stem

random.seed($seed)
src = Path("$src_dir")
dst = Path("$dst_dir")
target = $target

# Group by site
by_site = defaultdict(list)
for f in src.glob("*.npz"):
    by_site[extract_site(f.name)].append(f)

total = sum(len(v) for v in by_site.values())
if total == 0:
    print(f"  No tiles found in {src}")
    exit(0)

# Calculate proportional samples
dst.mkdir(parents=True, exist_ok=True)
sampled = 0
for site, files in by_site.items():
    proportion = len(files) / total
    n_sample = max(1, round(proportion * target))
    n_sample = min(n_sample, len(files))
    selected = random.sample(files, n_sample)
    for f in selected:
        shutil.copy2(f, dst / f.name)
    sampled += len(selected)
    print(f"    {site}: {len(selected)}/{len(files)}")

print(f"  Total: {sampled} tiles (target: {target})")
EOF
}

# MAIN -----
echo "=============================================="
echo "Sampling CV Splits (Proportional by Site)"
echo "=============================================="
echo "Target train: $TARGET_TRAIN tiles"
echo "Target val: $TARGET_VAL tiles"
echo "Seed: $SEED"
echo ""

for cv in "${CV_SPLITS[@]}"; do
    echo "=============================================="
    echo "$cv"
    echo "=============================================="

    SRC="$SRC_BASE/$cv/chips_1024"
    DST="$SRC_BASE/${cv}_sampled/chips_1024"

    # Clear destination
    rm -rf "$DST"
    mkdir -p "$DST/train" "$DST/val" "$DST/test"

    # Sample train
    echo ""
    echo "Sampling train (target: $TARGET_TRAIN):"
    sample_proportional "$SRC/train" "$DST/train" "$TARGET_TRAIN" "$SEED"

    # Sample val
    echo ""
    echo "Sampling val (target: $TARGET_VAL):"
    sample_proportional "$SRC/val" "$DST/val" "$TARGET_VAL" "$SEED"

    # Copy test (no sampling - keep all for evaluation)
    echo ""
    echo "Copying test (all tiles):"
    cp "$SRC/test/"*.npz "$DST/test/" 2>/dev/null || echo "  No test tiles"
    test_count=$(ls "$DST/test/"*.npz 2>/dev/null | wc -l || echo 0)
    echo "  Total: $test_count tiles"

    echo ""
done

echo "=============================================="
echo "Sampling complete"
echo "=============================================="
echo ""
echo "Sampled data saved to:"
for cv in "${CV_SPLITS[@]}"; do
    echo "  $SRC_BASE/${cv}_sampled/chips_1024/"
done
