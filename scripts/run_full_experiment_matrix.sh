#!/usr/bin/env bash
# Runs the full ablation matrix + quality-aware variant + robustness analyses,
# sequentially, on the remote A100. Meant to be run inside tmux so it survives
# an SSH disconnect. Each step's stdout/stderr is also tee'd to logs/ so you
# can inspect it even mid-run or after the session ends.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== [1/5] image_only ==="
python train.py --variant image_only --data-dir "$DATA_DIR" 2>&1 | tee logs/train_image_only.log

echo "=== [2/5] late_fusion ==="
python train.py --variant late_fusion --data-dir "$DATA_DIR" 2>&1 | tee logs/train_late_fusion.log

echo "=== [3/5] channel_gated (main method) ==="
python train.py --variant channel_gated --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated.log

echo "=== [4/5] channel_gated + quality-aware (robustness analysis 2) ==="
python train.py --variant channel_gated --quality-aware --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_quality_aware.log

echo "=== [5/5] robustness_analysis.py (analyses 1, 3, 4) ==="
python robustness_analysis.py --variant channel_gated --data-dir "$DATA_DIR" 2>&1 | tee logs/robustness_analysis.log

echo "=== ALL DONE ==="
