#!/usr/bin/env bash
# SAM (Sharpness-Aware Minimization) + AdamW, one more tracked experiment.
# ~2x slower per epoch than the others (two forward-backward passes per
# step) — expect roughly 130 min for the 5-fold run instead of ~65.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated + SAM(AdamW) optimizer ==="
python train.py --variant channel_gated --run-tag channel_gated_sam_adamw --optimizer sam_adamw \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_sam_adamw.log

echo "=== eval-time tricks (TTA + multi-image) on top of the SAM checkpoints, free (no retraining) ==="
python scripts/reeval_eval_time_options.py --base-run-tag channel_gated_sam_adamw \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/reeval_channel_gated_sam_adamw.log

echo "=== DONE ==="
echo "Run 'python robustness_analysis.py --data-dir $DATA_DIR' afterward to refresh results/summary_table.csv."
