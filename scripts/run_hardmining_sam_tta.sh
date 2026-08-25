#!/usr/bin/env bash
# Stacks the best-performing quality-reweighting mechanism (hard_mining) with
# the two other independently-proven-positive interventions on this dataset
# (SAM optimizer, TTA at eval). Unlike the earlier LDAM-margin+grad-clip
# attempt (which combined two UNTESTED mechanisms and made things worse),
# both pieces here are already individually validated on channel_gated:
#   - SAM alone: 0.810 balanced accuracy (channel_gated_sam_adamw)
#   - TTA alone: free, no retraining, +sensitivity/specificity across configs
# SAM's flatter-minima objective may also directly help the diagnosed
# checkpoint-selection noise problem (noisy val_bacc every epoch), since SAM
# is designed to avoid sharp/noisy loss-landscape regions.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated + hard_mining quality weight + SAM optimizer + TTA (eval-time only) ==="
python train.py --variant channel_gated --run-tag channel_gated_hardmining_sam_tta \
  --quality-weight-mode hard_mining --optimizer sam_adamw --tta \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_hardmining_sam_tta.log

echo "=== DONE ==="
echo "Run 'python robustness_analysis.py --data-dir $DATA_DIR' afterward to refresh results/summary_table.csv."
