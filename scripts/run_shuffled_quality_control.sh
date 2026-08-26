#!/usr/bin/env bash
# Shuffled-quality control (validity check, not a new search) — see the
# "shuffled-quality control" task spec. Tests whether trust/hard_mining's
# results come from the quality signal's actual information content, or from
# generic per-sample reweighting regardless of what the weights mean. Trains
# both variants with mean_image_rating permuted within each fold's training
# lesions only (fixed seed=test_fold, --shuffle-quality-control), evaluated
# on the true, unshuffled test fold — exactly as trust/hard_mining were
# trained otherwise.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated + quality-adaptive loss reweighting: trust (SHUFFLED CONTROL) ==="
python train.py --variant channel_gated --run-tag channel_gated_qweight_trust_shuffled \
  --quality-weight-mode trust --shuffle-quality-control --data-dir "$DATA_DIR" \
  2>&1 | tee logs/train_channel_gated_qweight_trust_shuffled.log

echo "=== channel_gated + quality-adaptive loss reweighting: hard_mining (SHUFFLED CONTROL) ==="
python train.py --variant channel_gated --run-tag channel_gated_qweight_hardmining_shuffled \
  --quality-weight-mode hard_mining --shuffle-quality-control --data-dir "$DATA_DIR" \
  2>&1 | tee logs/train_channel_gated_qweight_hardmining_shuffled.log

echo "=== analysis: shuffled-quality control comparison ==="
python scripts/shuffled_quality_control_analysis.py --data-dir "$DATA_DIR" \
  2>&1 | tee logs/shuffled_quality_control_analysis.log

echo "=== DONE ==="
echo "Results: results/robustness_shuffled_quality_control.csv"
