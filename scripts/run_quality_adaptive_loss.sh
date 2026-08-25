#!/usr/bin/env bash
# QUALITY_ADAPTIVE_LOSS_TASK.md — the headline novel-contribution experiment.
# Trains both directions of quality-adaptive loss reweighting on the
# channel-gated architecture (the project's main method only), under the
# exact same subject-disjoint 5-fold protocol as every other run, then runs
# the Step 3 robustness-analysis extension. Requires the core config's
# tercile CSVs (robustness_quality_tercile_channel_gated_quality{False,True}.csv)
# to already exist — i.e. `python robustness_analysis.py` must have been run
# at least once already.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated + quality-adaptive loss reweighting: trust ==="
python train.py --variant channel_gated --run-tag channel_gated_qweight_trust \
  --quality-weight-mode trust --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_qweight_trust.log

echo "=== channel_gated + quality-adaptive loss reweighting: hard_mining ==="
python train.py --variant channel_gated --run-tag channel_gated_qweight_hard_mining \
  --quality-weight-mode hard_mining --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_qweight_hard_mining.log

echo "=== Step 3: extending the quality-tercile robustness analysis to both variants ==="
python scripts/quality_adaptive_loss_analysis.py --data-dir "$DATA_DIR" 2>&1 | tee logs/quality_adaptive_loss_analysis.log

echo "=== DONE ==="
echo "Results: results/robustness_quality_reweighting_comparison.csv and results/robustness_quality_reweighting_gaps.csv"
echo "Run 'python robustness_analysis.py' afterward to refresh results/summary_table.csv with the two new run_tags."
