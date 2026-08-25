#!/usr/bin/env bash
# Tests Stochastic Weight Averaging (Izmailov et al. 2018) as a replacement
# for the single-noisy-epoch checkpoint-selection rule used everywhere else
# in this project. Motivated directly by channel_gated_qweight_{trust,
# hard_mining}'s training logs, which show val_bacc swinging 0.15-0.25
# between adjacent epochs in every fold — the current "pick the single best
# epoch" rule is likely capturing noise spikes, not converged states.
#
# Two runs, isolating two different questions:
#   1. channel_gated_swa (plain, no quality reweighting): how much of the
#      project's fold-to-fold variance is checkpoint-selection noise alone,
#      independent of any loss-function change?
#   2. channel_gated_hardmining_swa: does SWA's more stable checkpoint help
#      the best-performing loss variant (hard_mining) further?
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated + SWA alone (isolates checkpoint-selection-noise contribution) ==="
python train.py --variant channel_gated --run-tag channel_gated_swa \
  --use-swa --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_swa.log

echo "=== channel_gated + hard_mining quality weight + SWA ==="
python train.py --variant channel_gated --run-tag channel_gated_hardmining_swa \
  --quality-weight-mode hard_mining --use-swa --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_hardmining_swa.log

echo "=== DONE ==="
echo "Run 'python robustness_analysis.py --data-dir $DATA_DIR' afterward to refresh results/summary_table.csv."
