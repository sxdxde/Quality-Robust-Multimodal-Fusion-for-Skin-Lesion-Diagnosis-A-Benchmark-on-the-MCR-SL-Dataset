#!/usr/bin/env bash
# Runs the follow-up experiments discussed after the core ablation matrix:
# focal loss, dermoscopy preprocessing, contrastive loss, an alternate
# optimizer, and eval-time-only TTA/multi-image averaging. Each is logged
# under its own run_tag in results/results_ledger.csv, alongside (not
# replacing) the original 4 core runs. Run inside tmux — see the earlier
# run_full_experiment_matrix.sh for the tmux workflow, same idea here.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== [1/5] channel_gated + focal loss ==="
python train.py --variant channel_gated --run-tag channel_gated_focal --focal-gamma 2.0 \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_focal.log

echo "=== [2/5] channel_gated + dermoscopy preprocessing (hair removal + color norm) ==="
python train.py --variant channel_gated --run-tag channel_gated_preprocessed --use-preprocessing \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_preprocessed.log

echo "=== [3/5] channel_gated + supervised contrastive auxiliary loss ==="
python train.py --variant channel_gated --run-tag channel_gated_contrastive --use-contrastive \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_contrastive.log

echo "=== [4/5] channel_gated + AdamW/cosine/discriminative-LR optimizer ==="
python train.py --variant channel_gated --run-tag channel_gated_optimizerv2 --optimizer adamw_cosine_discriminative \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_optimizerv2.log

echo "=== [5/5] eval-time-only TTA + multi-image averaging (no retraining, reuses existing checkpoints) ==="
python scripts/reeval_eval_time_options.py --data-dir "$DATA_DIR" 2>&1 | tee logs/reeval_eval_time_options.log

echo "=== ALL EXTENDED EXPERIMENTS DONE ==="
echo "Run 'python robustness_analysis.py --data-dir $DATA_DIR' afterward to refresh results/summary_table.csv with every run_tag now in the ledger."
