#!/usr/bin/env bash
# Step 2 prerequisite (see the close-out task): plain channel_gated and
# channel_gated_qweight_trust never saved per-epoch checkpoints, only the
# single best/final one -- so prediction-level top-k ensembling needs a
# fresh rerun with checkpoint saving enabled. Uses NEW run_tags
# (channel_gated_topk, channel_gated_qweight_trust_topk) specifically to
# avoid appending duplicate rows to the existing channel_gated /
# channel_gated_qweight_trust ledger entries -- the exact footgun that
# caused the earlier ledger dedup incident. Only plain and trust are
# rerun here, per the task's explicit scope (not hard_mining or any
# hard_mining-derived config -- three independent stacking attempts on
# hard_mining already failed).
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated (plain), rerun with top-5 checkpoint saving ==="
python train.py --variant channel_gated --run-tag channel_gated_topk \
  --save-topk 5 --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_topk.log

echo "=== channel_gated + trust quality weight, rerun with top-5 checkpoint saving ==="
python train.py --variant channel_gated --run-tag channel_gated_qweight_trust_topk \
  --quality-weight-mode trust --save-topk 5 \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_qweight_trust_topk.log

echo "=== DONE training. Now run: python scripts/checkpoint_ensemble_eval.py --data-dir $DATA_DIR ==="
