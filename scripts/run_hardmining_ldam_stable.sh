#!/usr/bin/env bash
# Follow-up to QUALITY_ADAPTIVE_LOSS_TASK.md: stacks the best-performing
# mechanism so far (quality-adaptive loss reweighting, hard_mining direction)
# with two training-stability/decision-boundary changes motivated directly by
# channel_gated_qweight_{trust,hard_mining}'s training logs:
#   - val_bacc oscillated wildly epoch-to-epoch in every fold (noisy
#     checkpoint selection) with occasional destabilizing train_loss spikes
#     -> gradient clipping (clip_grad_norm_, max_norm=1.0)
#   - sensitivity trailed specificity by ~0.11 even in the best config
#     -> LDAM-style (Cao et al. 2019) class margin on the binary head,
#        standard C=0.5, not tuned/searched
# Both are opt-in flags (default off), so no already-reported run is affected.
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== channel_gated + hard_mining quality weight + grad clip + LDAM margin ==="
python train.py --variant channel_gated --run-tag channel_gated_hardmining_ldam_stable \
  --quality-weight-mode hard_mining --grad-clip-norm 1.0 --use-ldam-margin \
  --data-dir "$DATA_DIR" 2>&1 | tee logs/train_channel_gated_hardmining_ldam_stable.log

echo "=== DONE ==="
echo "Run 'python robustness_analysis.py --data-dir $DATA_DIR' afterward to refresh results/summary_table.csv."
echo "Also run 'python scripts/optimal_threshold_eval.py --data-dir $DATA_DIR --run-tags channel_gated_hardmining_ldam_stable' to check the free threshold-calibration gain on top."
