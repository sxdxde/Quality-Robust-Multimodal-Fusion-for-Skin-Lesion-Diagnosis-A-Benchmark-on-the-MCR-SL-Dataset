#!/usr/bin/env bash
# Run this ON THE REMOTE A100 PC, with the `brats` conda env active.
#
# Two things, neither of which trains anything:
#
# 1. Step 0 verification (scripts/verify_image_level_training.py) — asserts
#    every image of a lesion lands in exactly one fold, and prints the
#    evidence on what the train split actually enumerates today. This is the
#    check that settles the "all-images-per-lesion" question: training is
#    already image-level (config.py:train_on_all_dermoscopic_images defaults
#    to True), so there is no data-utilization gap to close.
#
# 2. The one genuinely untested, free item that task did surface — eval-time
#    TTA / multi-image averaging applied to the VALIDATED hard_mining
#    checkpoints. Every prior eval-time-trick run was on the plain baseline
#    or SAM; none on hard_mining. Pure inference on existing checkpoints, no
#    retraining, ~minutes not hours.
#
# Usage:
#   conda activate brats
#   bash ~/mcrsl_project/scripts/run_verify_and_hardmining_eval_variants.sh [DATA_DIR]
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
BASE_RUN_TAG="channel_gated_qweight_hard_mining"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== Step 0 verification: fold safety + what the train split actually yields ==="
python scripts/verify_image_level_training.py --data-dir "$DATA_DIR" \
  2>&1 | tee logs/verify_image_level_training.log

echo
echo "=== Eval-time variants on the validated hard_mining checkpoints (no retraining) ==="
python scripts/reeval_eval_time_options.py --base-run-tag "$BASE_RUN_TAG" --data-dir "$DATA_DIR" \
  2>&1 | tee logs/reeval_hardmining_eval_options.log

echo
echo "=== Aggregated comparison: hard_mining, with and without the eval-time tricks ==="
python scripts/report_ledger_rows.py \
  "$BASE_RUN_TAG" \
  "${BASE_RUN_TAG}_tta" \
  "${BASE_RUN_TAG}_multiimage" \
  "${BASE_RUN_TAG}_tta_multiimage" \
  2>&1 | tee logs/report_hardmining_eval_variants.log

echo
echo "=== DONE ==="
echo "Pull results back with ./pull_remote_results.sh from the local machine."
