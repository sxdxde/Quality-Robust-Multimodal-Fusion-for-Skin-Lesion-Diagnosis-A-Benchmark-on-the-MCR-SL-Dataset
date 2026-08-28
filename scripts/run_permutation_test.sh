#!/usr/bin/env bash
# Multi-seed permutation test for the shuffled-quality control.
#
# The published control used ONE permutation, which is a single draw from the
# null and cannot on its own establish that the real gain is unusual. This runs
# N independent permutations so the real hard_mining result can be placed
# against a null DISTRIBUTION and given an empirical p-value.
#
# ~67 min per permutation (5 folds each). N=20 gives resolution to p=1/21.
set -uo pipefail
DATA_DIR="${1:-$HOME/mcrsl_project/data/raw/extracted/MCR-SL_dataset}"
N="${2:-20}"
cd "$(dirname "$0")/.."
mkdir -p logs
for s in $(seq 1 "$N"); do
  TAG="channel_gated_qweight_hm_perm${s}"
  if grep -q "^.*,${TAG},False,4," results/results_ledger.csv 2>/dev/null; then
    echo "=== permutation $s already complete, skipping ==="; continue
  fi
  echo "=== permutation $s/$N (seed $s) ==="
  python train.py --variant channel_gated --run-tag "$TAG" \
    --quality-weight-mode hard_mining --shuffle-quality-control --shuffle-seed "$s" \
    --data-dir "$DATA_DIR" > "logs/train_${TAG}.log" 2>&1
  echo "    done: $(grep -E '^total time' logs/train_${TAG}.log | tail -1)"
done
echo "=== ALL $N PERMUTATIONS DONE ==="
