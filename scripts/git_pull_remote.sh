#!/usr/bin/env bash
# Run this ON THE REMOTE A100 PC to sync code from GitHub into ~/mcrsl_project.
# Pure git+rsync — doesn't need the `brats` conda env active.
#
# Keeps a persistent checkout at ~/mcrsl_project_git (`git pull` there on
# repeat runs, `git clone` only the first time) and rsyncs just the code into
# ~/mcrsl_project, leaving data/, checkpoints/, results/, logs/ untouched.
#
# Those last two ARE tracked in the GitHub repo (results/logs get committed
# from the local dev machine after pull_remote_results.sh), but on THIS
# machine they're live, growing, append-only run outputs — e.g.
# results/results_ledger.csv gets a new row appended by every training run
# here, and is never pushed from this machine. Running `git pull` straight
# inside ~/mcrsl_project would hit "local changes would be overwritten" on
# that file the moment any training has happened remotely. Keeping the git
# checkout in a separate directory and rsyncing only code across sidesteps
# that entirely — same shape as the old rsync-from-laptop workflow, just
# sourced from GitHub now instead of the dev machine.
#
# Usage:
#   ./git_pull_remote.sh
set -euo pipefail

REPO_URL="https://github.com/sxdxde/Quality-Robust-Multimodal-Fusion-for-Skin-Lesion-Diagnosis-A-Benchmark-on-the-MCR-SL-Dataset.git"
GIT_DIR="$HOME/mcrsl_project_git"
PROJECT_DIR="$HOME/mcrsl_project"

if [ -d "$GIT_DIR/.git" ]; then
  echo "=== git pull (existing checkout at $GIT_DIR) ==="
  git -C "$GIT_DIR" pull origin main
else
  echo "=== git clone (first time) ==="
  git clone "$REPO_URL" "$GIT_DIR"
fi

echo "=== syncing code into $PROJECT_DIR (data/checkpoints/results/logs left untouched) ==="
mkdir -p "$PROJECT_DIR"
rsync -av --exclude 'data/' --exclude 'checkpoints/' --exclude 'results/' --exclude 'logs/' \
  --exclude '.git/' --exclude '__pycache__/' \
  "$GIT_DIR/" "$PROJECT_DIR/"

echo "=== DONE — $PROJECT_DIR is now up to date with origin/main ==="
