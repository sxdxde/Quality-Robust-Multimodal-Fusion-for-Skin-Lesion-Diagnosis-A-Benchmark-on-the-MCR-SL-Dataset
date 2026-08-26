#!/usr/bin/env bash
# Run this ON THE REMOTE A100 PC (not locally), with the `brats` conda env
# already active (conda activate brats — same as every other run_*.sh in
# this project). Pulls the latest code from GitHub into ~/mcrsl_project
# (without touching data/checkpoints/results/logs, which live remote-only),
# then runs the shuffled-quality-control experiment.
#
# Usage:
#   conda activate brats
#   ./remote_pull_and_run_shuffled_control.sh [DATA_DIR]
#   (DATA_DIR defaults to ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset)
set -euo pipefail

REPO_URL="https://github.com/sxdxde/Quality-Robust-Multimodal-Fusion-for-Skin-Lesion-Diagnosis-A-Benchmark-on-the-MCR-SL-Dataset.git"
PROJECT_DIR="$HOME/mcrsl_project"
DATA_DIR="${1:-$PROJECT_DIR/data/raw/extracted/MCR-SL_dataset}"

echo "=== pulling latest code from GitHub ==="
cd "$PROJECT_DIR"
rm -rf mcrsl_project_git
git clone "$REPO_URL" mcrsl_project_git
rsync -av --exclude 'data/' --exclude 'checkpoints/' --exclude 'results/' --exclude 'logs/' \
  --exclude '.git/' --exclude '__pycache__/' \
  mcrsl_project_git/ "$PROJECT_DIR/"
rm -rf mcrsl_project_git

echo "=== running shuffled-quality control (trust + hard_mining, 5-fold each) ==="
cd "$PROJECT_DIR"
bash scripts/run_shuffled_quality_control.sh "$DATA_DIR"

echo "=== DONE — pull results back to local with pull_remote_results.sh ==="
