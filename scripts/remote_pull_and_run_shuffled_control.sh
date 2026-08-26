#!/usr/bin/env bash
# Run this ON THE REMOTE A100 PC (not locally), with the `brats` conda env
# already active (conda activate brats — same as every other run_*.sh in
# this project). Pulls the latest code from GitHub via git_pull_remote.sh
# into ~/mcrsl_project (data/checkpoints/results/logs untouched), then runs
# the shuffled-quality-control experiment.
#
# ONE-TIME BOOTSTRAP (only needed the very first time — this script and
# git_pull_remote.sh don't exist on the remote until pulled at least once):
#   git clone https://github.com/sxdxde/Quality-Robust-Multimodal-Fusion-for-Skin-Lesion-Diagnosis-A-Benchmark-on-the-MCR-SL-Dataset.git ~/mcrsl_project_git
#   bash ~/mcrsl_project_git/scripts/git_pull_remote.sh
#
# Usage (every time after that):
#   conda activate brats
#   bash ~/mcrsl_project/scripts/remote_pull_and_run_shuffled_control.sh [DATA_DIR]
#   (DATA_DIR defaults to ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset)
set -euo pipefail

PROJECT_DIR="$HOME/mcrsl_project"
DATA_DIR="${1:-$PROJECT_DIR/data/raw/extracted/MCR-SL_dataset}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/git_pull_remote.sh"

echo "=== running shuffled-quality control (trust + hard_mining, 5-fold each) ==="
cd "$PROJECT_DIR"
bash scripts/run_shuffled_quality_control.sh "$DATA_DIR"

echo "=== DONE — pull results back to local with pull_remote_results.sh ==="
