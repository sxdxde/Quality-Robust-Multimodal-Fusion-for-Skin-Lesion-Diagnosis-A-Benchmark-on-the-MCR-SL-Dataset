#!/usr/bin/env bash
# Push local code changes to the remote A100 machine.
# Excludes data/raw/ (the actual dataset) and checkpoints/, which should live
# remote-only (large, and the dataset is downloaded directly on the remote —
# never routed through this machine). Does NOT exclude all of data/, since
# data/*.py (loaders, schema validation, fold splitting) is real project code
# that lives in that same top-level directory.
#
# CRITICAL: also excludes results/ and logs/ — those are run OUTPUTS that
# only flow remote -> local via pull_remote_results.sh. Pushing them here
# would silently overwrite the remote's real accumulated results (e.g. the
# ledger) with a stale/empty local copy, which happened once already.
set -euo pipefail

REMOTE_HOST="cs24d0010@172.16.1.199"
REMOTE_PATH="~/mcrsl_project/"
LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)/"

rsync -avz --progress \
  --exclude 'data/raw/' \
  --exclude 'checkpoints/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.pt' \
  --exclude '*.npy' \
  "$LOCAL_PATH" "$REMOTE_HOST:$REMOTE_PATH"

echo "Synced code to $REMOTE_HOST:$REMOTE_PATH"
