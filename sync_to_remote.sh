#!/usr/bin/env bash
# Push local code changes to the remote A100 machine.
# Excludes data/ and checkpoints/, which should live remote-only (large, and the
# dataset is downloaded directly on the remote — never routed through this machine).
set -euo pipefail

REMOTE_HOST="cs24d0010@172.16.1.199"
REMOTE_PATH="~/mcrsl_project/"
LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)/"

rsync -avz --progress \
  --exclude 'data/' \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.pt' \
  --exclude '*.npy' \
  "$LOCAL_PATH" "$REMOTE_HOST:$REMOTE_PATH"

echo "Synced code to $REMOTE_HOST:$REMOTE_PATH"
