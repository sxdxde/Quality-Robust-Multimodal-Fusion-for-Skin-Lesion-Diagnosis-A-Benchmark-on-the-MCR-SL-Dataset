#!/usr/bin/env bash
# Pull results and logs back from the remote A100 machine.
# Deliberately does NOT pull raw data or full checkpoints by default — those are large
# and usually don't need to leave the remote. Pull a specific checkpoint manually if needed.
set -euo pipefail

REMOTE_HOST="cs24d0010@172.16.1.199"
REMOTE_PATH="~/mcrsl_project/"
LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)/"

mkdir -p "$LOCAL_PATH/results" "$LOCAL_PATH/logs"

rsync -avz --progress "$REMOTE_HOST:$REMOTE_PATH/results/" "$LOCAL_PATH/results/"
rsync -avz --progress "$REMOTE_HOST:$REMOTE_PATH/logs/" "$LOCAL_PATH/logs/"

echo "Pulled results and logs from $REMOTE_HOST:$REMOTE_PATH"
