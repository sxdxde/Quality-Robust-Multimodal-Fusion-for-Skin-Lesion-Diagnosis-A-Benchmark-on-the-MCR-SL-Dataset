#!/usr/bin/env bash
# Downloads the MCR-SL dataset directly on the remote A100 machine.
# Run this ON THE REMOTE only (never route the dataset through the local machine):
#   ssh cs24d0010@172.16.1.199
#   conda activate brats
#   bash ~/mcrsl_project/scripts/download_dataset.sh
set -euo pipefail

DATA_DIR="${1:-$HOME/mcrsl_project/data/raw}"
ZENODO_DOI="10.5281/zenodo.17306338"

mkdir -p "$DATA_DIR"

# zenodo_get isn't in the shared `brats` env (used by the sister TextBraTS project) —
# install it standalone, don't touch anything else in the env.
if ! python -c "import zenodo_get" >/dev/null 2>&1; then
  pip install zenodo_get
fi

echo "Downloading MCR-SL (DOI $ZENODO_DOI) into $DATA_DIR ..."
zenodo_get "$ZENODO_DOI" -o "$DATA_DIR"

echo
echo "Download complete. Contents of $DATA_DIR:"
ls -la "$DATA_DIR"

echo
echo "Next: verify the downloaded file/folder names (image folders + metadata"
echo "spreadsheets) match what the MCR-SL dataset paper describes BEFORE writing"
echo "any loader/schema-validation code."
