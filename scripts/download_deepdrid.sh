#!/usr/bin/env bash
# STEP 0 of the DeepDRiD cross-dataset gate. Run ON THE REMOTE.
#
# This is a GATE, not a download-and-hope. It runs in two phases so that a
# repo that turns out to be LFS-quota-gated, externally hosted, or far too
# large costs seconds to discover rather than an hour of stalled transfer.
#
#   PHASE A (seconds, ~few MB): metadata-only clone with LFS smudge DISABLED.
#            Reveals the real directory structure, whether the image files are
#            actually in the repo or are LFS pointers / external links, and the
#            exact total LFS payload size. Then STOPS and reports.
#
#   PHASE B (only if you decide Phase A's numbers are acceptable): fetches the
#            actual image payload for the two folders the task needs.
#
# Usage:
#   bash scripts/download_deepdrid.sh          # Phase A only (default, safe)
#   bash scripts/download_deepdrid.sh --fetch  # Phase A then Phase B
#
# Time-box: the task allows 2 hours total for Step 0. Phase A should take
# well under a minute. Every phase prints elapsed time so the box is visible.
set -uo pipefail

REPO_URL="https://github.com/deepdrdoc/DeepDRiD.git"
DEST="${DEEPDRID_DIR:-$HOME/deepdrid}"
WANT_FETCH=0
[ "${1:-}" = "--fetch" ] && WANT_FETCH=1

T0=$(date +%s)
elapsed() { echo "[elapsed $(( $(date +%s) - T0 ))s]"; }
hr() { printf '=%.0s' {1..78}; echo; }

hr
echo "PHASE A — metadata-only inspection (no image payload downloaded yet)"
hr

if [ -d "$DEST/.git" ]; then
  echo "Existing checkout at $DEST — reusing it (delete it to start clean)."
else
  echo "Cloning metadata only into $DEST ..."
  echo "  GIT_LFS_SKIP_SMUDGE=1 => LFS files arrive as small pointer text files."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$REPO_URL" "$DEST"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo
    echo "*** GATE FAILED at Phase A: clone returned $rc. $(elapsed)"
    echo "*** Report this and fall back to the future-work framing. Do not retry blindly."
    exit 1
  fi
fi
echo "$(elapsed) clone done."

echo
echo "--- on-disk size of the metadata checkout ---"
du -sh "$DEST" 2>/dev/null

echo
echo "--- top-level contents ---"
ls -la "$DEST"

echo
echo "--- directory tree (2 levels, dirs only) ---"
find "$DEST" -maxdepth 2 -type d -not -path '*/.git*' | sed "s|$DEST|.|" | sort

echo
echo "--- every non-image file (the schema/readme surface) ---"
find "$DEST" -type f -not -path '*/.git/*' \
  \( -iname '*.csv' -o -iname '*.xlsx' -o -iname '*.docx' -o -iname '*.md' \
     -o -iname '*.txt' -o -iname '*.json' \) \
  | sed "s|$DEST|.|" | sort

echo
echo "--- is this repo actually using Git LFS? ---"
if [ -f "$DEST/.gitattributes" ]; then
  echo ".gitattributes found:"
  grep -i lfs "$DEST/.gitattributes" || echo "  (no lfs filters listed)"
else
  echo "no .gitattributes — repo likely does NOT use LFS"
fi

echo
echo "--- LFS payload inventory (this is the number that decides the gate) ---"
if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
  ( cd "$DEST" && git lfs ls-files -s 2>/dev/null | head -25 )
  echo "..."
  TOTAL=$( cd "$DEST" && git lfs ls-files -s 2>/dev/null | \
    sed -n 's/.*(\([0-9.]*\) *\([KMG]*B\)).*/\1 \2/p' | \
    awk '{u=$2; v=$1; if(u=="KB")v/=1024; else if(u=="GB")v*=1024; s+=v} END {printf "%.1f", s}' )
  N=$( cd "$DEST" && git lfs ls-files 2>/dev/null | wc -l )
  echo "LFS files: ${N:-0}   approximate total: ${TOTAL:-0} MB"
else
  echo "git-lfs is NOT installed in this environment."
  echo "If the repo needs LFS, that is a gate finding — report it rather than"
  echo "installing system packages on a shared machine."
fi

echo
echo "--- do the two folders the task needs actually exist here? ---"
for d in regular_fundus_images \
         regular_fundus_images/regular-fundus-training \
         regular_fundus_images/regular-fundus-validation \
         "Online-Challenge1&2-Evaluation"; do
  if [ -e "$DEST/$d" ]; then echo "  FOUND   $d"; else echo "  MISSING $d"; fi
done

echo
echo "--- sample: is a supposed image file real, or an LFS pointer? ---"
SAMPLE=$(find "$DEST" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
         -not -path '*/.git/*' 2>/dev/null | head -1)
if [ -n "$SAMPLE" ]; then
  echo "sample file: ${SAMPLE#$DEST/}  ($(du -h "$SAMPLE" | cut -f1))"
  if head -c 100 "$SAMPLE" | grep -q "git-lfs.github.com"; then
    echo "  -> LFS POINTER (not real image data). Phase B is required to get pixels."
  else
    echo "  -> looks like real image data already."
  fi
else
  echo "no image files present in the metadata checkout."
  echo "  -> images are probably hosted externally; check the README output next."
fi

hr
echo "PHASE A COMPLETE $(elapsed)"
hr
cat <<'NOTE'
DECIDE NOW, before any large transfer:

  * If the two required folders are MISSING and no images are present, the data
    is hosted off-GitHub. Read the extracted README (next command) for the real
    host, and judge whether it is reachable from this machine WITHOUT a separate
    account/approval. If it needs one -> GATE FAILS, stop, use future-work framing.

  * If LFS total size is large relative to the time left, or git-lfs is absent
    -> GATE FAILS. Report the actual number. Do not "try anyway".

Next, regardless: dump the schema and README text (fast, no download):

    python scripts/inspect_deepdrid.py

Only if the numbers above are clearly acceptable, fetch the payload:

    bash scripts/download_deepdrid.sh --fetch
NOTE

if [ "$WANT_FETCH" -eq 0 ]; then
  exit 0
fi

hr
echo "PHASE B — fetching actual payload for the two required folders"
hr
echo "Starting at $(elapsed). Watch this against the 2-hour Step 0 box."
cd "$DEST" || exit 1
git lfs pull --include="regular_fundus_images/regular-fundus-training/**" 2>&1
git lfs pull --include="regular_fundus_images/regular-fundus-validation/**" 2>&1
rc=$?
echo
echo "--- size after fetch ---"
du -sh "$DEST"
du -sh "$DEST"/regular_fundus_images/* 2>/dev/null
echo
if [ $rc -ne 0 ]; then
  echo "*** Phase B returned $rc — treat as a gate failure and report it. $(elapsed)"
  exit 1
fi
echo "PHASE B COMPLETE $(elapsed)"
echo "Now run:  python scripts/inspect_deepdrid.py"
