"""Parallel, resumable DeepDRiD fetcher — engineered around the measured
failure mode of `git clone` on this machine.

WHY THIS EXISTS
    `git clone` failed twice here: RPC failed (result=92) / early EOF /
    index-pack failed, after 50 minutes. Diagnosis:
      * single-stream throughput is 40-64 KB/s (to kernel.org too, so it is a
        general network constraint, not GitHub-specific)
      * git transfers are SINGLE-STREAM, so they cannot exceed that
      * one long-lived connection is fragile here and dies before finishing
      * 16 parallel streams reach ~700 KB/s => it is PER-CONNECTION
        throttling, not an aggregate bandwidth cap
      * the GitHub archive endpoint ignores HTTP Range, so a tarball cannot
        be chunked either

    So: fetch the files INDIVIDUALLY, in PARALLEL, from raw.githubusercontent
    .com. Each request is short-lived (dodges the early-EOF death) and many
    run at once (dodges the per-connection cap). Fully resumable, so an
    interruption costs only the files still missing.

STDLIB ONLY - deliberately no pip install into the shared `brats` conda env.

USAGE
    # 1. See the manifest and TOTAL SIZE first (no bulk download):
    python scripts/fetch_deepdrid_parallel.py --dry-run

    # 2. If the size is acceptable, fetch:
    python scripts/fetch_deepdrid_parallel.py --workers 16

    # Resume after any interruption - just run it again, same command.
    # Already-complete files are skipped by size check.

    # If the GitHub API is rate-limited (shared campus IP), either wait for
    # the hourly reset or supply a token (any classic PAT, no scopes needed):
    GITHUB_TOKEN=ghp_xxx python scripts/fetch_deepdrid_parallel.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OWNER = "deepdrdoc"
REPO = "DeepDRiD"
RAW = "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
API_TREE = "https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

# Only these subtrees are needed for the cross-dataset experiment. Everything
# else in the repo (ultra-widefield images especially) is not used, and
# skipping it is the single biggest saving available.
DEFAULT_PREFIXES = [
    "regular_fundus_images/regular-fundus-training",
    "regular_fundus_images/regular-fundus-validation",
]

_print_lock = threading.Lock()


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return "{:.1f} {}".format(n, unit)
        n /= 1024.0
    return "{:.1f} TB".format(n)


def api_get(url, token=None, timeout=60):
    headers = {"User-Agent": "deepdrid-fetch", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_manifest(branch, token):
    """One API call returns the ENTIRE file list with sizes. This is also what
    finally answers the gate's 'how big is it, actually?' question."""
    url = API_TREE.format(owner=OWNER, repo=REPO, branch=branch)
    try:
        data = api_get(url, token)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise SystemExit(
                "\nGitHub API rate limit hit (60/hr per IP unauthenticated, and this is a\n"
                "shared campus IP).\n\n"
                "  Option A: wait for the hourly reset and re-run.\n"
                "  Option B: supply any classic personal access token (no scopes needed):\n"
                "            GITHUB_TOKEN=ghp_xxx python scripts/fetch_deepdrid_parallel.py --dry-run\n"
                "            That raises the limit to 5000/hr and needs exactly ONE call.\n"
            )
        if e.code == 404:
            raise SystemExit("Branch '{}' not found. Try --branch main (or master).".format(branch))
        raise
    if data.get("truncated"):
        print("  WARNING: the API truncated this tree; some files may be missing from the manifest.")
    return [e for e in data.get("tree", []) if e.get("type") == "blob"]


def download_one(entry, dest_root, branch, retries, timeout):
    """Fetch one file. Returns (path, status, bytes). Resumable by size check."""
    path = entry["path"]
    size = entry.get("size", 0)
    out = dest_root / path

    if out.exists() and out.stat().st_size == size and size > 0:
        return (path, "skip", 0)

    url = RAW.format(owner=OWNER, repo=REPO, branch=branch,
                     path=urllib.parse.quote(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "deepdrid-fetch"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            got = tmp.stat().st_size
            if size and got != size:
                raise IOError("size mismatch: expected {} got {}".format(size, got))
            tmp.replace(out)
            return (path, "ok", got)
        except Exception as e:                     # noqa: BLE001 - want every failure retried
            last_err = e
            try:
                tmp.unlink()
            except OSError:
                pass
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 15))  # backoff, capped
    return (path, "FAIL: {}".format(last_err), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=Path.home() / "deepdrid")
    p.add_argument("--branch", default="master")
    p.add_argument("--workers", type=int, default=16,
                   help="parallel connections; 16 measured ~700 KB/s here vs 64 KB/s single")
    p.add_argument("--prefixes", nargs="*", default=DEFAULT_PREFIXES,
                   help="only fetch paths starting with these; pass with no values to fetch ALL")
    p.add_argument("--dry-run", action="store_true", help="show manifest + total size, download nothing")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    print("=" * 74)
    print("DeepDRiD parallel fetch  (branch={}, workers={})".format(args.branch, args.workers))
    print("=" * 74)
    print("fetching file manifest via one API call{}...".format(" [authenticated]" if token else ""))

    blobs = get_manifest(args.branch, token)
    print("  repo contains {} files total".format(len(blobs)))

    if args.prefixes:
        wanted = [b for b in blobs if any(b["path"].startswith(pre) for pre in args.prefixes)]
        print("  filtering to prefixes: {}".format(args.prefixes))
    else:
        wanted = blobs
        print("  NO prefix filter - fetching the entire repository")

    if not wanted:
        print("\n  No files matched. Top-level directories present:")
        tops = sorted({b["path"].split("/")[0] for b in blobs})
        for t in tops[:40]:
            print("    ", t)
        raise SystemExit("\nAdjust --prefixes to match the real layout above.")

    total = sum(b.get("size", 0) for b in wanted)
    have = sum(1 for b in wanted
               if (args.dest / b["path"]).exists()
               and (args.dest / b["path"]).stat().st_size == b.get("size", 0))
    remaining = sum(b.get("size", 0) for b in wanted
                    if not ((args.dest / b["path"]).exists()
                            and (args.dest / b["path"]).stat().st_size == b.get("size", 0)))

    print()
    print("  files selected : {}".format(len(wanted)))
    print("  already have   : {}".format(have))
    print("  TOTAL SIZE     : {}".format(human(total)))
    print("  still to fetch : {}".format(human(remaining)))
    print()
    for rate, label in ((64 * 1024, "single-stream 64 KB/s (what git got)"),
                        (700 * 1024, "16 parallel ~700 KB/s (measured)")):
        secs = remaining / rate if rate else 0
        print("  ETA @ {:<38}: {:.0f} min".format(label, secs / 60))

    # A few of the largest files, so an unexpectedly huge payload is obvious.
    print("\n  largest selected files:")
    for b in sorted(wanted, key=lambda x: -x.get("size", 0))[:5]:
        print("    {:>10}  {}".format(human(b.get("size", 0)), b["path"]))

    if args.dry_run:
        print("\n--dry-run: nothing downloaded. Re-run without it to fetch.")
        return

    if remaining == 0:
        print("\nAll selected files already present. Nothing to do.")
        return

    print("\nstarting parallel fetch into {} ...".format(args.dest))
    args.dest.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = ok = skipped = failed = 0
    got_bytes = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, b, args.dest, args.branch,
                          args.retries, args.timeout): b for b in wanted}
        for fut in as_completed(futs):
            path, status, nbytes = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                got_bytes += nbytes
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                failures.append((path, status))

            if done % 25 == 0 or done == len(wanted):
                el = time.time() - t0
                rate = got_bytes / el if el > 0 else 0
                eta = (remaining - got_bytes) / rate if rate > 0 else 0
                with _print_lock:
                    print("  [{}/{}] ok={} skip={} fail={} | {} @ {}/s | ETA {:.0f} min"
                          .format(done, len(wanted), ok, skipped, failed,
                                  human(got_bytes), human(rate), eta / 60))

    el = time.time() - t0
    print("\n" + "=" * 74)
    print("fetched {} in {:.1f} min  (avg {}/s)".format(
        human(got_bytes), el / 60, human(got_bytes / el if el else 0)))
    print("  ok={}  already-had={}  failed={}".format(ok, skipped, failed))
    if failures:
        print("\n  {} file(s) failed - RE-RUN THE SAME COMMAND to retry only these:".format(len(failures)))
        for path, status in failures[:10]:
            print("    {}  <- {}".format(path, status))
        if len(failures) > 10:
            print("    ... and {} more".format(len(failures) - 10))
        print("\n  (the fetch is resumable; completed files are skipped by size check)")
    else:
        print("\n  ALL FILES COMPLETE.")
        print("  Next:  python scripts/inspect_deepdrid.py --root {}".format(args.dest))
    print("=" * 74)


if __name__ == "__main__":
    main()
