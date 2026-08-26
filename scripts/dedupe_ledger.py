"""Audited de-duplication of results_ledger.csv.

Context: this project has ALREADY been burned once by a careless dedup — a
`drop_duplicates(subset=['variant','fold'])` keyed on the wrong columns
silently deleted the plain `channel_gated` baseline's 5 rows, because
`variant` (really run_tag) is shared between the plain and auxiliary-head
configs and is only disambiguated by `quality_aware`. See FINDINGS.md's
"Data-integrity incident". This script is written to make that class of
mistake impossible:

- The identity key is (variant, quality_aware, fold) — all three, always.
- Dry-run by default. Nothing is written without --apply.
- --apply refuses to proceed unless a timestamped backup is written first.
- Rows are only ever removed when another row shares the full identity key;
  the surviving row is the most recent by timestamp (an interrupted-then-
  restarted run leaves a stale earlier row, so latest-wins is correct).
- Refuses to run if any duplicate group has DIFFERING metrics, since that
  means two genuinely different results collided and a human has to decide
  — it is not a mechanical duplicate.

Usage:
    python scripts/dedupe_ledger.py                  # dry run, report only
    python scripts/dedupe_ledger.py --apply          # back up, then rewrite
"""
import argparse
import datetime
import shutil
from pathlib import Path

import pandas as pd

KEY = ["variant", "quality_aware", "fold"]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "sensitivity_malignant", "specificity", "auroc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=Path("results/results_ledger.csv"))
    parser.add_argument("--apply", action="store_true", help="actually rewrite the ledger (default: dry run)")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                         help="max abs metric difference within a duplicate group still treated as identical")
    args = parser.parse_args()

    df = pd.read_csv(args.ledger)
    print(f"ledger: {args.ledger}  ({len(df)} rows, {df['variant'].nunique()} distinct run_tags)")

    dup_mask = df.duplicated(subset=KEY, keep=False)
    dups = df[dup_mask]

    if dups.empty:
        print("\nNo duplicate (variant, quality_aware, fold) rows found. Nothing to do.")
        return

    print(f"\n{len(dups)} rows across {len(dups.groupby(KEY))} duplicated identity keys:\n")
    conflicting = []
    for key, group in dups.groupby(KEY):
        spread = {m: float(group[m].max() - group[m].min()) for m in METRICS if m in group}
        worst = max(spread.values()) if spread else 0.0
        status = "identical" if worst <= args.tolerance else f"DIFFERING (max spread {worst:.4g})"
        print(f"  {key}  x{len(group)}  -> {status}")
        for _, r in group.iterrows():
            print(f"      {r['timestamp']}  bal_acc={r['balanced_accuracy']:.6f}  auroc={r['auroc']:.6f}")
        if worst > args.tolerance:
            conflicting.append(key)

    if conflicting:
        print(f"\nREFUSING TO PROCEED: {len(conflicting)} duplicate group(s) have differing metrics.")
        print("These are not mechanical duplicates — two different results collided under one")
        print("identity key. Resolve by hand (decide which run is authoritative) before deduping.")
        raise SystemExit(1)

    deduped = df.sort_values("timestamp").drop_duplicates(subset=KEY, keep="last")
    removed = len(df) - len(deduped)
    print(f"\nWould remove {removed} stale row(s); {len(deduped)} would remain.")

    print("\nPer-run_tag fold counts after dedup (anything != 5 needs a look):")
    counts = deduped.groupby(["variant", "quality_aware"]).size().reset_index(name="n_folds")
    for _, r in counts.iterrows():
        flag = "" if r["n_folds"] == 5 else "   <-- CHECK"
        print(f"  {r['variant']:<52} quality_aware={str(r['quality_aware']):<5} n={r['n_folds']}{flag}")

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to back up and rewrite.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.ledger.with_name(f"{args.ledger.stem}.backup_{stamp}.csv")
    shutil.copy2(args.ledger, backup)
    print(f"\nBacked up original to {backup}")

    deduped.to_csv(args.ledger, index=False)
    print(f"Rewrote {args.ledger} ({len(deduped)} rows).")


if __name__ == "__main__":
    main()
