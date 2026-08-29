"""Permutation test for the shuffled-quality control, plus its figure.

The originally reported control used ONE permutation. A single draw from the
null cannot establish that the real result is unusual, only that one shuffle
happened to score lower. This aggregates N independent permutations
(scripts/run_permutation_test.sh) into a null DISTRIBUTION, places the real
`hard_mining` result against it, and reports an empirical one-sided p-value

    p = (#{null >= real} + 1) / (N + 1)

which is the standard conservative estimator (it can never report p = 0).

Outputs a single-column IEEE figure: the null histogram with the real result
and the plain baseline marked, so the causal claim is visual rather than a
table row.

Usage:
    python scripts/permutation_analysis.py
    python scripts/permutation_analysis.py --metric sensitivity_malignant
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_TAG = "channel_gated_qweight_hard_mining"
BASE_TAG = "channel_gated"
PERM_PREFIX = "channel_gated_qweight_hm_perm"
NICE = {"balanced_accuracy": "Balanced accuracy",
        "sensitivity_malignant": "Malignant sensitivity",
        "auroc": "AUROC"}


def fold_mean(rows, tag, metric):
    """Mean over the 5 folds, taking the latest row per fold so an
    interrupted-and-restarted run cannot double-count a fold."""
    sel = [r for r in rows if r["variant"] == tag and r["quality_aware"] == "False"]
    latest = {}
    for r in sorted(sel, key=lambda a: a["timestamp"]):
        latest[r["fold"]] = r
    if len(latest) != 5:
        return None
    return st.mean(float(v[metric]) for v in latest.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("results/results_ledger.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", default="balanced_accuracy", choices=list(NICE))
    ap.add_argument("--max-perms", type=int, default=200)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.ledger)))
    real = fold_mean(rows, REAL_TAG, args.metric)
    base = fold_mean(rows, BASE_TAG, args.metric)
    if real is None or base is None:
        raise SystemExit("real or baseline run incomplete in the ledger")

    nulls = []
    for i in range(1, args.max_perms + 1):
        v = fold_mean(rows, f"{PERM_PREFIX}{i}", args.metric)
        if v is not None:
            nulls.append((i, v))
    if not nulls:
        raise SystemExit(f"no complete permutations found (prefix {PERM_PREFIX})")

    vals = [v for _, v in nulls]
    n = len(vals)
    ge = sum(1 for v in vals if v >= real)
    p = (ge + 1) / (n + 1)
    z = (real - st.mean(vals)) / st.pstdev(vals) if st.pstdev(vals) > 0 else float("nan")

    print(f"metric        : {args.metric}")
    print(f"permutations  : {n}")
    print(f"real          : {real:.4f}")
    print(f"plain baseline: {base:.4f}")
    print(f"null          : mean={st.mean(vals):.4f}  sd={st.pstdev(vals):.4f} "
          f"min={min(vals):.4f} max={max(vals):.4f}")
    print(f"nulls >= real : {ge}/{n}")
    print(f"empirical p   : {p:.4f}   (floor at this N is {1/(n+1):.4f})")
    print(f"real is {z:.2f} sd above the null mean")

    out_csv = args.results_dir / "permutation_test.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n_permutations", "real", "plain_baseline",
                    "null_mean", "null_sd", "null_min", "null_max",
                    "n_null_ge_real", "empirical_p", "z_vs_null"])
        w.writerow([args.metric, n, f"{real:.4f}", f"{base:.4f}",
                    f"{st.mean(vals):.4f}", f"{st.pstdev(vals):.4f}",
                    f"{min(vals):.4f}", f"{max(vals):.4f}", ge, f"{p:.4f}", f"{z:.2f}"])
        w.writerow([])
        w.writerow(["permutation_seed", args.metric])
        for i, v in nulls:
            w.writerow([i, f"{v:.4f}"])
    print(f"wrote {out_csv}")

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.hist(vals, bins=max(6, min(12, n // 2)), color="0.72", edgecolor="white",
            label="shuffled ratings (n=%d)" % n)
    ax.axvline(base, color="tab:blue", ls=":", lw=1.4,
               label=f"plain baseline ({base:.3f})")
    ax.axvline(real, color="firebrick", lw=1.8,
               label=f"real ratings ({real:.3f})")
    ax.set_xlabel(NICE[args.metric], fontsize=7.5)
    ax.set_ylabel("permutations", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.8, loc="upper left", framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    stem = args.results_dir / f"permutation_hist_{args.metric}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf (+ .png)")


if __name__ == "__main__":
    main()
