"""Quick, no-checkpoint-loading way to read aggregated mean+-std for one or
more run_tags straight out of results_ledger.csv -- faster than running the
full robustness_analysis.py when all you need is a specific comparison.

Usage:
    python scripts/report_ledger_rows.py channel_gated channel_gated_swa
    python scripts/report_ledger_rows.py --ledger results/results_ledger.csv channel_gated_swa
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tags", nargs="+", help="one or more run_tag values (the ledger's 'variant' column) to report")
    parser.add_argument("--ledger", type=Path, default=Path("results/results_ledger.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.ledger)
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "sensitivity_malignant", "specificity", "auroc"]

    for tag in args.run_tags:
        sub = df[df["variant"] == tag]
        if sub.empty:
            print(f"\n=== {tag}: NOT FOUND in ledger ===")
            continue
        print(f"\n=== {tag} (n_folds={len(sub)}) ===")
        if len(sub) != 5:
            print(f"  WARNING: expected 5 fold rows, found {len(sub)} -- check for duplicate/missing entries before trusting this")
        print(sub[["fold"] + metric_cols].to_string(index=False))
        for col in metric_cols:
            print(f"  {col}: {sub[col].mean():.4f} +/- {sub[col].std():.4f}")


if __name__ == "__main__":
    main()
