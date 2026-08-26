"""Shuffled-quality control (validity check, not a new search).

Tests whether `trust`/`hard_mining`'s results come from the quality signal's
actual information content, or from generic per-sample reweighting regardless
of what the weights mean. No training happens here — reads the 5-fold
checkpoints already saved by scripts/run_shuffled_quality_control.sh
(channel_gated_qweight_{trust_shuffled,hardmining_shuffled}_qualityFalse_fold{0..4}.pt),
plus the real (unshuffled) trust/hard_mining ledger rows and tercile CSVs
already produced by scripts/run_quality_adaptive_loss.sh.

Produces the deliverable table: real vs. shuffled-control balanced accuracy,
sensitivity, and high-minus-low quality-tercile accuracy gap, for both
mechanisms.

Usage:
    python scripts/shuffled_quality_control_analysis.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset

Outputs (under results/):
    oof_predictions_channel_gated_qweight_{trust,hardmining}_shuffled.csv
    confusion_matrix_channel_gated_qweight_{trust,hardmining}_shuffled.csv / .png
    robustness_quality_tercile_channel_gated_qweight_{trust,hardmining}_shuffled.csv / .png
    robustness_shuffled_quality_control.csv / .png   - the real-vs-shuffled comparison table
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import build_image_index, build_lesion_table, load_raw_tables
from robustness_analysis import (
    analysis_1_quality_stratified,
    collect_oof_predictions,
    get_fold_assignment,
    save_confusion_matrix,
)

# (real run_tag, shuffled run_tag, display mechanism name)
MECHANISMS = [
    ("channel_gated_qweight_trust", "channel_gated_qweight_trust_shuffled", "trust"),
    ("channel_gated_qweight_hard_mining", "channel_gated_qweight_hardmining_shuffled", "hard_mining"),
]


def tercile_gap(summary: pd.DataFrame) -> float:
    s = summary.set_index("tercile")["accuracy"]
    return float(s["high"] - s["low"])


def load_or_compute_tercile(run_tag: str, ckpt_tag: str, lesion_df, image_index_df, subjects_by_fold,
                             n_folds: int, checkpoint_dir: Path, image_size: int, batch_size: int,
                             device, results_dir: Path) -> pd.DataFrame:
    """Reuses the tercile CSV already on disk for the real (unshuffled) runs
    (produced by scripts/quality_adaptive_loss_analysis.py); computes fresh
    for the shuffled-control runs, which don't exist yet."""
    existing = results_dir / f"robustness_quality_tercile_{run_tag}.csv"
    if existing.exists():
        return pd.read_csv(existing)

    print(f"\n=== collecting OOF predictions: {run_tag} (ckpt_tag={ckpt_tag}) ===")
    oof_df = collect_oof_predictions(
        "channel_gated", False, lesion_df, image_index_df, subjects_by_fold, n_folds,
        checkpoint_dir, image_size, batch_size, device, ckpt_tag=ckpt_tag,
    )
    oof_df.to_csv(results_dir / f"oof_predictions_{run_tag}.csv", index=False)
    save_confusion_matrix(oof_df, run_tag, results_dir)
    return analysis_1_quality_stratified(oof_df, lesion_df, results_dir, run_tag)


def load_ledger_means(ledger_path: Path, run_tag: str) -> dict:
    df = pd.read_csv(ledger_path)
    rows = df[df["variant"] == run_tag]
    if len(rows) == 0:
        raise ValueError(f"no ledger rows found for run_tag={run_tag!r} in {ledger_path}")
    return {
        "balanced_acc": rows["balanced_accuracy"].mean(),
        "sensitivity": rows["sensitivity_malignant"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    images_root = args.images_root or args.data_dir
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ledger_path = args.results_dir / "results_ledger.csv"

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    rows = []
    for real_tag, shuffled_tag, mechanism in MECHANISMS:
        real_tercile = load_or_compute_tercile(
            real_tag, real_tag, lesion_df, image_index_df, subjects_by_fold, args.n_folds,
            args.checkpoint_dir, args.image_size, args.batch_size, device, args.results_dir,
        )
        shuffled_tercile = load_or_compute_tercile(
            shuffled_tag, f"{shuffled_tag}_qualityFalse", lesion_df, image_index_df, subjects_by_fold, args.n_folds,
            args.checkpoint_dir, args.image_size, args.batch_size, device, args.results_dir,
        )

        real_ledger = load_ledger_means(ledger_path, real_tag)
        shuffled_ledger = load_ledger_means(ledger_path, shuffled_tag)

        rows.append({
            "mechanism": f"{mechanism} (real)",
            "balanced_acc": real_ledger["balanced_acc"],
            "sensitivity": real_ledger["sensitivity"],
            "high_minus_low_tercile_gap": tercile_gap(real_tercile),
        })
        rows.append({
            "mechanism": f"{mechanism} (shuffled control)",
            "balanced_acc": shuffled_ledger["balanced_acc"],
            "sensitivity": shuffled_ledger["sensitivity"],
            "high_minus_low_tercile_gap": tercile_gap(shuffled_tercile),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.results_dir / "robustness_shuffled_quality_control.csv", index=False)

    print("\n=== Shuffled-quality control: real vs. shuffled comparison ===")
    print(comparison.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(comparison))
    colors = ["tab:blue" if "real" in m else "tab:red" for m in comparison["mechanism"]]
    ax[0].bar(x, comparison["balanced_acc"], color=colors)
    ax[0].set_xticks(x); ax[0].set_xticklabels(comparison["mechanism"], rotation=30, ha="right")
    ax[0].set_ylabel("Balanced accuracy")
    ax[0].set_title("Real vs. shuffled: balanced accuracy")

    ax[1].bar(x, comparison["high_minus_low_tercile_gap"], color=colors)
    ax[1].set_xticks(x); ax[1].set_xticklabels(comparison["mechanism"], rotation=30, ha="right")
    ax[1].set_ylabel("High - low tercile accuracy gap")
    ax[1].set_title("Real vs. shuffled: quality-tercile gap")
    ax[1].axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(args.results_dir / "robustness_shuffled_quality_control.png", dpi=150)
    plt.close(fig)

    print("\n=== DONE ===")
    print("Results: results/robustness_shuffled_quality_control.csv / .png")


if __name__ == "__main__":
    main()
