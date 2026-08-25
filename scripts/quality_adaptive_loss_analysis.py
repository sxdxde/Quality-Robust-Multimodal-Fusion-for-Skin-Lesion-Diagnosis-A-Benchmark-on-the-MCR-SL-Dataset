"""Step 3 of QUALITY_ADAPTIVE_LOSS_TASK.md: extends the existing
quality-tercile robustness analysis to the two quality-adaptive loss
reweighting variants (trust / hard_mining), without rebuilding it.

No training happens here — reads the 5-fold checkpoints already saved by
scripts/run_quality_adaptive_loss.sh (channel_gated_qweight_{trust,hard_mining}
_qualityFalse_fold{0..4}.pt) plus the pre-existing plain channel_gated and
aux-head quality_aware checkpoints/tercile CSVs from the core run.

Produces the paper's actual open-question deliverable: one table comparing
the high-minus-low quality-tercile accuracy gap across all four mechanisms
(plain / auxiliary quality-prediction head / loss-reweight-trust /
loss-reweight-hard-mining).

Usage:
    python scripts/quality_adaptive_loss_analysis.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset

Outputs (under results/):
    oof_predictions_channel_gated_qweight_trust.csv / _hard_mining.csv
    confusion_matrix_channel_gated_qweight_{trust,hard_mining}.csv / .png
    aux_9class_channel_gated_qweight_{trust,hard_mining}.csv
    robustness_quality_tercile_channel_gated_qweight_{trust,hard_mining}.csv / .png
    robustness_quality_reweighting_comparison.csv / .png   - the 4-way table
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
    save_aux_table,
    save_confusion_matrix,
)

RUN_TAGS = ["channel_gated_qweight_trust", "channel_gated_qweight_hard_mining"]
MODE_LABELS = {
    "channel_gated_qualityFalse": "plain (no quality-awareness)",
    "channel_gated_qualityTrue": "auxiliary quality-prediction head",
    "channel_gated_qweight_trust": "loss reweight: trust (down-weight low-quality)",
    "channel_gated_qweight_hard_mining": "loss reweight: hard-mining (up-weight low-quality)",
}


def load_existing_tercile_summary(results_dir: Path, cfg_tag: str) -> pd.DataFrame:
    path = results_dir / f"robustness_quality_tercile_{cfg_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python robustness_analysis.py` first to produce the "
            f"core config's tercile summaries (plain and auxiliary-head channel_gated) before "
            f"running this script."
        )
    return pd.read_csv(path)


def high_minus_low_gap(summary: pd.DataFrame) -> float:
    s = summary.set_index("tercile")["accuracy"]
    return float(s["high"] - s["low"])


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

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    tercile_summaries = {
        "channel_gated_qualityFalse": load_existing_tercile_summary(args.results_dir, "channel_gated_qualityFalse"),
        "channel_gated_qualityTrue": load_existing_tercile_summary(args.results_dir, "channel_gated_qualityTrue"),
    }

    for run_tag in RUN_TAGS:
        ckpt_tag = f"{run_tag}_qualityFalse"
        print(f"\n=== collecting OOF predictions: {run_tag} ===")
        oof_df = collect_oof_predictions(
            "channel_gated", False, lesion_df, image_index_df, subjects_by_fold, args.n_folds,
            args.checkpoint_dir, args.image_size, args.batch_size, device, ckpt_tag=ckpt_tag,
        )
        oof_df.to_csv(args.results_dir / f"oof_predictions_{run_tag}.csv", index=False)

        save_confusion_matrix(oof_df, run_tag, args.results_dir)
        save_aux_table(oof_df, run_tag, args.results_dir)
        tercile_summaries[run_tag] = analysis_1_quality_stratified(oof_df, lesion_df, args.results_dir, run_tag)

    rows = []
    gap_rows = []
    for cfg_tag, summary in tercile_summaries.items():
        gap = high_minus_low_gap(summary)
        gap_rows.append({"mode": cfg_tag, "label": MODE_LABELS[cfg_tag], "high_minus_low_accuracy_gap": gap})
        for _, r in summary.iterrows():
            rows.append({"mode": cfg_tag, "label": MODE_LABELS[cfg_tag], **r.to_dict()})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.results_dir / "robustness_quality_reweighting_comparison.csv", index=False)

    gap_summary = pd.DataFrame(gap_rows)
    gap_summary.to_csv(args.results_dir / "robustness_quality_reweighting_gaps.csv", index=False)

    print("\n=== Four-way quality-tercile comparison (results/robustness_quality_reweighting_comparison.csv) ===")
    print(comparison.to_string(index=False))
    print("\n=== High-minus-low tercile accuracy gap, all four mechanisms (results/robustness_quality_reweighting_gaps.csv) ===")
    print(gap_summary.to_string(index=False))

    order = ["channel_gated_qualityFalse", "channel_gated_qualityTrue"] + RUN_TAGS
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    width = 0.2
    for i, cfg_tag in enumerate(order):
        s = tercile_summaries[cfg_tag].set_index("tercile")["accuracy"].reindex(["low", "mid", "high"])
        ax.bar(x + (i - 1.5) * width, s.values, width, label=MODE_LABELS[cfg_tag])
    ax.set_xticks(x); ax.set_xticklabels(["low", "mid", "high"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Quality-tercile accuracy: plain vs. aux-head vs. loss-reweighted (trust/hard-mining)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results_dir / "robustness_quality_reweighting_comparison.png", dpi=150)
    plt.close(fig)

    print("\n=== DONE ===")
    print("Re-run `python robustness_analysis.py` afterward if you also want summary_table.csv refreshed with these two new run_tags.")


if __name__ == "__main__":
    main()
