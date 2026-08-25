"""One-off recovery script: re-appends the plain channel_gated (quality_aware=
False) baseline's 5 ledger rows, which were accidentally wiped by a flawed
drop_duplicates(subset=['variant','fold']) dedup that didn't account for the
ledger's 'variant' column actually being run_tag, and channel_gated's plain
and auxiliary-head (quality_aware=True) runs sharing that same run_tag.

Does NOT retrain anything — re-evaluates the existing, untouched
checkpoints/channel_gated_qualityFalse_fold{0..4}.pt files and appends
their metrics back to results_ledger.csv, exactly reproducing what
train.py's run_cv would have logged originally.

Usage:
    python scripts/recover_plain_baseline_ledger.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import build_image_index, build_lesion_table, load_raw_tables
from evaluate import append_to_ledger, compute_binary_metrics
from robustness_analysis import collect_oof_predictions, get_fold_assignment


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ledger_path = str(args.results_dir / "results_ledger.csv")

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    ckpts = sorted(args.checkpoint_dir.glob("channel_gated_qualityFalse_fold*.pt"))
    print(f"Found checkpoints: {[c.name for c in ckpts]}")
    if len(ckpts) != args.n_folds:
        raise SystemExit(
            f"expected {args.n_folds} checkpoints (channel_gated_qualityFalse_fold0..{args.n_folds-1}.pt), "
            f"found {len(ckpts)} — stopping before writing anything to the ledger."
        )

    oof_df = collect_oof_predictions(
        "channel_gated", False, lesion_df, image_index_df, subjects_by_fold, args.n_folds,
        args.checkpoint_dir, args.image_size, args.batch_size, device,
    )

    for test_fold in range(args.n_folds):
        fold_df = oof_df[(oof_df["fold"] == test_fold) & (oof_df["has_binary_label"])]
        metrics = compute_binary_metrics(
            fold_df["binary_label"].astype(int).to_numpy(),
            fold_df["pred_label"].to_numpy(),
            fold_df["pred_prob"].to_numpy(),
        )
        append_to_ledger(
            ledger_path, "channel_gated", False, test_fold, args.n_folds, args.seed, metrics,
            notes="recovered after accidental ledger dedup bug (2026-08) — re-evaluated from "
                  "existing checkpoints, no retraining; values should match the original run.",
        )
        print(f"[fold {test_fold}] re-appended: accuracy={metrics['accuracy']:.4f} "
              f"balanced_accuracy={metrics['balanced_accuracy']:.4f} auroc={metrics['auroc']:.4f}")

    print("\nDone. Now run `python robustness_analysis.py` again to confirm summary_table.csv "
          "shows channel_gated (quality_aware=False) with n_folds=5 again.")


if __name__ == "__main__":
    main()
