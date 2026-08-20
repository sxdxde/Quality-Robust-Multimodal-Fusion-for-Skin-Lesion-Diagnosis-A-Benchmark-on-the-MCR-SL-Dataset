"""Re-evaluates an already-trained checkpoint set with eval-time-only
options that don't require retraining — test-time augmentation (flip
averaging) and multi-image averaging (using every dermoscopic image per
lesion instead of just diagnosis_image_id). Each combination is logged as
its own run_tag in the ledger, derived from --base-run-tag.

No training happens here — pure inference on the 5 existing fold
checkpoints for --base-run-tag (defaults to the plain non-quality-aware
channel_gated baseline; pass e.g. --base-run-tag channel_gated_sam_adamw to
apply the same eval-time tricks on top of the SAM run instead).

Usage:
    python scripts/reeval_eval_time_options.py --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
    python scripts/reeval_eval_time_options.py --base-run-tag channel_gated_sam_adamw --data-dir ...
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MCRSLDataset, collate_fn, fit_numeric_stats
from data.loader import build_image_index, build_lesion_table, load_raw_tables
from evaluate import aggregate_fold_metrics, append_to_ledger
from models.model import MCRSLModel
from robustness_analysis import get_fold_assignment
from train import metrics_from_predictions, predict

ARCHITECTURE_VARIANT = "channel_gated"  # all eval-time-option runs so far are on this architecture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--base-run-tag", default="channel_gated", help="run_tag whose checkpoints to re-evaluate; checkpoint files are <base-run-tag>_qualityFalse_fold<n>.pt")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    images_root = args.images_root or args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_checkpoint_tag = f"{args.base_run_tag}_qualityFalse"
    run_configs = [
        (f"{args.base_run_tag}_tta", {"tta": True, "multi_image_eval": False}),
        (f"{args.base_run_tag}_multiimage", {"tta": False, "multi_image_eval": True}),
        (f"{args.base_run_tag}_tta_multiimage", {"tta": True, "multi_image_eval": True}),
    ]

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    ledger_path = str(args.results_dir / "results_ledger.csv")

    for run_tag, opts in run_configs:
        print(f"\n=== {run_tag} (tta={opts['tta']}, multi_image_eval={opts['multi_image_eval']}) ===")
        fold_metrics = []
        for test_fold in range(args.n_folds):
            val_fold = (test_fold + 1) % args.n_folds
            train_folds = [f for f in range(args.n_folds) if f not in (test_fold, val_fold)]
            train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
            test_subjects = subjects_by_fold[test_fold]

            numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
            test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, args.image_size, "test",
                                    False, verbose=False, multi_image_eval=opts["multi_image_eval"])
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

            ckpt_path = args.checkpoint_dir / f"{base_checkpoint_tag}_fold{test_fold}.pt"
            model = MCRSLModel(variant=ARCHITECTURE_VARIANT, quality_aware=False).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

            df = predict(model, test_loader, device, tta=opts["tta"])
            metrics = metrics_from_predictions(df, aggregate_by_lesion=opts["multi_image_eval"])
            fold_metrics.append(metrics)
            append_to_ledger(ledger_path, run_tag, False, test_fold, args.n_folds, args.seed, metrics,
                              notes=f"eval-time only, reusing {base_checkpoint_tag} checkpoints, no retraining")
            print(f"[{run_tag} fold {test_fold}] accuracy={metrics['accuracy']:.4f} "
                  f"balanced_accuracy={metrics['balanced_accuracy']:.4f} auroc={metrics['auroc']:.4f}")

        agg = aggregate_fold_metrics(fold_metrics)
        print(f"[{run_tag}] AGGREGATE: accuracy={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f} "
              f"balanced_accuracy={agg['balanced_accuracy_mean']:.4f}+/-{agg['balanced_accuracy_std']:.4f} "
              f"auroc={agg['auroc_mean']:.4f}+/-{agg['auroc_std']:.4f}")


if __name__ == "__main__":
    main()
