"""Free (no-retraining) check: is balanced accuracy being left on the table by
the fixed 0.5 probability threshold used everywhere else in this project?

For each fold, picks the classification threshold that maximizes
sensitivity+specificity (Youden's J statistic) on that fold's VAL split only,
then applies that fixed threshold to the held-out TEST fold and recomputes
metrics. No test-fold peeking: threshold selection never sees test-fold
labels, matching every other checkpoint-selection decision in this project.

Run against any already-trained run_tag's checkpoints (default: the two
quality-adaptive-loss variants, since that's what we're trying to push past
0.836 right now).

Usage:
    python scripts/optimal_threshold_eval.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset \
        --run-tags channel_gated_qweight_hard_mining channel_gated_qweight_trust channel_gated_sam_adamw_tta channel_gated
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MCRSLDataset, collate_fn, fit_numeric_stats
from data.loader import build_image_index, build_lesion_table, load_raw_tables
from evaluate import aggregate_fold_metrics, compute_binary_metrics
from models.model import MCRSLModel
from robustness_analysis import get_fold_assignment
from train import move_batch


@torch.no_grad()
def get_probs(model, loader, device, tta: bool = False):
    model.eval()
    probs, labels, has_label = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])
        p = torch.sigmoid(out["binary_logits"])
        if tta:
            flipped = torch.flip(batch["image"], dims=[-1])
            out_flip = model(flipped, batch["categorical"], batch["numerical"], batch["numerical_missing"])
            p = (p + torch.sigmoid(out_flip["binary_logits"])) / 2
        probs.append(p.cpu().numpy())
        labels.append(batch["binary_label"].cpu().numpy())
        has_label.append(batch["has_binary_label"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(has_label)


def youden_threshold(y_true, y_score) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-tags", nargs="+", default=[
        "channel_gated_qweight_hard_mining", "channel_gated_qweight_trust",
        "channel_gated_sam_adamw_tta", "channel_gated",
    ])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    images_root = args.images_root or args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    all_rows = []
    for run_tag in args.run_tags:
        ckpt_tag = f"{run_tag}_qualityFalse"
        fold_metrics_default, fold_metrics_optimal = [], []
        thresholds_used = []

        for test_fold in range(args.n_folds):
            val_fold = (test_fold + 1) % args.n_folds
            train_folds = [f for f in range(args.n_folds) if f not in (test_fold, val_fold)]
            train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
            val_subjects = subjects_by_fold[val_fold]
            test_subjects = subjects_by_fold[test_fold]

            numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
            val_ds = MCRSLDataset(lesion_df, image_index_df, val_subjects, numeric_stats, args.image_size, "val", False, verbose=False)
            test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, args.image_size, "test", False, verbose=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

            ckpt_path = args.checkpoint_dir / f"{ckpt_tag}_fold{test_fold}.pt"
            if not ckpt_path.exists():
                print(f"[{run_tag}] missing checkpoint {ckpt_path}, skipping this run_tag")
                fold_metrics_default = fold_metrics_optimal = None
                break

            model = MCRSLModel(variant="channel_gated", quality_aware=False).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

            val_probs, val_labels, val_has_label = get_probs(model, val_loader, device)
            val_probs, val_labels = val_probs[val_has_label], val_labels[val_has_label]
            threshold = youden_threshold(val_labels.astype(int), val_probs)
            thresholds_used.append(threshold)

            test_probs, test_labels, test_has_label = get_probs(model, test_loader, device)
            test_probs, test_labels = test_probs[test_has_label], test_labels[test_has_label]

            default_metrics = compute_binary_metrics(test_labels.astype(int), (test_probs >= 0.5).astype(int), test_probs)
            optimal_metrics = compute_binary_metrics(test_labels.astype(int), (test_probs >= threshold).astype(int), test_probs)
            fold_metrics_default.append(default_metrics)
            fold_metrics_optimal.append(optimal_metrics)

            print(f"[{run_tag} fold {test_fold}] val-optimal threshold={threshold:.3f} | "
                  f"default(0.5) bal_acc={default_metrics['balanced_accuracy']:.4f} -> "
                  f"optimal bal_acc={optimal_metrics['balanced_accuracy']:.4f}")

        if fold_metrics_default is None:
            continue

        agg_default = aggregate_fold_metrics(fold_metrics_default)
        agg_optimal = aggregate_fold_metrics(fold_metrics_optimal)
        print(f"\n=== {run_tag}: default(0.5) vs. val-optimal-threshold, mean +/- std over {args.n_folds} folds ===")
        print(f"  balanced_accuracy: {agg_default['balanced_accuracy_mean']:.4f}+/-{agg_default['balanced_accuracy_std']:.4f} "
              f"-> {agg_optimal['balanced_accuracy_mean']:.4f}+/-{agg_optimal['balanced_accuracy_std']:.4f}")
        print(f"  sensitivity:       {agg_default['sensitivity_malignant_mean']:.4f} -> {agg_optimal['sensitivity_malignant_mean']:.4f}")
        print(f"  specificity:       {agg_default['specificity_mean']:.4f} -> {agg_optimal['specificity_mean']:.4f}")
        print(f"  thresholds used per fold: {[round(t, 3) for t in thresholds_used]}\n")

        all_rows.append({
            "run_tag": run_tag,
            "balanced_accuracy_default": agg_default["balanced_accuracy_mean"],
            "balanced_accuracy_optimal_threshold": agg_optimal["balanced_accuracy_mean"],
            "sensitivity_default": agg_default["sensitivity_malignant_mean"],
            "sensitivity_optimal_threshold": agg_optimal["sensitivity_malignant_mean"],
            "specificity_default": agg_default["specificity_mean"],
            "specificity_optimal_threshold": agg_optimal["specificity_mean"],
            "mean_threshold_used": float(np.mean(thresholds_used)),
        })

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.results_dir / "optimal_threshold_comparison.csv", index=False)
    print("=== Summary (results/optimal_threshold_comparison.csv) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
