"""Step 2 of the close-out task: prediction-level checkpoint ensembling.

Mechanistically distinct from SWA (weight-space averaging, already tried and
negative when stacked on hard_mining) and multi-image TTA (image-space
averaging): this averages PREDICTED PROBABILITIES across the top-k
val-balanced-accuracy epoch checkpoints per fold, then thresholds once.
Applied to `channel_gated` (plain) and `channel_gated_qweight_trust` ONLY,
per the task's explicit scope -- not hard_mining or any hard_mining-derived
config (three independent stacking attempts on hard_mining already failed).

Requires scripts/run_topk_checkpoint_rerun.sh to have already been run (that
script trains fresh channel_gated_topk / channel_gated_qweight_trust_topk
configs with per-epoch top-k checkpoint saving enabled -- the original runs
never saved per-epoch checkpoints, only the single best).

Usage:
    python scripts/checkpoint_ensemble_eval.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset

Outputs (under results/):
    oof_predictions_{config}_topk_ensemble.csv          - per-lesion OOF predictions
    robustness_quality_tercile_{config}_topk_ensemble.csv/.png  - tercile breakdown
    checkpoint_ensemble_six_way_comparison.csv           - extends the four-mechanism
                                                            tercile-gap table to six rows
    Appends {config}_topk_ensemble rows to results/results_ledger.csv (one per fold),
    exactly like every other run_tag.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MCRSLDataset, collate_fn, fit_numeric_stats
from data.loader import build_image_index, build_lesion_table, load_raw_tables
from evaluate import aggregate_fold_metrics, append_to_ledger, compute_binary_metrics
from models.model import MCRSLModel
from robustness_analysis import analysis_1_quality_stratified, get_fold_assignment
from train import move_batch

# (rerun run_tag used for checkpoint saving, label used for ledger/output naming)
CONFIGS = [
    ("channel_gated_topk", "channel_gated_topk_ensemble"),
    ("channel_gated_qweight_trust_topk", "channel_gated_qweight_trust_topk_ensemble"),
]


@torch.no_grad()
def ensemble_predict_fold(ckpt_paths, test_loader, device):
    """Averages sigmoid probabilities across all checkpoints in ckpt_paths for
    every sample in test_loader. Returns a DataFrame with one row per sample."""
    all_probs = None
    lesion_ids, labels, has_label = None, None, None

    for ckpt_path in ckpt_paths:
        model = MCRSLModel(variant="channel_gated", quality_aware=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        probs_this_ckpt, ids_this, labels_this, has_label_this = [], [], [], []
        for batch in test_loader:
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])
            probs_this_ckpt.append(torch.sigmoid(out["binary_logits"]).cpu())
            ids_this.extend(batch["lesion_id"])
            labels_this.append(batch["binary_label"].cpu())
            has_label_this.append(batch["has_binary_label"].cpu())

        probs_this_ckpt = torch.cat(probs_this_ckpt)
        if all_probs is None:
            all_probs = probs_this_ckpt
            lesion_ids = ids_this
            labels = torch.cat(labels_this)
            has_label = torch.cat(has_label_this)
        else:
            assert ids_this == lesion_ids, "checkpoint eval order mismatch across ensemble members"
            all_probs = all_probs + probs_this_ckpt

    all_probs = all_probs / len(ckpt_paths)
    return pd.DataFrame({
        "lesion_id": lesion_ids,
        "pred_prob": all_probs.numpy(),
        "binary_label": labels.numpy(),
        "has_binary_label": has_label.numpy(),
    })


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
    ledger_path = str(args.results_dir / "results_ledger.csv")

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    tercile_summaries = {}
    for rerun_tag, label in CONFIGS:
        print(f"\n=== {label} (ensembling top-k checkpoints per fold from {rerun_tag}) ===")
        fold_metrics = []
        oof_rows = []

        for test_fold in range(args.n_folds):
            val_fold = (test_fold + 1) % args.n_folds
            train_folds = [f for f in range(args.n_folds) if f not in (test_fold, val_fold)]
            train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
            test_subjects = subjects_by_fold[test_fold]

            numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
            test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, args.image_size, "test", False, verbose=False)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

            ckpt_paths = sorted(args.checkpoint_dir.glob(f"{rerun_tag}_qualityFalse_fold{test_fold}_top*.pt"))
            if not ckpt_paths:
                raise SystemExit(
                    f"no top-k checkpoints found for {rerun_tag} fold {test_fold} "
                    f"(expected {rerun_tag}_qualityFalse_fold{test_fold}_top*.pt) -- "
                    f"run scripts/run_topk_checkpoint_rerun.sh first."
                )
            print(f"[{label} fold {test_fold}] ensembling {len(ckpt_paths)} checkpoints: {[p.name for p in ckpt_paths]}")

            fold_df = ensemble_predict_fold(ckpt_paths, test_loader, device)
            fold_df["fold"] = test_fold
            fold_df["pred_label"] = (fold_df["pred_prob"] >= 0.5).astype(int)
            fold_df["correct"] = fold_df["pred_label"] == fold_df["binary_label"].astype(int)
            oof_rows.append(fold_df)

            labeled = fold_df[fold_df["has_binary_label"].astype(bool)]
            metrics = compute_binary_metrics(
                labeled["binary_label"].astype(int).to_numpy(),
                labeled["pred_label"].to_numpy(),
                labeled["pred_prob"].to_numpy(),
            )
            fold_metrics.append(metrics)
            append_to_ledger(
                ledger_path, label, False, test_fold, args.n_folds, args.seed, metrics,
                notes=f"prediction-level ensemble of top-{len(ckpt_paths)} val-bacc checkpoints from {rerun_tag}, no test-fold peeking in checkpoint selection",
            )
            print(f"[{label} fold {test_fold}] accuracy={metrics['accuracy']:.4f} "
                  f"balanced_accuracy={metrics['balanced_accuracy']:.4f} auroc={metrics['auroc']:.4f}")

        agg = aggregate_fold_metrics(fold_metrics)
        print(f"[{label}] AGGREGATE: accuracy={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f} "
              f"balanced_accuracy={agg['balanced_accuracy_mean']:.4f}+/-{agg['balanced_accuracy_std']:.4f} "
              f"sensitivity={agg['sensitivity_malignant_mean']:.4f} specificity={agg['specificity_mean']:.4f} "
              f"auroc={agg['auroc_mean']:.4f}")

        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_df.to_csv(args.results_dir / f"oof_predictions_{label}.csv", index=False)
        tercile_summaries[label] = analysis_1_quality_stratified(oof_df, lesion_df, args.results_dir, label)

    # Extend the existing four-mechanism tercile-gap table to six rows
    existing = {
        "channel_gated_qualityFalse": "plain (no quality-awareness)",
        "channel_gated_qualityTrue": "auxiliary quality-prediction head",
        "channel_gated_qweight_trust": "loss reweight: trust",
        "channel_gated_qweight_hard_mining": "loss reweight: hard_mining",
    }
    rows = []
    for cfg_tag, lbl in existing.items():
        path = args.results_dir / f"robustness_quality_tercile_{cfg_tag}.csv"
        if not path.exists():
            print(f"warning: {path} not found, skipping from six-way table")
            continue
        summary = pd.read_csv(path)
        gap = float(summary.set_index("tercile")["accuracy"]["high"] - summary.set_index("tercile")["accuracy"]["low"])
        rows.append({"mechanism": lbl, "high_minus_low_accuracy_gap": gap})
    for rerun_tag, label in CONFIGS:
        summary = tercile_summaries[label]
        gap = float(summary.set_index("tercile")["accuracy"]["high"] - summary.set_index("tercile")["accuracy"]["low"])
        rows.append({"mechanism": f"{label} (prediction-level top-k ensemble)", "high_minus_low_accuracy_gap": gap})

    six_way = pd.DataFrame(rows)
    six_way.to_csv(args.results_dir / "checkpoint_ensemble_six_way_comparison.csv", index=False)
    print("\n=== Six-way quality-tercile gap comparison (results/checkpoint_ensemble_six_way_comparison.csv) ===")
    print(six_way.to_string(index=False))

    print("\n=== DONE ===")
    print("Run 'python robustness_analysis.py' afterward to refresh results/summary_table.csv with the new *_topk_ensemble run_tags.")


if __name__ == "__main__":
    main()
