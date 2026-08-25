"""Post-hoc analysis over already-trained checkpoints. No training happens
here — everything reads the 5-fold checkpoints train.py already saved.

Does NOT write to results_ledger.csv — train.py is the sole writer of the
ledger (one append per fold, on every run, under that run's run_tag). This
script only READS it for the summary table. (Earlier versions of this script
deleted and rebuilt the whole ledger from a hardcoded 4-config list — that
was a one-time repair for a sync-script bug that had corrupted the ledger's
history; doing that on every run would now silently wipe out every
experiment run beyond those original 4. Don't reintroduce that.)

Does three things in one remote run, for the CONFIGS list below (the
"core" configs from CLAUDE.md's ablation matrix, not every follow-up
experiment — extend CONFIGS if you want the full battery on a new one):
1. Saves per-config confusion matrices and the aux 9-class exploratory table
   (small-N classes flagged per CLAUDE.md) for every listed config.
2. Prints/saves the master summary table read fresh from the ledger (which
   by now includes every run_tag anyone has trained, not just CONFIGS).
3. Runs the four robustness analyses against the best model (channel_gated,
   non-quality-aware — the empirical best by macro-F1/balanced accuracy),
   PLUS analysis 2 (quality-aware vs. non-aware tercile comparison), which
   needs both channel_gated configs' predictions.

Usage:
    python robustness_analysis.py --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset

Outputs (under results/):
    summary_table.csv                                    - one row per run_tag in the ledger, mean+-std (paper table)
    confusion_matrix_<config>.csv / .png                  - per CONFIGS entry
    aux_9class_<config>.csv                               - per CONFIGS entry, small-N classes flagged
    oof_predictions_<config>.csv                          - per-lesion out-of-fold predictions, per CONFIGS entry
    robustness_quality_tercile_<config>.csv / .png        - analysis 1, for both channel_gated configs
    robustness_quality_aware_comparison.csv / .png        - analysis 2
    robustness_histopath.csv                              - analysis 3 (best model only)
    robustness_metadata_importance.csv / .png             - analysis 4 (best model only)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from data.dataset import MCRSLDataset, collate_fn, fit_numeric_stats
from data.folds import make_subject_disjoint_folds
from data.loader import build_image_index, build_lesion_table, load_raw_tables
from data.schema import CATEGORICAL_FIELDS, NUMERICAL_FIELDS, SMALL_N_CLASSES, UNIFIED_DIAGNOSIS_CLASSES
from evaluate import compute_binary_metrics
from models.model import MCRSLModel
from train import move_batch

CONFIGS = [
    ("image_only", False),
    ("late_fusion", False),
    ("channel_gated", False),
    ("channel_gated", True),
]
BEST_VARIANT, BEST_QUALITY_AWARE = "channel_gated", False


def tag(variant: str, quality_aware: bool) -> str:
    return f"{variant}_quality{quality_aware}"


def get_fold_assignment(lesion_df, n_folds: int, seed: int):
    valid_binary = lesion_df.dropna(subset=["binary_label"])
    subject_malignant_count = valid_binary.groupby("subject_id")["binary_label"].sum().to_dict()
    for sid in lesion_df["subject_id"].unique():
        subject_malignant_count.setdefault(sid, 0)
    assignment = make_subject_disjoint_folds(subject_malignant_count, n_folds=n_folds, seed=seed)
    subjects_by_fold = {f: set() for f in range(n_folds)}
    for sid, f in assignment.items():
        subjects_by_fold[f].add(sid)
    return subjects_by_fold


@torch.no_grad()
def collect_oof_predictions(cfg_variant: str, quality_aware: bool, lesion_df, image_index_df,
                             subjects_by_fold: dict, n_folds: int, checkpoint_dir: Path,
                             image_size: int, batch_size: int, device, ckpt_tag: str = None) -> pd.DataFrame:
    """One row per (lesion, fold-it-was-tested-in), with binary AND aux
    predictions — every lesion gets exactly one out-of-fold prediction.

    `ckpt_tag` overrides the checkpoint filename prefix (default: tag(cfg_variant,
    quality_aware)) for run_tags that don't follow the variant_qualityBool
    pattern, e.g. follow-up experiments like channel_gated_qweight_trust
    (see scripts/quality_adaptive_loss_analysis.py)."""
    rows = []
    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in (test_fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
        test_subjects = subjects_by_fold[test_fold]

        numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
        test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, image_size, "test", False, verbose=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        ckpt_path = checkpoint_dir / f"{ckpt_tag or tag(cfg_variant, quality_aware)}_fold{test_fold}.pt"
        model = MCRSLModel(variant=cfg_variant, quality_aware=quality_aware).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        for batch in test_loader:
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])
            probs = torch.sigmoid(out["binary_logits"]).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            aux_preds = out["aux_logits"].argmax(dim=-1).cpu().numpy()

            for i, lesion_id in enumerate(batch["lesion_id"]):
                has_aux = int(batch["aux_label"][i].item()) != -100
                rows.append({
                    "lesion_id": lesion_id,
                    "fold": test_fold,
                    "has_binary_label": bool(batch["has_binary_label"][i].item()),
                    "binary_label": float(batch["binary_label"][i].item()),
                    "pred_prob": float(probs[i]),
                    "pred_label": int(preds[i]),
                    "correct": bool(preds[i] == int(batch["binary_label"][i].item())) if batch["has_binary_label"][i] else None,
                    "histo_confirmed": bool(batch["histo_confirmed"][i].item()),
                    "has_aux_label": has_aux,
                    "aux_label": int(batch["aux_label"][i].item()) if has_aux else None,
                    "aux_pred": int(aux_preds[i]),
                })

    return pd.DataFrame(rows)


def save_confusion_matrix(oof_df: pd.DataFrame, cfg_tag: str, out_dir: Path):
    df = oof_df[oof_df["has_binary_label"]]
    metrics = compute_binary_metrics(df["binary_label"].astype(int).to_numpy(), df["pred_label"].to_numpy(), df["pred_prob"].to_numpy())
    cm = metrics["confusion_matrix"]
    pd.DataFrame(cm, index=["true_non_malignant", "true_malignant"], columns=["pred_non_malignant", "pred_malignant"]) \
        .to_csv(out_dir / f"confusion_matrix_{cfg_tag}.csv")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred non-mal", "pred malignant"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true non-mal", "true malignant"])
    ax.set_title(f"Confusion matrix ({cfg_tag}, aggregated across folds)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{cfg_tag}.png", dpi=150)
    plt.close(fig)
    return cm


def save_aux_table(oof_df: pd.DataFrame, cfg_tag: str, out_dir: Path):
    """9-class exploratory table. Several classes have <10 lesions — flagged
    per class, per CLAUDE.md, not presented as statistically robust."""
    df = oof_df[oof_df["has_aux_label"]]
    y_true = df["aux_label"].astype(int).to_numpy()
    y_pred = df["aux_pred"].astype(int).to_numpy()
    labels = list(range(len(UNIFIED_DIAGNOSIS_CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    rows = []
    for idx, cls in enumerate(UNIFIED_DIAGNOSIS_CLASSES):
        rows.append({
            "class": cls, "support": int(support[idx]), "precision": precision[idx],
            "recall": recall[idx], "f1": f1[idx],
            "small_N_flag": cls in SMALL_N_CLASSES,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"aux_9class_{cfg_tag}.csv", index=False)
    return summary


def analysis_1_quality_stratified(oof_df: pd.DataFrame, lesion_df: pd.DataFrame, out_dir: Path, cfg_tag: str):
    """Bucket lesions by mean image_rating into terciles; report
    accuracy/sensitivity per tercile + Spearman(rating, confidence/error)."""
    df = oof_df[oof_df["has_binary_label"]].merge(
        lesion_df[["lesion_id", "mean_image_rating"]], on="lesion_id", how="left"
    ).dropna(subset=["mean_image_rating"])

    n = len(df)
    print(f"[analysis 1 | {cfg_tag}] N with both prediction and quality rating: {n} (report as-is — expect ~238)")

    df["tercile"] = pd.qcut(df["mean_image_rating"], 3, labels=["low", "mid", "high"])
    df["error"] = 1 - df["correct"].astype(int)
    df["confidence"] = np.abs(df["pred_prob"] - 0.5) * 2  # 0=uncertain, 1=confident

    rows = []
    for t in ["low", "mid", "high"]:
        sub = df[df["tercile"] == t]
        malignant = sub[sub["binary_label"] == 1]
        sens = malignant["correct"].mean() if len(malignant) > 0 else float("nan")
        rows.append({"tercile": t, "n": len(sub), "accuracy": sub["correct"].mean(), "sensitivity_malignant": sens})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"robustness_quality_tercile_{cfg_tag}.csv", index=False)
    print(summary)

    rho_error, p_error = spearmanr(df["mean_image_rating"], df["error"])
    rho_conf, p_conf = spearmanr(df["mean_image_rating"], df["confidence"])
    print(f"[analysis 1 | {cfg_tag}] Spearman(rating, error) = {rho_error:.3f} (p={p_error:.3f}, N={n})")
    print(f"[analysis 1 | {cfg_tag}] Spearman(rating, confidence) = {rho_conf:.3f} (p={p_conf:.3f}, N={n})")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary["tercile"], summary["accuracy"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by image-quality tercile ({cfg_tag}, N={n})")
    fig.tight_layout()
    fig.savefig(out_dir / f"robustness_quality_tercile_{cfg_tag}.png", dpi=150)
    plt.close(fig)

    return summary


def analysis_2_quality_aware_comparison(summary_no_qa: pd.DataFrame, summary_qa: pd.DataFrame, out_dir: Path):
    """Does explicit quality-awareness flatten the accuracy gap across
    quality terciles, vs. the same architecture without it?"""
    merged = summary_no_qa.merge(summary_qa, on="tercile", suffixes=("_no_quality_aware", "_quality_aware"))
    merged.to_csv(out_dir / "robustness_quality_aware_comparison.csv", index=False)

    gap_no_qa = summary_no_qa.set_index("tercile")["accuracy"]["low"] - summary_no_qa.set_index("tercile")["accuracy"]["high"]
    gap_qa = summary_qa.set_index("tercile")["accuracy"]["low"] - summary_qa.set_index("tercile")["accuracy"]["high"]
    print(f"[analysis 2] low-vs-high tercile accuracy gap: no-quality-aware={gap_no_qa:+.3f}, quality-aware={gap_qa:+.3f}")
    print(f"[analysis 2] {'quality-awareness FLATTENS the gap' if abs(gap_qa) < abs(gap_no_qa) else 'quality-awareness does NOT flatten the gap'} "
          f"(|gap| {abs(gap_no_qa):.3f} -> {abs(gap_qa):.3f})")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, summary_no_qa["accuracy"], width, label="non-quality-aware")
    ax.bar(x + width / 2, summary_qa["accuracy"], width, label="quality-aware")
    ax.set_xticks(x); ax.set_xticklabels(summary_no_qa["tercile"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Quality-aware vs. non-aware, by quality tercile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_quality_aware_comparison.png", dpi=150)
    plt.close(fig)
    return merged


def analysis_3_histopath_vs_panel(oof_df: pd.DataFrame, out_dir: Path):
    """Qualitative comparison only (n=28 histo-confirmed) — no p-value."""
    df = oof_df[oof_df["has_binary_label"]]
    histo = df[df["histo_confirmed"]]
    panel = df[~df["histo_confirmed"]]

    summary = pd.DataFrame([
        {"group": "histopathology_confirmed", "n": len(histo), "accuracy": histo["correct"].mean(),
         "mean_confidence": np.abs(histo["pred_prob"] - 0.5).mean() * 2},
        {"group": "panel_consensus_only", "n": len(panel), "accuracy": panel["correct"].mean(),
         "mean_confidence": np.abs(panel["pred_prob"] - 0.5).mean() * 2},
    ])
    print(f"[analysis 3] qualitative only — histo n={len(histo)} is too small for a confidence interval")
    print(summary)
    summary.to_csv(out_dir / "robustness_histopath.csv", index=False)
    return summary


@torch.no_grad()
def analysis_4_metadata_importance(cfg_variant: str, quality_aware: bool, lesion_df, image_index_df,
                                    subjects_by_fold: dict, checkpoint_dir: Path, image_size: int,
                                    batch_size: int, n_folds: int, device, out_dir: Path):
    """Ablation-by-field importance: for each metadata field, mask it to its
    "unknown"/missing value and measure the resulting |delta binary logit|,
    averaged over every test-fold sample. Compares against the dataset
    paper's own Tables 3-4 significant fields: location_group, sex,
    referral_diagnosis, diameter (all p<0.01).

    Chosen over gradient x input: that metric sums over each categorical
    field's whole embedding vector (12 dims) but a numeric field is a single
    scalar, so categorical fields get a structurally larger raw score
    regardless of true importance — not a fair comparison across field
    types. Ablation's |delta logit| is directly comparable across both.
    """
    field_scores = {f: [] for f in list(CATEGORICAL_FIELDS.keys()) + NUMERICAL_FIELDS}

    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in (test_fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
        test_subjects = subjects_by_fold[test_fold]

        numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
        test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, image_size, "test", False, verbose=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        ckpt_path = checkpoint_dir / f"{tag(cfg_variant, quality_aware)}_fold{test_fold}.pt"
        model = MCRSLModel(variant=cfg_variant, quality_aware=quality_aware).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        for batch in test_loader:
            batch = move_batch(batch, device)
            base_logits = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])["binary_logits"]

            for field, vocab in CATEGORICAL_FIELDS.items():
                ablated_categorical = dict(batch["categorical"])
                ablated_categorical[field] = torch.full_like(ablated_categorical[field], fill_value=len(vocab))
                out = model(batch["image"], ablated_categorical, batch["numerical"], batch["numerical_missing"])
                delta = (out["binary_logits"] - base_logits).abs()
                field_scores[field].extend(delta.cpu().numpy().tolist())

            for field in NUMERICAL_FIELDS:
                ablated_numerical = dict(batch["numerical"])
                ablated_numerical[field] = torch.zeros_like(ablated_numerical[field])
                ablated_missing = dict(batch["numerical_missing"])
                ablated_missing[field] = torch.ones_like(ablated_missing[field])
                out = model(batch["image"], batch["categorical"], ablated_numerical, ablated_missing)
                delta = (out["binary_logits"] - base_logits).abs()
                field_scores[field].extend(delta.cpu().numpy().tolist())

    summary = pd.DataFrame([
        {"field": f, "mean_abs_logit_delta": float(np.mean(scores)) if scores else float("nan")}
        for f, scores in field_scores.items()
    ]).sort_values("mean_abs_logit_delta", ascending=False)

    paper_significant = {"location_group", "sex", "referral_diagnosis", "diameter"}
    summary["in_paper_significant_fields"] = summary["field"].isin(paper_significant)

    print("[analysis 4] metadata field importance (ablation, |delta logit|), vs. dataset paper's Tables 3-4 significant fields")
    print(summary)
    summary.to_csv(out_dir / "robustness_metadata_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:orange" if s else "tab:blue" for s in summary["in_paper_significant_fields"]]
    ax.barh(summary["field"], summary["mean_abs_logit_delta"], color=colors)
    ax.set_xlabel("mean |delta logit| when field is ablated")
    ax.set_title("Metadata field importance (orange = paper's significant fields)")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_metadata_importance.png", dpi=150)
    plt.close(fig)

    return summary


def build_master_summary_table(ledger_path: str, out_dir: Path):
    df = pd.read_csv(ledger_path)
    summary = df.groupby(["variant", "quality_aware"]).agg(
        n_folds=("fold", "count"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        sensitivity_malignant_mean=("sensitivity_malignant", "mean"), sensitivity_malignant_std=("sensitivity_malignant", "std"),
        specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
    ).round(4).reset_index()
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print("\n=== Master summary table (results/summary_table.csv) ===")
    print(summary.to_string(index=False))
    return summary


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

    ledger_path = str(args.results_dir / "results_ledger.csv")

    oof_by_config = {}
    tercile_by_config = {}
    for variant, quality_aware in CONFIGS:
        cfg_tag = tag(variant, quality_aware)
        print(f"\n=== collecting OOF predictions: {cfg_tag} ===")
        oof_df = collect_oof_predictions(
            variant, quality_aware, lesion_df, image_index_df, subjects_by_fold, args.n_folds,
            args.checkpoint_dir, args.image_size, args.batch_size, device,
        )
        oof_df.to_csv(args.results_dir / f"oof_predictions_{cfg_tag}.csv", index=False)
        oof_by_config[(variant, quality_aware)] = oof_df

        save_confusion_matrix(oof_df, cfg_tag, args.results_dir)
        save_aux_table(oof_df, cfg_tag, args.results_dir)

        if variant == "channel_gated":
            tercile_by_config[quality_aware] = analysis_1_quality_stratified(oof_df, lesion_df, args.results_dir, cfg_tag)

    build_master_summary_table(ledger_path, args.results_dir)

    analysis_2_quality_aware_comparison(tercile_by_config[False], tercile_by_config[True], args.results_dir)

    best_oof = oof_by_config[(BEST_VARIANT, BEST_QUALITY_AWARE)]
    analysis_3_histopath_vs_panel(best_oof, args.results_dir)
    analysis_4_metadata_importance(
        BEST_VARIANT, BEST_QUALITY_AWARE, lesion_df, image_index_df, subjects_by_fold,
        args.checkpoint_dir, args.image_size, args.batch_size, args.n_folds, device, args.results_dir,
    )

    print("\n=== ALL ANALYSES DONE ===")


if __name__ == "__main__":
    main()
