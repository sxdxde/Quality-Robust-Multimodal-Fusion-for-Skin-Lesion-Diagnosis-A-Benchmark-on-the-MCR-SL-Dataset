"""The four robustness analyses (CLAUDE.md's actual novelty), run against a
trained model's per-lesion test-set predictions aggregated across all 5 CV
folds (every lesion gets exactly one out-of-fold prediction this way, so N
matches the full usable lesion count rather than a single fold).

Usage:
    python robustness_analysis.py --variant channel_gated --data-dir ... --images-root ...
    (loads the 5 fold checkpoints written by train.py under checkpoints/)

Outputs (under results/):
    robustness_quality_tercile.csv / .png       - analysis 1 (+ 2 if quality-aware ckpts exist)
    robustness_histopath.csv                     - analysis 3
    robustness_metadata_importance.csv / .png    - analysis 4
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
from torch.utils.data import DataLoader

from data.dataset import MCRSLDataset, collate_fn, fit_numeric_stats
from data.folds import make_subject_disjoint_folds
from data.loader import build_image_index, build_lesion_table, load_raw_tables
from data.schema import CATEGORICAL_FIELDS, NUMERICAL_FIELDS
from models.model import MCRSLModel
from train import move_batch


@torch.no_grad()
def collect_oof_predictions(cfg_variant: str, quality_aware: bool, lesion_df, image_index_df, n_folds: int,
                             seed: int, checkpoint_dir: Path, image_size: int, batch_size: int, device) -> pd.DataFrame:
    """Runs each fold's saved checkpoint on its own held-out test subjects,
    returns one row per lesion with prediction, probability, and — for
    metadata-importance — the raw input tensors' gradient sensitivity.
    """
    valid_binary = lesion_df.dropna(subset=["binary_label"])
    subject_malignant_count = valid_binary.groupby("subject_id")["binary_label"].sum().to_dict()
    for sid in lesion_df["subject_id"].unique():
        subject_malignant_count.setdefault(sid, 0)
    assignment = make_subject_disjoint_folds(subject_malignant_count, n_folds=n_folds, seed=seed)
    subjects_by_fold = {f: set() for f in range(n_folds)}
    for sid, f in assignment.items():
        subjects_by_fold[f].add(sid)

    rows = []
    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in (test_fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
        test_subjects = subjects_by_fold[test_fold]

        numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
        test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, image_size, "test", False, verbose=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        ckpt_path = checkpoint_dir / f"{cfg_variant}_quality{quality_aware}_fold{test_fold}.pt"
        model = MCRSLModel(variant=cfg_variant, quality_aware=quality_aware).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for batch in test_loader:
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])
            probs = torch.sigmoid(out["binary_logits"]).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            for i, lesion_id in enumerate(batch["lesion_id"]):
                rows.append({
                    "lesion_id": lesion_id,
                    "fold": test_fold,
                    "has_binary_label": bool(batch["has_binary_label"][i].item()),
                    "binary_label": float(batch["binary_label"][i].item()),
                    "pred_prob": float(probs[i]),
                    "pred_label": int(preds[i]),
                    "correct": bool(preds[i] == int(batch["binary_label"][i].item())) if batch["has_binary_label"][i] else None,
                    "histo_confirmed": bool(batch["histo_confirmed"][i].item()),
                })

    return pd.DataFrame(rows)


def analysis_1_quality_stratified(oof_df: pd.DataFrame, lesion_df: pd.DataFrame, out_dir: Path):
    """Bucket lesions by mean image_rating into terciles; report
    accuracy/sensitivity per tercile + Spearman(rating, confidence/error)."""
    df = oof_df[oof_df["has_binary_label"]].merge(
        lesion_df[["lesion_id", "mean_image_rating"]], on="lesion_id", how="left"
    ).dropna(subset=["mean_image_rating"])

    n = len(df)
    print(f"[analysis 1] N with both prediction and quality rating: {n} (report as-is, don't overstate — expect ~238)")

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
    summary.to_csv(out_dir / "robustness_quality_tercile.csv", index=False)
    print(summary)

    rho_error, p_error = spearmanr(df["mean_image_rating"], df["error"])
    rho_conf, p_conf = spearmanr(df["mean_image_rating"], df["confidence"])
    print(f"[analysis 1] Spearman(rating, error) = {rho_error:.3f} (p={p_error:.3f}, N={n})")
    print(f"[analysis 1] Spearman(rating, confidence) = {rho_conf:.3f} (p={p_conf:.3f}, N={n})")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary["tercile"], summary["accuracy"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by image-quality tercile (N={n}, N/tercile~{n//3})")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_quality_tercile.png", dpi=150)
    plt.close(fig)

    return summary, (rho_error, p_error), (rho_conf, p_conf)


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
                                    checkpoint_dir: Path, image_size: int, batch_size: int, seed: int,
                                    n_folds: int, device, out_dir: Path):
    """Gradient x input sensitivity per metadata field, averaged across all
    5 fold checkpoints' test sets. Compares against the dataset paper's own
    Tables 3-4 significant fields: location_group, sex, referral_diagnosis,
    diameter (all p<0.01)."""
    valid_binary = lesion_df.dropna(subset=["binary_label"])
    subject_malignant_count = valid_binary.groupby("subject_id")["binary_label"].sum().to_dict()
    for sid in lesion_df["subject_id"].unique():
        subject_malignant_count.setdefault(sid, 0)
    assignment = make_subject_disjoint_folds(subject_malignant_count, n_folds=n_folds, seed=seed)
    subjects_by_fold = {f: set() for f in range(n_folds)}
    for sid, f in assignment.items():
        subjects_by_fold[f].add(sid)

    field_scores = {f: [] for f in list(CATEGORICAL_FIELDS.keys()) + NUMERICAL_FIELDS}

    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in (test_fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
        test_subjects = subjects_by_fold[test_fold]

        numeric_stats = fit_numeric_stats(lesion_df, train_subjects)
        test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, image_size, "test", False, verbose=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        ckpt_path = checkpoint_dir / f"{cfg_variant}_quality{quality_aware}_fold{test_fold}.pt"
        model = MCRSLModel(variant=cfg_variant, quality_aware=quality_aware).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for batch in test_loader:
            batch = move_batch(batch, device)
            with torch.enable_grad():
                cat_embeds = {}
                for field in CATEGORICAL_FIELDS:
                    emb = model.metadata_encoder.embeddings[field](batch["categorical"][field])
                    emb = emb.detach().requires_grad_(True)
                    cat_embeds[field] = emb
                num_vals = {f: batch["numerical"][f].detach().requires_grad_(True) for f in NUMERICAL_FIELDS}

                parts = [cat_embeds[f.name] for f in model.metadata_encoder.categorical_fields]
                for f in model.metadata_encoder.numerical_fields:
                    parts.append(num_vals[f.name].unsqueeze(-1))
                    parts.append(batch["numerical_missing"][f.name].unsqueeze(-1))
                metadata_vec = model.metadata_encoder.mlp(torch.cat(parts, dim=-1))

                pooled, feature_map = model.image_encoder(batch["image"])
                fused = model.fusion(pooled, metadata_vec, feature_map)
                logits = model.binary_head(fused)
                logits.sum().backward()

            for field in CATEGORICAL_FIELDS:
                grad_x_input = (cat_embeds[field].grad * cat_embeds[field]).sum(dim=-1).abs()
                field_scores[field].extend(grad_x_input.cpu().numpy().tolist())
            for field in NUMERICAL_FIELDS:
                grad_x_input = (num_vals[field].grad * num_vals[field]).abs()
                field_scores[field].extend(grad_x_input.cpu().numpy().tolist())

    summary = pd.DataFrame([
        {"field": f, "mean_abs_grad_x_input": float(np.mean(scores)) if scores else float("nan")}
        for f, scores in field_scores.items()
    ]).sort_values("mean_abs_grad_x_input", ascending=False)

    paper_significant = {"location_group", "sex", "referral_diagnosis", "diameter"}
    summary["in_paper_significant_fields"] = summary["field"].isin(paper_significant)

    print("[analysis 4] metadata field importance (gradient x input), vs. dataset paper's Tables 3-4 significant fields")
    print(summary)
    summary.to_csv(out_dir / "robustness_metadata_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:orange" if s else "tab:blue" for s in summary["in_paper_significant_fields"]]
    ax.barh(summary["field"], summary["mean_abs_grad_x_input"], color=colors)
    ax.set_xlabel("mean |grad x input|")
    ax.set_title("Metadata field importance (orange = paper's significant fields)")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_metadata_importance.png", dpi=150)
    plt.close(fig)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="channel_gated")
    parser.add_argument("--quality-aware", action="store_true")
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

    oof_df = collect_oof_predictions(
        args.variant, args.quality_aware, lesion_df, image_index_df, args.n_folds, args.seed,
        args.checkpoint_dir, args.image_size, args.batch_size, device,
    )
    oof_df.to_csv(args.results_dir / "oof_predictions.csv", index=False)

    analysis_1_quality_stratified(oof_df, lesion_df, args.results_dir)
    analysis_3_histopath_vs_panel(oof_df, args.results_dir)
    analysis_4_metadata_importance(
        args.variant, args.quality_aware, lesion_df, image_index_df, args.checkpoint_dir,
        args.image_size, args.batch_size, args.seed, args.n_folds, device, args.results_dir,
    )
    print("\nNote: analysis 2 (quality-aware training comparison) requires running train.py twice — "
          "once with --quality-aware and once without — then comparing their robustness_quality_tercile.csv "
          "outputs side by side (not computed here, since it's a between-run comparison, not a single-run analysis).")


if __name__ == "__main__":
    main()
