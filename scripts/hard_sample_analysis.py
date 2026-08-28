"""Qualitative hard-sample analysis with Grad-CAM — the final analysis increment.

No retraining. Uses the existing `hard_mining` out-of-fold predictions and the
existing per-fold checkpoints for inference + Grad-CAM only.

SAMPLE SELECTION is a systematic 4-category grid, not a cherry-pick:
  1. low-quality  + CORRECT    - model succeeding despite a poor photograph
  2. low-quality  + WRONG      - the genuine quality-linked failure mode
  3. high-quality + WRONG      - errors that are NOT a quality problem
                                 (genuinely ambiguous lesions) - the key contrast
  4. borderline confidence     - |p - 0.5| smallest; real uncertainty rather
                                 than confident-and-wrong
Within each category malignant lesions are preferred, since missed malignancies
are the clinically important error.

GRAD-CAM TARGET — the POST-GATE feature map.
    This model is multimodal, and its decision is made on the conv feature map
    *after* the metadata-derived channel gate has been applied:

        gate       = sigmoid(Linear(metadata_vec))          # (B, 1280)
        gated_map  = feature_map * gate[:, :, None, None]   # <-- CAM target
        fused      = mlp(pool(gated_map))
        logit      = binary_head(fused)

    Explaining the raw EfficientNet output instead would show generic backbone
    saliency, not what THIS model used. `gated_map` is a local variable inside
    ChannelGatedFusion.forward, so rather than modifying shared model code we
    replicate the forward pass here with the gate applied explicitly.

    The metadata vector is DETACHED before the gate, so gradients flow only
    with respect to the image feature map — the metadata branch's contribution
    is held fixed, per the task spec.

Usage:
    python scripts/hard_sample_analysis.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
    # optional stretch: baseline-vs-hard_mining CAMs side by side
    python scripts/hard_sample_analysis.py --data-dir ... --compare-baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, fit_numeric_stats
from data.loader import build_image_index, build_lesion_table, encode_categorical, load_raw_tables, parse_numeric_with_unknown
from data.schema import CATEGORICAL_FIELDS, NUMERICAL_FIELDS
from models.model import MCRSLModel
from robustness_analysis import get_fold_assignment

HARD_MINING_TAG = "channel_gated_qweight_hard_mining"
BASELINE_TAG = "channel_gated"


# --------------------------------------------------------------------------
# Grad-CAM on the post-gate feature map
# --------------------------------------------------------------------------
def gradcam_post_gate(model, image, categorical, numerical, numerical_missing):
    """Returns (cam HxW in [0,1], predicted probability).

    Replicates MCRSLModel.forward for the channel_gated variant, but keeps a
    handle on `gated_map` so gradients can be taken w.r.t. it.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    pooled, feature_map = model.image_encoder(image)

    # Metadata branch held FIXED — detached, so no gradient flows into it.
    with torch.no_grad():
        metadata_vec = model.metadata_encoder(categorical, numerical, numerical_missing)
    gate = torch.sigmoid(model.fusion.gate(metadata_vec.detach()))

    gated_map = feature_map * gate.unsqueeze(-1).unsqueeze(-1)   # CAM target
    gated_map.retain_grad()

    fused = model.fusion.mlp(model.fusion.pool(gated_map).flatten(1))
    logit = model.binary_head(fused)

    # Explain the PREDICTED class, not always "malignant". With a single
    # logit, evidence for the benign class is the negative direction — using
    # +logit for a benign prediction backprops the wrong way and produces a
    # near-empty map after the ReLU (verified in testing).
    prob = torch.sigmoid(logit.detach())[0].item()
    signed = logit if prob >= 0.5 else -logit
    signed.sum().backward()

    grads = gated_map.grad[0]          # (C, H, W)
    acts = gated_map.detach()[0]       # (C, H, W)
    weights = grads.mean(dim=(1, 2))   # (C,) channel importance
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        print("    WARNING: degenerate (all-zero) CAM — heatmap will be flat")
    return cam.cpu().numpy(), prob


def build_inputs(row, numeric_stats, image_size, device):
    """One sample's model inputs, using the evaluation transform."""
    img = Image.open(row["path"]).convert("RGB")
    tf = build_transforms(image_size, train=False)
    x = tf(img).unsqueeze(0).to(device)

    categorical = {f: torch.tensor([encode_categorical(row.get(f), v)], dtype=torch.long, device=device)
                   for f, v in CATEGORICAL_FIELDS.items()}
    numerical, missing = {}, {}
    for f in NUMERICAL_FIELDS:
        val, miss = parse_numeric_with_unknown(row.get(f))
        mean, std = numeric_stats[f]
        z = 0.0 if miss else (val - mean) / std
        numerical[f] = torch.tensor([z], dtype=torch.float32, device=device)
        missing[f] = torch.tensor([1.0 if miss else 0.0], dtype=torch.float32, device=device)

    # Undo normalization for display rather than re-opening at a different crop,
    # so the heatmap aligns exactly with what the model saw.
    disp = tf(img).permute(1, 2, 0).numpy()
    disp = np.clip(disp * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
    return x, categorical, numerical, missing, disp


# --------------------------------------------------------------------------
# Systematic sample selection
# --------------------------------------------------------------------------
def select_samples(df, per_category=2):
    """Four categories; prefer malignant within each; never reuse a lesion."""
    used, picked = set(), []

    def take(sub, category, n, sort_cols, ascending):
        sub = sub[~sub["lesion_id"].isin(used)]
        if sub.empty:
            return
        # malignant first, then the category's own ordering
        sub = sub.sort_values(["binary_label"] + sort_cols,
                              ascending=[False] + ascending)
        for _, r in sub.head(n).iterrows():
            picked.append({**r.to_dict(), "category": category})
            used.add(r["lesion_id"])

    low, high = df[df["tercile"] == "low"], df[df["tercile"] == "high"]
    # 1&2: low quality, correct / wrong — most confident first, so the example is clear
    take(low[low["correct"]], "low-quality, CORRECT", per_category, ["confidence"], [False])
    take(low[~low["correct"]], "low-quality, WRONG", per_category, ["confidence"], [False])
    # 3: high quality but wrong — the "not a quality problem" contrast
    take(high[~high["correct"]], "high-quality, WRONG", per_category, ["confidence"], [False])
    # 4: genuine uncertainty, any tier
    take(df, "borderline confidence", per_category, ["confidence"], [True])
    return pd.DataFrame(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--images-root", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--oof-csv", type=Path,
                    default=Path("results/oof_predictions_channel_gated_qweight_hard_mining.csv"))
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4,
                    help="samples per figure block; wraps so the aspect ratio suits a paper column")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-baseline", action="store_true",
                    help="stretch: also render the plain-baseline CAM for each sample")
    args = ap.parse_args()

    images_root = args.images_root or args.data_dir
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.oof_csv.exists():
        raise SystemExit(
            f"{args.oof_csv} not found. Produce it first with:\n"
            f"  python scripts/quality_adaptive_loss_analysis.py --data-dir {args.data_dir}")

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    subjects_by_fold = get_fold_assignment(lesion_df, args.n_folds, args.seed)

    oof = pd.read_csv(args.oof_csv)
    oof = oof[oof["has_binary_label"]].copy()

    # Merge ONLY the lesion columns not already present in the OOF file.
    # `binary_label`, `aux_label` and `histo_confirmed` exist in both, and a
    # blanket merge silently suffixes them to _x/_y, breaking every later
    # reference. Take the OOF copies (which are what was actually scored).
    need = ["lesion_id", "mean_image_rating", "unified_diagnosis", "diagnosis_image_id"]
    need += [c for c in list(CATEGORICAL_FIELDS.keys()) + NUMERICAL_FIELDS if c in lesion_df.columns]
    need = [c for c in dict.fromkeys(need) if c in lesion_df.columns]
    overlap = (set(need) & set(oof.columns)) - {"lesion_id"}
    assert not overlap, f"column collision would corrupt the merge: {overlap}"

    df = oof.merge(lesion_df[need], on="lesion_id", how="left").dropna(subset=["mean_image_rating"])
    # Same tercile boundaries as robustness analysis 1.
    df["tercile"] = pd.qcut(df["mean_image_rating"], 3, labels=["low", "mid", "high"])
    df["confidence"] = (df["pred_prob"] - 0.5).abs() * 2
    df["correct"] = df["correct"].astype(bool)

    # Attach the diagnosis image path.
    idx = image_index_df.set_index("image_id")["path"]
    df["path"] = df["diagnosis_image_id"].map(idx)
    df = df.dropna(subset=["path"])
    print(f"candidate pool: {len(df)} lesions with prediction, rating and image")

    sel = select_samples(df, args.per_category)
    print(f"\nselected {len(sel)} samples:")
    for _, r in sel.iterrows():
        print(f"  [{r['category']:<24}] {r['lesion_id']}  rating={r['mean_image_rating']:.1f} "
              f"({r['tercile']})  true={'MAL' if r['binary_label'] == 1 else 'ben'} "
              f"pred={'MAL' if r['pred_label'] == 1 else 'ben'} p={r['pred_prob']:.3f}")

    # ---- CAMs -----------------------------------------------------------
    models_to_run = [(HARD_MINING_TAG, "hard_mining")]
    if args.compare_baseline:
        models_to_run.append((BASELINE_TAG, "baseline"))

    cams = {tag: [] for tag, _ in models_to_run}
    displays, cache = [], {}

    for _, r in sel.iterrows():
        fold = int(r["fold"])
        val_fold = (fold + 1) % args.n_folds
        train_folds = [f for f in range(args.n_folds) if f not in (fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
        numeric_stats = fit_numeric_stats(lesion_df, train_subjects)

        x, cat, num, miss, disp = build_inputs(r, numeric_stats, args.image_size, device)
        displays.append(disp)

        for tag, _ in models_to_run:
            key = (tag, fold)
            if key not in cache:
                ckpt = args.checkpoint_dir / f"{tag}_qualityFalse_fold{fold}.pt"
                if not ckpt.exists():
                    raise SystemExit(f"missing checkpoint {ckpt}")
                m = MCRSLModel(variant="channel_gated", quality_aware=False).to(device)
                m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                cache[key] = m
            cam, prob = gradcam_post_gate(cache[key], x, cat, num, miss)
            cams[tag].append((cam, prob))

    # ---- Figure ---------------------------------------------------------
    # Wrapped into chunks of `--cols` so the aspect ratio suits a paper column;
    # a single 8-wide strip is unreadable once scaled to page width.
    n = len(sel)
    cols = min(args.cols, n)
    block = 1 + len(models_to_run)          # image row + one CAM row per model
    n_chunks = int(np.ceil(n / cols))
    fig, axs = plt.subplots(n_chunks * block, cols,
                            figsize=(3.0 * cols, 3.25 * n_chunks * block), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")

    for j, (_, r) in enumerate(sel.iterrows()):
        chunk, col = divmod(j, cols)
        base = chunk * block
        gt = "MALIGNANT" if r["binary_label"] == 1 else "benign"
        pr = "MALIGNANT" if r["pred_label"] == 1 else "benign"
        mark = "OK" if r["correct"] else "WRONG"
        colour = "darkgreen" if r["correct"] else "firebrick"

        axs[base][col].imshow(displays[j])
        axs[base][col].set_title(
            f"{r['category']}\n{r['lesion_id']} · {r['unified_diagnosis']} · "
            f"quality {r['mean_image_rating']:.1f}/10 ({r['tercile']})\n"
            f"true {gt} → pred {pr}  [{mark}]\nP(malignant)={r['pred_prob']:.3f}",
            fontsize=7.5, color=colour)

        for i, (tag, label) in enumerate(models_to_run, start=1):
            cam, _ = cams[tag][j]
            cam_up = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                              .resize((args.image_size, args.image_size), Image.BILINEAR)) / 255.0
            axs[base + i][col].imshow(displays[j])
            axs[base + i][col].imshow(cam_up, cmap="jet", alpha=0.45)
            axs[base + i][col].set_title(f"Grad-CAM ({label})", fontsize=7.5)

    fig.suptitle("Hard-sample analysis: channel-gated + hard_mining — Grad-CAM on the "
                 "post-gate feature map\n(out-of-fold predictions; each lesion scored by the "
                 "fold in which it was held out)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    # tight_layout alone leaves the next block's 4-line caption sitting on top
    # of the CAM row above it; give the rows explicit breathing room.
    fig.subplots_adjust(hspace=0.34)
    out_fig = args.results_dir / ("hard_samples_gradcam_compare.png" if args.compare_baseline
                                  else "hard_samples_gradcam.png")
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)
    print(f"\nwrote {out_fig}")

    # ---- Table ----------------------------------------------------------
    cols = ["category", "lesion_id", "unified_diagnosis", "mean_image_rating", "tercile",
            "binary_label", "pred_label", "pred_prob", "correct", "fold", "histo_confirmed"]
    tbl = sel[[c for c in cols if c in sel.columns]].copy()
    tbl["mean_image_rating"] = tbl["mean_image_rating"].round(2)
    tbl["pred_prob"] = tbl["pred_prob"].round(3)
    tbl = tbl.rename(columns={"binary_label": "true_malignant", "pred_label": "pred_malignant"})
    out_csv = args.results_dir / "hard_samples_table.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}\n")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
