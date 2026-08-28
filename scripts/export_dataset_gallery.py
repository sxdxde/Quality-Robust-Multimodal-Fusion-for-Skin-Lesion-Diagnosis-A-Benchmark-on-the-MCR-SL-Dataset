"""Builds a DATASET gallery — what the model actually trains on.

Distinct from scripts/export_paper_examples.py, which shows the model's
best-case *predictions*. This one shows the *data*, with no reference to model
output at all, so it can be presented as an honest picture of the input
distribution rather than a highlight reel.

Three rows, each answering a question a reviewer will ask:
  Row 1  Malignant lesions, one per unified-diagnosis class present
  Row 2  Non-malignant lesions, one per class
  Row 3  The QUALITY spread — lowest-rated to highest-rated diagnosis image.
         This is the paper's whole premise made visual: it shows what a
         rating of ~3/10 versus ~9/10 actually looks like.

Plus a second figure contrasting the two modalities for a single lesion
(dermoscopic, which we train on, vs clinical, which we do not).

Selection is deterministic (sorted, no sampling), so re-running reproduces
the same figure.

Usage:
    python scripts/export_dataset_gallery.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import build_image_index, build_lesion_table, load_raw_tables

N_PER_ROW = 4


def diag_image_path(row, image_index_df):
    """The diagnosis_image_id image — the one experts actually rated."""
    m = image_index_df[image_index_df["image_id"] == row["diagnosis_image_id"]]
    return (Path(m.iloc[0]["path"]), m.iloc[0]["modality"]) if len(m) else (None, None)


def pick_by_class(df, image_index_df, want_malignant, n):
    """One lesion per distinct unified_diagnosis, largest classes first, so the
    row shows variety rather than n near-identical nevi."""
    sub = df[df["binary_label"] == (1.0 if want_malignant else 0.0)]
    sub = sub.dropna(subset=["unified_diagnosis", "mean_image_rating"])
    order = sub["unified_diagnosis"].value_counts().index.tolist()
    picked = []
    for cls in order:
        if len(picked) >= n:
            break
        cands = sub[sub["unified_diagnosis"] == cls].sort_values("lesion_id")
        for _, r in cands.iterrows():
            p, mod = diag_image_path(r, image_index_df)
            if p and p.exists():
                picked.append((r, p, mod))
                break
    return picked


def pick_quality_spread(df, image_index_df, n):
    """Evenly spaced across the observed rating range, lowest to highest."""
    sub = df.dropna(subset=["mean_image_rating"]).sort_values("mean_image_rating")
    rows = []
    if len(sub) == 0:
        return rows
    idxs = [int(i * (len(sub) - 1) / max(n - 1, 1)) for i in range(n)]
    for i in idxs:
        r = sub.iloc[i]
        p, mod = diag_image_path(r, image_index_df)
        if p and p.exists():
            rows.append((r, p, mod))
    return rows


def draw_row(axes, items, row_label, show_rating_prominently=False):
    for ax in axes:
        ax.axis("off")
    for ax, (r, path, mod) in zip(axes, items):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
        rating = r["mean_image_rating"]
        rtxt = "n/a" if pd.isna(rating) else "{:.1f}/10".format(rating)
        if show_rating_prominently:
            title = "quality {}\n{} · {}".format(rtxt, r["lesion_id"], r["unified_diagnosis"])
        else:
            title = "{} · {}\nquality {}".format(r["lesion_id"], r["unified_diagnosis"], rtxt)
        ax.set_title(title, fontsize=7.5)
    if len(axes):
        axes[0].text(-0.08, 0.5, row_label, transform=axes[0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--images-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    images_root = args.images_root or args.data_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)

    # Gallery restricted to the diagnosis image of each lesion — the image the
    # experts rated, and the one used at evaluation time.
    mal = pick_by_class(lesion_df, image_index_df, True, N_PER_ROW)
    ben = pick_by_class(lesion_df, image_index_df, False, N_PER_ROW)
    qual = pick_quality_spread(lesion_df, image_index_df, N_PER_ROW)

    print("selected: {} malignant, {} non-malignant, {} quality-spread".format(
        len(mal), len(ben), len(qual)))
    for label, items in (("malignant", mal), ("non-malignant", ben), ("quality", qual)):
        for r, p, mod in items:
            print("  {:14} {} {:4} rating={} modality={}".format(
                label, r["lesion_id"], r["unified_diagnosis"],
                "n/a" if pd.isna(r["mean_image_rating"]) else round(r["mean_image_rating"], 1), mod))

    fig, axs = plt.subplots(3, N_PER_ROW, figsize=(3.1 * N_PER_ROW, 9.6))
    draw_row(axs[0], mal, "MALIGNANT")
    draw_row(axs[1], ben, "NON-MALIGNANT")
    draw_row(axs[2], qual, "QUALITY  low $\\rightarrow$ high", show_rating_prominently=True)
    fig.suptitle("MCR-SL: what the model trains on\n"
                 "dermoscopic diagnosis images, by malignancy, diagnosis class, and expert quality rating",
                 fontsize=11)
    fig.tight_layout(rect=[0.02, 0.0, 1, 0.94])
    out = args.out_dir / "dataset_gallery.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)

    # --- modality contrast: same lesion, dermoscopic vs clinical -----------
    pair = None
    for lid, g in image_index_df.groupby("lesion_id"):
        mods = set(g["modality"])
        if {"dermoscopy", "clinical"} <= mods:
            d = g[g["modality"] == "dermoscopy"].iloc[0]
            c = g[g["modality"] == "clinical"].iloc[0]
            if Path(d["path"]).exists() and Path(c["path"]).exists():
                pair = (lid, Path(d["path"]), Path(c["path"]))
                break

    if pair:
        lid, dpath, cpath = pair
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0))
        axs[0].imshow(Image.open(dpath).convert("RGB")); axs[0].axis("off")
        axs[0].set_title("Dermoscopic — 1352 images\n" r"$\bf{used\ for\ training}$", fontsize=9)
        axs[1].imshow(Image.open(cpath).convert("RGB")); axs[1].axis("off")
        axs[1].set_title("Clinical — 779 images\n" r"$\bf{not\ used}$ (different domain)", fontsize=9)
        fig.suptitle("Same lesion ({}), two modalities".format(lid), fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out2 = args.out_dir / "modality_comparison.png"
        fig.savefig(out2, dpi=160)
        plt.close(fig)
        print("wrote", out2)
    else:
        print("no lesion found with both modalities on disk — skipped modality figure")


if __name__ == "__main__":
    main()
