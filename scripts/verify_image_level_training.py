"""Step 0 verification for the "all-images-per-lesion" task, plus the
diagnostic that settles what that task's remaining scope actually is.

Runs no training. Four checks, in order:

1. **Fold-safety property (the task's Step 0, the one that silently
   invalidates everything if wrong)**: every image of a given lesion must
   land in exactly one fold. Asserted explicitly, per lesion.

2. **Images-per-lesion census**: how many dermoscopy vs. clinical images
   each lesion actually has, and how many are reachable on disk.

3. **What the train split actually yields today**: replicates
   data/dataset.py:_build_samples' train-split enumeration exactly, per
   fold, and reports lesion count vs. image-sample count. This is the check
   that settles whether training is already image-level (it is — see
   config.py:train_on_all_dermoscopic_images, default True) or lesion-level.

4. **pos_weight: lesion-level vs. image-level**: train.py's
   compute_binary_pos_weight derives the binary class weight from lesion_df
   (one row per lesion) while the loader yields image-level samples. If
   malignant and non-malignant lesions carry systematically different image
   counts, the applied pos_weight is miscalibrated against the batch
   composition the model actually sees. This quantifies that gap per fold —
   it is the one genuine, un-implemented item in the task's Step 1.

Usage:
    python scripts/verify_image_level_training.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.folds import make_subject_disjoint_folds
from data.loader import build_image_index, build_lesion_table, load_raw_tables


def build_fold_assignment(lesion_df: pd.DataFrame, n_folds: int, seed: int) -> dict:
    """Mirrors train.py:run_cv's fold construction exactly."""
    valid_binary = lesion_df.dropna(subset=["binary_label"])
    subject_malignant_count = valid_binary.groupby("subject_id")["binary_label"].sum().to_dict()
    for sid in lesion_df["subject_id"].unique():
        subject_malignant_count.setdefault(sid, 0)
    return make_subject_disjoint_folds(subject_malignant_count, n_folds=n_folds, seed=seed)


def train_split_images(lesion_df: pd.DataFrame, image_index_df: pd.DataFrame, subject_ids: set) -> pd.DataFrame:
    """Replicates data/dataset.py:_build_samples with use_all_images=True
    (the train-split path when train_on_all_dermoscopic_images is True):
    every dermoscopy image per lesion, falling back to the
    diagnosis_image_id image when a lesion has no dermoscopy image."""
    subset = lesion_df[lesion_df["subject_id"].isin(subject_ids)]
    images_by_lesion = {lid: g for lid, g in image_index_df.groupby("lesion_id")}

    rows = []
    for _, row in subset.iterrows():
        imgs = images_by_lesion.get(row["lesion_id"], image_index_df.iloc[0:0])
        use_imgs = imgs[imgs["modality"] == "dermoscopy"]
        if len(use_imgs) == 0:
            use_imgs = imgs[imgs["image_id"] == row["diagnosis_image_id"]]
        for _, img_row in use_imgs.iterrows():
            rows.append({"lesion_id": row["lesion_id"], "image_id": img_row["image_id"],
                         "binary_label": row["binary_label"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_root = args.images_root or args.data_dir
    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)
    assignment = build_fold_assignment(lesion_df, args.n_folds, args.seed)

    subjects_by_fold = {f: set() for f in range(args.n_folds)}
    for sid, f in assignment.items():
        subjects_by_fold[f].add(sid)

    # --- Check 1: fold-safety property -------------------------------------
    print("\n" + "=" * 78)
    print("CHECK 1 — fold safety: every image of a lesion lands in exactly one fold")
    print("=" * 78)
    lesion_to_subject = lesion_df.set_index("lesion_id")["subject_id"].to_dict()
    img_with_fold = image_index_df.copy()
    img_with_fold["subject_id"] = img_with_fold["lesion_id"].map(lesion_to_subject)
    img_with_fold["fold"] = img_with_fold["subject_id"].map(assignment)

    folds_per_lesion = img_with_fold.groupby("lesion_id")["fold"].nunique()
    violations = folds_per_lesion[folds_per_lesion != 1]
    n_unassigned = int(img_with_fold["fold"].isna().sum())

    print(f"lesions with images: {len(folds_per_lesion)}")
    print(f"lesions whose images span >1 fold: {len(violations)}")
    print(f"image rows with no fold assignment: {n_unassigned}")
    assert len(violations) == 0, f"FOLD LEAK: {violations.to_dict()}"
    assert n_unassigned == 0, f"{n_unassigned} image rows unassigned to any fold"
    print("PASS — subject-disjoint folds already guarantee lesion-disjoint images.")
    print("No fold-assignment code change is needed for image-level training.")

    # --- Check 2: images-per-lesion census ---------------------------------
    print("\n" + "=" * 78)
    print("CHECK 2 — images-per-lesion census (on-disk, post-join)")
    print("=" * 78)
    by_modality = image_index_df.groupby("modality").size()
    print(f"total usable image rows: {len(image_index_df)}")
    for modality, n in by_modality.items():
        print(f"  {modality}: {n}")

    derm = image_index_df[image_index_df["modality"] == "dermoscopy"]
    derm_per_lesion = derm.groupby("lesion_id").size()
    print(f"\ndermoscopy images per lesion (over {len(derm_per_lesion)} lesions with >=1):")
    print(f"  mean={derm_per_lesion.mean():.2f}  median={derm_per_lesion.median():.0f}  "
          f"min={derm_per_lesion.min()}  max={derm_per_lesion.max()}")
    print(f"  lesions with zero dermoscopy images: {len(lesion_df) - len(derm_per_lesion)}")

    # --- Check 3: what the train split actually yields ----------------------
    print("\n" + "=" * 78)
    print("CHECK 3 — train-split enumeration today (replicates _build_samples)")
    print("=" * 78)
    print(f"{'fold':>5} {'train lesions':>14} {'train image samples':>21} {'ratio':>7}")
    total_lesions = total_images = 0
    for test_fold in range(args.n_folds):
        val_fold = (test_fold + 1) % args.n_folds
        train_folds = [f for f in range(args.n_folds) if f not in (test_fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])

        n_lesions = int(lesion_df["subject_id"].isin(train_subjects).sum())
        samples = train_split_images(lesion_df, image_index_df, train_subjects)
        total_lesions += n_lesions
        total_images += len(samples)
        print(f"{test_fold:>5} {n_lesions:>14} {len(samples):>21} {len(samples)/max(n_lesions,1):>6.2f}x")

    print(f"\nAveraged over folds: {total_images/args.n_folds:.0f} image samples "
          f"from {total_lesions/args.n_folds:.0f} lesions "
          f"({total_images/max(total_lesions,1):.2f}x)")
    if total_images > total_lesions * 1.5:
        print("\n>>> Training is ALREADY image-level (all dermoscopic images per lesion).")
        print(">>> config.py:train_on_all_dermoscopic_images defaults to True and every")
        print(">>> logged run recorded it as True. The task's premise that existing runs")
        print(">>> use one image per lesion does NOT hold — no data-utilization gap here.")
    else:
        print("\n>>> Training appears lesion-level — the task's premise holds.")

    # --- Check 4: pos_weight lesion-level vs image-level --------------------
    print("\n" + "=" * 78)
    print("CHECK 4 — pos_weight: lesion-level (as applied today) vs image-level (as seen)")
    print("=" * 78)
    print(f"{'fold':>5} {'lesion-level':>14} {'image-level':>13} {'rel. diff':>11}")
    diffs = []
    for test_fold in range(args.n_folds):
        val_fold = (test_fold + 1) % args.n_folds
        train_folds = [f for f in range(args.n_folds) if f not in (test_fold, val_fold)]
        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])

        # exactly train.py:compute_binary_pos_weight
        lab = lesion_df[lesion_df["subject_id"].isin(train_subjects)]["binary_label"].dropna()
        pw_lesion = (lab == 0.0).sum() / max((lab == 1.0).sum(), 1)

        samples = train_split_images(lesion_df, image_index_df, train_subjects)
        slab = samples["binary_label"].dropna()
        pw_image = (slab == 0.0).sum() / max((slab == 1.0).sum(), 1)

        rel = (pw_image - pw_lesion) / pw_lesion * 100
        diffs.append(rel)
        print(f"{test_fold:>5} {pw_lesion:>14.3f} {pw_image:>13.3f} {rel:>+10.1f}%")

    print(f"\nmean |relative difference|: {np.mean(np.abs(diffs)):.1f}%")
    print("\nThis is the one genuinely un-implemented item from the task's Step 1:")
    print("compute_binary_pos_weight uses lesion-level counts while the loader yields")
    print("image-level samples. The size of the gap above says whether fixing it is a")
    print("real correction or a numerical no-op.")

    print("\n" + "=" * 78)
    print("ALL CHECKS DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
