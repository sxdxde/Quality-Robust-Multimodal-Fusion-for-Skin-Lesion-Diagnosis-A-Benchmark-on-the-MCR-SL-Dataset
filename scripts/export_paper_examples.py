"""Exports the model's highest-confidence CORRECT predictions for the
paper's qualitative example figure: real metadata, real ground truth, real
channel-gated predictions, and a copy of the actual diagnosis image file.

These are best-case examples, chosen to be visually clear and illustrate a
confident correct call in each direction -- NOT a representative or random
sample of overall performance. The paper's actual performance numbers
(sensitivity ~0.67-0.74, the real false-negative/false-positive rates, etc.)
are reported in full, unfiltered, elsewhere (Tables III-VIII); this figure
does not stand in for those numbers and must not be captioned as if it did.

Selection: among correct predictions, the two most confident malignant
calls (highest P(malignant)) and the two most confident non-malignant calls
(lowest P(malignant)).

Usage:
    python scripts/export_paper_examples.py --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import build_image_index, build_lesion_table, load_raw_tables

DISPLAY_FIELDS = ["age", "sex", "location_group", "referral_diagnosis", "diameter"]
N_PER_CLASS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--oof-csv", type=Path, default=Path("results/oof_predictions_channel_gated_qualityFalse.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_examples"))
    args = parser.parse_args()

    images_root = args.images_root or args.data_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables, verbose=False)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root, verbose=False)
    oof = pd.read_csv(args.oof_csv)

    # oof already carries binary_label/aux_label/histo_confirmed (the ground
    # truth used to produce the predictions) -- drop lesion_df's copies
    # before merging so the merge doesn't silently suffix them to _x/_y.
    lesion_meta = lesion_df.drop(columns=["binary_label", "aux_label", "histo_confirmed"])
    merged = oof.merge(lesion_meta, on="lesion_id", how="left")
    merged = merged[merged["has_binary_label"] & merged["correct"]]

    def has_image(row):
        return len(image_index_df[
            (image_index_df["lesion_id"] == row["lesion_id"]) &
            (image_index_df["image_id"] == row["diagnosis_image_id"])
        ]) > 0

    merged = merged[merged.apply(has_image, axis=1)]

    best_malignant = merged[merged["binary_label"] == 1].sort_values("pred_prob", ascending=False).head(N_PER_CLASS)
    best_nonmalignant = merged[merged["binary_label"] == 0].sort_values("pred_prob", ascending=True).head(N_PER_CLASS)

    rows = []
    for label, df in [("high_confidence_malignant", best_malignant), ("high_confidence_nonmalignant", best_nonmalignant)]:
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            diag_img_rows = image_index_df[
                (image_index_df["lesion_id"] == row["lesion_id"]) &
                (image_index_df["image_id"] == row["diagnosis_image_id"])
            ]
            img_path = diag_img_rows.iloc[0]["path"]
            dest_name = f"{row['lesion_id']}_{row['diagnosis_image_id']}.png"
            shutil.copy(img_path, args.out_dir / dest_name)

            record = {"category": f"{label}_{rank}", "lesion_id": row["lesion_id"], "image_file": dest_name,
                      "modality": diag_img_rows.iloc[0]["modality"]}
            for f in DISPLAY_FIELDS:
                record[f] = row.get(f)
            record["mean_image_rating"] = row.get("mean_image_rating")
            record["histo_confirmed"] = row.get("histo_confirmed")
            record["ground_truth"] = "Malignant" if row["binary_label"] == 1 else "Non-malignant"
            record["predicted"] = "Malignant" if row["pred_label"] == 1 else "Non-malignant"
            record["pred_prob_malignant"] = round(float(row["pred_prob"]), 3)
            rows.append(record)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_dir / "examples.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved {len(rows)} example images + examples.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
