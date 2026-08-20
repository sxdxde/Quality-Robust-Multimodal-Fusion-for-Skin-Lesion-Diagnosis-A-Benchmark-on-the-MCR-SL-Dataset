"""Exports a small, representative (not cherry-picked) set of example
lesions for the paper's input/output figure and table: real metadata, real
ground truth, real channel-gated predictions, and a copy of the actual
diagnosis image file for each.

Selection rule, deterministic and not chosen to flatter the model: within
each of the four confusion-matrix categories (true positive, true negative,
false negative, false positive), take the lesion with the lowest lesion_id
alphabetically that has a usable image on disk. This surfaces at least one
real failure case (the false negative) alongside successes, rather than
only showing wins.

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
from data.schema import CATEGORICAL_FIELDS, NUMERICAL_FIELDS

DISPLAY_FIELDS = ["age", "sex", "location_group", "referral_diagnosis", "diameter"]


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

    merged = oof.merge(lesion_df, on="lesion_id", how="left")
    merged = merged[merged["has_binary_label"]]

    def category(row):
        if row["binary_label"] == 1 and row["pred_label"] == 1:
            return "true_positive"
        if row["binary_label"] == 0 and row["pred_label"] == 0:
            return "true_negative"
        if row["binary_label"] == 1 and row["pred_label"] == 0:
            return "false_negative"
        return "false_positive"

    merged["category"] = merged.apply(category, axis=1)

    rows = []
    for cat in ["true_positive", "true_negative", "false_negative", "false_positive"]:
        candidates = merged[merged["category"] == cat].sort_values("lesion_id")
        for _, row in candidates.iterrows():
            diag_img_rows = image_index_df[
                (image_index_df["lesion_id"] == row["lesion_id"]) &
                (image_index_df["image_id"] == row["diagnosis_image_id"])
            ]
            if len(diag_img_rows) == 0:
                continue
            img_path = diag_img_rows.iloc[0]["path"]
            dest_name = f"{row['lesion_id']}_{row['diagnosis_image_id']}.png"
            shutil.copy(img_path, args.out_dir / dest_name)

            record = {"category": cat, "lesion_id": row["lesion_id"], "image_file": dest_name,
                      "modality": diag_img_rows.iloc[0]["modality"]}
            for f in DISPLAY_FIELDS:
                record[f] = row.get(f)
            record["mean_image_rating"] = row.get("mean_image_rating")
            record["histo_confirmed"] = row.get("histo_confirmed")
            record["ground_truth"] = "Malignant" if row["binary_label"] == 1 else "Non-malignant"
            record["predicted"] = "Malignant" if row["pred_label"] == 1 else "Non-malignant"
            record["pred_prob_malignant"] = round(float(row["pred_prob"]), 3)
            rows.append(record)
            break  # one example per category

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_dir / "examples.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved {len(rows)} example images + examples.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
