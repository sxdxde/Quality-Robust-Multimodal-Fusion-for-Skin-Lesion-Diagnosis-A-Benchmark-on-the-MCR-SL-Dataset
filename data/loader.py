"""Loads and joins the six MCR-SL tables into one per-lesion DataFrame ready
for fold splitting and Dataset construction. See data/schema.py for the
verified column names/dtypes this assumes, and the design decisions
(dropped fields, label derivation) documented there.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import (
    CATEGORICAL_FIELDS,
    MALIGNANT,
    NON_MALIGNANT,
    UNIFIED_DIAGNOSIS_CLASSES,
    USABLE_RATING_EXPERTS,
)

MODALITY_TO_DIR = {"clinical": "clinical", "dermoscopy": "dermoscopic"}


def parse_numeric_with_unknown(value) -> tuple[float, bool]:
    """Returns (value_or_nan, is_missing). Handles the literal "unknown"
    string marker used throughout MCR-SL's numeric-looking object columns,
    plus genuine NaN/None.
    """
    if value is None:
        return float("nan"), True
    if isinstance(value, float) and math.isnan(value):
        return float("nan"), True
    if isinstance(value, str) and value.strip().lower() == "unknown":
        return float("nan"), True
    try:
        return float(value), False
    except (TypeError, ValueError):
        return float("nan"), True


def encode_categorical(value, vocab: list[str]) -> int:
    """Maps a raw categorical value to its vocab index, or `len(vocab)`
    ("unknown" slot) for the literal "unknown" string, NaN, or any value not
    in the fixed vocab. Never imputes.
    """
    if isinstance(value, str) and value in vocab:
        return vocab.index(value)
    return len(vocab)


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        "lesion": pd.read_excel(data_dir / "lesion.xlsx"),
        "subject": pd.read_excel(data_dir / "subject.xlsx"),
        "image": pd.read_excel(data_dir / "image.xlsx"),
        "dermatology_diagnosis": pd.read_excel(data_dir / "dermatology_diagnosis.xlsx"),
        "histopathology_diagnosis": pd.read_excel(data_dir / "histopathology_diagnosis.xlsx"),
        "unified_diagnosis": pd.read_excel(data_dir / "unified_diagnosis.xlsx"),
    }


def compute_mean_image_rating(lesion_df: pd.DataFrame, derm_df: pd.DataFrame) -> pd.Series:
    """Mean image_rating across usable experts (E001/E003/E004), restricted
    to each lesion's diagnosis_image_id row (experts only rate the single
    image used for diagnosis). Returns a Series indexed by lesion_id, NaN for
    lesions with no such row (verified: L0013, L0205).
    """
    diag_pairs = set(zip(lesion_df["lesion_id"], lesion_df["diagnosis_image_id"]))
    is_diag = derm_df.apply(lambda r: (r["lesion_id"], r["image_id"]) in diag_pairs, axis=1)
    usable = derm_df[is_diag & derm_df["expert_id"].isin(USABLE_RATING_EXPERTS)]
    return usable.groupby("lesion_id")["image_rating"].mean()


def build_lesion_table(tables: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """One row per lesion (240 total), with subject metadata merged in,
    binary/aux labels derived, mean quality rating, and histopath-confirmed
    flag. Malignancy=="unknown" lesions keep binary_label=NaN (excluded from
    the binary task, not dropped from the table — they may still be usable
    for the aux/quality analyses).
    """
    lesion = tables["lesion"].copy()
    subject = tables["subject"].copy()
    unified = tables["unified_diagnosis"].copy()
    histo = tables["histopathology_diagnosis"].copy()
    derm = tables["dermatology_diagnosis"].copy()

    df = lesion.merge(subject, on="subject_id", how="left", validate="many_to_one")

    unified_map = unified.set_index("lesion_id")["unified_diagnosis"]
    df["unified_diagnosis"] = df["lesion_id"].map(unified_map)

    df["binary_label"] = df["malignancy"].map({MALIGNANT: 1.0, NON_MALIGNANT: 0.0})
    n_excluded_binary = df["binary_label"].isna().sum()

    aux_class_to_idx = {c: i for i, c in enumerate(UNIFIED_DIAGNOSIS_CLASSES)}
    df["aux_label"] = df["unified_diagnosis"].map(aux_class_to_idx)  # NaN for "UNK"/missing
    n_excluded_aux = df["aux_label"].isna().sum()

    df["mean_image_rating"] = df["lesion_id"].map(compute_mean_image_rating(lesion, derm))

    histo_lesion_ids = set(histo["lesion_id"])
    df["histo_confirmed"] = df["lesion_id"].isin(histo_lesion_ids)

    if verbose:
        print(f"[build_lesion_table] {len(df)} lesions total")
        print(f"  binary task: {n_excluded_binary} excluded (malignancy=='unknown')")
        print(f"  aux 9-class task: {n_excluded_aux} excluded (unified_diagnosis=='UNK' or missing)")
        print(f"  histopath-confirmed: {df['histo_confirmed'].sum()}")
        print(f"  lesions missing a quality rating: {df['mean_image_rating'].isna().sum()}")

    return df


def build_image_index(tables: dict[str, pd.DataFrame], lesion_ids: set[str], images_root: Path, verbose: bool = True) -> pd.DataFrame:
    """Inner-joins image.xlsx to the given lesion_id set, drops rows whose
    file isn't actually on disk (defensive — expected to be redundant with
    the lesion_id join per the verified 263-row orphan/missing-file match),
    and resolves each row's absolute file path.
    """
    image = tables["image"].copy()
    n_before = len(image)
    image = image[image["lesion_id"].isin(lesion_ids)].copy()
    n_after_join = len(image)

    images_root = Path(images_root)
    image["path"] = image.apply(
        lambda r: images_root / MODALITY_TO_DIR[r["modality"]] / f"{r['image_id']}.png", axis=1
    )
    exists_mask = image["path"].apply(lambda p: p.exists())
    image = image[exists_mask].copy()

    if verbose:
        print(f"[build_image_index] {n_before} raw image rows -> {n_after_join} after lesion_id join "
              f"-> {len(image)} after file-existence check "
              f"({n_after_join - len(image)} missing files dropped)")

    return image
