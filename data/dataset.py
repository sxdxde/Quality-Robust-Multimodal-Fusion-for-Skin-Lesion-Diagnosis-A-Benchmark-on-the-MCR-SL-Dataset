"""PyTorch Dataset for MCR-SL lesion classification.

Design decisions (logged here per CLAUDE.md's "log the choice" instruction):
- Training samples: at N=240 lesions, `train_on_all_dermoscopic_images=True`
  (config.py) uses every dermoscopy-modality image per lesion as a separate
  training sample (more signal via natural augmentation across viewpoints/
  lighting), falling back to the lesion's diagnosis_image_id image (any
  modality) when a lesion has zero dermoscopy images. When False, or for
  val/test splits (always), exactly one image per lesion is used — the
  diagnosis_image_id image — so evaluation always reports one prediction per
  lesion, matching the eval protocol.
- Numeric fields (age/height/weight/diameter) are z-score normalized using
  train-fold statistics only, refit per fold (data/dataset.py:
  fit_numeric_stats), never leaking val/test-fold values into the mean/std.
- Categorical vocabularies are fixed globally from data/schema.py (a
  structural fact about the field's domain, not fold-specific).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.loader import parse_numeric_with_unknown, encode_categorical
from data.preprocessing import apply_dermoscopy_preprocessing
from data.schema import CATEGORICAL_FIELDS, NUMERICAL_FIELDS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def fit_numeric_stats(lesion_df: pd.DataFrame, subject_ids: set) -> dict[str, tuple[float, float]]:
    """Fit (mean, std) per numeric field on train-fold subjects only."""
    train_df = lesion_df[lesion_df["subject_id"].isin(subject_ids)]
    stats = {}
    for field in NUMERICAL_FIELDS:
        parsed = train_df[field].apply(lambda v: parse_numeric_with_unknown(v)[0])
        valid = parsed.dropna()
        mean = float(valid.mean()) if len(valid) > 0 else 0.0
        std = float(valid.std()) if len(valid) > 1 and valid.std() > 0 else 1.0
        stats[field] = (mean, std)
    return stats


def _build_samples(lesion_df: pd.DataFrame, image_index_df: pd.DataFrame, subject_ids: set,
                    use_all_images: bool, verbose: bool) -> list[dict]:
    """use_all_images=True: every dermoscopy-modality image per lesion is a
    separate sample (falls back to the diagnosis_image_id image, any
    modality, if a lesion has zero dermoscopy images). False: exactly the
    diagnosis_image_id image, one sample per lesion.
    """
    subset = lesion_df[lesion_df["subject_id"].isin(subject_ids)]
    images_by_lesion = {lid: g for lid, g in image_index_df.groupby("lesion_id")}

    samples = []
    n_no_image = 0
    for _, row in subset.iterrows():
        lesion_id = row["lesion_id"]
        imgs = images_by_lesion.get(lesion_id, image_index_df.iloc[0:0])

        if use_all_images:
            use_imgs = imgs[imgs["modality"] == "dermoscopy"]
            if len(use_imgs) == 0:
                use_imgs = imgs[imgs["image_id"] == row["diagnosis_image_id"]]
        else:
            use_imgs = imgs[imgs["image_id"] == row["diagnosis_image_id"]]

        if len(use_imgs) == 0:
            n_no_image += 1
            continue

        row_dict = row.to_dict()
        for _, img_row in use_imgs.iterrows():
            samples.append({**row_dict, "image_id": img_row["image_id"], "path": img_row["path"]})

    if verbose and n_no_image > 0:
        print(f"[_build_samples] {n_no_image}/{len(subset)} lesions had no usable image on disk for this split — dropped")
    return samples


class MCRSLDataset(Dataset):
    def __init__(
        self,
        lesion_df: pd.DataFrame,
        image_index_df: pd.DataFrame,
        subject_ids: set,
        numeric_stats: dict[str, tuple[float, float]],
        image_size: int,
        split: str,  # "train" | "val" | "test"
        use_all_dermoscopic: bool = True,
        verbose: bool = True,
        use_preprocessing: bool = False,
        multi_image_eval: bool = False,
    ):
        """use_all_dermoscopic: for split=="train" only, use every dermoscopy
        image per lesion as a separate sample (more training signal at
        N=240). multi_image_eval: for split in ("val","test"), same
        multi-image sampling instead of the single diagnosis_image_id image —
        predictions must be averaged back to one-per-lesion by the caller
        (train.py / robustness_analysis.py), this class does not aggregate.
        use_preprocessing: apply dermoscopy hair-removal + color
        normalization (data/preprocessing.py) before the standard transform.
        """
        assert split in ("train", "val", "test")
        self.split = split
        self.numeric_stats = numeric_stats
        self.use_preprocessing = use_preprocessing
        self.transform = build_transforms(image_size, train=(split == "train"))
        use_all_images = (split == "train" and use_all_dermoscopic) or (split != "train" and multi_image_eval)
        self.samples = _build_samples(
            lesion_df, image_index_df, subject_ids, use_all_images=use_all_images, verbose=verbose,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = Image.open(s["path"]).convert("RGB")
        if self.use_preprocessing:
            image = apply_dermoscopy_preprocessing(image)
        image = self.transform(image)

        categorical = {}
        for field, vocab in CATEGORICAL_FIELDS.items():
            categorical[field] = torch.tensor(encode_categorical(s.get(field), vocab), dtype=torch.long)

        numerical = {}
        numerical_missing = {}
        for field in NUMERICAL_FIELDS:
            val, missing = parse_numeric_with_unknown(s.get(field))
            mean, std = self.numeric_stats[field]
            z = 0.0 if missing else (val - mean) / std
            numerical[field] = torch.tensor(z, dtype=torch.float32)
            numerical_missing[field] = torch.tensor(1.0 if missing else 0.0, dtype=torch.float32)

        binary_label = s.get("binary_label")
        has_binary_label = not (binary_label is None or (isinstance(binary_label, float) and np.isnan(binary_label)))

        aux_label = s.get("aux_label")
        has_aux_label = not (aux_label is None or (isinstance(aux_label, float) and np.isnan(aux_label)))

        rating = s.get("mean_image_rating")
        has_rating = not (rating is None or (isinstance(rating, float) and np.isnan(rating)))

        return {
            "image": image,
            "categorical": categorical,
            "numerical": numerical,
            "numerical_missing": numerical_missing,
            "binary_label": torch.tensor(float(binary_label) if has_binary_label else 0.0, dtype=torch.float32),
            "has_binary_label": torch.tensor(has_binary_label, dtype=torch.bool),
            "aux_label": torch.tensor(int(aux_label) if has_aux_label else -100, dtype=torch.long),
            "quality_target": torch.tensor(float(rating) / 10.0 if has_rating else 0.0, dtype=torch.float32),
            "has_quality_rating": torch.tensor(has_rating, dtype=torch.bool),
            "lesion_id": s["lesion_id"],
            "histo_confirmed": torch.tensor(bool(s.get("histo_confirmed", False)), dtype=torch.bool),
        }


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "binary_label": torch.stack([b["binary_label"] for b in batch]),
        "has_binary_label": torch.stack([b["has_binary_label"] for b in batch]),
        "aux_label": torch.stack([b["aux_label"] for b in batch]),
        "quality_target": torch.stack([b["quality_target"] for b in batch]),
        "has_quality_rating": torch.stack([b["has_quality_rating"] for b in batch]),
        "lesion_id": [b["lesion_id"] for b in batch],
        "histo_confirmed": torch.stack([b["histo_confirmed"] for b in batch]),
    }
    out["categorical"] = {
        field: torch.stack([b["categorical"][field] for b in batch]) for field in CATEGORICAL_FIELDS
    }
    out["numerical"] = {
        field: torch.stack([b["numerical"][field] for b in batch]) for field in NUMERICAL_FIELDS
    }
    out["numerical_missing"] = {
        field: torch.stack([b["numerical_missing"][field] for b in batch]) for field in NUMERICAL_FIELDS
    }
    return out
