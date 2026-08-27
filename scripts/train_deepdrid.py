"""Cross-dataset check: does the MCR-SL `hard_mining` quality-adaptive loss
transfer to a second medical imaging domain?

SECONDARY, SUPPORTING EXPERIMENT — deliberately smaller in scope and weaker in
evidentiary standard than the MCR-SL result:
  * DeepDRiD's OWN provided train/validation split, used as a SINGLE SPLIT.
    No k-fold CV, so there is no fold-to-fold variance and no error bar. This
    is explicitly NOT the shuffled-control standard used on MCR-SL.
  * Image encoder only (EfficientNet-B0) — no metadata encoder, no fusion.
    DeepDRiD has no MCR-SL-style structured metadata, and dropping fusion
    isolates the loss function, which is the only thing under test.

TASK
    Referable DR: per-image DR grade >= 2 (moderate NPDR or worse), the
    standard binarisation in DR-screening literature. Each image is ONE eye
    (image_id like `1_l1` = patient 1, left eye, photo 1), so the label comes
    from left_eye_DR_Level or right_eye_DR_Level accordingly — exactly one of
    which is non-null per row. Using patient_DR_Level instead would wrongly
    give both eyes the same label when one eye can be worse than the other.

QUALITY SIGNAL — `Clarity`, chosen over three alternatives, justified:
    DeepDRiD exposes FOUR quality sub-scores, not one:
      Overall quality  binary 0/1        - most direct in meaning, but only two
                                           levels, so hard_mining would collapse
                                           to just {1.5, 0.5} with no gradation
      Clarity          1/4/6/8/10        - CHOSEN
      Field definition 1/4/6/8/10        - about framing (disc/macula centring),
                                           not legibility
      Artifact         0/1/4/6/8/10      - INVERTED (higher = worse) and
                                           zero-inflated (599/1200 at 0)
    `Clarity` is chosen because (a) it shares MCR-SL's exact [1,10] range, so
    the hard_mining formula applies UNCHANGED rather than needing an invented
    remapping; (b) it is genuinely graded (5 levels); and (c) its rubric is
    explicitly about how much diagnostic detail is visible ("can identify
    Level N vascular arch and X lesions") — the closest analogue to what
    MCR-SL's experts rated. Higher = better, same direction as MCR-SL.

        w = 1.5 - (clarity - 1) / 9      # [1,10] -> [1.5, 0.5]

TWO CONFIGS ONLY (per the task spec — no third, no sweep):
    1. control : encoder + SAM + TTA, NO quality weighting
    2. test    : encoder + SAM + TTA, + hard_mining quality weighting
    Everything else identical, same seed, so the loss is the only difference.

Usage:
    python scripts/train_deepdrid.py --root ~/deepdrid --config control
    python scripts/train_deepdrid.py --root ~/deepdrid --config hard_mining
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import compute_binary_metrics
from models.heads import BinaryHead
from models.image_encoder import EfficientNetB0Encoder
from models.sam_optimizer import SAM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REFERABLE_THRESHOLD = 2  # >= moderate NPDR
QUALITY_COL = "Clarity"


def build_split(root: Path, split: str) -> pd.DataFrame:
    """One row per image with a referable-DR label and a Clarity score."""
    base = root / "regular_fundus_images" / f"regular-fundus-{split}"
    df = pd.read_csv(base / f"regular-fundus-{split}.csv")

    # Each row is one eye; exactly one of the two eye columns is populated.
    eye = df["image_id"].str.extract(r"_(l|r)\d+$")[0]
    grade = np.where(eye == "l", df["left_eye_DR_Level"], df["right_eye_DR_Level"])
    df["dr_grade"] = pd.to_numeric(pd.Series(grade, index=df.index), errors="coerce")

    n_missing = int(df["dr_grade"].isna().sum())
    if n_missing:
        print(f"  [{split}] dropping {n_missing} rows with no eye-specific DR grade")
        df = df.dropna(subset=["dr_grade"])

    df["label"] = (df["dr_grade"] >= REFERABLE_THRESHOLD).astype(float)
    # Paths in the CSV are Windows-style and omit the Images/ component —
    # rebuild from ids instead of trusting the string.
    df["path"] = df.apply(
        lambda r: base / "Images" / str(r["patient_id"]) / f"{r['image_id']}.jpg", axis=1)

    missing = [p for p in df["path"] if not p.exists()]
    if missing:
        raise SystemExit(f"  [{split}] {len(missing)} image files missing on disk, e.g. {missing[:3]}")

    print(f"  [{split}] {len(df)} images, {len(df['patient_id'].unique())} patients, "
          f"referable={int(df['label'].sum())} ({df['label'].mean():.1%})")
    return df.reset_index(drop=True)


def quality_weight(clarity: np.ndarray) -> np.ndarray:
    """hard_mining, formula unchanged from MCR-SL: up-weight low quality."""
    return 1.5 - (clarity - 1.0) / 9.0


class DeepDRiDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, train: bool, use_quality: bool):
        self.df = df
        self.use_quality = use_quality
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self.tf(Image.open(r["path"]).convert("RGB"))
        w = float(quality_weight(np.array(r[QUALITY_COL], dtype=float))) if self.use_quality else 1.0
        return {
            "image": img,
            "label": torch.tensor(float(r["label"]), dtype=torch.float32),
            "weight": torch.tensor(w, dtype=torch.float32),
        }


class DRModel(nn.Module):
    """Image encoder + binary head. No metadata, no fusion — deliberately."""

    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=True)
        self.head = BinaryHead(EfficientNetB0Encoder.FEATURE_DIM)

    def forward(self, x):
        pooled, _ = self.encoder(x)
        return self.head(pooled)


@torch.no_grad()
def evaluate(model, loader, device, tta: bool):
    model.eval()
    probs, labels = [], []
    for b in loader:
        x = b["image"].to(device)
        p = torch.sigmoid(model(x))
        if tta:
            p = (p + torch.sigmoid(model(torch.flip(x, dims=[-1])))) / 2
        probs.append(p.cpu().numpy())
        labels.append(b["label"].numpy())
    y_score = np.concatenate(probs)
    y_true = np.concatenate(labels).astype(int)
    return compute_binary_metrics(y_true, (y_score >= 0.5).astype(int), y_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "deepdrid")
    ap.add_argument("--config", choices=["control", "hard_mining"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--sam-rho", type=float, default=0.05)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ledger", type=Path, default=Path("results/deepdrid_ledger.csv"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_quality = args.config == "hard_mining"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 74)
    print(f"DeepDRiD cross-dataset check — config={args.config}  (SINGLE SPLIT, no CV)")
    print(f"  quality weighting: {'hard_mining on ' + QUALITY_COL if use_quality else 'NONE (control)'}")
    print("=" * 74)

    train_df = build_split(args.root, "training")
    val_df = build_split(args.root, "validation")

    if use_quality:
        w = quality_weight(train_df[QUALITY_COL].to_numpy(dtype=float))
        print(f"  quality weights: min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f}")

    train_ds = DeepDRiDDataset(train_df, args.image_size, True, use_quality)
    val_ds = DeepDRiDDataset(val_df, args.image_size, False, False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_pos = float(train_df["label"].sum())
    n_neg = float(len(train_df) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32, device=device)
    print(f"  pos_weight (image-level): {pos_weight.item():.3f}")

    model = DRModel().to(device)
    opt = SAM(model.parameters(), torch.optim.AdamW, rho=args.sam_rho,
              lr=args.lr, weight_decay=args.weight_decay)

    best_bacc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in train_ld:
            x = b["image"].to(device)
            y = b["label"].to(device)
            sw = b["weight"].to(device) if use_quality else None

            loss = BinaryHead.loss(model(x), y, pos_weight=pos_weight, sample_weight=sw)
            loss.backward()
            opt.first_step(zero_grad=True)
            BinaryHead.loss(model(x), y, pos_weight=pos_weight, sample_weight=sw).backward()
            opt.second_step(zero_grad=True)
            tot += loss.item()
            nb += 1

        m = evaluate(model, val_ld, device, tta=False)
        print(f"  epoch {ep+1}/{args.epochs} train_loss={tot/max(nb,1):.4f} "
              f"val_bacc={m['balanced_accuracy']:.4f} val_auroc={m['auroc']:.4f}")
        if m["balanced_accuracy"] > best_bacc:
            best_bacc, best_epoch = m["balanced_accuracy"], ep
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final = evaluate(model, val_ld, device, tta=True)
    mins = (time.time() - t0) / 60

    print("\n" + "=" * 74)
    print(f"FINAL ({args.config}, best epoch {best_epoch+1}, TTA on, single split)")
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity_malignant", "specificity", "auroc"):
        print(f"  {k:24s} {final[k]:.4f}")
    print(f"  trained in {mins:.1f} min")
    print("=" * 74)

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not args.ledger.exists()
    with open(args.ledger, "a", newline="") as f:
        wtr = csv.writer(f)
        if new:
            wtr.writerow(["timestamp", "config", "quality_signal", "split", "n_train", "n_val",
                          "accuracy", "balanced_accuracy", "macro_f1", "sensitivity",
                          "specificity", "auroc", "best_epoch", "notes"])
        wtr.writerow([
            datetime.datetime.now(datetime.timezone.utc).isoformat(), args.config,
            QUALITY_COL if use_quality else "none", "provided train/val (single split)",
            len(train_df), len(val_df),
            f"{final['accuracy']:.4f}", f"{final['balanced_accuracy']:.4f}",
            f"{final['macro_f1']:.4f}", f"{final['sensitivity_malignant']:.4f}",
            f"{final['specificity']:.4f}", f"{final['auroc']:.4f}", best_epoch + 1,
            "single split, NOT cross-validated; secondary supporting check",
        ])
    print(f"appended to {args.ledger}")


if __name__ == "__main__":
    main()
