"""Subject-disjoint stratified 5-fold CV training entrypoint.

For each of the 5 CV rotations: one fold is held out as the final test fold
(reported metrics), a second fold is held out as a validation fold used only
for checkpoint selection (best epoch by val balanced accuracy) and early
stopping — never for the reported numbers, so there's no test-fold peeking
and no hyperparameter tuning against the final metrics.

Validation during training is always the fast/plain path (no TTA, no
multi-image averaging, no preprocessing beyond what the run itself uses) so
checkpoint selection stays honest and cheap. `--tta` / `--multi-image-eval`
only affect the FINAL test-fold evaluation that gets logged to the ledger.

Usage:
    python train.py --variant channel_gated --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset
    python train.py --variant channel_gated --run-tag channel_gated_focal --focal-gamma 2.0 --data-dir ...
    python train.py --variant channel_gated --run-tag channel_gated_preprocessed --use-preprocessing --data-dir ...
    python train.py --variant channel_gated --run-tag channel_gated_contrastive --use-contrastive --data-dir ...
    python train.py --variant channel_gated --run-tag channel_gated_optimizerv2 --optimizer adamw_cosine_discriminative --data-dir ...
"""
import argparse
import copy
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from data.dataset import MCRSLDataset, collate_fn, fit_numeric_stats
from data.folds import fold_summary, make_subject_disjoint_folds
from data.loader import build_image_index, build_lesion_table, load_raw_tables
from evaluate import aggregate_fold_metrics, append_to_ledger, compute_binary_metrics
from models.heads import AuxDiagnosisHead, BinaryHead, QualityHead, supervised_contrastive_loss
from models.model import MCRSLModel
from models.sam_optimizer import SAM


def compute_binary_pos_weight(lesion_df, subject_ids) -> torch.Tensor:
    labels = lesion_df[lesion_df["subject_id"].isin(subject_ids)]["binary_label"].dropna()
    n_pos = (labels == 1.0).sum()
    n_neg = (labels == 0.0).sum()
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def compute_aux_class_weights(lesion_df, subject_ids, num_classes: int) -> torch.Tensor:
    labels = lesion_df[lesion_df["subject_id"].isin(subject_ids)]["aux_label"].dropna().astype(int)
    counts = np.array([(labels == c).sum() for c in range(num_classes)])
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def move_batch(batch, device):
    batch["image"] = batch["image"].to(device)
    batch["binary_label"] = batch["binary_label"].to(device)
    batch["has_binary_label"] = batch["has_binary_label"].to(device)
    batch["aux_label"] = batch["aux_label"].to(device)
    batch["quality_target"] = batch["quality_target"].to(device)
    batch["has_quality_rating"] = batch["has_quality_rating"].to(device)
    batch["categorical"] = {k: v.to(device) for k, v in batch["categorical"].items()}
    batch["numerical"] = {k: v.to(device) for k, v in batch["numerical"].items()}
    batch["numerical_missing"] = {k: v.to(device) for k, v in batch["numerical_missing"].items()}
    return batch


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    """'adam': plain Adam, single LR (original behavior).
    'adamw_cosine_discriminative': AdamW with a lower LR on the pretrained
    EfficientNet-B0 backbone than on the newly-initialized heads/fusion/
    metadata encoder, plus a cosine LR schedule over cfg.epochs.
    'sam_adamw': Sharpness-Aware Minimization wrapping AdamW (single LR, no
    schedule — kept isolated from the discriminative-LR/cosine combo above
    so its effect isn't confounded with that already-tested variant).
    """
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = None
    elif cfg.optimizer == "adamw_cosine_discriminative":
        backbone_params = list(model.image_encoder.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
            {"params": head_params, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.optimizer == "sam_adamw":
        optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=cfg.sam_rho, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = None
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    return optimizer, scheduler


def compute_loss(model, batch, device, cfg: TrainConfig, pos_weight, aux_weights) -> torch.Tensor:
    out = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])

    mask = batch["has_binary_label"]
    if mask.sum() > 0:
        binary_loss = BinaryHead.loss(
            out["binary_logits"][mask], batch["binary_label"][mask],
            pos_weight=pos_weight, focal_gamma=cfg.focal_gamma,
        )
    else:
        binary_loss = torch.tensor(0.0, device=device)

    aux_loss = AuxDiagnosisHead.loss(out["aux_logits"], batch["aux_label"], class_weights=aux_weights)
    loss = binary_loss + cfg.aux_loss_weight * aux_loss

    if cfg.quality_aware:
        qmask = batch["has_quality_rating"]
        if qmask.sum() > 0:
            quality_loss = QualityHead.loss(out["quality_pred"][qmask], batch["quality_target"][qmask])
            loss = loss + cfg.quality_loss_weight * quality_loss

    if cfg.use_contrastive and mask.sum() > 1:
        contrastive_loss = supervised_contrastive_loss(out["fused_embedding"][mask], batch["binary_label"][mask])
        loss = loss + cfg.contrastive_weight * contrastive_loss

    return loss


def run_epoch(model, loader, device, optimizer, cfg: TrainConfig, pos_weight, aux_weights, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0
    is_sam = isinstance(optimizer, SAM)

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = move_batch(batch, device)

            if train and is_sam:
                loss = compute_loss(model, batch, device, cfg, pos_weight, aux_weights)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                compute_loss(model, batch, device, cfg, pos_weight, aux_weights).backward()
                optimizer.second_step(zero_grad=True)
            else:
                loss = compute_loss(model, batch, device, cfg, pos_weight, aux_weights)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(model, loader, device, tta: bool = False) -> pd.DataFrame:
    """One row per sample with lesion_id/prob/label — NOT aggregated to one
    row per lesion (do that separately if the loader yields multiple images
    per lesion, e.g. multi_image_eval=True on the dataset).
    """
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["categorical"], batch["numerical"], batch["numerical_missing"])
        probs = torch.sigmoid(out["binary_logits"])

        if tta:
            flipped = torch.flip(batch["image"], dims=[-1])
            out_flip = model(flipped, batch["categorical"], batch["numerical"], batch["numerical_missing"])
            probs = (probs + torch.sigmoid(out_flip["binary_logits"])) / 2

        probs = probs.cpu().numpy()
        has_label = batch["has_binary_label"].cpu().numpy()
        labels = batch["binary_label"].cpu().numpy()
        for i, lesion_id in enumerate(batch["lesion_id"]):
            rows.append({"lesion_id": lesion_id, "prob": float(probs[i]), "has_binary_label": bool(has_label[i]), "binary_label": float(labels[i])})
    return pd.DataFrame(rows)


def metrics_from_predictions(df: pd.DataFrame, aggregate_by_lesion: bool = False) -> dict:
    df = df[df["has_binary_label"]]
    if aggregate_by_lesion:
        df = df.groupby("lesion_id").agg(prob=("prob", "mean"), binary_label=("binary_label", "first")).reset_index()
    y_true = df["binary_label"].astype(int).to_numpy()
    y_score = df["prob"].to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    return compute_binary_metrics(y_true, y_pred, y_score)


def evaluate_loader(model, loader, device) -> dict:
    """Fast/plain path used during training for val-fold checkpoint selection."""
    return metrics_from_predictions(predict(model, loader, device, tta=False), aggregate_by_lesion=False)


def run_cv(cfg: TrainConfig, lesion_df, image_index_df, images_root: Path, device: torch.device):
    run_tag = cfg.resolved_run_tag()
    valid_binary = lesion_df.dropna(subset=["binary_label"])
    subject_malignant_count = (
        valid_binary.groupby("subject_id")["binary_label"].sum().to_dict()
    )
    # subjects with lesions but none carrying a valid binary label still need a fold assignment
    for sid in lesion_df["subject_id"].unique():
        subject_malignant_count.setdefault(sid, 0)

    assignment = make_subject_disjoint_folds(subject_malignant_count, n_folds=cfg.n_folds, seed=cfg.seed)
    print(fold_summary(subject_malignant_count, assignment, cfg.n_folds))

    subjects_by_fold = {f: set() for f in range(cfg.n_folds)}
    for sid, f in assignment.items():
        subjects_by_fold[f].add(sid)

    fold_metrics = []
    for test_fold in range(cfg.n_folds):
        val_fold = (test_fold + 1) % cfg.n_folds
        train_folds = [f for f in range(cfg.n_folds) if f not in (test_fold, val_fold)]

        train_subjects = set().union(*[subjects_by_fold[f] for f in train_folds])
        val_subjects = subjects_by_fold[val_fold]
        test_subjects = subjects_by_fold[test_fold]

        numeric_stats = fit_numeric_stats(lesion_df, train_subjects)

        train_ds = MCRSLDataset(lesion_df, image_index_df, train_subjects, numeric_stats, cfg.image_size, "train",
                                 cfg.train_on_all_dermoscopic_images, use_preprocessing=cfg.use_dermoscopy_preprocessing)
        val_ds = MCRSLDataset(lesion_df, image_index_df, val_subjects, numeric_stats, cfg.image_size, "val",
                               False, use_preprocessing=cfg.use_dermoscopy_preprocessing)
        test_ds = MCRSLDataset(lesion_df, image_index_df, test_subjects, numeric_stats, cfg.image_size, "test",
                                False, use_preprocessing=cfg.use_dermoscopy_preprocessing, multi_image_eval=cfg.multi_image_eval)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        pos_weight = compute_binary_pos_weight(lesion_df, train_subjects).to(device)
        aux_weights = compute_aux_class_weights(lesion_df, train_subjects, cfg.num_aux_classes).to(device)

        model = MCRSLModel(
            variant=cfg.variant,
            num_aux_classes=cfg.num_aux_classes,
            quality_aware=cfg.quality_aware,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            fusion_dropout=cfg.fusion_dropout,
        ).to(device)
        optimizer, scheduler = build_optimizer(model, cfg)

        best_val_bacc = -1.0
        best_state = None
        for epoch in range(cfg.epochs):
            train_loss = run_epoch(model, train_loader, device, optimizer, cfg, pos_weight, aux_weights, train=True)
            if scheduler is not None:
                scheduler.step()
            val_metrics = evaluate_loader(model, val_loader, device)
            print(f"[{run_tag} fold {test_fold}] epoch {epoch+1}/{cfg.epochs} train_loss={train_loss:.4f} "
                  f"val_bacc={val_metrics['balanced_accuracy']:.4f} val_auroc={val_metrics['auroc']:.4f}")
            if val_metrics["balanced_accuracy"] > best_val_bacc:
                best_val_bacc = val_metrics["balanced_accuracy"]
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        test_df = predict(model, test_loader, device, tta=cfg.use_tta)
        test_metrics = metrics_from_predictions(test_df, aggregate_by_lesion=cfg.multi_image_eval)
        print(f"[{run_tag} fold {test_fold}] TEST accuracy={test_metrics['accuracy']:.4f} "
              f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} auroc={test_metrics['auroc']:.4f}")

        fold_metrics.append(test_metrics)
        append_to_ledger(cfg.results_ledger_path, run_tag, cfg.quality_aware, test_fold, cfg.n_folds, cfg.seed, test_metrics, notes=cfg.notes)

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / f"{run_tag}_quality{cfg.quality_aware}_fold{test_fold}.pt")

    agg = aggregate_fold_metrics(fold_metrics)
    print(f"\n=== Aggregated ({run_tag}, mean +/- std across folds) ===")
    for k, v in agg.items():
        if k == "confusion_matrix_sum":
            print(f"confusion_matrix_sum:\n{v}")
        else:
            print(f"{k}: {v:.4f}")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="channel_gated", choices=["image_only", "late_fusion", "channel_gated"])
    parser.add_argument("--run-tag", default="", help="identifies this experiment condition in the ledger/checkpoints; defaults to --variant")
    parser.add_argument("--quality-aware", action="store_true")
    parser.add_argument("--data-dir", required=True, type=Path, help="dir containing lesion.xlsx etc.")
    parser.add_argument("--images-root", type=Path, default=None, help="defaults to --data-dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--use-preprocessing", action="store_true")
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=None)
    parser.add_argument("--optimizer", default=None, choices=["adam", "adamw_cosine_discriminative", "sam_adamw"])
    parser.add_argument("--backbone-lr-mult", type=float, default=None)
    parser.add_argument("--sam-rho", type=float, default=None)
    parser.add_argument("--tta", action="store_true", help="TTA (flip-averaged) on the final test-fold eval only")
    parser.add_argument("--multi-image-eval", action="store_true", help="average predictions across all dermoscopic images per lesion on the final test-fold eval only")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    cfg = TrainConfig(variant=args.variant, run_tag=args.run_tag, quality_aware=args.quality_aware, notes=args.notes)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.focal_gamma is not None:
        cfg.focal_gamma = args.focal_gamma
    if args.use_preprocessing:
        cfg.use_dermoscopy_preprocessing = True
    if args.use_contrastive:
        cfg.use_contrastive = True
    if args.contrastive_weight is not None:
        cfg.contrastive_weight = args.contrastive_weight
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.backbone_lr_mult is not None:
        cfg.backbone_lr_mult = args.backbone_lr_mult
    if args.sam_rho is not None:
        cfg.sam_rho = args.sam_rho
    if args.tta:
        cfg.use_tta = True
    if args.multi_image_eval:
        cfg.multi_image_eval = True

    images_root = args.images_root or args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: {asdict(cfg)}")

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)
    image_index_df = build_image_index(tables, set(lesion_df["lesion_id"]), images_root)

    t0 = time.time()
    run_cv(cfg, lesion_df, image_index_df, images_root, device)
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
