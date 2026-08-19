"""Metrics computation, shared by train.py (per-fold reporting) and
robustness_analysis.py. Schema-independent — operates on plain arrays.
"""
import csv
import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """y_true/y_pred: 0/1 int arrays. y_score: malignant-class probability, for AUROC."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")  # single-class fold edge case

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "sensitivity_malignant": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "confusion_matrix": cm,
    }


def aggregate_fold_metrics(per_fold_metrics: list[dict]) -> dict:
    """Mean +/- std across folds for every scalar metric."""
    keys = [k for k in per_fold_metrics[0] if k != "confusion_matrix"]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in per_fold_metrics], dtype=float)
        agg[f"{k}_mean"] = np.nanmean(vals)
        agg[f"{k}_std"] = np.nanstd(vals)
    agg["confusion_matrix_sum"] = sum(m["confusion_matrix"] for m in per_fold_metrics)
    return agg


LEDGER_FIELDS = [
    "timestamp", "variant", "quality_aware", "fold", "n_folds", "seed",
    "accuracy", "balanced_accuracy", "macro_f1", "sensitivity_malignant",
    "specificity", "auroc", "notes",
]


def append_to_ledger(ledger_path: str, variant: str, quality_aware: bool, fold: int, n_folds: int, seed: int, metrics: dict, notes: str = ""):
    """Append-only. Never overwrite prior rows — negative results are paper
    material (limitations/discussion), not something to hide.
    """
    path = Path(ledger_path)
    write_header = not path.exists()
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": variant,
        "quality_aware": quality_aware,
        "fold": fold,
        "n_folds": n_folds,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "sensitivity_malignant": metrics["sensitivity_malignant"],
        "specificity": metrics["specificity"],
        "auroc": metrics["auroc"],
        "notes": notes,
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
