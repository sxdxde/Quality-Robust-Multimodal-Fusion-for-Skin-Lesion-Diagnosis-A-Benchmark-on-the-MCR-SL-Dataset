"""Training/experiment config. Kept as a single dataclass rather than a YAML
layer — this is a short conference-scoped project with a small, fixed
ablation matrix (CLAUDE.md), not a hyperparameter search; a config file
format would be overhead, not clarity.
"""
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    variant: str = "channel_gated"  # one of models.model.VARIANTS
    quality_aware: bool = False  # robustness analysis #2

    n_folds: int = 5
    seed: int = 42

    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Loss weighting (CLAUDE.md): aux 9-class head down-weighted, quality
    # regression head weighted low when quality_aware=True.
    aux_loss_weight: float = 0.4
    quality_loss_weight: float = 0.15
    focal_gamma: float = 0.0  # 0 = plain weighted BCE; >0 switches binary head to focal

    # If True, train on all dermoscopic images per lesion (more signal at
    # N=240); eval always aggregates back to one prediction per lesion using
    # the diagnosis_image_id image, per CLAUDE.md eval protocol. Logged here
    # rather than hardcoded so the choice shows up in the results ledger.
    train_on_all_dermoscopic_images: bool = True

    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.3
    metadata_hidden_dim: int = 128
    metadata_out_dim: int = 128

    num_aux_classes: int = 9

    results_ledger_path: str = "results/results_ledger.csv"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    notes: str = ""
