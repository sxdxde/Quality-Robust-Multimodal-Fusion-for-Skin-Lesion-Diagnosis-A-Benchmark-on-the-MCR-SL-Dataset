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

    # --- post-baseline experiment knobs (each its own tracked run, not part
    # of the core CLAUDE.md ablation matrix — see run_tag) ---
    use_dermoscopy_preprocessing: bool = False  # data/preprocessing.py: hair removal + color norm
    use_contrastive: bool = False  # supervised contrastive aux loss on the fused embedding
    contrastive_weight: float = 0.1
    optimizer: str = "adam"  # "adam" | "adamw_cosine_discriminative" | "sam_adamw"
    backbone_lr_mult: float = 0.1  # only used by adamw_cosine_discriminative
    sam_rho: float = 0.05  # SAM neighborhood size, only used by sam_adamw

    # Final-test-only eval-time options (never applied during training/val —
    # val must stay fast and unaugmented for honest checkpoint selection).
    use_tta: bool = False  # average sigmoid probs over original + horizontal flip
    multi_image_eval: bool = False  # average predictions across all of a lesion's dermoscopic images, not just diagnosis_image_id

    # run_tag identifies this experiment condition in the ledger/checkpoint
    # filenames, decoupled from `variant` (the architecture). Defaults to
    # variant when unset so the original 3-variant runs are unaffected.
    run_tag: str = ""

    results_ledger_path: str = "results/results_ledger.csv"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    notes: str = ""

    def resolved_run_tag(self) -> str:
        return self.run_tag or self.variant
