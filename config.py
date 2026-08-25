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

    # Quality-adaptive loss reweighting (QUALITY_ADAPTIVE_LOSS_TASK.md) — a
    # per-sample multiplicative weight on the binary loss, derived from each
    # lesion's expert-rated image quality (distinct from quality_aware,
    # which instead adds an auxiliary quality-prediction head).
    # "none": no reweighting (default, existing behavior unchanged).
    # "trust": down-weight low-quality (less reliable) samples, [1,10]->[0.5,1.5].
    # "hard_mining": up-weight low-quality samples, [1,10]->[1.5,0.5].
    quality_weight_mode: str = "none"

    # Diagnosed from channel_gated_qweight_{trust,hard_mining}'s training logs:
    # val_bacc oscillates wildly epoch-to-epoch (checkpoint selection is
    # capturing noise spikes, not converged states) and train_loss shows
    # occasional destabilizing jumps mid-training. Both opt-in (default off)
    # so no already-reported run's methodology silently changes.
    grad_clip_norm: float = 0.0  # 0 = disabled; clip_grad_norm_ max_norm otherwise
    use_ldam_margin: bool = False  # LDAM-style (Cao et al. 2019) class margin on the binary head
    ldam_margin_c: float = 0.5  # standard LDAM constant, not tuned/searched here

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
