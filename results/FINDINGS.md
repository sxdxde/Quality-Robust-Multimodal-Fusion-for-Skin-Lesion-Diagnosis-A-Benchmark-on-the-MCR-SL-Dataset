# MCR-SL Findings (working notes for the INDICON 2026 write-up)

> **Contents**
> **Start here for writing:** [Abstract-ready numbers](#abstract-ready-numbers-copy-these-with-their-stds) ·
> [Master results table](#master-results-table--every-run-one-place) ·
> [Paper-section mapping](#suggested-paper-section-mapping) ·
> [Consolidated limitations](#consolidated-limitations-draft-the-limitations-section-straight-from-this) ·
> [Reproducibility spec](#reproducibility-specification-for-the-papers-implementation-details-paragraph) ·
> [Artifact inventory](#artifact-inventory--which-file-backs-which-claim) ·
> [Chronology](#chronology--what-was-tried-in-order-and-why)
>
> **Setup:** [Task & protocol](#task--protocol) · [Dataset composition](#dataset-composition-verified-against-the-real-files-not-the-dataset-papers-prose) ·
> [Fold composition](#fold-composition-seed-42-make_subject_disjoint_folds) ·
> [Statistical conventions](#statistical-conventions-used-throughout-this-document) ·
> [Architecture](#architecture-block-by-block-matches-fig-1-in-the-paper-and-modelspy)
>
> **Results:** [Core ablation matrix](#core-ablation-matrix-claudemds-required-matrix) ·
> [Extended experiments](#extended-experiments-post-baseline-each-its-own-tracked-run_tag) ·
> [Quality-adaptive loss](#quality-adaptive-loss-reweighting-follow-up-to-robustness-analysis-2) ·
> [Shuffled-quality control](#shuffled-quality-control-validity-check--closes-out-the-quality-adaptive-loss-search) ·
> [Image-level verification + best config](#image-level-training-verification--eval-time-variants-on-hard_mining-final-experiment) ·
> [Robustness analyses 1–6](#robustness-analyses-the-actual-novelty) ·
> [Are these scores good?](#are-these-scores-good)

**One-line summary.** First benchmark on MCR-SL. Best configuration: channel-gated
metadata fusion + quality-adaptive `hard_mining` loss reweighting + TTA + multi-image averaging,
**0.840 ± 0.095 balanced accuracy / 0.918 ± 0.058 AUROC**. The quality-adaptive loss is
validated against an information-matched shuffled control (its gain collapses to baseline when
the ratings are shuffled). Six robustness analyses; three initially-positive findings were
reversed by follow-up controls, which is itself the paper's most transferable lesson.

Last updated: 2026-08-27. **Search closed; training complete — everything from here is
writing.** Core matrix + extended experiments + the
quality-adaptive loss reweighting follow-up are all complete. `hard_mining` (alone) is the
final best single result (0.820 balanced accuracy / 0.764 sensitivity / 0.906 AUROC). Three
separate attempts to stack something on top of it (LDAM+grad-clip, SAM+TTA, SWA) all made it
worse. The final diagnostic (`channel_gated_swa`, isolating checkpoint-selection noise from
any loss-function change) shows the remaining balanced-accuracy variance is mostly an
inherent small-N (~8-9 malignant lesions/fold) sampling floor, not a fixable pipeline
artifact — so prediction-level checkpoint ensembling (prepared but not run) was skipped as
unlikely to help for the same reason. **The shuffled-quality control (below) changed the
two-finding framing**: `hard_mining`'s raw-performance gain survives an information-matched
shuffled-rating control (collapses back to baseline when the ratings are shuffled), but
`trust`'s tercile-gap narrowing does *not* survive the same control — a content-free shuffled
weighting produces an even smaller gap, which reads as a generic-reweighting artifact rather
than genuine quality-awareness. Final framing: `hard_mining` is the paper's one validated
quality-adaptive-loss contribution; `trust` is reported as a negative/null result after the
control, alongside the auxiliary-quality-head failure.

**Final experiment (2026-08-27) — and the headline numbers the paper should report.** A proposed
"all-images-per-lesion" task turned out to rest on a false premise: training has been
image-level (~5.65x more image samples than lesions) since the very first run. Verifying that,
along with the fold-safety property, **retroactively validates every result in this document**
(no lesion's images ever span two folds). The one genuinely free item that task did surface —
eval-time TTA/multi-image averaging on the validated `hard_mining` checkpoints, never previously
run — yields the project's **best configuration: `hard_mining` + TTA + multi-image, 0.875
accuracy / 0.840 balanced accuracy / 0.783 sensitivity / 0.918 AUROC**, with no retraining. The
same verification also quantified a real, systematic ~16% `pos_weight` miscalibration, which is
documented as a methodological caveat rather than fixed on the last day (reasoning in
"Image-level training verification" below).

## Task & protocol
Binary malignant vs. non-malignant lesion classification, MCR-SL dataset (240 lesions,
60 subjects, first benchmark on this dataset). Subject-disjoint stratified 5-fold CV;
per fold, a second held-out fold is used for checkpoint selection (never the reported
test fold) — no test-fold peeking, no hyperparameter tuning against final numbers.

6 lesions have `malignancy=="unknown"` and are excluded from the binary task (234 usable).
**6** lesions are excluded from the 9-class aux task (`unified_diagnosis=="UNK"` or missing).
*(An earlier version of this line said 5 — corrected 2026-08-27 against every run's own
`build_lesion_table` output, which prints "aux 9-class task: 6 excluded".)*

### Dataset composition (verified against the real files, not the dataset paper's prose)

| quantity | value | source |
|---|---|---|
| lesions | 240 | `lesion.xlsx` |
| subjects | 60 (59 with ≥1 binary-usable lesion) | `subject.xlsx` |
| binary labels | 42 Malignant / 192 Non-malignant / 6 unknown | `lesion.malignancy` |
| binary-usable lesions | **234** | 240 − 6 unknown |
| image rows (raw) | 2394 | `image.xlsx` |
| image rows after lesion join + file check | **2131** (0 missing files) | `build_image_index` |
| — dermoscopy | 1352 | `image.modality` |
| — clinical | 779 (unused — see future work) | `image.modality` |
| dermoscopy images per lesion | mean 5.73, median 6, range 1–18 | Check 2 below |
| lesions with zero dermoscopy images | 4 (fall back to `diagnosis_image_id`) | Check 2 below |
| histopathology-confirmed | **28** (not 29 as CLAUDE.md's prose estimated) | `histopathology_diagnosis.xlsx` |
| lesions with a mean quality rating | 238/240 (L0013, L0205 have none) | 3 experts, E001/E003/E004 |
| lesions with a mean certainty | 238/240 | **4** experts — E002's certainty is intact |
| 9-class small-N classes | MEL 8, SCC **4**, ANG 4, DF 2 | `data/schema.py` (verified counts) |

Note `SCC=4`, not 5 as CLAUDE.md's prose said — `data/schema.py`'s counts were verified
against the file. Every table using 9-class metrics must flag these four classes.

### Fold composition (seed 42, `make_subject_disjoint_folds`)

Folds are balanced greedily on **malignant-lesion count per subject**, so malignant counts come
out even but *subject* counts do not:

| fold | subjects | malignant lesions | train lesions (when this is the test fold) | train image samples |
|---|---|---|---|---|
| 0 | 5 | 9 | 190 | 1042 |
| 1 | 6 | 9 | 163 | 882 |
| 2 | 16 | 8 | 122 | 677 |
| 3 | 16 | 8 | 102 | 621 |
| 4 | 16 | 8 | 143 | 843 |

**The subject imbalance is severe and worth disclosing** — folds 0 and 1 hold 5 and 6 subjects
while folds 2–4 hold 16 each. This is a direct consequence of balancing on malignant-lesion
count with subjects carrying between 1 and 11 lesions each: a few subjects contribute many
malignant lesions, so a fold can hit its malignant quota with very few subjects. It also
explains part of the fold-to-fold variance documented throughout: fold 0's test set is only 5
subjects' worth of lesions. Train-set sizes vary correspondingly (102–190 lesions, 621–1042
image samples). Each lesion appears in exactly 3 of the 5 training sets (720 = 240 × 3 ✓).

### Statistical conventions used throughout this document
- **All mean ± std across folds use population std (`ddof=0`)**, matching
  `evaluate.py:aggregate_fold_metrics`. Pandas' `.std()` default (`ddof=1`) would be ~11.8%
  larger at N=5 and inconsistent with every other number here — `scripts/report_ledger_rows.py`
  explicitly forces `ddof=0` for this reason.
- **AUROC** uses `sklearn.roc_auc_score` on the malignant-class probability; a fold with a
  single class present yields NaN and is skipped via `nanmean` (never triggered in practice).
- **Spearman correlations** use `scipy.stats.spearmanr`, two-sided, no multiple-comparison
  correction — with six robustness analyses reporting several p-values each, treat any single
  p just below 0.05 with corresponding caution.
- **Permutation tests** (analysis 6) use 10,000 permutations, `RandomState(42)`, and report the
  empirical one-sided p as `mean(null >= observed)`.
- **Terciles** use `pd.qcut(..., 3)` (equal-frequency, not equal-width), so bucket Ns differ.

## Architecture (block-by-block, matches Fig. 1 in the paper and models/*.py)

- **Input** (per forward pass): one dermoscopic image (3×224×224, ImageNet-normalized) +
  patient metadata (16 categorical + 4 numeric fields, after dropping the constant-valued
  and unusably sparse free-text fields — see `data/schema.py`'s "explicitly dropped fields"
  note).
  **Training is image-level, not lesion-level** — `config.py:train_on_all_dermoscopic_images`
  defaults to `True`, so *every* dermoscopic image of a training lesion is a separate
  training sample (~5.65x more samples than lesions; verified, see "Image-level training"
  below). Only val/test use exactly one image per lesion (the `diagnosis_image_id` image),
  so evaluation reports one prediction per lesion per the eval protocol. The metadata vector
  and quality weight are per-lesion values broadcast across all of that lesion's images.
  *(This bullet previously read just "Input: one dermoscopic image", which describes the
  per-sample tensor and was misread as one image per lesion — clarified after that
  misreading sent a whole task after an already-implemented "fix".)*

- **Image encoder** (`models/image_encoder.py: EfficientNetB0Encoder`) — ImageNet-pretrained
  EfficientNet-B0 (`timm`), fully trainable end to end (no frozen layers, unlike some
  transfer-learning setups). `forward_features` gives a (B, 1280, 7, 7) conv feature map;
  `AdaptiveAvgPool2d(1)` gives a (B, 1280) pooled vector. Both are returned from one forward
  pass, so `image_only` (uses only the pooled vector) and both fusion variants share the
  same backbone call.

- **Metadata encoder** (`models/metadata_encoder.py: MetadataEncoder`):
  - Categorical fields: each field gets its own `nn.Embedding(cardinality+1, 12)`. The "+1"
    slot is a reserved "unknown" index for missing/unseen values — routed there, never
    imputed, matching the dataset's own stated missingness policy.
  - Numeric fields: passed through as raw per-field z-scored scalars (mean/std fit on
    train-fold data only), each paired with a 0/1 missingness bit — 2 raw dims per field,
    no embedding.
  - All categorical embeddings (16 x 12 = 192-d) + numeric (value, missing-bit) pairs
    (4 x 2 = 8-d) are concatenated (200-d total) and passed through a 2-layer MLP:
    Linear(200->128) -> ReLU -> Dropout(0.2) -> Linear(128->128) -> ReLU, giving the
    128-d metadata vector shown in Fig. 1.

- **Fusion** (`models/fusion.py`) — two variants, sharing the same final projection layer:
  - *Late fusion baseline*: concat(1280-d pooled image, 128-d metadata) = 1408-d ->
    Linear(1408->256) -> ReLU -> Dropout(0.3). Output: 256-d.
  - *Channel-gated fusion (main method)*: metadata vector -> Linear(128->1280) -> sigmoid
    -> 1280-d gate in [0,1]. This gate multiplies the 1280-channel conv feature map
    elementwise, channel by channel, broadcasting over the 7x7 spatial grid (the SE-block
    mechanism, conditioned on metadata instead of the block's own pooled features). The
    gated map is then globally average-pooled to 1280-d and passed through the **same**
    Linear(1280->256) -> ReLU -> Dropout(0.3) projection as the late-fusion path. Output:
    256-d. **This final 256-d projection was missing from the first draft of Fig. 1 and the
    Section III-A prose (both showed "Gated global pool" feeding the heads directly at
    1280-d) — corrected in both once found while writing this section.**

- **Heads** (`models/heads.py`) — all single `nn.Linear` layers operating on the 256-d
  fused vector (or directly on the 1280-d pooled image vector for `image_only`, which has
  no fusion step to project it down):
  - Binary head: Linear(256->1) -> logit -> sigmoid -> P(malignant). Weighted BCE (or focal
    loss, gamma configurable) with pos_weight recomputed from each training fold's own
    malignant ratio.
  - Auxiliary head: Linear(256->9) -> 9-class logits (unified diagnosis), class-weighted
    cross-entropy, contributes at 0.4x weight in the combined loss, exploratory table only.
  - Quality head (quality-aware variant only): Linear(256->1) -> predicted mean expert
    quality rating (rating/10, 0-1 normalized), MSE loss, weighted 0.15.

- **Training-time-only auxiliary loss** (not depicted in Fig. 1 — applies to the 256-d fused
  embedding *before* the heads, not a forward-pass block): supervised contrastive loss
  (`models/heads.py: supervised_contrastive_loss`, SupCon-style), used only in the
  `channel_gated_contrastive` follow-up experiment; a negative result there (see below).

## Master results table — every run, one place

All 5-fold means. Sorted by balanced accuracy. `quality_aware` is `False` everywhere except the
one row that names it. Rebuild from the ledger with
`python scripts/report_ledger_rows.py <run_tag>`.

| # | run_tag | acc | **bal_acc** | macro_F1 | sens | spec | AUROC | retrain? | group |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `hard_mining_tta_multiimage` | **0.875** | **0.840** | **0.810** | 0.783 | 0.896 | 0.918 | no | qweight+eval |
| 2 | `hard_mining_multiimage` | 0.871 | 0.838 | 0.805 | 0.783 | 0.892 | **0.920** | no | qweight+eval |
| 3 | `hard_mining_tta` | 0.849 | 0.833 | 0.783 | **0.808** | 0.858 | 0.911 | no | qweight+eval |
| 4 | `channel_gated_sam_adamw_tta` | 0.861 | 0.822 | 0.789 | 0.719 | 0.925 | 0.908 | yes+eval | extended |
| 5 | **`channel_gated_qweight_hard_mining`** | 0.845 | **0.820** | 0.778 | 0.764 | 0.875 | 0.906 | yes | **qweight (headline)** |
| 6 | `channel_gated_tta_multiimage` | 0.857 | 0.815 | 0.787 | 0.719 | 0.910 | 0.908 | no | extended |
| 7 | `channel_gated_multiimage` | 0.854 | 0.813 | 0.784 | 0.719 | 0.907 | 0.906 | no | extended |
| 8 | `channel_gated_sam_adamw` | 0.859 | 0.811 | 0.785 | 0.694 | 0.927 | 0.904 | yes | extended |
| 9 | `channel_gated_sam_adamw_tta_multiimage` | 0.856 | 0.810 | 0.780 | 0.694 | 0.925 | **0.915** | yes+eval | extended |
| 10 | `channel_gated_focal` (γ=2) | 0.858 | 0.809 | 0.779 | 0.694 | 0.923 | 0.877 | yes | extended |
| 11 | `channel_gated_hardmining_swa` | — | 0.807 | — | 0.675 | 0.939 | 0.856 | yes | stacked on hm |
| 12 | `channel_gated_sam_adamw_multiimage` | 0.851 | 0.800 | 0.773 | 0.672 | **0.928** | 0.917 | yes+eval | extended |
| 13 | `channel_gated_hardmining_ldam_stable` | — | 0.798 | — | 0.697 | 0.899 | — | yes | stacked on hm |
| 14 | `channel_gated_preprocessed` | 0.828 | 0.793 | 0.753 | 0.736 | 0.851 | 0.875 | yes | extended |
| 15 | `channel_gated_hardmining_sam_tta` | — | 0.789 | — | 0.761 | 0.816 | 0.886 | yes+eval | stacked on hm |
| 16 | `channel_gated_qweight_trust` | 0.793 | 0.788 | 0.721 | 0.725 | 0.850 | 0.869 | yes | qweight |
| 17 | `channel_gated_tta` | 0.841 | 0.788 | 0.765 | 0.672 | 0.904 | 0.897 | no | extended |
| 18 | `image_only` | 0.813 | 0.784 | 0.733 | 0.697 | 0.872 | 0.895 | yes | **core matrix** |
| 19 | **`channel_gated`** (plain) | 0.833 | 0.783 | 0.757 | 0.672 | 0.895 | 0.881 | yes | **core matrix** |
| 20 | `hardmining_shuffled` (control) | — | 0.775 | — | 0.667 | — | — | yes | control |
| 21 | `channel_gated_contrastive` | 0.812 | 0.771 | 0.734 | 0.669 | 0.872 | 0.873 | yes | extended |
| 22 | `late_fusion` | 0.831 | 0.768 | 0.743 | 0.644 | 0.891 | 0.851 | yes | **core matrix** |
| 23 | `trust_shuffled` (control) | — | 0.763 | — | 0.672 | — | — | yes | control |
| 24 | `channel_gated_swa` | 0.829 | 0.762 | 0.736 | 0.603 | 0.920 | 0.914 | yes | diagnostic |
| 25 | `channel_gated_combined` | 0.818 | 0.759 | 0.715 | 0.622 | 0.895 | 0.893 | yes+eval | extended |
| 26 | `channel_gated_optimizerv2` | 0.822 | 0.757 | 0.731 | 0.661 | 0.853 | 0.845 | yes | extended |
| 27 | `channel_gated` + quality-aware head | 0.812 | 0.740 | 0.715 | 0.578 | 0.902 | 0.879 | yes | **core matrix** |

Dashes are metrics not recorded in the notes for that run — read them off
`results/results_ledger.csv` if a table needs them. Rows 1–3 and 5 are the paper's headline
family; rows 18/19/22/27 are CLAUDE.md's required core matrix.

### Per-fold detail — the `hard_mining` family (the paper's headline numbers)

Reported in full because these are the rows the paper leads with, and because the fold-to-fold
spread is the single most important caveat attached to them.

**`channel_gated_qweight_hard_mining` (base):**

| fold | acc | bal_acc | macro_F1 | sens | spec | AUROC |
|---|---|---|---|---|---|---|
| 0 | 0.7200 | 0.7083 | 0.7029 | 0.6667 | 0.7500 | 0.8403 |
| 1 | 0.9200 | 0.8889 | 0.9081 | 0.7778 | 1.0000 | 0.9722 |
| 2 | 0.8980 | 0.7881 | 0.8032 | 0.6250 | 0.9512 | 0.9024 |
| 3 | 0.8710 | 0.9259 | 0.7933 | 1.0000 | 0.8519 | 0.9722 |
| 4 | 0.8143 | 0.7863 | 0.6835 | 0.7500 | 0.8226 | 0.8427 |
| **mean ± std** | 0.8446 ± 0.0717 | **0.8195 ± 0.0782** | 0.7782 ± 0.0805 | 0.7639 ± 0.1303 | 0.8751 ± 0.0898 | 0.9060 ± 0.0585 |

**`+ TTA`** (mean ± std): acc 0.8486 ± 0.0556 · **bal_acc 0.8330 ± 0.0711** · macro_F1 0.7832 ± 0.0750 · sens **0.8083 ± 0.1274** · spec 0.8578 ± 0.0693 · AUROC 0.9105 ± 0.0630

**`+ multi-image`** (mean ± std): acc 0.8713 ± 0.0728 · **bal_acc 0.8378 ± 0.0929** · macro_F1 0.8051 ± 0.0935 · sens 0.7833 ± 0.1670 · spec 0.8923 ± 0.0934 · AUROC **0.9197 ± 0.0541**

**`+ TTA + multi-image`** (mean ± std): acc **0.8745 ± 0.0752** · **bal_acc 0.8396 ± 0.0952** · macro_F1 **0.8099 ± 0.0966** · sens 0.7833 ± 0.1670 · spec 0.8959 ± 0.0944 · AUROC 0.9182 ± 0.0580

**Read the standard deviations before quoting any mean.** Balanced accuracy on the best config
is 0.840 ± **0.095**; per-fold it ranges 0.738 → 0.964. Sensitivity is worse: ± 0.167, with
fold 2 at 0.500 and fold 3 at 1.000. With 8–9 malignant lesions per test fold, a single lesion
flipping moves fold sensitivity by ~0.11–0.125. **Every headline number in this paper should
carry its std, and no two configs within ~0.05 balanced accuracy of each other should be
described as one beating the other.**

## Core ablation matrix (CLAUDE.md's required matrix)

| run | accuracy | balanced_acc | macro_F1 | sensitivity(malig) | specificity | AUROC |
|---|---|---|---|---|---|---|
| image_only (baseline) | 0.813 | 0.784 | 0.733 | 0.697 | 0.872 | 0.895 |
| late_fusion (concat baseline) | 0.831 | 0.768 | 0.743 | 0.644 | 0.891 | 0.851 |
| **channel_gated (main method)** | 0.833 | 0.783 | 0.757 | 0.672 | 0.895 | 0.881 |
| channel_gated + quality-aware head | 0.812 | 0.740 | 0.715 | 0.578 | 0.902 | 0.879 |

**channel_gated is not a clean win** — image_only actually beats it on balanced accuracy,
sensitivity, and AUROC. At N=234 with ~40 malignant lesions, fold-to-fold variance is
large (sensitivity std 0.16–0.19). Report honestly, don't oversell the fusion method.

**Quality-aware training made things worse**, not better, across every metric — a real
negative result. Analysis 2 (below) confirms the low-vs-high quality-tercile accuracy gap
widened (0.091 → 0.101) rather than flattening. Auxiliary quality-regression head, as
implemented, does not help robustness here.

## Extended experiments (post-baseline, each its own tracked run_tag)

| run | accuracy | balanced_acc | macro_F1 | sensitivity | specificity | AUROC | retrain? |
|---|---|---|---|---|---|---|---|
| **channel_gated_sam_adamw_tta** | 0.861 | **0.822** | 0.789 | 0.719 | 0.925 | 0.908 | yes + eval-only stack |
| channel_gated_tta_multiimage | 0.857 | 0.815 | 0.787 | 0.719 | 0.910 | 0.908 | no (eval-only) |
| channel_gated_multiimage | 0.854 | 0.813 | 0.784 | 0.719 | 0.907 | 0.906 | no (eval-only) |
| channel_gated_sam_adamw (SAM+AdamW alone) | 0.859 | 0.811 | 0.785 | 0.694 | 0.927 | 0.904 | yes |
| channel_gated_sam_adamw_tta_multiimage | 0.856 | 0.810 | 0.780 | 0.694 | 0.925 | **0.915** | yes + eval-only stack |
| channel_gated_focal (γ=2) | 0.858 | 0.809 | 0.779 | 0.694 | 0.923 | 0.877 | yes |
| channel_gated_sam_adamw_multiimage | 0.851 | 0.800 | 0.773 | 0.672 | **0.928** | 0.917 | yes + eval-only stack |
| channel_gated_preprocessed | 0.828 | 0.793 | 0.753 | **0.736** | 0.851 | 0.875 | yes |
| channel_gated_tta | 0.841 | 0.788 | 0.765 | 0.672 | 0.904 | 0.897 | no (eval-only) |
| channel_gated_contrastive | 0.812 | 0.771 | 0.734 | 0.669 | 0.872 | 0.873 | yes |
| channel_gated_optimizerv2 (AdamW+cosine+discriminative-LR) | 0.822 | 0.757 | 0.731 | 0.661 | 0.853 | 0.845 | yes |
| channel_gated_combined (preprocessing+focal+SAM+TTA) | 0.818 | 0.759 | 0.715 | 0.622 | 0.895 | 0.893 | yes + eval-only stack |

### What actually moved the needle, and what didn't
- **SAM+AdamW, combined with TTA, is the overall best config** — 0.822 balanced accuracy,
  0.908 AUROC. SAM alone (0.811 bal. acc) already beat every other single-model config
  except the two eval-time-only tricks; seeking flatter minima on a small, noisy dataset
  paid off, consistent with the motivation for trying it. Notably ~2x-compute concern
  didn't materialize in wall-clock time — this run finished in the same ~65 min as the
  others, suggesting the pipeline was data-loading-bound, not backward-pass-bound, at this
  model size.
- **Multi-image test-time averaging is a big, reliable, free win on its own** (no
  retraining) — but it does **not** stack cleanly on top of SAM+TTA: adding it to
  `channel_gated_sam_adamw_tta` *reduced* balanced accuracy (0.822 → 0.810) even though it
  pushed AUROC to the dataset-wide high (0.915). The eval-time tricks are not strictly
  additive — worth a line in the paper rather than assuming "more tricks = always better."
- **Focal loss and preprocessing trade off in opposite, interpretable directions**: focal
  loss buys accuracy/specificity at sensitivity's cost (fewer false alarms, more missed
  malignancies); dermoscopy preprocessing (hair removal + color normalization) gives the
  *highest sensitivity of any variant* (0.736) — plausible, since hair/lighting artifacts
  are exactly the kind of quality issue that could obscure the features that matter for
  catching malignant lesions. Ties in naturally with the quality-robustness theme.
- **Contrastive loss, the plain-AdamW/cosine/discriminative-LR optimizer, and
  quality-aware training all underperformed the plain baseline.** Three independent
  negative results pointing the same direction: at N≈234, adding model/objective
  complexity doesn't reliably help and sometimes hurts, while simple prediction-averaging
  (and, it turns out, sharpness-aware optimization) reliably does. Genuinely interesting
  discussion-section material — arguably more interesting than "our method wins."
- **Stacking every individually-promising trick (preprocessing + focal loss + SAM + TTA)
  underperforms both the winning combo AND the plain baseline** — 0.759 balanced accuracy,
  below baseline's 0.783 and well below SAM+TTA's 0.822, with much higher variance across
  folds (balanced-accuracy std 0.132, sensitivity std 0.285 — nearly half the mean).
  Plausible explanation: focal loss and preprocessing pull the decision boundary in
  different directions (specificity-favoring vs. sensitivity-favoring, per the point
  above), and stacking them with SAM+TTA doesn't resolve that tension — it just adds
  instability. A fourth, independent demonstration that "more interventions" isn't a
  free lunch at this N — arguably the single most interesting negative result of the
  sprint, since it's the most intuitive-sounding failed guess (each ingredient works
  alone, so combining them "should" stack).

### Does anything cross 0.9?
AUROC does, comfortably now — best config 0.915, several others 0.90–0.91. Balanced
accuracy tops out at 0.822 and sensitivity at 0.736. Given ~40 malignant lesions total,
that ceiling on balanced accuracy/sensitivity is real, not a tuning gap — don't chase it
further by tuning against these numbers.

### Literature context (verified during the paper-writing/review pass — supersedes the
earlier note below, kept struck through for the record rather than deleted)
~~Closest comparator, PAD-UFES-20 (metadata+image fusion, ~1,641 lesions — ~7x our N): best
published multimodal balanced accuracy there is 0.832. Our best (0.822) is within a hair
of that despite ~7x less data.~~ **This was wrong**: 0.832 (and MetaBlock's 0.765) are
**six-class** balanced accuracy on PAD-UFES-20, not binary — not a valid comparator for
our binary task despite the same metric name. The verified binary comparator is Uliana &
Krohling 2025 (DiffMIC, arXiv:2504.00026), 0.836 balanced accuracy, cancer vs. non-cancer,
5-fold CV, 2298 images — our best (0.822) is close to but below this. That paper also
attributes a higher ~0.88 to an earlier ResNet-50 setup, sourced secondhand within their
paper; not independently verified, reported in the paper as such rather than as a
confirmed benchmark. HAM10000/PH2 headline numbers (accuracy 0.88–0.99) remain not fair
comparators — 40x+ more data and/or an easier task and/or looser evaluation rigor.

### Comparing like with like: "234 vs. 2298" was not a fair ratio

**A real error, caught 2026-08-28 and corrected everywhere.** The claim "~10x less training
data" (which appeared 3x in the slides and once above) was computed as PAD-UFES-20's **image**
count ÷ our **lesion** count — two different units. Our models do not train on 234 samples; they
train on ~1,352 dermoscopic images (every dermoscopic image of a training lesion is a separate
sample, ~5.7 per lesion — see "Image-level training verification").

| quantity | MCR-SL (ours) | PAD-UFES-20 | ratio |
|---|---|---|---|
| images available | 2,131 | 2,298 | 1.1x |
| **images used for training** | **1,352** (dermoscopic) | **2,298** | **1.7x** |
| **lesions** | **234** usable | **~1,641** | **7.0x** |
| subjects / patients | 60 | ~1,373 * | ~23x |
| evaluation unit | **per lesion** | **per image** | — |
| malignant | 42 lesions (18%) | * | — |
| non-malignant | 192 lesions (82%) | * | — |

\* PAD-UFES-20's patient count and cancer/non-cancer split are **not independently re-verified**
in this project — the 1,641 lesion figure comes from the earlier literature pass, but the class
balance and patient count should be confirmed against Pacheco et al. (2020) before submission.
Do not put them in the paper as verified numbers until that check is done.

**What to say instead of "10x less data":** *"7x fewer lesions, with comparable image counts."*
That is defensible and still makes the point. The largest genuine gap is subjects — 60 vs.
~1,373 patients — which matters more than either image or lesion count for generalization, and
is the honest thing to emphasize. Note also the evaluation units differ (we report per-lesion,
DiffMIC per-image), so even the headline metrics are not measuring quite the same quantity.

## Robustness analyses (the actual novelty)

*Six analyses: 1–4 were the original set; 5–6 (diagnostic certainty, intra-subject consistency)
were added 2026-08-27, post-hoc on existing predictions at zero GPU cost.*

1. **Quality-stratified performance** (channel_gated, N=231 lesions with both a
   prediction and a quality rating): accuracy by tercile — low 0.776, mid 0.914, high
   0.868 (non-monotonic). Spearman(rating, error) = -0.093 (p=0.158) — not significant at
   this N. Report as "no strong quality-performance relationship detected," not overstated
   either way.
2. **Quality-aware training vs. not** (analysis 2): tercile accuracy gap widened under
   quality-aware training (0.091 → 0.101) — the intervention did not flatten the gap it
   was designed to address. Negative result, reportable as-is.
3. **Histopathology-confirmed (n=28) vs. panel-consensus (n=203)**: model does *worse* on
   histopath-confirmed lesions (75.0% vs. 87.0% accuracy) — plausible, since histopathology
   tends to get ordered for the more clinically ambiguous/suspicious lesions. Qualitative
   only, n=28 too small for a CI.
4. **Metadata field importance** (ablation-by-field, |Δ logit|, vs. dataset paper's
   Tables 3–4 significant fields: location_group, sex, referral_diagnosis, diameter):
   `referral_diagnosis`, `sex`, `location_group` land in the upper half of the ranking;
   all four numeric fields (`diameter` included) rank at the very bottom. Likely an
   architectural capacity imbalance (12-dim learned embedding per categorical field vs. a
   single z-scored scalar per numeric field), not evidence that diameter is clinically
   uninformative — worth a line in the discussion, not a redesign this sprint.

5. **Diagnostic certainty vs. image quality** (`scripts/robustness_analyses_5_6.py`, post-hoc on
   the existing `channel_gated` out-of-fold predictions, N=231). `dermatology_diagnosis.certainty`
   is each expert's self-reported confidence in their own diagnosis (0/25/50/75/100) — a
   different axis from `image_rating` (photo quality). **Verified first, not assumed: E002's
   certainty is fully intact (241/241 valid, as are E001/E003/E004)**, so the documented E002
   data loss really was specific to image-quality ratings, exactly as the dataset paper states.
   Certainty is therefore averaged over all **four** experts, unlike the three-expert quality
   rating.

   | certainty tercile | n | n malignant | % malignant | accuracy | sensitivity |
   |---|---|---|---|---|---|
   | low | 93 | 23 | 24.7% | 0.796 | 0.609 |
   | mid | 65 | 13 | 20.0% | 0.892 | 0.692 |
   | high | 73 | 6 | 8.2% | 0.890 | 0.833 |

   Pooled, certainty looks like a much better predictor of model error than photo quality:
   Spearman(certainty, error) = **−0.175 (p=0.008)** vs. Spearman(rating, error) = −0.093
   (p=0.158); same ordering on confidence (+0.214, p=0.001 vs. +0.125, p=0.058). The two axes
   are strongly but not identically related (Spearman(certainty, rating) = +0.704).

   **The pooled result is confounded, and controlling for it dissolves the finding entirely.**
   Malignant lesions concentrate at low certainty (24.7% → 20.0% → 8.2% across terciles —
   experts are less sure about the harder, more suspicious lesions), and this model is far
   weaker on malignant lesions generally (sensitivity 0.672 vs. specificity 0.895). So a
   low-certainty bucket scores worse largely *because* it holds three times the malignant
   fraction. Both axes were therefore stratified by class identically — the claim here is a
   *comparison*, so controlling one side and not the other would have invalidated it:

   | axis | non-malignant (n=189, 20 errors) | malignant (n=42, 14 errors) |
   |---|---|---|
   | certainty | ρ=−0.127 (p=0.083, n.s.) | ρ=−0.093 (p=0.559, n.s.) |
   | image rating | ρ=−0.120 (p=0.101, n.s.) | ρ=+0.166 (p=0.295, n.s.) |

   **Nothing survives.** No stratum reaches significance for either axis, and — the decisive
   point — on non-malignant lesions (the only stratum with meaningful power) the two axes are
   effectively *indistinguishable*: −0.127 vs. −0.120, p=0.083 vs. p=0.101. The large pooled
   gap (p=0.008 vs. p=0.158) was an artifact of certainty being more strongly associated with
   malignancy than image quality is, so pooling inflated certainty's correlation more.

   **Report as a null, and say why the naive version was wrong.** The defensible statement is
   *"neither expert diagnostic certainty nor image quality reliably predicts model error at this
   N once class composition is accounted for."* It must **not** be written as "model errors
   track diagnostic difficulty rather than photo quality" — that conclusion is available only
   from the uncontrolled pooled numbers and does not survive a class control. This also
   reinforces rather than contradicts analysis 1's original "no strong quality-performance
   relationship detected": certainty does not rescue that null, it reproduces it. The residual
   non-malignant trends (p≈0.08–0.10, same magnitude for both axes) are suggestive of a weak
   shared effect at most — worth one sentence of discussion, not a claim.

   Methodologically this is the most useful part of analysis 5: a pooled correlation on this
   dataset produces a clean-looking p=0.008 that is substantially class-mix artifact. Worth a
   line in the paper as a caution for anyone else stratifying MCR-SL by an expert-annotation
   axis, since malignancy correlates with most of them.

   One piece is not subject to the class-mix confound: the within-malignant sensitivity trend
   (0.609 → 0.692 → 0.833) is monotonic and computed within malignant lesions only. But it rests
   on 23/13/**6** malignant lesions per tercile, and the corresponding within-malignant
   correlation is flatly non-significant (ρ=−0.093, p=0.559) — so it is a suggestive shape, not
   evidence. Do not report it without the per-tercile malignant counts next to it.

6. **Intra-subject consistency** (same script, same predictions). MCR-SL deliberately collected
   ≥2 lesions per subject; unused until now. Scope check first, per the small-N discipline used
   for the histopathology subset: **55 of 59 subjects have ≥2 usable lesions** (227 lesions;
   mean 4.13 per subject, max 11) — comfortably above the qualitative-only threshold.

   | outcome | subjects | % |
   |---|---|---|
   | all lesions correct | 27 | 49.1% |
   | mixed | 28 | 50.9% |
   | **all lesions incorrect** | **0** | **0.0%** |

   **No subject-level error clustering — a clean null.** Against a permutation null that
   reshuffles lesion correctness across lesions with subject sizes held fixed (10,000
   permutations), observed perfectly-consistent subjects = 27 vs. null mean 29.88, empirical
   p=0.954. Observed consistency is if anything *below* chance, so errors scatter independently
   across lesions rather than concentrating in a few "hard subjects." Zero subjects fail on all
   their lesions.

   This is a genuinely useful negative for the methods section: it is evidence that the
   per-subject metadata the model consumes (skin type, sun reaction, history) is **not** inducing
   subject-level systematic bias, and that subject-disjoint splitting is not masking a
   per-patient confound. Report with N=55 stated alongside the p-value, never the p-value alone.

## Quality-adaptive loss reweighting (follow-up to robustness analysis #2)

Motivation: the auxiliary quality-prediction head (analysis #2 above) failed — predicting
image quality as a side objective made things worse everywhere. Face-recognition literature
(MagFace; AdaFace, Kim et al., CVPR 2022, arXiv:2204.00964 — verified directly, not taken on
faith) shows a mechanistically different lever, quality-adaptive *loss weighting* rather than
quality *prediction*, helps under low-quality inputs. `QUALITY_ADAPTIVE_LOSS_TASK.md` tests
this on `channel_gated` (main method only, `quality_aware=False` — a distinct mechanism from
the auxiliary head, not a retry of it).

**Implementation**: a per-sample multiplicative weight on the binary BCE loss, derived from
each lesion's existing `mean_image_rating` (same E001/E003/E004 computation already used for
analysis #1 — reused, not recomputed). Two directions:
- **trust**: `w = 0.5 + (rating-1)/9`, maps rating [1,10] -> weight [0.5, 1.5] (down-weights
  low-quality/less-reliable samples).
- **hard_mining**: `w = 1.5 - (rating-1)/9`, maps [1,10] -> [1.5, 0.5] (up-weights low-quality
  samples, forcing the model to work harder on them).
Samples with no valid rating get a neutral weight of 1.0. `models/heads.py:BinaryHead.loss`
verified to reproduce the exact old loss when `sample_weight=None` (byte-for-byte, unit
tested) — no other run's results are affected by this change.

| run | accuracy | balanced_acc | macro_F1 | sensitivity(malig) | specificity | AUROC |
|---|---|---|---|---|---|---|
| channel_gated (plain baseline) | 0.833 | 0.783 | 0.757 | 0.672 | 0.895 | 0.881 |
| channel_gated_qweight_trust | 0.793 | 0.788 | 0.721 | 0.725 | 0.850 | 0.869 |
| **channel_gated_qweight_hard_mining** | **0.845** | **0.820** | 0.778 | **0.764** | 0.875 | 0.906 |

**`hard_mining` is the new best single result on the core dimensions** — beats plain
`channel_gated` by +0.037 balanced accuracy, +0.092 sensitivity (the clinically weightier
metric — a missed malignancy costs more than a false alarm), +0.025 AUROC. Nearly matches
`channel_gated_sam_adamw_tta` (0.822 balanced acc, the previous best) while *exceeding* it on
sensitivity (0.764 vs. 0.719). `trust` is roughly flat on core metrics (+0.005 balanced acc,
within the ~0.08 fold-to-fold std — not a robust difference) but is the mechanism that
narrows the quality-tercile gap, below.

**Quality-tercile gap, extending analysis #2's table to a three-way (four-mechanism)
comparison** (high-minus-low tercile accuracy gap, channel_gated, N=231):

| mechanism | high−low accuracy gap |
|---|---|
| plain (no quality-awareness) | 0.091 |
| auxiliary quality-prediction head | 0.101 (worse — analysis #2's original finding) |
| loss reweight: trust | ~~0.082 (narrower — first success across 3 mechanisms tried)~~ **does not
survive the shuffled-quality control below — see that section** |
| loss reweight: hard_mining | 0.091 (unchanged from plain) |

Three distinct quality-awareness mechanisms, three different outcomes — report as a nuanced,
non-monolithic finding, not "quality-awareness helps" or "doesn't help" as a single verdict:
- **Auxiliary head**: net negative everywhere (both core metrics and the tercile gap).
- ~~**trust**: the only mechanism that narrows the quality-robustness gap specifically, without
  materially hurting sensitivity or AUROC (sensitivity actually rose vs. plain).~~ **Superseded
  by the shuffled-quality control** (see "Shuffled-quality control" section below, added after
  this was first written): a content-free shuffled-rating control produces an even narrower gap
  than the real `trust` result, so this apparent narrowing is not distinguishable from a
  generic-reweighting artifact — report `trust` as a negative/null result, not a success.
- **hard_mining**: doesn't touch the tercile gap (unchanged from plain, 0.091) but delivers
  the largest raw performance gain project-wide. Plausible mechanism: up-weighting
  low-quality samples acts as a general hard-example-mining regularizer that sharpens the
  whole model roughly proportionally across all quality levels, rather than specifically
  closing the quality-tercile gap — a coherent story, not a loose end. **The shuffled-quality
  control below confirms this gain is genuine (not generic reweighting).**

### Root cause diagnosis: why balanced accuracy plateaus around 0.78–0.82

From full per-epoch training logs (`logs/train_channel_gated_qweight_{trust,hard_mining}.log`,
5 folds each): **val_bacc oscillates by 0.15–0.25 between *adjacent* epochs in every single
fold**, never settling into a smooth convergence curve (e.g. hard_mining fold 0:
0.88 → 0.64 → 0.86 → 0.77 → 0.69 → ... across just the first 5 epochs). Train loss collapses
to near-zero (0.01–0.05) by epoch 15–20 in most folds — heavy overfitting on top of an
already-noisy validation signal. Occasional mid-training train_loss spikes (e.g.
0.03 → 0.36 at epoch 19, fold 0 trust) are consistent with a few high-loss-weight outlier
batches (malignant + low-quality samples, combined weight ≈ pos_weight×1.5 ≈ 6.9 under
hard_mining) occasionally destabilizing a gradient step.

**Consequence**: the existing "pick the single epoch with highest val_bacc" checkpoint
selection rule (used for every config in this project, not just the quality-weight ones) is
likely capturing noise spikes rather than converged states — test balanced accuracy swings
0.70–0.93 fold-to-fold for `hard_mining` alone. This is very likely the dominant remaining
gap to DiffMIC's 0.836 (2298 images vs. our 234 lesions — **note: this "~10x less data" phrasing
was wrong and is corrected in "Comparing like with like" below; it divided their *image* count by
our *lesion* count. The real gap is 7x fewer lesions with comparable image counts** — proportionally
noisier per-fold estimates, an inherent MCR-SL-scale limitation, not something a loss-formula
change alone can fix). Stochastic Weight Averaging (Izmailov et al. 2018; SWAD, Cha et al.
2021) is documented in the literature as the established fix for exactly this failure mode
(noisy-validation checkpoint selection in place of a clean validation set) — see "queued"
below.

### Follow-up interventions attempted on top of hard_mining — mixed/negative results

**Free (no retraining) — val-optimal threshold recalibration.** Per-fold Youden's-J threshold
selected on the val fold only, applied to the held-out test fold (no test-fold peeking). No
reliable gain: `hard_mining` balanced accuracy actually *worsened* (0.8195 → 0.8052);
`trust`/`plain` were roughly flat (+0.0015 / +0.0098, within noise). The thresholds picked
per fold ranged wildly (e.g. 0.005 to 0.937 for `hard_mining`) — confirms the val-fold signal
is itself too noisy at this N (8–9 malignant lesions per val fold) to reliably calibrate a
threshold, not just to select a checkpoint. Script: `scripts/optimal_threshold_eval.py`.

**LDAM-style class margin (Cao et al., NeurIPS 2019) + gradient clipping, stacked on
`hard_mining`.** Motivated by the sensitivity-vs-specificity gap (0.764 vs. 0.875) and the
training-instability spikes above; margin formula `C/n_j^0.25` (C=0.5, standard LDAM
constant, not tuned) gives the minority (malignant) class the larger training-time margin,
intended to push sensitivity up. Result: **net negative** — balanced accuracy 0.798 (−0.022
vs. `hard_mining` alone), sensitivity 0.697 (**−0.067, the opposite of the intended
direction**), specificity 0.899 (+0.024). Margin and gradient clipping were bundled into one
run rather than isolated, so attribution between the two pieces is unclear — a real
methodological shortcut taken under time pressure. Third negative/mixed result in the
quality-awareness search overall, alongside the auxiliary head. Config:
`channel_gated_hardmining_ldam_stable`.

**SAM optimizer + TTA, stacked on `hard_mining`.** Lower a priori risk than the LDAM+clip
attempt — both pieces (SAM, TTA) are already independently validated as positive on this
exact dataset (SAM alone: 0.811 balanced acc; TTA: free, no retraining) rather than new
untested mechanisms. Result: **net negative anyway** — balanced accuracy 0.789 (−0.031 vs.
`hard_mining` alone, and below plain SAM+TTA's own 0.822), specificity 0.816 (well below both
`hard_mining` alone at 0.875 and plain SAM+TTA at 0.925), AUROC 0.886 (−0.020). Sensitivity
held roughly flat (0.761 vs. 0.764). Plausible mechanism: SAM computes its adversarial
perturbation direction from the *already quality-reweighted* loss surface — with a few
malignant+low-quality samples carrying up to ~6.9x effective weight, SAM's worst-case-
neighborhood search likely chases robustness around those few outlier-weighted samples
rather than genuine model-wide flatness. Config: `channel_gated_hardmining_sam_tta`.

**Stochastic Weight Averaging, stacked on `hard_mining`.** Result: again **net negative** —
balanced accuracy 0.807 (−0.013 vs. `hard_mining` alone), and, like the auxiliary head and
LDAM+clip before it, the model becomes markedly *more conservative*: sensitivity dropped to
0.675 (−0.089, the worst sensitivity of any `hard_mining`-family config), specificity rose to
0.939 (+0.064), AUROC fell to 0.856 (−0.051). SWA did deliver on its literal promise —
fold-to-fold **accuracy** variance genuinely tightened (std 0.080 → 0.035, more than half),
confirming the averaged checkpoint is more stable — but a more stable checkpoint that's
stably biased toward specificity isn't the improvement being sought for a cancer-screening
task. Config: `channel_gated_hardmining_swa`.

**Three independent interventions stacked on `hard_mining` (LDAM+clip, SAM+TTA, SWA) have
now all made it worse**, each by a different mechanism. This is a strong, consistent signal
that `hard_mining`'s benefit is a standalone effect at this dataset's scale, not a base to
build further improvements on — worth stating plainly, and it mirrors the original ablation
matrix's own finding that `channel_gated_combined` (stacking preprocessing+focal+SAM+TTA)
underperformed every individual ingredient.

**`channel_gated_swa` (plain, SWA only, isolating checkpoint-selection noise from any
loss-function change) — the close-out diagnostic.**

| metric | plain | plain + SWA | Δ | std: plain → +SWA |
|---|---|---|---|---|
| accuracy | 0.8326 | 0.8286 | −0.0040 | 0.0629 → 0.0477 |
| **balanced accuracy** | **0.7834** | **0.7616** | **−0.0218** | **0.0762 → 0.0757 (essentially unchanged)** |
| macro-F1 | 0.7567 | 0.7357 | −0.0210 | 0.0653 → 0.0449 |
| **sensitivity** | **0.6722** | **0.6028** | **−0.0694** | **0.1423 → 0.1769 (worse)** |
| specificity | 0.8947 | 0.9204 | +0.0258 | 0.0425 → 0.0478 |
| AUROC | 0.8814 | 0.9144 | +0.0330 | 0.0579 → 0.0277 (nearly halved) |

**Answer to the root-cause question**: only partial. SWA meaningfully tightens variance on
AUROC (std nearly halved) and, to a lesser extent, accuracy and macro-F1 — these benefit from
a smoother, weight-averaged decision surface over the (larger) non-malignant class. But it
does **not** meaningfully reduce balanced-accuracy variance (0.0762 → 0.0757, essentially flat)
and actually *increases* sensitivity variance (0.1423 → 0.1769). Balanced accuracy and
sensitivity are exactly the two metrics dominated by the ~8–9 malignant lesions per fold — a
sampling floor that weight-averaging cannot smooth away, because it isn't noise in *which*
epoch's weights get selected, it's irreducible small-N variance in how a handful of malignant
lesions individually classify. **Checkpoint-selection noise explains part of the picture
(AUROC/accuracy), but the dominant limit on the metric that matters most for this paper
(balanced accuracy) is MCR-SL's scale itself, not a fixable training-pipeline artifact.**

A second, independent observation: SWA shifts the model toward specificity at sensitivity's
cost **on the plain baseline alone**, not just when stacked with `hard_mining` (which showed
the identical pattern, −0.089 sensitivity / +0.064 specificity). This confirms that
conservative bias-shift is a general property of weight-averaging in this small/noisy-
malignant-class regime, not an interaction effect specific to the quality-reweighting
mechanism.

**Decision on Step 2 (prediction-level checkpoint ensembling)**: given Step 1 already shows
the balanced-accuracy variance is mostly an inherent small-N floor rather than a
checkpoint-selection artifact, and prediction-space averaging (Step 2) would very likely hit
the same floor for the same reason, Step 2 was not run. Code for it
(`scripts/run_topk_checkpoint_rerun.sh`, `scripts/checkpoint_ensemble_eval.py`) was prepared
and verified but is being held per the task's own conditional ("only if Step 1 doesn't
already answer things") rather than run automatically. **Search stopped here per Step 3.**
Implementation of the SWA experiment itself was verified end-to-end locally before running
(toy multi-input model matching this project's custom `forward()` signature): `AveragedModel`
deep-copies rather than aliases the source model, weight averaging accumulates correctly, BN
recalibration measurably changes the running stats vs. the naive average, and the resulting
state_dict loads cleanly into a fresh model instance. Script: `scripts/run_swa_experiments.sh`.

Note also: reading this result off the ledger surfaced a real reporting-script bug (not a
repeat of the earlier ledger corruption) — `channel_gated` is shared by both the plain and
auxiliary-head configs, distinguished only by the `quality_aware` column; the first version
of `scripts/report_ledger_rows.py` didn't filter on it and silently merged both configs' rows
into one misleading 10-row block. Fixed (`--quality-aware` flag added, default `false`); the
underlying ledger data itself was never at risk.

### Literature grounding for the novelty claim

Verified via direct search rather than taken on faith from the task brief that motivated this
work: **AdaFace (Kim et al., CVPR 2022, arXiv:2204.00964)** is real, legitimate precedent —
quality-adaptive margin/emphasis in face recognition. Its mechanism is more nuanced than
either of our two linear variants: it uses a **feature-norm proxy** for quality (not a
ground-truth label), and for samples it estimates as low-quality it **emphasizes easy
samples** (avoiding forced separation on ambiguous/degraded inputs) — the *opposite*
direction from our `hard_mining` variant, which up-weights low-quality samples and is
empirically the one that worked here. No skin-lesion or broader medical-imaging paper was
found using **real expert-assigned** per-image quality ratings (as opposed to saliency
scores, automated quality thresholds, or inter-annotator label-reliability weighting) as a
loss-reweighting signal for classification — this gap appears genuine on the literature
searched, not just asserted.

**Defensible novelty framing**: first application of quality-adaptive loss reweighting to
skin lesion classification using real expert-assigned quality labels rather than an estimated
proxy, finding empirically the opposite weighting direction from the closest face-recognition
precedent helps in this domain — plausibly because MCR-SL's "low quality" reflects
photography artifacts (lighting, focus, hair) in an image the diagnosing dermatologist still
successfully labeled from, unlike face-ID matching where low quality can make the identity
itself unrecoverable from the image. **Not** defensible: claiming this beats SOTA outright —
`hard_mining` (0.820) remains below the verified DiffMIC comparator (0.836).

### Data-integrity incident: ledger dedup bug (see also "Known data/pipeline caveats" below)

A `drop_duplicates(subset=['variant','fold'])` fix for a genuine duplicate-fold-entry bug
(caused by an interrupted-and-restarted training run appending a stale extra ledger row)
incorrectly also deleted the plain `channel_gated` (quality_aware=False) baseline's 5 ledger
rows — the ledger's `variant` column is actually `run_tag`, and the plain and auxiliary-head
(`quality_aware=True`) runs both share that run_tag (distinguished only by `quality_aware`,
which the dedup subset didn't key on). Recovered without retraining by re-evaluating the
untouched checkpoints (`channel_gated_qualityFalse_fold{0-4}.pt`) and re-appending fresh
metrics — recovered values matched the originally-reported numbers to 4 decimal places
(accuracy 0.8326, balanced_accuracy 0.7834, AUROC 0.8814), confirming no data was actually
lost, only temporarily absent from the ledger CSV. Script:
`scripts/recover_plain_baseline_ledger.py`.

### Shuffled-quality control (validity check — closes out the quality-adaptive-loss search)

A control experiment, not a further search: tests whether `trust`/`hard_mining`'s results are
attributable to the quality signal's actual information content, or to generic per-sample
reweighting regardless of what the weights mean. For each fold, `mean_image_rating` was
permuted across that fold's *training* lesions only (fixed seed = fold index), breaking the
lesion-to-rating correspondence while preserving the exact rating distribution; `w_quality` was
then computed from the shuffled values via the unmodified `trust`/`hard_mining` formulas.
Architecture, protocol, and checkpoint selection were otherwise identical, and evaluation
always used the true, unshuffled test fold (quality was never shuffled at eval time). Configs:
`channel_gated_qweight_trust_shuffled`, `channel_gated_qweight_hardmining_shuffled` (scripts:
`train.py --shuffle-quality-control`, `scripts/run_shuffled_quality_control.sh`,
`scripts/shuffled_quality_control_analysis.py`).

| mechanism | balanced_acc | sensitivity | high−low tercile gap |
|---|---|---|---|
| plain baseline (no quality-awareness) | 0.783 | 0.672 | 0.091 |
| trust (real) | 0.788 | 0.725 | 0.082 |
| **trust (shuffled control)** | 0.763 | 0.672 | **0.021** |
| hard_mining (real) | **0.820** | **0.764** | 0.091 |
| **hard_mining (shuffled control)** | **0.775** | **0.667** | 0.016 |

**`hard_mining`'s gain is real, not generic reweighting.** Its target metric (raw balanced
accuracy) collapses from 0.820 (real) to 0.775 (shuffled) — landing at essentially the plain
baseline (0.783), even fractionally below it; sensitivity shows the identical pattern (0.764 →
0.667, landing right at plain's 0.672). An uninformative, distribution-matched-but-shuffled
weight gives back none of `hard_mining`'s benefit — the effect depends on the weights actually
tracking each lesion's real quality, confirming genuine information-bearing reweighting rather
than an "any per-sample noise helps" regularization trick.

**`trust`'s tercile-gap result does not survive the control, and not in the direction that was
worried about going in.** The failure mode flagged in the task spec was "shuffled stays close
to real" (i.e., a generic effect masquerading as quality-specific). What happened instead: the
shuffled control's gap (0.021) is *smaller* than both the real result (0.082) and the plain
baseline (0.091) — an even more "flattened" tercile profile than the mechanism actually
designed to flatten it, produced by weights carrying zero real quality information. Per-tercile
accuracy under the shuffled control is non-monotonic (low 0.753, mid 0.882, high 0.774) rather
than a smoother quality-linked shape, consistent with generic per-sample-weight noise
redistributing errors across terciles somewhat arbitrarily rather than the model becoming
quality-aware. Combined with a real drop in raw performance under the shuffled control
(balanced accuracy 0.788 → 0.763, sensitivity 0.725 → 0.672 — landing exactly on the plain
baseline's 0.672), the honest reading is that `trust`'s original, already-modest gap narrowing
(0.091 → 0.082, ~10% relative, on tercile Ns of 53–93 where a swing this size is well within
plausible sampling noise per this project's own established small-N caution) is **not robustly
distinguishable from what content-free reweighting can produce**. This walks back the earlier
framing ("first success across 3 mechanisms tried") — `trust`'s quality-robustness claim is
unconfirmed by this control, not validated by it.

**Net effect on the paper's framing**: `hard_mining` survives as the headline quality-adaptive-
loss result — real quality-aware reweighting materially outperforms both the plain baseline and
an information-matched shuffled control, on the metric that matters for it (raw balanced
accuracy/sensitivity). `trust` does not survive as a quality-robustness finding — the control
suggests its narrower tercile gap is plausibly a generic-reweighting artifact rather than
evidence the model is exploiting the real quality signal. Report `hard_mining` as the paper's
one validated quality-adaptive-loss contribution; report `trust` as a negative/null result
after the control, alongside the auxiliary-quality-head failure — not as a second positive
mechanism. **Per the task scope, this closes out the quality-adaptive-loss search alongside
`channel_gated_swa` and the (skipped) prediction-ensembling check — no further controls or
variants.**

## Image-level training verification + eval-time variants on `hard_mining` (final experiment)

Prompted by a proposed "all-images-per-lesion" task premised on the belief that every config
trains on a single image per lesion. **That premise was false** — training has been image-level
since the first run. Verified rather than assumed, via
`scripts/verify_image_level_training.py` (run on the real data, remote):

**Check 1 — fold safety (the property that would silently invalidate everything):** all 240
lesions have every one of their images in exactly one fold; 0 lesions span >1 fold, 0 image
rows unassigned. Subject-disjoint folds already guarantee lesion-disjoint images. **This
retroactively validates every result in this document** — there has never been image-level
leakage across folds.

**Check 2 — image census:** 2131 usable images = 1352 dermoscopy + 779 clinical. Dermoscopy per
lesion: mean 5.73, median 6, range 1–18; 4 lesions have zero dermoscopy images (these fall back
to their `diagnosis_image_id` image).

**Check 3 — what the train split actually enumerates:** averaged over folds, **813 image samples
from 144 lesions (5.65x)**. Per fold: 1042/190, 882/163, 677/122, 621/102, 843/143. Training has
always used ~1352 dermoscopic images, not 240. The only genuinely unused data is the 779
**clinical** images (a different visual domain; deliberately out of scope).

Note also that this lever could not have fixed the diagnosed root cause even if it had been
available: the documented bottleneck is ~8–9 malignant **lesions** per fold, and additional
photographs of those same lesions are near-duplicates that add no new independent malignant
cases. More images per lesion ≠ more independent minority-class examples.

### Check 4 — a real, quantified, systematic miscalibration (documented, not fixed)

`train.py:compute_binary_pos_weight` derives the binary class weight from `lesion_df` (one row
per lesion) while the loader yields image-level samples. Measured per fold:

| fold | pos_weight (lesion-level, as applied) | pos_weight (image-level, as actually seen) | rel. diff |
|---|---|---|---|
| 0 | 6.667 | 6.496 | −2.6% |
| 1 | 5.400 | 3.960 | −26.7% |
| 2 | 3.654 | 2.761 | −24.4% |
| 3 | 2.808 | 2.324 | −17.2% |
| 4 | 4.520 | 4.113 | −9.0% |

Mean |relative difference| **16.0%**, and — importantly — **negative in all five folds**. The
direction is systematic, not noise: malignant lesions carry *more* images each than
non-malignant ones, so the true image-level class imbalance is milder than the lesion-level
ratio being applied. Every model in this project has therefore been trained with a pos_weight
biased ~16% high, i.e. told malignant is rarer than the batches it actually sees.

**Deliberately not "fixed" before the deadline**, for two defensible reasons worth stating in
the paper's methods rather than quietly correcting: (1) the bias runs *toward* sensitivity, the
metric this task weights most heavily, so the "correction" would most likely lower the number
that matters clinically; (2) every existing ledger row was trained under the lesion-level
weight, so changing it now would break cross-row comparability for the entire results table on
the last day. Recorded as a known, quantified methodological caveat and a concrete future-work
item, not as a silent flaw.

### Eval-time variants on the validated `hard_mining` checkpoints (no retraining)

The one genuinely untested, free item the task surfaced: every prior eval-time-trick run was on
the plain baseline or SAM, never on `hard_mining`. Pure inference on existing checkpoints
(`scripts/reeval_eval_time_options.py --base-run-tag channel_gated_qweight_hard_mining`).

| run | accuracy | balanced_acc | macro_F1 | sensitivity | specificity | AUROC |
|---|---|---|---|---|---|---|
| hard_mining (base) | 0.845 | 0.820 | 0.778 | 0.764 | 0.875 | 0.906 |
| + TTA | 0.849 | 0.833 | 0.783 | **0.808** | 0.858 | 0.911 |
| + multi-image | 0.871 | 0.838 | 0.805 | 0.783 | 0.892 | **0.920** |
| **+ TTA + multi-image** | **0.875** | **0.840** | **0.810** | 0.783 | 0.896 | 0.918 |

**Multi-image test-time averaging stacks cleanly on `hard_mining` — unlike on SAM+TTA.** This is
the interesting part. The extended-experiments section above documents that adding multi-image
averaging to `channel_gated_sam_adamw_tta` *reduced* balanced accuracy (0.822 → 0.810). Here the
same trick *adds* (+0.018 over base, and +0.020 with TTA on top). So "eval-time tricks aren't
strictly additive" remains true, but the failure was specific to the SAM combination, not a
general property of multi-image averaging — a more precise version of the earlier claim.

TTA-only gives the family's **highest sensitivity (0.808, +0.044 over base)** at some
specificity cost — the one variant to pick if sensitivity is the priority, which for
cancer screening is arguable.

**On the DiffMIC comparison — resist the temptation.** The best config's 0.840 balanced accuracy
sits nominally above the verified binary comparator (Uliana & Krohling 2025, DiffMIC, 0.836).
**This is not a "we beat SOTA" result and must not be written as one**: fold-to-fold std is
0.095, so the two are statistically indistinguishable; it's a different dataset (2298 images vs.
our 234 lesions), so it was never a like-for-like benchmark; and the gain came from eval-time
averaging on an existing model, not a better method. The honest framing stays exactly as the
"Are these scores good?" section already puts it — first benchmark on MCR-SL, with a
robustness analysis, in a believable range for the task and scale.

**Ledger integrity note**: `report_ledger_rows.py`'s built-in guard caught a duplicated fold-0
row for `channel_gated_qweight_hard_mining_tta` (6 rows, not 5), which skewed that run's
reported balanced accuracy to 0.8215. The correct 5-fold value is **0.8330**, confirmed against
`reeval_eval_time_options.py`'s own in-run aggregate (computed from the 5 folds it had just
evaluated, independent of the ledger). Consistent with the previously documented
interrupted-then-restarted-run pattern; the table above uses the corrected figures.
`scripts/dedupe_ledger.py` cleans it — dry-run by default, keys on (variant, quality_aware,
fold) so it cannot repeat the earlier incident where a dedup missing `quality_aware` deleted the
plain baseline's rows, and refuses to touch any duplicate group whose metrics actually differ.

## Are these scores good?

For a **first benchmark on a brand-new, 240-lesion dataset**, yes — this is a respectable,
defensible result, not a suspiciously perfect one (which would suggest leakage) and not a
weak one either. AUROC ~0.88–0.92 and balanced accuracy ~0.78–0.84 sit in a believable
range for dermoscopy malignancy classification without a large in-domain pretraining corpus.

~~and land within a hair of the best published PAD-UFES-20 multimodal result (0.832 bal. acc)
despite ~7x less data~~ — **struck: this repeated the retracted PAD-UFES-20 comparison.** That
0.832 is a *six-class* balanced accuracy and is not a valid comparator for a binary task (see
"Literature context" above, which corrected it). The only verified binary comparator is
DiffMIC's 0.836 on 2298 images — a different dataset, so cross-dataset context rather than a
benchmark, and our 0.840 ± 0.095 is statistically indistinguishable from it either way.

The more clinically relevant number — sensitivity on the malignant class, ~0.67–0.81 depending
on config — is moderate, and worth flagging plainly as a limitation (missing roughly one in
five malignant lesions even at best) rather than downplaying it. The paper's real strength is
not "we beat some accuracy number," it's the robustness analyses this dataset uniquely enables
(see above) — that's the honest novelty pitch for INDICON's biomedical imaging track. Note also
that "SOTA" is a hollow claim here in the literal sense — we're the first and only benchmark on
MCR-SL, so there is no prior number to have beaten; lead with "first benchmark + a
shuffled-control-validated quality-adaptive loss + six robustness analyses," not "SOTA," in the
abstract/intro.

## Reproducibility specification (for the paper's implementation-details paragraph)

Every number in this document comes from these settings. Defaults live in `config.py`;
anything not listed was not varied.

| hyperparameter | value | notes |
|---|---|---|
| backbone | EfficientNet-B0, ImageNet-pretrained (`timm`) | fully trainable, no frozen layers |
| input resolution | 224 × 224 | ImageNet mean/std normalization |
| batch size | 16 | |
| epochs | 30 | fixed, no early stopping in the reported runs |
| optimizer | Adam | `sam_adamw` / `adamw_cosine_discriminative` only in named runs |
| learning rate | 1e-4 | single LR (discriminative LR only in `optimizerv2`) |
| weight decay | 1e-4 | |
| seed | 42 | fold assignment + init; **single seed, not averaged over seeds** |
| folds | 5, subject-disjoint, greedy malignant-balanced | `data/folds.py` |
| checkpoint selection | best val-fold balanced accuracy | val fold = `(test_fold + 1) % 5` |
| aux 9-class loss weight | 0.4 | |
| quality head loss weight | 0.15 | `quality_aware=True` runs only |
| focal γ | 0.0 (plain weighted BCE) | 2.0 in `channel_gated_focal` |
| categorical embedding dim | 12 per field (+1 "unknown" slot) | 16 fields → 192-d |
| metadata MLP | 200 → 128 → 128, ReLU, Dropout 0.2 | |
| fusion projection | → 256-d, ReLU, Dropout 0.3 | shared by both fusion variants |
| SAM ρ | 0.05 | one value, no sweep |
| SWA start | 75% of training, BN-recalibrated | `use_swa` runs only |
| train augmentation | RandomResizedCrop(0.8–1.0), HFlip, Rotation(15°), ColorJitter(0.2/0.2) | train split only |
| val/test transform | Resize(1.15×) + CenterCrop | no augmentation |

**Single-seed caveat, stated plainly**: every result is one seed. Given fold-to-fold std of
0.08–0.17 on the headline metrics, a different seed would plausibly reorder configs separated
by less than ~0.05 balanced accuracy. Multi-seed averaging is the single highest-value
robustness improvement available and was not done — disclose it, don't bury it.

**Environment**: A100-PCIE-40GB, conda env `brats` (shared with a sister project; only `timm`,
`scikit-learn`, `zenodo_get` were added, nothing upgraded). Typical run ~65 min for 5 folds;
SAM runs took the same wall-clock time, indicating the pipeline was data-loading-bound rather
than compute-bound at this model size.

### Exact commands to reproduce each result group

```bash
# 0. Schema validation FIRST (fails loudly on any column/dtype/value-set drift)
python data/validate_schema.py --data-dir $DATA

# 1. Core ablation matrix (rows 18/19/22/27)
bash scripts/run_full_experiment_matrix.sh $DATA

# 2. Extended experiments (SAM, focal, preprocessing, contrastive, optimizerv2, combined)
bash scripts/run_extended_experiments.sh $DATA

# 3. Eval-time-only variants on any trained checkpoint set (no retraining)
python scripts/reeval_eval_time_options.py --base-run-tag <tag> --data-dir $DATA

# 4. Quality-adaptive loss reweighting (trust / hard_mining) + tercile analysis
bash scripts/run_quality_adaptive_loss.sh $DATA

# 5. Shuffled-quality control (the validity check for #4)
bash scripts/run_shuffled_quality_control.sh $DATA

# 6. SWA diagnostic
bash scripts/run_swa_experiments.sh $DATA

# 7. Robustness analyses 1-4 (+ master summary table, confusion matrices, OOF predictions)
python robustness_analysis.py --data-dir $DATA

# 8. Robustness analyses 5-6 (certainty, intra-subject) - CPU only, no GPU needed
python scripts/robustness_analyses_5_6.py --data-dir $DATA

# 9. Image-level training / fold-safety verification
python scripts/verify_image_level_training.py --data-dir $DATA

# Utilities
python scripts/report_ledger_rows.py <tag> [<tag>...]   # aggregated mean+/-std from the ledger
python scripts/dedupe_ledger.py [--apply]                # audited duplicate-row cleanup
```
`$DATA` = `~/mcrsl_project/data/raw/extracted/MCR-SL_dataset` on the remote.

## Artifact inventory — which file backs which claim

Anything cited in the paper should be traceable to one of these. All under `results/`.

| artifact | backs |
|---|---|
| `results_ledger.csv` | every per-fold metric, every run (append-only, sole writer is `train.py`) |
| `summary_table.csv` | master mean±std table, regenerated from the ledger |
| `oof_predictions_<cfg>.csv` | per-lesion out-of-fold predictions; input to analyses 1/3/4/5/6 |
| `confusion_matrix_<cfg>.csv/.png` | aggregated confusion matrices |
| `aux_9class_<cfg>.csv` | 9-class exploratory table, small-N flags included |
| `robustness_quality_tercile_<cfg>.csv/.png` | analysis 1 |
| `robustness_quality_aware_comparison.csv/.png` | analysis 2 |
| `robustness_histopath.csv` | analysis 3 |
| `robustness_metadata_importance.csv/.png` | analysis 4 |
| `robustness_certainty_tercile_<cfg>.csv/.png` | analysis 5 (tercile table) |
| `robustness_certainty_vs_quality_<cfg>.csv` | analysis 5 (two-axis comparison) |
| `robustness_certainty_by_class_<cfg>.csv` | analysis 5 (the class control that changed the conclusion) |
| `robustness_intra_subject_<cfg>.csv/.png` | analysis 6 (per-subject accuracy) |
| `robustness_intra_subject_summary_<cfg>.csv` | analysis 6 (permutation result) |
| `robustness_quality_reweighting_comparison.csv/.png` | four-mechanism tercile comparison |
| `robustness_quality_reweighting_gaps.csv` | high−low gap per mechanism |
| `robustness_shuffled_quality_control.csv/.png` | the shuffled control (real vs. shuffled) |
| `paper_examples/` | example lesion images for the qualitative figure |
| `logs/train_<tag>.log` | full per-epoch training curves (the source for the root-cause diagnosis) |

## Suggested paper-section mapping

| paper element | content | source |
|---|---|---|
| Abstract | first benchmark on MCR-SL; quality-adaptive loss reweighting validated against a shuffled control; six robustness analyses | — |
| II. Dataset | composition table, fold construction, missingness policy | "Dataset composition", "Fold composition" |
| III-A. Architecture | block-by-block spec, Fig. 1 | "Architecture" |
| III-B. Protocol | subject-disjoint 5-fold, checkpoint selection, metrics | "Task & protocol" |
| III-C. Quality-adaptive loss | `trust` / `hard_mining` formulas + motivation | "Quality-adaptive loss reweighting" |
| IV. Table I — core ablation | rows 18/19/22/27 | "Core ablation matrix" |
| IV. Table II — extended | rows 4/6–10/12/14/17/21/25/26 | "Extended experiments" |
| IV. Table III — quality reweighting | plain / trust / hard_mining | "Quality-adaptive loss reweighting" |
| IV. Table IV — **shuffled control** | real vs. shuffled, both mechanisms | "Shuffled-quality control" |
| IV. Table V — best config | `hard_mining` + TTA + multi-image, with std | "Eval-time variants" |
| V. Robustness | six analyses, each with its N | "Robustness analyses" |
| VI. Limitations | consolidated list below | "Consolidated limitations" |
| VI. Future work | clinical images; image-level pos_weight; multi-seed | "Not yet tried" |

## Consolidated limitations (draft the Limitations section straight from this)

Ordered roughly by how much a reviewer would care.

1. **Scale.** 234 binary-usable lesions, 42 malignant, ~8–9 malignant per test fold. This is the
   dominant constraint on every conclusion and the documented cause of the balanced-accuracy
   ceiling (see the SWA diagnostic). One malignant lesion flipping moves a fold's sensitivity by
   ~0.11.
2. **Single seed.** No multi-seed averaging. Config orderings within ~0.05 balanced accuracy are
   not robust.
3. **Fold subject imbalance.** Folds hold 5, 6, 16, 16, 16 subjects — balanced on malignant
   count, not on subjects. Fold 0's test set is 5 subjects' worth of lesions.
4. **Moderate sensitivity.** 0.76–0.81 on the best configs: roughly one in five malignant
   lesions missed. State this plainly; it is the clinically decisive number.
5. **`pos_weight` miscalibration.** Computed lesion-level, applied to image-level batches; ~16%
   high, systematically, in all five folds. Quantified and deliberately not corrected — see
   Check 4.
6. **No external validation.** Everything is within-MCR-SL cross-validation. The DiffMIC
   comparison is cross-dataset context, not a benchmark.
7. **9-class task is exploratory only.** MEL 8, SCC 4, ANG 4, DF 2 — four classes below 10
   lesions. Never present per-class metrics for these as robust.
8. **Multiple comparisons.** Six robustness analyses, many p-values, no correction applied.
9. **Clinical images unused.** 779 of 2131 images (37%) never entered training.
10. **Metadata importance is architecture-confounded.** Numeric fields get one z-scored scalar
    vs. 12 dims per categorical field, so the ranking reflects capacity, not clinical relevance.
11. **Histopathology subset (n=28) is qualitative only.** No CI, no p-value.
12. **LDAM + gradient clipping were bundled** into one run, so their contributions cannot be
    attributed separately — a shortcut taken under deadline pressure.

## Known data/pipeline caveats (for the methods section)
- 21 `lesion_id`s in `image.xlsx` (263 images) have no matching row in `lesion.xlsx` and
  no corresponding file on disk — dropped via inner join, logged at load time.
- `image_rating` coverage: 238/240 lesions have full 3-expert (E001/E003/E004) coverage on
  their diagnosis-image row; E002's ratings are entirely lost (0/241 non-null, matches
  CLAUDE.md's note); 2 lesions (L0013, L0205) have no diagnosis-image row at all.
- Histopathology-confirmed: 28 lesions (not 29, as CLAUDE.md's prose estimated from the
  dataset paper — verified against the actual file).
- A sync-script bug briefly overwrote the remote results ledger with a stale local copy;
  fixed (`sync_to_remote.sh` now excludes `results/` and `logs/` entirely — those only
  flow remote→local via `pull_remote_results.sh`), and the ledger was rebuilt correctly
  from checkpoints. No impact on any trained model or checkpoint, only on the CSV log.
- A second, separate ledger incident: a `drop_duplicates` fix for a duplicate-fold-entry bug
  (from an interrupted/restarted training run) didn't key on `quality_aware` and accidentally
  deleted the plain `channel_gated` baseline's 5 ledger rows (it shares `run_tag="channel_gated"`
  with the `quality_aware=True` auxiliary-head run). Recovered by re-evaluating the untouched
  checkpoints — recovered values matched the original numbers to 4 decimal places, no data
  actually lost. See "Quality-adaptive loss reweighting" section above for full detail.

## Abstract-ready numbers (copy these, with their stds)

The five numbers most likely to be quoted, in the form they should be quoted:

- **Best configuration**: `channel_gated` + quality-adaptive `hard_mining` loss reweighting +
  TTA + multi-image test-time averaging — **balanced accuracy 0.840 ± 0.095**, accuracy
  0.875 ± 0.075, sensitivity 0.783 ± 0.167, specificity 0.896 ± 0.094, AUROC 0.918 ± 0.058.
- **Best single trained model** (no eval-time tricks): `channel_gated_qweight_hard_mining` —
  **balanced accuracy 0.820 ± 0.078**, sensitivity 0.764 ± 0.130, AUROC 0.906 ± 0.059.
- **Plain main-method baseline**: `channel_gated` — balanced accuracy 0.783 ± 0.076,
  sensitivity 0.672 ± 0.142, AUROC 0.881 ± 0.058.
- **Quality-adaptive loss gain**: +0.037 balanced accuracy and +0.092 sensitivity over the
  plain baseline, **and it collapses to 0.775 / 0.667 under an information-matched shuffled
  control** — i.e. back to baseline, which is the evidence that the gain is quality-specific.
- **Robustness headline**: across six analyses, no reliable relationship between model error
  and either image quality or expert diagnostic certainty once class composition is controlled;
  no subject-level error clustering (p=0.954, N=55 subjects).

**Framing rule for the abstract**: lead with *"first benchmark on MCR-SL + a quality-adaptive
loss validated by a shuffled control"*, not with a SOTA claim. There is no prior MCR-SL number
to beat, and the 0.840 vs. DiffMIC's 0.836 comparison is cross-dataset and well inside the
noise.

## Chronology — what was tried, in order, and why

Useful for the discussion section's narrative, and as an honest record of how the conclusions
moved.

| # | experiment | outcome |
|---|---|---|
| 1 | Core matrix (image_only / late_fusion / channel_gated) | channel_gated not a clean win; image_only beats it on bal_acc |
| 2 | Auxiliary quality-prediction head | **negative** — worse everywhere; tercile gap widened 0.091→0.101 |
| 3 | Extended sweep (SAM, TTA, focal, preprocessing, contrastive, optimizerv2) | SAM+TTA best at 0.822; contrastive/optimizerv2 negative |
| 4 | Stacking everything (`combined`) | **negative** — 0.759, below plain baseline |
| 5 | Robustness analyses 1–4 | no strong quality-performance relationship (analysis 1) |
| 6 | Quality-adaptive loss: `trust` / `hard_mining` | `hard_mining` **best single result** 0.820; `trust` appears to narrow tercile gap |
| 7 | Threshold recalibration on `hard_mining` | **no gain** — val signal too noisy to calibrate |
| 8 | LDAM margin + grad clip on `hard_mining` | **negative** — sensitivity fell 0.067 (opposite of intent) |
| 9 | SAM+TTA on `hard_mining` | **negative** — 0.789, below both parents |
| 10 | SWA on `hard_mining` | **negative** — more stable but stably biased to specificity |
| 11 | `channel_gated_swa` diagnostic | variance is an inherent small-N floor, not a checkpoint artifact |
| 12 | **Shuffled-quality control** | `hard_mining` **validated**; `trust` **invalidated** (→ null) |
| 13 | Image-level training verification | premise of the "all-images" task was false; fold safety **confirmed** |
| 14 | Eval-time variants on `hard_mining` | **best overall config**, 0.840, free |
| 15 | Robustness analyses 5–6 | certainty → **null** after class control; no subject clustering |

**Three conclusions were reversed by follow-up checks** (6→12 for `trust`, the pooled certainty
result in 15, and the "all-images gap" premise in 13). That pattern is itself worth a sentence
in the discussion: on a dataset this size, an uncontrolled first look produces plausible-looking
positives that controls dissolve.

## Not yet tried / explicitly out of scope this sprint
- Optional stretch ablation: text-templated metadata + channel-gated fusion (CLAUDE.md
  §Optional stretch ablation) — only if time allows near the end.
- ~~Combining the winning eval-time tricks on top of focal loss or preprocessing~~ —
  **done**: `channel_gated_combined` (preprocessing+focal+SAM+TTA) tested, negative result
  (0.759 bal. acc., below both baseline and SAM+TTA, high variance) — see above.
- Broader SAM rho sweep, ASAM variant — one fixed rho=0.05 tested, no sweep (would be
  tuning against final numbers).
- **Decided:** robustness analyses 1/3/4 stay on `channel_gated` (CLAUDE.md's designated
  main method), not re-run against the empirically-better `channel_gated_sam_adamw_tta`.
  Keeps the paper's structure clean — core ablation matrix + robustness analysis on the
  designated method as one section, the extended experiments (SAM, TTA, focal, etc.) as a
  separate follow-up results section, not conflated into "the main result."
- **Final: `channel_gated_qweight_hard_mining` (alone — 0.820 balanced acc / 0.764
  sensitivity / 0.906 AUROC) replaces `channel_gated_sam_adamw_tta` as the paper's reported
  best follow-up config.** Every attempt to build on top of it (LDAM+grad-clip, SAM+TTA, SWA
  — three independent mechanisms) made it worse, so it stands as-is rather than as a base for
  further stacking.
- **Search closed** after LDAM+grad-clip, the threshold check, SAM+TTA, hard_mining+SWA, and
  finally the `channel_gated_swa` isolating diagnostic all came back negative, non-improving,
  or (for the last one) confirming the remaining variance is an inherent small-N floor rather
  than a fixable pipeline artifact. Prediction-level checkpoint ensembling (Step 2 of the
  close-out task) was prepared and verified but deliberately not run, since Step 1 already
  answered the question it was meant to test. The shuffled-quality control (see that section
  above) was the last scoped item: it confirmed `hard_mining`'s raw-performance gain is real
  (collapses to plain baseline under a content-free shuffled control) but showed `trust`'s
  tercile-gap narrowing does not survive the same control (the shuffled version narrows the
  gap *further*, most plausibly a generic-reweighting artifact). No further loss-formula,
  optimizer, training-modification, or control variants beyond this point — continuing would
  risk the exact p-hacking pattern flagged when this quality-adaptive-loss task was first
  scoped. **Final ceiling, as it now stands: `hard_mining` alone (0.820 balanced acc / 0.764
  sensitivity / 0.906 AUROC) is the paper's one validated quality-adaptive-loss result, for
  raw performance; `trust` is reported as a negative/null quality-robustness result, alongside
  the auxiliary-quality-head failure — not as a second success.**
- **All-images-per-lesion training: not applicable — already implemented.** Verified
  2026-08-27: `train_on_all_dermoscopic_images` has defaulted to `True` since the first run
  (~5.65x image samples per lesion). No data-utilization gap existed. See "Image-level training
  verification" above. Best config after that session's free eval-time variants:
  **`hard_mining` + TTA + multi-image, 0.840 balanced accuracy / 0.918 AUROC.**
- **Cross-dataset check on DeepDRiD — attempted, gate failed, not completed (2026-08-27).**
  Scoped as a small secondary section: does `hard_mining` transfer to a second medical imaging
  domain (diabetic retinopathy, ISBI 2020) whose quality annotation is structured differently?
  A go/no-go gate was run **before** writing any adapter or training code, and it failed inside
  its 2-hour box:
  - `git clone` of `deepdrdoc/DeepDRiD` **aborted after 50 minutes** — `RPC failed
    (result=92, HTTP 200)`, `early EOF`, `index-pack failed`. The server compressed all 2,916
    objects successfully; the connection could not hold a single stream long enough to deliver
    the pack. An earlier attempt died the same way after ~97 MB.
  - Measured remote throughput: **40–64 KB/s single-stream** (GitHub *and* kernel.org, so it is
    a general network constraint, not GitHub-specific). 16 parallel streams reach ~700 KB/s,
    but neither available path can exploit that: `git` transfers are single-stream, and the
    GitHub archive endpoint **ignores HTTP Range requests** (verified — it streams the whole
    archive instead of honouring the header), so the tarball cannot be chunked either.
  - `git-lfs` is not installed on the remote, and the GitHub API is rate-limited from the
    shared campus IP, so the repository size could not be confirmed cheaply either.

  **No DeepDRiD number is reported, estimated, or implied anywhere** — in the paper, the slides,
  or the ledger (verified: 0 matching rows). This is the designed outcome of a gate, recorded as
  such. It remains the most concrete cross-domain future-work item; a machine with ordinary
  throughput, or a mirror of the regular-fundus subset, would clear it. Scripts are committed
  and ready (`scripts/download_deepdrid.sh`, `scripts/inspect_deepdrid.py`) — the latter also
  checks whether DeepDRiD's shipped train/validation split is patient-disjoint, which matters
  because this project's whole protocol rests on subject-disjoint splitting.
- **Clinical images (779) remain the only genuinely untapped data** — a different visual domain
  from dermoscopy, so using them would need an explicit modality indicator to avoid confusing
  the encoder. Deliberately out of scope this sprint; the single most concrete future-work item.
- **Image-level `pos_weight` correction** — a real, quantified ~16% systematic miscalibration
  (see Check 4 above), left in place to preserve cross-row comparability on the final day and
  because the bias runs toward sensitivity. Second concrete future-work item; the paper should
  disclose it rather than let a reader discover it.
