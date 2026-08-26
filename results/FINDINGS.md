# MCR-SL Findings (working notes for the INDICON 2026 write-up)

Last updated: 2026-08-26. **Search closed.** Core matrix + extended experiments + the
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

## Task & protocol
Binary malignant vs. non-malignant lesion classification, MCR-SL dataset (240 lesions,
60 subjects, first benchmark on this dataset). Subject-disjoint stratified 5-fold CV;
per fold, a second held-out fold is used for checkpoint selection (never the reported
test fold) — no test-fold peeking, no hyperparameter tuning against final numbers.

6 lesions have `malignancy=="unknown"` and are excluded from the binary task (234 usable).
5 lesions have `unified_diagnosis=="UNK"` and are excluded from the 9-class aux task.

## Architecture (block-by-block, matches Fig. 1 in the paper and models/*.py)

- **Input**: one dermoscopic image (3×224×224, ImageNet-normalized) + patient metadata
  (16 categorical + 4 numeric fields, after dropping the constant-valued and unusably
  sparse free-text fields — see `data/schema.py`'s "explicitly dropped fields" note).

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

## Robustness analyses (the actual novelty)

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
gap to DiffMIC's 0.836 (2298 images vs. our 234 lesions — ~10x less data means proportionally
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

## Are these scores good?

For a **first benchmark on a brand-new, 240-lesion dataset**, yes — this is a respectable,
defensible result, not a suspiciously perfect one (which would suggest leakage) and not a
weak one either. AUROC ~0.88–0.92 and balanced accuracy ~0.78–0.82 sit in a believable
range for dermoscopy malignancy classification without the benefit of a large pretraining
corpus in-domain, and land within a hair of the best published PAD-UFES-20 multimodal
result (0.832 bal. acc) despite ~7x less data (see Literature context above). The more
clinically relevant number — sensitivity on the malignant class, ~0.67–0.74 depending on
config — is moderate, and worth flagging plainly as a limitation (missing roughly a
quarter to a third of malignant lesions) rather than downplaying it. The paper's real
strength is not "we beat some accuracy number," it's the robustness analyses this dataset
uniquely enables (see above) — that's the honest novelty pitch for INDICON's biomedical
imaging track (see prior discussion in this session). Note also that "SOTA" is a hollow
claim here in the literal sense — we're the first and only benchmark on MCR-SL, so there
is no prior number to have beaten; lead with "first benchmark + robustness analysis," not
"SOTA," in the abstract/intro.

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
