# Task: Quality-Adaptive Loss Reweighting (headline novel contribution)

Companion file to `CLAUDE.md` — read that first for full project context. This file is a
narrow, time-boxed task spec for one specific experiment. Do not expand scope beyond what's
written here without checking in first; the deadline is 31 August 2026 and this needs to be
fully trained, evaluated, and logged with 2–3 days left over for writing.

## Why this exists (one paragraph, for context only — don't re-derive this)
The project's existing contributions (channel-gated fusion, SAM+TTA, the 5-fold protocol) are
solid but not novel — none of them are new mechanisms, and none require MCR-SL specifically.
The one thing that IS novel and dataset-specific: MCR-SL is the only public skin lesion dataset
with real, expert-labeled per-image quality ratings (not an estimated proxy). Face-recognition
research (MagFace, AdaFace, QAFace) has shown that using a quality signal to reweight the
training loss — not to predict it as an auxiliary output — measurably improves robustness on
low-quality inputs. This project already tried quality-as-auxiliary-prediction-target (the
`quality-aware` variant in the existing results) and it made things worse. This task tries
quality-as-loss-weighting instead, which is mechanistically different, not a retry of the same
idea with different hyperparameters.

## Objective
Add a per-sample loss weight, derived from each lesion's expert-rated image quality, to the
existing binary malignancy loss on the **channel-gated fusion** architecture (the project's
main method — do not apply this to image-only or late-fusion; those stay as-is for comparison).
Train and evaluate two directions of this weighting, add both to the results ledger, and re-run
the quality-tercile stratified analysis for both.

## Step 0 — Locate before you modify
Before writing any new code:
1. Find the existing weighted-BCE loss implementation for the binary head (likely in `train.py`
   or a `losses.py`/`models/` module). Confirm exactly how `w_class_i` (the per-fold malignant
   class weight) is currently applied — this task adds a multiplicative term alongside it, not
   a replacement.
2. Find where per-lesion quality data is already loaded — this project already computed
   `mean_image_rating` per lesion for the existing quality-tercile robustness analysis (using
   E001/E003/E004, excluding lost E002 ratings, linked via `diagnosis_image_id`). Reuse that
   exact computation; do not recompute quality scores from scratch or from a different subset
   of experts.
3. Confirm which 231 (or 234) lesions have a valid quality score, matching what the existing
   robustness analysis already used, so the new variant's results are directly comparable to
   the existing Table IV-style breakdown.

Report what you find (file names, function names, the exact quality-score computation) before
proceeding, so the plan can be sanity-checked against the actual codebase rather than assumed.

## Step 1 — Implement the weighting (two variants, both required)

```python
# quality_i: mean expert rating (1-10) for lesion i, from the existing computation (Step 0.2)
# w_class_i: existing per-fold class weight, computed exactly as it already is — unchanged

# Variant A — "trust": down-weight low-quality (less reliable) samples
w_quality_i_A = 0.5 + (quality_i - 1) / 9 * 1.0   # maps [1,10] -> [0.5, 1.5]

# Variant B — "hard-mining": up-weight low-quality samples to force robustness to them
w_quality_i_B = 1.5 - (quality_i - 1) / 9 * 1.0   # maps [1,10] -> [1.5, 0.5]

loss_i = w_class_i * w_quality_i * BCE(pred_i, y_i)
```

- Implement this as a configurable flag/parameter on the existing training script (e.g.
  `--quality-weight-mode {none, trust, hard_mining}`), not a new script — this must reuse the
  exact same data loading, fold splitting, checkpoint selection, and evaluation code as every
  other variant in the project, or the results won't be comparable.
- Do **not** parameterize the weight range further (no grid search over [0.5, 1.5] vs other
  ranges). Fixed range, two directions, that's the whole experiment. If there's a compelling
  reason to deviate, stop and flag it rather than expanding the search silently.
- Do **not** touch the architecture, the metadata encoder, the fusion mechanism, or any other
  hyperparameter. This is a loss-function-only change on top of the existing best architecture.

## Step 2 — Train and evaluate
- Both variants (A and B), channel-gated architecture only, under the exact same subject-disjoint
  stratified 5-fold protocol already used for every other row in the project's ablation table
  (same fold assignments — do not regenerate folds).
- Same checkpoint selection rule (val-fold balanced accuracy) as the rest of the project.
- Report the same metric set as the existing core ablation table: accuracy, balanced accuracy,
  macro-F1, sensitivity, specificity, AUROC, mean ± std over the 5 test folds.
- This dataset is small — both variants should be well within the existing per-run time budget
  observed for the channel-gated method. No new VRAM or runtime concerns expected.

## Step 3 — Extend the existing robustness analyses (don't rebuild them)
1. Add both new variants to the quality-tercile stratified table (the project's existing
   Table IV-equivalent) — same tercile boundaries, same 231-lesion evaluation set, same
   accuracy/sensitivity breakdown per tercile.
2. Recompute the Spearman correlation (rating vs. prediction error) for each new variant.
3. Report the high-minus-low tercile accuracy gap for both variants side-by-side with the
   existing plain channel-gated gap (0.091) and the existing failed auxiliary-quality-head gap
   (0.101). This three-way (plain / aux-head / loss-reweighted) comparison is the deliverable
   that actually answers the paper's open question — make sure it ends up in one table.

## Step 4 — Log everything, regardless of outcome
Append both variants to the project's master results ledger, exactly like every other run.
**All three outcomes are reportable and none should be hidden or re-run repeatedly to chase a
positive result:**
- If either variant narrows the tercile gap and doesn't hurt sensitivity/AUROC materially →
  headline positive result, becomes the paper's primary reported method.
- If both variants are neutral (gap unchanged, metrics ~flat) → report as a null result,
  distinct from the auxiliary-head's negative result (that one actively hurt; this one simply
  didn't help) — still an informative finding about which mechanisms do and don't work at this
  data scale.
- If both variants hurt performance → report as a second, convergent negative result alongside
  the auxiliary-head failure — two structurally different quality-awareness mechanisms both
  failing is itself a citable finding, not a dead end.

Do not attempt a third weighting scheme or additional hyperparameter search based on how A and
B turn out — one round, both directions, report honestly, move to writing.

## Time box
Steps 0–1: locate + implement, well under a day.
Step 2: train + evaluate both variants — this dataset trains fast; don't let this run long.
Steps 3–4: analysis + logging, same day as training given the small evaluation set.
Target: fully done within 2 days of starting, leaving the remainder of the timeline for the
architecture × quality-tercile comparison (if not already complete) and paper writing.
