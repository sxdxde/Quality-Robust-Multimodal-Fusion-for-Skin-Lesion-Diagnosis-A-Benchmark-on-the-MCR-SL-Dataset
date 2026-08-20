# MCR-SL Findings (working notes for the INDICON 2026 write-up)

Last updated: 2026-08-20. All planned experiments complete, including SAM(AdamW).

## Task & protocol
Binary malignant vs. non-malignant lesion classification, MCR-SL dataset (240 lesions,
60 subjects, first benchmark on this dataset). Subject-disjoint stratified 5-fold CV;
per fold, a second held-out fold is used for checkpoint selection (never the reported
test fold) — no test-fold peeking, no hyperparameter tuning against final numbers.

6 lesions have `malignancy=="unknown"` and are excluded from the binary task (234 usable).
5 lesions have `unified_diagnosis=="UNK"` and are excluded from the 9-class aux task.

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

### Does anything cross 0.9?
AUROC does, comfortably now — best config 0.915, several others 0.90–0.91. Balanced
accuracy tops out at 0.822 and sensitivity at 0.736. Given ~40 malignant lesions total,
that ceiling on balanced accuracy/sensitivity is real, not a tuning gap — don't chase it
further by tuning against these numbers.

### Literature context (see prior discussion this session)
Closest comparator, PAD-UFES-20 (metadata+image fusion, ~1,641 lesions — ~7x our N): best
published multimodal balanced accuracy there is 0.832. Our best (0.822) is within a hair
of that despite ~7x less data — a legitimate, citable line for the paper. HAM10000/PH2
headline numbers (accuracy 0.88–0.99) are not fair comparators — 40x+ more data and/or an
easier task and/or looser evaluation rigor; don't cite them without heavy caveats.

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

## Not yet tried / explicitly out of scope this sprint
- Optional stretch ablation: text-templated metadata + channel-gated fusion (CLAUDE.md
  §Optional stretch ablation) — only if time allows near the end.
- Combining the winning eval-time tricks on top of focal loss or preprocessing (only ever
  tested against plain baseline and SAM) — untested combination, possible but not run.
- Broader SAM rho sweep, ASAM variant — one fixed rho=0.05 tested, no sweep (would be
  tuning against final numbers).
- **Decided:** robustness analyses 1/3/4 stay on `channel_gated` (CLAUDE.md's designated
  main method), not re-run against the empirically-better `channel_gated_sam_adamw_tta`.
  Keeps the paper's structure clean — core ablation matrix + robustness analysis on the
  designated method as one section, the extended experiments (SAM, TTA, focal, etc.) as a
  separate follow-up results section, not conflated into "the main result."
