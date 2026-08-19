# Project Brief: Quality-Robust Multimodal Fusion for Skin Lesion Diagnosis (MCR-SL)

## Context
IEEE INDICON 2026 submission. Deadline **31 August 2026**. This is a short, tightly-scoped
conference paper (~13 working days from kickoff to submission) — depth over breadth. Do not
build an exhaustive architecture search; pick clean, defensible choices and spend remaining
time on the two robustness analyses below, which are the actual novelty.

**Dataset:** MCR-SL (Castro-Fernandez et al., *Data* 2025) — 240 skin lesions, 60 subjects,
779 clinical + 1352 dermoscopic images, rich per-subject/per-lesion structured metadata,
CC-BY license. Source: https://zenodo.org/records/17306338
This is a brand-new dataset descriptor paper with **zero prior benchmark/method papers** —
we are the first. Ground every design decision in the dataset paper's own schema (tables
below) — do not assume field names; verify against the actual downloaded CSVs first.

**Sister project:** a related project (TextBraTS, brain tumor segmentation) established
working patterns we're reusing here: offline-cached auxiliary-modality embeddings, an
SE-style channel-gating fusion block (`TextSemanticChannelMod`), and a discipline of logging
every experiment (including negative results) in one master results table. Reuse the coding
patterns, not the domain code — this is a classification task, not segmentation, on a
completely different (much smaller) dataset.

---

## Task Definition
- **Primary task:** binary malignant vs. non-malignant lesion classification.
- **Secondary/auxiliary task:** 9-class unified diagnosis (NEV, SK, BCC, AK, ATY, MEL, SCC,
  ANG, DF) — report as an exploratory table only. Several classes have <10 lesions (MEL=8,
  SCC=5, ANG=4, DF=2); do not present per-class metrics for these as if statistically robust —
  flag explicitly in every table/caption that uses them.
- Ground truth: `unified_diagnosis` field (Table 10 in the dataset paper) — histopathology-
  confirmed where available (29/240 lesions), expert-panel consensus otherwise.

## Data Schema Notes (VERIFY AGAINST ACTUAL FILES BEFORE CODING)
The dataset paper describes these entities/fields — confirm exact column names, encodings,
and file names once downloaded, since prose descriptions can drift from the real CSVs:
- **Lesion** table: `lesion_id`, `referral_diagnosis`, `lesion_status_when_captured`,
  `location`, `location_group`, `diameter`, `malignancy`, `lesion_diagnosis`,
  `diagnosis_image_id` (the specific image_id used for the expert diagnosis — important, see
  robustness note below).
- **Subject** table: `subject_id`, `derived_from`, `age`, `sex`, `height`, `weight`,
  `natural_hair_color`, `skin_reaction_to_sun`, mole counts, sunburn history, `sunbed`,
  `h_cancer`, `h_skin_cancer`, `h_skin_cancer_relatives`, `organ_transplant`,
  `immunosuppresion`.
- **Image** table: `image_id`, `lesion_id`, `modality` (clinical/dermoscopic).
- **Dermatology diagnosis** table: `diagnosis_id`, `lesion_id`, `image_id`, `expert_id`,
  `diagnosis`, `2nd_option`, `certainty`, **`image_rating`** (1–10, per expert per image),
  `time`.
- **Histopathology diagnosis** table: `lesion_id`, `procedure`, `tumor_thickness`,
  `diagnosis` — only 29 rows.
- **Unified diagnosis** table: final `unified_diagnosis` per lesion.

**Critical detail for the robustness analysis:** `image_rating` exists per (image, expert)
pair, but experts only rated **the single image used for diagnosis** (one per lesion, via
`diagnosis_image_id` — usually dermoscopic, occasionally clinical if no dermoscopic image
existed). So the quality-stratification analysis operates on ~240 (image, rating) pairs, not
all 2131 images. Also: **expert E002's image-quality ratings were lost** (technical error) —
average across the 3 remaining experts (E001/E003/E004), or whichever IDs remain after
verifying the real expert_id values.

## Eval Protocol
- **Subject-disjoint stratified 5-fold cross-validation.** Never split at the lesion level —
  a subject can have multiple lesions, and images of the same lesion are near-duplicates.
  Use `subject_id` for fold assignment.
- Stratify folds to balance malignant-lesion count per fold as closely as possible (subjects
  can carry multiple lesions of mixed malignancy — a simple greedy balancing on
  per-subject malignant-lesion count is sufficient; don't over-engineer this).
- Report mean ± std across the 5 folds for every metric.
- Metrics: accuracy, balanced accuracy, macro-F1, sensitivity/recall on malignant class,
  specificity, AUROC. Confusion matrix aggregated across folds.
- **No test-set peeking, no hyperparameter tuning against fold-5 results** — pick
  hyperparameters via one held-out validation fold before running the final 5-fold report.

## Architecture

### Image encoder
`timm` EfficientNet-B0, ImageNet-pretrained, dermoscopic image as primary input (the
`diagnosis_image_id` image, or all dermoscopic images per lesion if you want more training
signal — decide based on how much data augmentation is needed at this scale; log the choice).
Keep both the pooled feature vector (1280-d) and the last conv feature map for the
channel-gating fusion variant.

### Metadata encoder
Tabular, NOT free-text (this was a deliberate choice — do not build a text-templating
pipeline unless explicitly asked for the stretch ablation below).
- Categorical fields → `nn.Embedding` per field (dim 8–16), sized to cardinality +1 for an
  explicit **"unknown"** category — missing values are NOT imputed (matches the dataset
  authors' own stated policy of leaving missingness to end users; do the same and encode it
  explicitly rather than silently filling values).
- Numerical fields → z-score normalized using **train-fold statistics only** (recompute per
  fold — leaking test-fold stats into normalization is a real risk here given N=240).
- Concatenate embeddings + numerics → 2-layer MLP → 128-d metadata vector.

### Fusion — build TWO variants, both required for the ablation table
1. **Late-fusion baseline:** concat(pooled image vector, metadata vector) → MLP → logits.
   This is the standard MetaBlock-style baseline from prior literature — needed as a fair
   comparison point, not the contribution.
2. **Channel-gated fusion (main method):** metadata vector → linear → sigmoid gate (1280-d)
   → elementwise-multiply the image feature map's channels (broadcast over spatial dims)
   before global pooling → classifier. Structurally the same idea as an SE-block conditioned
   on metadata instead of the block's own pooled features.

### Heads
- Binary head (main): weighted BCE or focal loss, class weights from train-fold malignant
  ratio (~1:4.6 malignant:non-malignant per the dataset's overall counts — recompute per fold).
- Auxiliary 9-class head: CE loss, class-weighted, weighted ~0.3–0.5x relative to the binary
  loss in the combined objective. Report separately, do not let it dominate training.

### Optional stretch ablation (only if time allows near the end — do not start early)
Template metadata into one sentence per lesion, encode with a frozen small sentence encoder
(e.g. `all-MiniLM-L6-v2`), feed through the same channel-gating fusion as a 3rd variant. This
is purely to test whether a text-shaped representation of the same information behaves
differently from the tabular one — interesting for discussion, not required for the paper to
be complete.

## Ablation / Experiment Matrix (target — keep this list short, don't let it grow)
| # | Variant | Purpose |
|---|---|---|
| 1 | Dermoscopic-only, image encoder + binary head | Baseline |
| 2 | + metadata, late-fusion (concat) | Standard fusion baseline |
| 3 | + metadata, channel-gated fusion | Main method |
| 4 | (optional) + text-templated metadata, channel-gated | Stretch, discussion only |

## Robustness Analyses (this is the actual novelty — prioritize these)
1. **Quality-stratified performance.** For the best model (from the matrix above), bucket
   lesions by mean `image_rating` (across available experts) into terciles; report
   accuracy/sensitivity per tercile, and Spearman correlation between rating and per-lesion
   error/predictive confidence. N per tercile will be modest (~80) — report it, don't
   overstate significance.
2. **Quality-aware training (upgrade from eval-only to methodological contribution).** Add a
   small auxiliary regression head predicting `image_rating` from image features, jointly
   trained with the classification heads (low loss weight, e.g. 0.1–0.2). Compare against the
   non-quality-aware version on the same quality terciles — does explicit quality-awareness
   flatten the performance gap across terciles?
3. **Histopathology-confirmed vs. panel-consensus-only.** Report accuracy/confidence
   separately for the 29 histopath-confirmed lesions vs. the other 211. Treat as a qualitative
   finding (n=29 is too small for confidence intervals) — frame accordingly, don't compute
   a p-value on 29 samples and call it significant.
4. **Metadata-attention sanity check.** Post-hoc (gradient×input or simple ablation-by-field
   importance) on the metadata MLP for the best model. Compare the top-ranked fields against
   the fields the dataset paper's own statistical tables (their Tables 3–4) found significantly
   associated with malignancy (location_group, sex, referral_diagnosis, diameter all had
   p<0.01 there). Report whether the model's learned importance aligns.

## Code / Deliverables Expected From Claude Code
- Clean, modular PyTorch code: `data/` (loaders, subject-disjoint fold splitting, schema
  validation against the actual CSVs), `models/` (image encoder, metadata encoder, fusion
  variants, heads), `train.py`, `evaluate.py`, `robustness_analysis.py`.
- A single master results ledger (CSV or markdown table) logging every run: variant, fold,
  all metrics, notes — append-only, never overwrite. Include negative results if any variant
  underperforms the baseline; that's paper material (limitations/discussion), not something
  to hide or silently drop.
- A schema-validation script that runs FIRST, before any training code, and asserts the
  actual column names/dtypes/value sets match what's assumed above — fail loudly and report
  the diff if not, rather than silently coercing.
- Fold-aggregated results table (mean±std) + confusion matrix + the 4 robustness analyses
  above as both raw data (CSV) and simple plots (matplotlib, no fancy dependencies).

## Remote Execution & Sync Workflow (A100)
Development happens locally — Claude Code edits code on this machine. All GPU work and the
dataset itself live on a remote A100 40GB PC. **Never route the dataset through the local
machine** — download it directly on the remote.

**Remote details (confirmed):**
- Host: `cs24d0010@172.16.1.199` (same machine as the sister TextBraTS project, password auth)
- Remote project path: `~/mcrsl_project/` (separate top-level directory — not nested under
  `~/BraTs/`, to keep this project's data/checkpoints fully independent)
- Conda env: `brats` (**reused** from the sister project — see caution below)
- GPU: A100-PCIE-40GB. This dataset is tiny (240 lesions) compared to the sister BraTS
  project — VRAM will not be the bottleneck here the way it was for the 3D volumetric runs.
  Don't spend time on memory optimization; prioritize code clarity and fast iteration.

**Shared-env caution:** the `brats` env is also used by the live TextBraTS project on the same
machine. Before installing anything, run `pip list` / `conda list` in that env and diff against
what's needed here (`timm`, `scikit-learn`, and `zenodo_get` are the likely gaps — `torch` and
CUDA should already be present). Install missing packages with `pip install <pkg>` only, and
do **not** upgrade or pin-change any already-installed package (especially `torch`,
`torchvision`, `transformers`) — a version bump here could silently break BraTS reproducibility
on a shared env. If a genuine version conflict shows up, stop and create a fresh env instead of
forcing it.

**Workflow:**
1. Edit code locally in this repo.
2. Push to remote with `sync_to_remote.sh` (rsync; excludes `data/` and `checkpoints/`, which
   live remote-only).
3. SSH into the remote, activate the conda env, run the schema-validation script **first**,
   then training/eval.
4. Pull results back with `pull_remote_results.sh` — only `results/` and `logs/`, never raw
   data or full checkpoints unless a specific checkpoint is needed locally for inspection.

**Dataset download — run this ON the remote, not locally:**
```bash
pip install zenodo_get
zenodo_get 10.5281/zenodo.17306338 -o ~/mcrsl_project/data/raw
```
`zenodo_get` pulls every file attached to the record in one command. After it finishes, confirm
the downloaded file/folder names match what the dataset paper describes (image folders +
metadata spreadsheets) before pointing the schema-validation script at them — do this check
before writing any loader code.

## What NOT to do
- Don't build a text-templating pipeline unless doing the optional stretch ablation.
- Don't tune hyperparameters against final-fold results.
- Don't present 9-class per-class metrics without flagging small-N classes in every table.
- Don't let the ablation matrix grow past what's listed above — if a new idea comes up
  mid-sprint, log it under "future work" rather than implementing it now.
- Don't impute missing metadata — encode missingness explicitly.
