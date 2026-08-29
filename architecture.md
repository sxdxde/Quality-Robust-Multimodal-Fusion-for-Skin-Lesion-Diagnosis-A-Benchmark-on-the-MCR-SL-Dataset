# Architecture Specification & Diagram Prompt

Reference document for the MCR-SL quality-adaptive loss paper. Part 1 is the verified
architecture (read off `models/*.py` and `data/schema.py`, not from the draft). Part 2 is a
copy-paste prompt for generating the figure.

---

## Part 1 — The architecture, verified

### Architectural style, in one line

A **two-stream (dual-encoder) multimodal classifier with SE-style channel gating**, trained
with a **per-sample quality-adaptive loss weight**. The novelty is the loss weighting only —
the backbone and the gating mechanism are both established components.

### Stream A — Image Encoder

| item | value |
|---|---|
| backbone | EfficientNet-B0, ImageNet-pretrained (`timm`) |
| input | dermoscopic image, `3 × 224 × 224` |
| trainable | fully, no frozen layers |
| output 1 | conv feature map `1280 × 7 × 7` (pre-pool) |
| output 2 | pooled vector `1280-d` (used by the late-fusion baseline only) |

### Stream B — Metadata Encoder

| item | value |
|---|---|
| categorical fields | **17**, each `nn.Embedding(cardinality + 1, 12)` |
| — the `+1` slot | reserved "unknown" index; missing values routed there, **never imputed** |
| categorical width | 17 × 12 = **204-d** |
| numeric fields | **4** (age, height, weight, diameter) |
| — encoding | z-scored on train-fold statistics only, each paired with a 0/1 missing-bit |
| numeric width | 4 × 2 = **8-d** |
| concatenated | **212-d** |
| MLP | `Linear(212→128) → ReLU → Dropout(0.2) → Linear(128→128) → ReLU` |
| output | **128-d** metadata vector |

### Fusion — Metadata-Conditioned Channel Gate

```
gate      = sigmoid( Linear(128 → 1280) )          # (B, 1280), values in [0,1]
gated_map = feature_map * gate[:, :, None, None]    # broadcast over the 7×7 grid
pooled    = AdaptiveAvgPool2d(1)(gated_map)         # (B, 1280)
fused     = Linear(1280 → 256) → ReLU → Dropout(0.3)  # (B, 256)
```

This is the Squeeze-and-Excitation mechanism, but the gating signal comes from **patient
metadata** instead of from the feature map's own statistics. Not claimed as novel.

### Prediction Heads (all on the same 256-d vector)

- **Binary head** — `Linear(256 → 1)` → sigmoid → `P(malignant)`. The reported task.
- **Auxiliary head** — `Linear(256 → 9)` → 9-class unified diagnosis, class-weighted CE,
  contributes at **0.4×**. Exploratory only.

### The Loss — the paper's contribution

For lesion *i* with expert quality rating `q_i ∈ [1, 10]`:

```
w_qual_i = 1.5 − (q_i − 1) / 9          # maps [1,10] → [1.5, 0.5]; low quality weighs MORE
w_cls_i  = n_neg / n_pos                # per-fold class weight, recomputed each fold
L_i      = w_cls_i · w_qual_i · BCE( p̂_i , y_i )
L_total  = L_binary + 0.4 · L_aux
```

Lesions with no rating get `w_qual = 1.0`.

**Critical property to convey in the diagram:** `w_qual_i` is a **precomputed, constant scalar**
— not a learned parameter, and never differentiated. Gradients flow back through the network by
the ordinary chain rule from the weighted BCE term. The quality path *enters* the loss; it is
not part of the backward path.

---

## Part 2 — Diagram generation prompt

> Copy everything in the block below.

```
Create a clean, publication-quality neural network architecture diagram for an A*
computer-vision conference paper (CVPR / MICCAI house style).

STYLE
- Flat 2D vector illustration. No 3D, no gradients, no drop shadows, no glow, no textures.
- White background. Thin (1–1.5 px) dark-grey strokes. Rounded rectangles, ~6 px corner radius.
- Muted pastel fills, one hue per functional group. Dark grey text, clean sans-serif
  (Helvetica / Inter / Arial). No italics, no all-caps except the group labels.
- Generous whitespace. The diagram must read as airy, not dense. Leave clear margins.

LAYOUT
- Strictly horizontal, left-to-right flow, in a wide banner (about 3:1 aspect ratio).
- Two parallel horizontal lanes that converge, then a third short path entering from below-right:
  - TOP LANE  = image stream
  - BOTTOM LANE = metadata stream
  - The two lanes MERGE at a circled multiply symbol in the middle-right of the figure.
- Solid arrows for the forward data path. One clearly DASHED arrow for the quality path.

GROUPING (draw these as light dashed rounded containers with a small label above each)
1. "IMAGE ENCODER"      — encloses the top-lane encoder block
2. "METADATA ENCODER"   — encloses the bottom-lane embedding + MLP blocks
3. "CHANNEL-GATED FUSION" — encloses the gate block and the multiply symbol
4. "QUALITY-ADAPTIVE LOSS" — encloses the loss block; give this container a subtly warmer
   fill so the reader's eye lands on it, since it is the paper's contribution

BLOCKS AND LABELS (use exactly this short text, nothing more)

Top lane, left to right:
  [ Dermoscopic image ]  small caption underneath: 3 x 224 x 224
      -> [ EfficientNet-B0 ]
      -> arrow labelled "1280 x 7 x 7"
      -> ( x )   a circled multiplication symbol
      -> [ Global avg pool + FC ]  caption: 256-d
      -> [ Binary head ]  caption: P(malignant)

Bottom lane, left to right:
  [ Patient metadata ]  small caption underneath: 17 categorical + 4 numeric
      -> [ Embeddings + MLP ]  caption: 212-d -> 128-d
      -> [ Sigmoid gate ]  caption: gate in [0,1]^1280
      -> a vertical arrow going UP into the ( x ) symbol, labelled "channel gate"

Loss path, right side:
  [ Binary head ] -> solid arrow DOWN into -> [ Weighted BCE loss ]
  [ Expert quality rating q ] -> DASHED arrow RIGHT into -> [ Weighted BCE loss ]
  Label the dashed arrow simply: "quality weight"
  Inside the loss block put only:  L = w_cls * w_qual * BCE
  Directly beneath the loss block, one small grey note: "w_qual precomputed, not learned"

SPACING RULES (important)
- Leave at least one full block-width of empty space between consecutive blocks.
- Never place a text label so it overlaps or touches a box; put arrow labels in clear space.
- No label longer than four words. No sentences anywhere inside the figure.
- Do not draw a legend, a title, axes, or a caption — those live outside the image.

DO NOT INCLUDE
- No formulas other than the single one inside the loss block.
- No layer-by-layer CNN stacks, no convolution cubes, no feature-map slabs.
- No photographs, no example lesion images, no heatmaps.
- No numbers other than the tensor shapes listed above.
- No decorative icons beyond one small camera or image glyph on the input block (optional).
```

### Notes for whoever runs the prompt

- If the generator crowds the labels, re-run asking for **fewer words per block** first; shrinking
  the text is the wrong fix, since the figure must stay legible at a 3.5-inch column width.
- The one thing that must survive any simplification is the **dashed quality path entering the
  loss block**, since that single arrow is the paper's contribution. If a generation drops or
  solidifies it, discard that generation.
- Generated raster output is a drafting aid. For camera-ready, redraw the accepted layout in
  TikZ or Inkscape so the figure is vector and the text is real text.
