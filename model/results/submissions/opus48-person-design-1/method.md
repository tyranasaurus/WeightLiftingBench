# Method

## Architecture
A small, fully hand-wired convolutional network (70 weights total). All numbers come from
formulas and hand-chosen constants in `build.py`; nothing is fitted to the training data.

Signal = **visible human skin** (face + hands/arms), made specific with two general-knowledge
properties that separate skin from reddish *objects* (wood, brick, meat, bread, tan surfaces,
warm light), which raw warm colour cannot distinguish:

1. **Colour (1×1 convs + relu).** Brightness-invariant *ratio* half-planes capturing skin's
   blood-red tint: `R≥1.13G`, `G≥1.02B`, `R≤1.85G`, `G≤1.5B`, `R≥0.20`, `R+G+B≤2.80`.
   Each broken rule is a "violation" `v_i=relu(linear)`; the skin map is `relu(1−Σv_i)∈[0,1]`.
   All skin tones (pale→dark) pass; beige/grey/snow/green/sky/orange/giraffe are rejected.
2. **Texture (3×3 high-pass + relu).** Skin is smooth; wood/brick/meat/bread are textured.
   Luminance → `identity−box3` high-pass → `|HF|=relu(HF)+relu(−HF)` → smooth gate
   `relu(1−6|HF|)∈[0,1]`.

Smooth-skin map = `AND(skin, smooth) = relu(skin+smooth−1)`. Its 6×6 local density is clamped
to a 0/1 "solid" indicator (`clamp(12·(d−0.4),0,1)`, drops scattered specks and thin edges),
then globally averaged to the **fraction of the photo that is solid smooth-skin**. The final
`dense` compares this fraction with a hand threshold (0.03 ≈ "a modestly visible person"):
person if the fraction exceeds it.

Primitive tricks (only conv/relu/avgpool/add/reshape/dense were used): violation-sum for the
colour AND-of-rules, `|x|` via two relus, map-AND via `relu(a+b−1)`, and a `[0,1]` clamp via
two relu pieces recombined with ±weights in the next linear layer. Valid (unpadded) design
choices keep every constant interpretable.

## How the weights were produced
Entirely by reasoning. The colour ratios and the "skin is smooth" texture cue are general facts
about human skin; the thresholds were chosen by hand and calibrated on *synthetic* colour
patches and blobs (never training images). The scale pyramid / pooling sizes follow from the
96×96 geometry. The five permitted checks were used only to understand the operating regime
(they revealed that raw warm-colour coverage is non-discriminative on this data — warm content
is equally common in both classes — which motivated the ratio + smoothness specialisation and a
conservative decision threshold). `build.py` is self-contained and rebuilds `network.json` and
`weights.npz` from its own source with no file or data access (verified by `check_rebuild.py`,
max relative difference 0.0).
