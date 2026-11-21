# Method

## Architecture (a hand-built "human face" detector)
All weights are computed in `build.py` from colour-science formulas, a fixed eye-band
template, and hand-chosen thresholds. No training image or label ever enters a weight.

Pipeline (46 layers, 170 non-zero weights):

1. **Pinkish human-skin map `S`** — a `1×1` conv turns each RGB pixel into 9 half-plane
   "violation" features of a skin box: YCbCr bounds (`Cr,Cb` warm red-dominant chroma),
   a brightness range, `R>G` and `G>B` ordering with minimum chroma (excludes washed-out
   beige/sand), and a **blue/red floor `B ≥ 0.55·R`**. A second `1×1` conv forms a soft
   AND (`1 − K·Σ relu(violations)`), so `S≈1` only on *pinkish* human skin. The B/R floor
   is the key step: it keeps light/medium skin but rejects the "orange" confounders that
   dominate the no-person class (wood, bread/food, brown animals).
2. **Luminance `Y`** — a `1×1` conv (`0.299,0.587,0.114`).
3. **Eye-band face evidence** at three scales (face ≈ 12/24/48 px, via `avgpool` factors
   2/4/8). At each scale a fixed `6×6` conv computes an eye-band contrast
   `mean(forehead+cheeks) − mean(eye row)`; a local `avgpool` of `S` gives the skin
   fraction over the same window. These are turned into `[0,1]` indicators and combined
   with a **`min` (built from `relu`/`add`): a dark eye-band that lies inside a pinkish
   skin region**. A global `maxpool` takes the best location per scale.
4. **Read-out** — the three scale scores are summed; `person = face_ev − 0.35`,
   `no_person = 0`. So the net predicts "person" only when it finds a human-skin-coloured
   face with an eye-band somewhere in the image.

## How the weights were produced
Every constant comes from general knowledge, not from the data: the YCbCr matrix, the
skin box / chroma / B-R thresholds (standard skin-detection colour science), the eye-band
template (a face is lighter at forehead/cheeks and darker at the eyes), and the slopes.
Images were only *looked at* (as PNG montages) and canonical skin/confounder colours were
tested *synthetically* to design the colour box; the detector was never run on training
pixels outside the 5 allowed checks.

## Rationale and the 5 checks
"No person" is the majority and is full of skin-coloured, face-structured things, so the
design aims for a person-specific cue (a human face) and for precision. The 5 checks were
used to validate principled design changes, not to fit weights: raw skin *area* did not
separate (acc .48) → localized skin *blob* suppressing full-frame warm (.50) → eye-band
faces, which fired on brown-animal/food faces (.48) → adding the pinkish-skin gate to the
blob (.50) → **pinkish-skin-gated eye-band faces (acc .522, TPR .495 > FPR .453)**, the
submitted network. It is the first configuration to beat the majority baseline, driven by
the B/R gate vetoing the animal/food "faces" that defeated the bare eye-band.
