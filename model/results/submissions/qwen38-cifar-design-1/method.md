# Method

## Architecture (all weights written by hand from formulas and rule tables — nothing fitted)

```
image (3,32,32)
 └ c1  conv 3→16, 5×5, pad 2   hand-designed filter bank:
       · 4 even-symmetric luminance bars (0°/45°/90°/135°, Gaussian windowed cosine)
       · 2 Gaussian-derivative step edges (fine vertical "bright-above", horizontal)
       · 8 double-opponent colour blobs (centre–surround difference-of-Gaussians with
         channel weightings: R−G, Y−B, white, black, green, blue, warm/tan/fur, red)
       · 1 coarse horizontal bar (horizons, vehicle sides)
       · 1 coarse bright-above-dark-below gradient (sky-over-ground layout)
 └ n1 norm → r1 relu → p1 maxpool 2×2          (contrast normalisation for lighting invariance)
 └ c2  conv 16→16, 3×3, pad 1  hand-written part/region detectors combining stage-1 channels:
       s0 wheel (dark disc centre + bright/textured ring, ring not dark), s1 bright round blob,
       s2 horizontal structure, s3 vertical structure, s4 fine structure (Gaussian-weighted edge
       density), s5 organic/diagonal texture, s6 sky (blue+bright+bright-above), s7 green foliage,
       s8 warm coat/ground, s9 light region, s10 dark mass, s11 red region, s12 large-scale
       layering, s13 local contrast, s14 water (blue + horizontal ripples + glints),
       s15 neutral/achromatic region (man-made grey/white/black).
 └ n2 norm → r2 relu → p2 maxpool 2×2 → (16,8,8)
 └ spatial-pyramid readout, three dense branches summed:
       d8: dense on the 8×8 map — per-class rule tables over (semantic channel × spatial profile),
           where every profile is made zero-sum over positions, so these terms fire only where a
           feature stands out relative to the rest of the same image (DC/contrast-baseline removal
           folded into the weights); rows are L1-mass-equalised across classes.
       d4: dense on an avgpool-2×2 (4×4) coarse copy of the same tables at half strength.
       d1: dense on the global average (1×1) — absolute scene gist (sky everywhere, forest green,
           water dominance) with hand-set biases centring a featureless image.
 └ out: add [d8, d4, d1] → (10,)
```

## How the weights were produced

Every number comes from code in `build.py`: analytic filter formulas (Gabor-type bars, Gaussian
derivatives, difference-of-Gaussians colour-opponent blobs), small hand-typed 3×3 combination
patterns encoding general shape knowledge (e.g. a wheel = dark disc with a textured ring near the
bottom of the frame), and per-class rule tables mapping semantic channels and spatial regions to
class scores from general knowledge of the ten classes (airplane: sky above, bright hard-edged
object centred, long horizontal wings; ship: water below, sky above, light superstructure at mid
height; automobile/truck: wheels at the bottom rows, horizontal body edges, neutral colours;
frog: green mass around a dark central body; deer: warm body centred in surrounding forest green;
horse: broad warm mass, horizontal body, vertical legs below; etc.). No training images, labels,
or statistics of them entered the weights; randomness is not used at all.

## What the five checks showed

1. First version: rule-table readout saturated — broad positive templates gave some classes a huge
   constant baseline (frog/cat won almost everything), 12%.
2. Added zero-sum (DC-removal) profiles so spatial terms respond only to relative placement: 13.6%,
   predictions spread out, ships and airplanes started being recognised (78–83 of their own class),
   but frog became the generic winner for centre-object images.
3. Balanced per-class L1 template mass and moved global gist to the 1×1 branch: 11.2% — frog still
   won on "any centred object" images because negative dense-channel terms backfire (features
   depleted at the centre turn penalties into bonuses).
4. Rewrote rules so negatives sit only on sparse, specific channels (wheel-at-bottom, water-at-
   bottom, sky-at-top) and green evidence for frog/deer uses surround profiles: 10% — the generic
   centre-object positive mass (dark-mass/structure/organic-texture at the centre) still favours
   frog and dog on most images, since every CIFAR-style photo has a centred object.
5. Final check (current weights): 11.3%. The remaining error is dominated by the generic
   centre-object pattern; per-class spatial evidence (sky/water/wheels/green) is too weak in the
   hand-set balance to override it without label-driven calibration, which the rules forbid.

The network is deterministic, data-free to build (`build.py` reproduces `network.json` and
`weights.npz` exactly, verified with `check_rebuild.py`), uses ~16.5k weights (limit 120M), and
every layer output shape was verified with `check_network.py`.
