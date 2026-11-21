# Method

**Architecture** (6 layers, 17.6M weights):

1. `feat` — 7x7 conv, 3 -> 4 channels, all from formulas: Gaussian-blurred brightness, red-vs-green/blue
   opponent, blue-vs-red/green opponent, and a difference-of-Gaussians of brightness (fine detail).
2. `fn` — `norm`, so each channel is zero-mean / unit-variance per image (handles dark and washed-out photos).
3. `match` — dense layer of 4,300 templates = 43 classes x 100 placements (4 sign sizes x 5x5 shifts).
4. `grp` / `best` / `scores` — reshape to (1, 43, 100), maxpool over the 100 placements, reshape to (43,):
   the class score is its best-matching placement.

**How the weights were produced.** No training or fitting of any kind. `build.py` draws each of the 43 signs
from general knowledge of the German sign catalogue (red ring + stroke-drawn digits, red-bordered triangles with
simple black pictograms, blue discs with white arrows, yellow priority diamond, stop octagon, "end of" discs with
a diagonal band, and so on) on a neutral grey background, with 3x3 supersampling. Each drawing is passed through
the same first-layer filters and becomes its own template, normalised by itself alone (per-channel mean/variance,
brightness channels restricted to the sign's own area and made zero-sum there because background brightness is
unknown, then scaled to unit length). All sizes, colours, stroke shapes, scales, shifts and channel weights were
typed by hand; nothing was measured on, copied from, or tuned against the training images or generated images.

**Checks used:** 1 of 5. Check 1 (43 training images, accuracy 0.58) told me only that the network runs end to
end and that the templates respond to the right sign families; no number was changed after it.

`python3 /app/check_rebuild.py /app/out` reports REBUILD OK (build.py reproduces the weights with no data).
