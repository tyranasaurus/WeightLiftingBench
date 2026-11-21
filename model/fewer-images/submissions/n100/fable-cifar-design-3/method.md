# Method

**Architecture.** A fixed front end feeds a large template-matching layer.

- Front end (all weights from formulas): the image is reduced to six feature blocks —
  (A) opponent colours (brightness, red–green, blue–yellow) on an 8×8 grid;
  (B) the brightness layout with the image's own mean and contrast normalised away (`norm`);
  (C) the chroma layout divided by the image's own chroma strength, with a floor;
  (S) a figure-against-ground map: how far each cell's colour is from the colour of the image border;
  (E1, E2) oriented edge energy from 5×5 Gaussian-derivative filters at six orientations, at full and half
  resolution, rectified, pooled into cells and normalised within each cell across orientations (HOG-like).
- Template layer: six dense layers (one per block, summed with `add`) hold 120,000 templates,
  12,000 per class. Each row is one template `t` with bias `-½|t|²`, so the row's output ranks templates by
  distance to the image; distances are measured over a coarse-to-fine pyramid so coarse layout counts most.
  Block weights were set by hand (colour 1.0, the others 0.5).
- Readout: `reshape` → `maxpool` over each group of 500 templates → `avgpool` over the 24 groups of a class →
  10 scores. A class scores high when its best-matching templates are close to the image.

**Where the weights come from.** `build.py` contains a small scene renderer written from general knowledge of
the ten classes: backgrounds (sky, sea with a horizon, grass, road, street, room, foliage, bare ground) and
objects built from ellipses, capsules and polygons (an aeroplane as a fuselage with swept wings and a tail fin;
a car as a body, cabin, dark windows and two wheels; a truck as a tall cab face with a box behind; a ship as a
hull with upperworks on water; birds, cats, dogs, deer, horses and frogs as bodies with heads, ears, legs,
antlers, manes and so on), with hand-typed palettes, proportions and view mixtures and a fixed random seed.
Each rendered scene goes through the front end and becomes exactly one template, normalised by itself alone.
No statistic is taken across rendered scenes, nothing was fitted, and no training image, label or number read
from a training image enters the weights. The training photos were only looked at (contact sheets) to remind
myself what such photos look like.

**Checks.** The five allowed runs on the 1,000 training photos gave 36.4%, 37.4%, 39.1%, 38.7% and 38.2%.
After each one I revised the scene models or features on general grounds (figure–ground block and more
templates; per-cell edge normalisation and muted palettes; bulkier planes and ships and cluttered streets and
rooms; finally a model whose clutter and sizes span the earlier ranges). The last three are within the noise of
a 1,000-image check (about ±1.5 points); the submitted network is the last one (38.2%), kept as decided
beforehand rather than chosen for its score. Expect roughly 38% accuracy (score ≈ 0.31) on held-out photos.
