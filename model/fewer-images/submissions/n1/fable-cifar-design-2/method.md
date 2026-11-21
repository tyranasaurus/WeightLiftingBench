# Method

**Architecture** (5 layers, about 1,200 weights):
`image -> conv 3x3 (10 ch) -> relu -> conv 1x1 (5 ch) -> avgpool 7x7 -> dense 10`.

- The first conv plus relu computes signed halves of three colour-opponent axes (luminance - 0.5, R - G,
  B - (R+G)/2) and of the Sobel x / y derivatives of luminance.
- The 1x1 conv recombines the halves into five maps: luminance, red-green, blue-yellow, vertical-edge energy
  |dL/dx| and horizontal-edge energy |dL/dy|, each minus a hand-chosen "ordinary photo" level.
- Average pooling turns each map into a 4x4 grid of cells (80 features).
- The dense readout is a hand-typed scene table: for every class I typed how far each feature is expected to
  sit from the ordinary level in each part of the picture (top / upper / lower / bottom rows, centre / side
  columns) — e.g. airplane = bright, blue, smooth border; ship = bright sky above, darker blue water below,
  horizontal edges in the middle; horse = sky above, dark red-brown centre, green-yellow below, vertical edges
  (legs) low down; frog = dark with very little blue; car / truck = grey road below, horizontal body lines,
  truck taller, brighter and boxier. The table is turned into weights with the diagonal-Gaussian formula
  `w = d / sigma^2`, `b = -sum d^2 / (2 sigma^2)` using hand-chosen spreads `sigma`.

**How the weights were produced:** entirely from formulas and constants typed in `build.py` from general
knowledge of what these classes' photos look like. No training, fitting, statistics of the training images,
or generated images were used. The 10 training photos were only viewed once as a montage.

**Status:** this is a simple fallback saved when time ran out; a richer design (a bank of procedurally drawn
scene templates) was planned but not built, so accuracy is expected to be modest (well above chance for
sky / water / grass scenes, weak for cat vs dog and car vs truck).
