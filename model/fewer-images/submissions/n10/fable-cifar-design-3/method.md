# Method

**Architecture** (17,667 weights): `image -> conv 3x3 + relu -> conv 1x1 + relu -> conv 1x1 -> avgpool 5x5 -> dense + relu -> dense (10)`.

- Layer `a` computes per-pixel colour-opponent differences (B-R, R-G, G-B and their negatives), brightness parts,
  and Sobel derivatives of luminance.
- Layers `b`/`c` turn these into ten soft maps in [0, 1]: blue (sky/water), green (vegetation), warm (brown/tan/fur),
  red (paint), white, grey, dark, strong horizontal edge, strong vertical edge, and flat (no edges). Each is a
  hand-written min/clip formula built from relus, with thresholds chosen from general colour knowledge.
- `cells` averages each map on a 6x6 grid; `regions` takes the mean over four regions (top, bottom, centre, border)
  and adds five conjunction units (sky everywhere, grass under sky, green everywhere, white everywhere,
  textured object on a smooth background).
- `scores` is a hand-typed evidence table: for every (map, region) and every conjunction, how much it speaks for or
  against each class (e.g. blue at the bottom -> ship, grey at the bottom -> automobile/truck, green at the bottom
  under sky -> horse, warm everywhere -> cat/dog/frog, smooth border -> airplane/bird).

**How the weights were produced**: every number is typed by hand in `build.py` from general knowledge of what the
classes and their usual surroundings look like. No training, fitting or statistics of any images were used, and no
generated images were used. The training photos were only viewed, to get a qualitative impression.

**Checks**: one labelled run on the 100 training images, used only to confirm the network runs and is not degenerate;
no number was changed after it.
