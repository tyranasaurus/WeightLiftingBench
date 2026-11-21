# Method

**Architecture.** One 28×28 convolution with 16,800 output channels (1,680 per class),
padding 2, giving 5×5 alignments (±2 px). A max-pool over the 25 alignments, a reshape to
(10, 1680, 1), and a max-pool over each class block produce the 10 class scores. Every
channel is a *prototype* of one digit, and the class score is the best match among its
prototypes. 13.2 M weights, all produced by `build.py` from typed pen paths and formulas.

**Where the weights come from.** No training, no fitting, no statistic of the training
images. I wrote each digit as a set of pen strokes — Catmull-Rom splines through control
points I typed from general knowledge of how the digits are written (an oval for 0, a bar
with optional flag/foot for 1, a curl + diagonal + base for 2, two bumps for 3, a left arm
+ crossbar + vertical for 4, and so on), with 6 style variants per digit. Sharp corners
(4's elbow, 5's shoulder, 3's pinch) are split into separate strokes so the spline cannot
overshoot into a blob. Each stroke is brushed with a disk whose radius varies along the
path. A prototype is then passed through the *same* normalisation the task describes:
shear for slant, an aspect scale, a smooth sinusoidal elastic warp (fixed seed), fit into a
20×20 box preserving proportions, and translated so its ink centre of mass is at the frame
centre. The bank is a grid over 7 slants × 3 aspects × 5 brush radii × 16 warp seeds.

**The matching score.** For a prototype I compute the exact distance from every pixel to
the brushed path (min over path samples of distance minus local radius). From it:
- `S(p) = exp(-max(0,d)^2 / 2σ²)`, σ=1.6 — a soft ridge rewarding image ink near the stroke;
- `relief = Gaussian(ρ=1.0) * ink`, divided by its own self-overlap so a perfect match
  refunds exactly the mass charge;
- kernel `S + λ·relief`, bias `−λ·mass`, λ=0.6.

This is a two-sided soft chamfer: image ink is rewarded for lying on the prototype, and the
prototype is charged only for strokes the image does not supply. The charge is what stops a
bare `1` bar (a subset of almost every digit) from winning everywhere.

**Checks used (5 of 5).** (1) 58.2% — a zero-sum kernel over-penalised complex digits, so
`1` won almost everything. (2) 71.0% — coverage minus a flat ink-mass charge fixed that.
(3) 76.9% — two-sided chamfer, corner-split strokes, wider deformation. (4) 77.3% — full-frame
kernels (prototypes were being cropped), σ=1.6, calibrated relief, 1,680 prototypes/class.
(5) 43.9% — adding a "how many prototypes match well" soft-vote branch collapsed to class 1,
because a small prototype's score is a large fraction of its own self-score on any digit;
I reverted to the check-4 network, which is what is submitted.

**Known limits.** Residual confusions are 4↔9 (centre-of-mass centring puts a 9's loop bottom
near a 4's crossbar), 2→4/1, 5→6 and 3→5. They are shape-fidelity limits of the typed models,
not optimisation artifacts; I had no checks left to verify further model edits, so I
submitted the best configuration I had actually measured.
