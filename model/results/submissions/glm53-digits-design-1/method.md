# Method

The network is a hand-crafted shape matcher; not a single weight was trained,
fitted, or copied from data. Everything comes from formulas and from typed
pen-path models of how people write digits.

**Features (conv, formulas only).** The first conv layer computes eight
oriented bar detectors (a second-derivative-of-Gaussian "bar": positive centre
lobe, negative flanks, hard-truncated across the stroke so the two sides of a
small loop do not suppress each other). Bars are orientation-selective and
nearly silent on blobs. Auxiliary channels: local ink density, local emptiness
(1 - ink), two rings, a small blob detector and a broad-ink channel. A relu +
2x2 max-pool thresholds and gains these into quasi-binary cell maps on a
14x14 grid. Because real stroke widths run from ~1.5 to ~6 px, a parallel
branch erodes the image with a 3x3 "fully-inked" conv (thick strokes collapse
to a ~2px core) and recomputes the same bar features on the core; the two
branches are combined later with an exact maximum.

**Templates (typed pen paths).** For each digit class I typed several
prototype pen paths (stadium-shaped loops with straight sides and rounded
corners, plus slanted copies), from general knowledge of digit shapes and the
way the 20x20 box and centre-of-mass centring place them in the frame. Each
prototype is rendered deterministically by a formula renderer, and its
template row is built from the rendering: positive deposits along the stroke
into the matching orientation channel (anisotropic spread: wide across the
stroke to absorb slant, tight along it), positive deposits on the
must-be-empty channel inside loops, negative weights on the ink channel where
a genuine digit of this class never has ink, and larger deposit weights on
the strokes that make the class recognisable (the 7's top bar, the 8's
left waist, the 9's loop wall, ...). Each row is self-calibrated: channels are
rescaled by the response they produce on the prototype's own rendering,
deposits the own rendering cannot detect are mostly pruned, and the row is
normalised so an ideal match scores about 1.

**Matching and readout.** Dense layers dot the pooled features against every
template in both branches; per-prototype scores are combined with an exact max
(A + relu(B - A)), floored, z-normalised per image (a norm layer over the
prototype axis), and each class scores through an exact running max over its
prototypes (chained dense+relu+add). A tiny mean-template branch breaks ties.
The result is a nearest-prototype classifier whose every component - filters,
paths, thresholds, zones, critical-stroke weights - was written by hand.
