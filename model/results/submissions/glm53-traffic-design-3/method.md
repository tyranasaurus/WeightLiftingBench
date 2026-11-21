# Method

The network is a hand-built template matcher — no training, no statistics of the
training images; every weight comes from general knowledge of German road signs
drawn in code.

**Feature stack.** The image is turned into 8 maps: positive colour opposites
relu(R−G), relu(R−B), relu(B−R), relu(B−G), a true-yellow map
relu((R−B)−(R−G)−(G−R)) with a small margin so red does not leak into it, a
per-image standardised grey map, and its negative (dark "ink") and positive
("bright") parts. Each colour map is standardised per image, so the very dark
photos (a fifth of the set) behave like normal ones.

**Templates.** Signs fill a known size window (22–27 px, roughly centred), so
each template is drawn once, at 3 diameters, as vector shapes (annuli, polygons,
thick strokes, a hand-written digit stroke font) on chosen feature channels.
Every template is zero-mean over its own footprint (orthogonal to any flat
background) and unit-L2 there. Three template banks convolve them over the
feature stack with a ±4 px position search and max-pool over position, scale
and (for pictograms and digits) ±14–20° rotation variants: sign families
(ring, filled disc, octagon, triangles, yellow diamond, blue disc/square, white
disc), the eight speed-limit numbers plus "80" on a white disc, and 76
pictograms/composite signs (warning pictograms, cars/trucks, blue arrows,
diagonal-stripe "end" signs, STOP letters, the no-entry bar). A fourth bank
computes three fixed-position averages (red inside the disc, ink inside the
disc, white bar) at 1×1.

**Fusion.** A hand-set rule table (dense layers summed): each class gets its
family evidence plus its pictogram evidence plus a small composite bonus;
non-members get negative family weights, and look-alike pairs (octagon vs disc,
white disc vs ring, blue disc vs square, bends, end-of-limit stripes) get extra
penalties; templates also carry "free" exclusions — a family that never shows a
colour gets a strong negative weight for it, which is exactly zero on the true
family and strongly negative on rivals.

Checked 3 of the 5 allowed runs on the training set (≈30% accuracy); the
remaining known weaknesses are the fine pictogram discrimination (digits,
keep-right/left arrows) and the dark "end of limit" signs.
