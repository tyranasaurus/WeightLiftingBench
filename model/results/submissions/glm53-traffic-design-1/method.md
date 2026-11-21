# Method

**Architecture** (11 layers): a 1x1 conv turns RGB into luminance / red-green /
blue-yellow maps, which are lightly smoothed and per-channel normalised (`norm`)
to remove illumination and colour cast. A 9x9 conv produces seven feature maps:
the three colour-opponent levels (shifted positive by a bias so they survive a
relu) plus thin/dark, thin/bright, thick/dark, thick/bright band-pass (DoG) maps
made one-sided by relu. A single "template bank" conv (441 channels, 28x28
kernels, stride 1, padding 3) then matches hand-drawn sign models against the
feature maps at 11x11 positions; a global max-pool collapses each channel to its
best match position, giving translation tolerance.

**How the weights were made**: every kernel is drawn programmatically from my
general knowledge of German sign geometry — red annulus rings, warning
triangles, octagons, diamonds, blue discs, digits built from stroke polylines
(a small hand-designed font), arrows, hatched end-of-limit discs, pictograms
(cars, truck, X-cross, exclamation mark, bends, bumps, traffic lights, people,
bicycle, snowflake, deer...). Each of the 43 classes has one content template;
each template exists at 3 sign radii and 3 rotations (9 channels), plus 8
"shape" templates (one per sign family) that model only the frame. Frames are
built as colour-contrast patterns (e.g. the ring requires red at the outer
annulus and *not* red at mid-radius, which separates rings from filled discs
and triangle borders). Content templates contain only the pictogram strokes
plus small penalties on unexpected content of either polarity (relu'd maps make
such penalties genuinely negative).

**Readout**: dense-1 turns each channel's score into a gate by subtracting the
per-image common mode (removing the noise floor) and a hand-set threshold
(shapes 30, content 8), then relu. Dense-2 is a hand-written rule table: each
class gets +3 from its own content gate (mean over its 9 variants), +1.5 from
its family's shape gate, -0.15 per same-family competitor, -0.6 per other
family's shape gate, plus small priors and a bonus for the two empty-interior
classes (15, 13) so they win when no content fires.

No training or statistics of the training images or labels were used for any
weight; the data was only viewed as text renderings to confirm the qualitative
appearance of each class. Calibration was done with a "self-simulation" in
which the bank's own kernels are used as pseudo-signs (pure weight-space
analysis, no data), plus four of the five allowed training-set runs.
