# Hand-designed galaxy morphology network

**Architecture (75 layers, 31.9M weights, all written by `build.py` from formulas).**
The network is a measurement pipeline followed by a hand-written rule table, built only from
conv / dense / relu / maxpool / avgpool / add / reshape:

1. **Luminance and colour.** 1x1 convolutions form luminance, red (z band) and blue (g band) above
   the sky level (0.14 in the 0..1 images, a property of the Legacy Survey rendering). Three 3x3
   binomial blurs smooth the luminance; a lightly smoothed copy is kept for the brightest levels.
2. **Soft isophote masks** at four surface-brightness levels above sky (0.06, 0.17, 0.38, 0.55),
   each a pair of ReLUs (`relu(k(L-t)) - relu(k(L-t)-1)`).
3. **Measurements as fixed dense layers**, each taken inside three concentric windows (radius 12,
   22, 36 px) so that field stars far from a small galaxy do not pollute it:
   - windowed area of each mask;
   - **elongation** of each mask: the amplitude of its cos(2 phi) angular moment, obtained as the
     max over 16 directions (reshape + maxpool). For an ellipse of axis ratio q this equals
     area x (1-q)/(1+q), so axis-ratio thresholds become linear inequalities `E - rho(q) A > 0`;
     also measured for the mid isophote inside an inner "bar" window;
   - red and blue flux inside the faint mask (ReLU gating, core excluded) for colour;
   - **structure**: positive difference-of-Gaussians between the faint and high isophotes, at
     knot scale (half resolution) and arm scale (quarter resolution);
   - central peak brightness, 180-degree rotational asymmetry of the mid mask (dense layer with
     mirror-pair weights, ReLU of both signs), and a secondary-nucleus detector (compact bright
     peak with extended light around it, max over an annulus).
4. **Soft comparisons**: every threshold is a ratio inequality written as a linear form on the
   measurement vector, turned into a 0..1 indicator by a ReLU pair. Thresholds come from Galaxy Zoo
   definitions and general knowledge (round q>0.76, in-between 0.6..0.76, cigar 0.48..0.6,
   edge-on q<0.48; red if z/g flux > 1.3, blue if < 1.0, etc.).
5. **Logic layers**: the window matching the galaxy's size is selected with soft ANDs; derived
   terms (bar = elongated bright core in a round symmetric galaxy, edge-on bulge = thin outer
   isophote with a rounder measurable core, merger = secondary nucleus AND asymmetry, ...) are
   soft AND / AND-NOT combinations. A final dense layer is the hand-written rule table of class
   scores.

**How the weights were produced.** Everything is typed from formulas (Gaussian kernels, cos(2 phi)
windows, mirror permutations, ellipse axis-ratio algebra) and hand-chosen thresholds/weights; no
training images, labels, statistics or fitted quantities enter the weights. Synthetic galaxies
rendered from analytic profiles (Sersic, exponential discs with log-spiral arms, bars, pairs) were
used to debug the pipeline. The five allowed checks on training images were used only to read the
confusion table and adjust rule weights by reasoning (e.g. gating the structure detector away from
saturated cores, using less-smoothed masks for small bright bars). Final check: 49.6% on 3000
training images (chance 10%).
