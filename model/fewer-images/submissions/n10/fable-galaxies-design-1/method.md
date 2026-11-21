# Method

**Architecture.** 4x4 average pooling (128 -> 32), a 1x1 conv + relu giving three maps (luminance above the sky,
blue-minus-red excess, red-minus-blue excess), then two dense branches that are added to give the 10 scores:
a "scalar" branch (windowed flux, core flux, blue flux, red flux) and a "harmonic" branch of 48 centred angular
templates (m = 2 elongation, m = 4 thinness, m = 1 lopsidedness; 16 phases each) whose modulus is taken by a
reshape + maxpool over the phases, which makes those features rotation invariant.

**Weights.** Nothing was trained or fitted. The sky level comes from the Legacy viewer's asinh stretch
(asinh(0.6)/sqrt(20) = 0.127), the templates are Gaussian windows times cos(m*theta) written as formulas, and the
readout is a nearest-prototype linear discriminant over six flux-normalised features, with the class prototypes,
feature tolerances and class frequencies typed by hand from general knowledge of the classes (e.g. m = 2 ratio
(1-q)/(1+q) for axis ratio q; red smooth galaxies, blue loose spirals, thin blue edge-on discs). All class scores
are proportional to the galaxy's flux, so the predicted class does not depend on brightness.

**Status.** This is a much simpler network than the one I had planned (isophote masks, central-component growth,
moment ratios through norm layers); time ran out before that was built. Prototype values are untuned first guesses.
