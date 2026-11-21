# Method

This is a fixed analytic morphology network; no optimization, regression, fitting, class averages, or image-derived values were used. The first convolution implements hand-written luminance isophotes, colour residuals, a Difference-of-Gaussians texture response, and four oriented ridge filters. A dense analytic projection then measures Gaussian apertures, annuli, angular Fourier modes, and two-handed logarithmic-spiral templates around the known image centre.

ReLU hinge tests compare those shape responses with local flux to obtain approximate scale- and brightness-robust concepts: concentration/extension, elongation, lopsidedness, disturbance, box/bar structure, spiral structure, colour, texture, and ridge evidence. The last dense layer is a hand-written astronomy rule table mapping those concepts to the ten morphology classes. All weights are generated from formulas and general class definitions in `build.py`; the training data is neither read nor required by the builder.
