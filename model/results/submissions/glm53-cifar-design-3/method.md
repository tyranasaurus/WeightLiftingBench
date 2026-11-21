# Method

**Architecture.** A hand-designed feature network (no training of any kind).
Three parallel convolutional branches extract formula-based filters from the
image: (1) colour-opponent maps turned into thresholded "colour words"
(red / green / blue / yellow / warm hue / grey / dark / bright / a
mild-warm-vs-strong-warm *band* that captures tan-brown fur but not vivid
red / deep blue = bluish-and-dark, i.e. sea water rather than sky); (2)
quadrature Gabor pairs at four orientations for small-scale texture energy,
and elongated Gabors plus a horizon filter (second derivative of a Gaussian,
wide along x) and DoG dark/bright-centre filters at a large scale; (3) a
wheel/blob bank (small and medium DoG dark-centre detectors at 9x9, a
darkness map, a small-bright-centre detector). Edge and blob maps are
thresholded and then spatially normalised (`norm`), so they measure where
structure is unusually strong *for that picture*, not how busy it is.
Everything is pooled onto coarse grids (4x4 cells of 8 px; the blob bank on
8x8 cells of 4 px), and centre-surround convolutions on the pooled grids
produce "island" features (what stands out in the middle vs the surround)
for colour and for edge structure (interior edges vs boundary ring).

**Readout.** Seven dense layers (one per pooled feature family) are summed
into the class scores. Their weights are written by hand as rule tables -
`(class, channel, spatial mask, weight)` entries encoding general knowledge:
fur is tan and fills the middle of animal pictures; deer and horses stand in
green fields, cats, dogs and frogs in warm indoor or dry scenes, frogs are
the most tan and least grey; vehicles sit on grey roads with a dark
shadow-and-wheel strip below and sky above, cars are low and wide, trucks
tall boxes with bigger wheels; ships have deep-blue water below, a horizon
band and a grey hull; planes have blue all around, a pale centre and the
least dark bottom; birds are small dark or tan things high in the picture
against sky or canopy. Families share a common core and differ by small
deltas, so evidence is not split inside a family.

**Normalisation of the readout** (this is what makes the hand weights
usable): every feature channel is rescaled by its typical magnitude
(measured on formula-drawn calibration scenes generated inside build.py -
no training data is read anywhere); each class's per-branch weight row is
made purely *redistributive* (zero sum across the ten classes for every
feature, so structure present in every picture lifts nobody and absence of
a feature can never score for a class that dislikes it), and each branch is
given exactly a hand-chosen share of the template's length (colour evidence
gets the largest share - it is the most reliable).  All ten class templates
end up with identical length, so scores are balanced matches between the
image's feature vector and each class template, and no class can win merely
by claiming broadly-firing features - the failure mode that plain
hand-balanced linear weights fall into.

**How the weights were produced.** Entirely from formulas (Gaussian
derivatives, Gabors, DoGs, colour-opponent arithmetic) and hand-written
rule tables chosen from general knowledge of the classes. The five allowed
training-image runs were used only to check that the network works; the
confusion they revealed (which classes absorbed which) guided qualitative
design decisions - e.g. that green surroundings do not identify animals
(planes are parked on grass), that a tan centre is the real animal
signature, and that "generic" claims must be structurally impossible - but
no statistic of the training data was ever turned into a weight, and no
pixel value was copied. build.py reads no files and reproduces the answer
exactly.
