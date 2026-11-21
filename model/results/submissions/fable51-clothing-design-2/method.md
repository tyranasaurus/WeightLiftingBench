# Method

The network is a hand-written rule system expressed in the allowed ops; no weight was fitted to or copied from any image.

**Architecture.** A 1x1 conv turns the image into the raw channel plus a soft binary mask (clip(8x-0.4, 0, 1)). A 13x13 conv
applies typed-in filters: thin-structure detectors, bright-pixel counts left/right/above/below, derivative filters, 9-row-averaged
dark/bright vertical-line detectors, a 3-wide mask sum (dilated mask) and a 3x3 dark-block detector. A 1x1 conv combines these
into "enclosed dark" detectors (leg gap, big openings) and the dilated mask, and another 13x13 conv adds a vertically consistent
gap map. A dense "profile" layer (plus relu) measures row widths, column heights, band widths (top, shoulders, upper arm, waist,
hem), left/right column-height asymmetry, neck notch, masses, thin/hole/gap/line sums; a second dense layer adds per-row
"width close to item width" terms. A dense layer then forms ~40 measurements minus hand-chosen thresholds (height 23 for the
shoe/garment split, fill ratio 0.48 for sandals, hem-minus-waist 3.5 for a flaring dress, etc.), each paired with a 0 and passed
through `norm`, whose output on a pair is a soft sign: a comparator bit in [-1, 1] with a chosen softness. A final dense layer
encodes AND-rules per class (e.g. trouser = tall & narrow & continuous dark gap reaching the hem & no neck notch; boot = wide
group & tall shaft side & solid; bag = many rows as wide as the item & symmetric; pullover = wide rectangular top with sleeves
to the hem and no centre line), each rule again through `norm`; a weight-free avgpool averages each class's rule votes and a
stride-2 maxpool keeps one score per class. Every path from the image to the scores passes through two comparator stages.

**How the weights were produced.** All constants come from general knowledge of the garment shapes in a size-normalised, centred
product photo (longest side 28): e.g. garments fill 28 rows while shoes are wider than tall, trousers are narrow with a leg gap,
a boot's shaft makes one side much taller, a sandal is sparse with enclosed strap openings, a bag has a rectangular body and a
thin handle loop. The design was debugged on synthetic silhouettes drawn from these descriptions and checked five times on
training images (final check: 59.8% on 10,000 images not used in earlier checks).
