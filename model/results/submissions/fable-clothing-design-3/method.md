# Method

**Architecture.** A hand-built "nearest drawn prototype" network.
1. Front end (1x1 convs + relu): soft foreground mask `clip((I-0.03)/0.035, 0, 1)` and a "bright" mask.
2. Morphology with max-pooling: closed mask (3x3 closing, fills print holes), eroded closed mask (thin parts vanish),
   and a thin-vertical-gap map (what closing filled, kept where it runs >= 5 px vertically: trouser gaps, sleeve slits).
   Each map is Gaussian-blurred and subsampled by 2 (conv 5x5, stride 2).
3. Four dense layers (one per map, 80,000 units each) whose rows are the same features of 80,000 procedurally drawn
   silhouettes (8,000 per class); with bias `-|p|^2/2` their sum is minus half the squared distance to each prototype.
4. Two small cue branches add class-table evidence: a centre-front-line detector on the `norm`-alised image (zip / button
   placket; min-of-two-sides logic, row segments, vetoed where the line is background) and a "mostly dim item" cue.
5. Read-out: reshape to (10, 1, 8000) and max-pool, so each class scores its best-matching prototype.

**Where the weights come from.** `build.py` contains shape models typed from general knowledge of garments, shoes and bags
(torso + sleeves + collar polygons, trouser legs, shoe profiles from sole/instep/shaft curves, bags with handle loops,
sandals from soles, straps and heels), rendered with fixed seeds exactly as the dataset was prepared (trim, longest side
to 28, centre). Thresholds, sub-type priors and cue tables are hand-set constants. No gradient descent, fitting, or
statistic of the training images or labels is used; training images were only viewed as PNG sheets to understand the classes.

**Checks used (3 of 5, all on the 60,000 training images).**
- Check 1, 64.5%: silhouette matching works for trousers/bags/shoes; tops were mis-modelled (neck dip, splayed sleeves).
- Check 2, 70.7%: new shape families helped; print holes amplified by raw erosion and a density-biased vote hurt dresses/T-shirts.
- Check 3, 73.4%: hole-robust channels, hang-configuration priors and best-match read-out. **This is the submitted network.**

A further revision (pattern cue for shirts, changed tables) was drafted but never run on training images, so it is not submitted.
Main remaining errors: coat/pullover/shirt are mutually confused, and solid-looking sandals go to sneaker/boot.
