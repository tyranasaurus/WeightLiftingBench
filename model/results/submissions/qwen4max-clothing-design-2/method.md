# Method

**Task.** 10-class classification of 28x28 greyscale clothing photos (Fashion-MNIST layout:
item light on black, trimmed, longest side 28, centred). Weights are entirely hand-designed;
no training, fitting, or statistics of the training images/labels are used anywhere.

**Architecture** (59 layers, ~84k weights, all from formulas + hand-chosen constants):

1. *Foreground mask.* `M = relu(image - 0.18)` (1x1 conv + relu). Background is ~0, item bright,
   so M is a soft silhouette with dark details (neckline, leg gap, handle hole, placket) removed.
2. *Structural features from M* (a 28-dim vector F):
   - regional means / ink fractions over hand-placed rectangles on the canonical centred canvas
     (total, top/bottom, centre-bottom vs side-bottom = leg gap, side-mid = sleeve length,
     shoulders, body, top-centre, topmost band = boot shaft / bag handle zone, ...);
   - occupied width/height fractions and per-band occupied widths, via avgpool -> saturating
     threshold `min(1, relu(K(x-t)))` (conv/relu/conv/relu/conv) -> avgpool. The saturation is what
     turns a row/column mean into a 0/1 occupancy indicator before averaging.
3. *Part detectors on the raw image* (formula kernels): enclosed-hole = center-surround
   (negative disk, positive annulus); front-opening / leg-gap = vertical dark-line (negative centre
   column, positive flanks); texture = positive Laplacian. Hole and opening are aggregated by
   region **max** (presence of a thin loop/line must not be diluted by a region mean); texture by
   region mean.
4. *Cascade readout.* Stage 1: twelve saturating gates S in [0,1] (shoe = wide&short,
   tall-garment, bag-handle, leg-gap, flare, long-sleeve, front-opening, texture, sparse, holey,
   shaft, solid). Stage 2 (last layer, linear): each class score is a signed sum of gates minus a
   threshold, so a class can only win inside its own group (wrong-group evidence is negative):
   shoes split sandal (sparse+holey) / sneaker (solid, no shaft) / boot (shaft); tall garments split
   trouser (leg-gap+narrow) / dress (flare, no handle) / T-shirt (short sleeve, no opening) /
   pullover (long sleeve, no opening) / coat (opening, smooth) / shirt (opening + pattern texture);
   bag = handle loop and not shoe/tall.

**Weight provenance.** Every kernel is a closed-form formula (center-surround ring, vertical
dark-line, Laplacian, box/region indicators, identity/placement matrices). Every threshold and
readout weight is a hand-chosen constant reasoned from general knowledge of what each class looks
like (sleeve length, front opening, leg gap, flare, handle loop, sole/shaft, sparsity). `build.py`
reads no files and regenerates `network.json` + `weights.npz` deterministically.

**Validation.** Five allowed checks were run on a 3000-image stratified subset (see checks.log):
0.105 (occupancy-scale bug), 0.305 / 0.305 (fixed extents; prototype readout), 0.123 (additive rule
table leaked across groups), 0.168 (cascade, but detector region-*mean* diluted thin handle/placket
so those gates never fired). After the last check I corrected, by code inspection and reasoning only
(no further image runs): detector aggregation to region-max, the shaft cue to threshold top-zone
mass alone, and the tall-garment/dress/handle balance. The submitted network is this corrected
cascade; it is validated for shape and data-free rebuild (`check_network.py`, `check_rebuild.py`
both pass) but its accuracy was not re-measured because the 5-check budget was exhausted.
