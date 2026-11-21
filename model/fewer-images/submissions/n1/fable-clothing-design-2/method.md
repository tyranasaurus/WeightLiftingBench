# Method

## Architecture

Two branches from the image, summed at the end.

**Silhouette matching (main branch).**
`image → 1x1 conv + relu + 1x1 conv + relu + 1x1 conv` turns grey levels into a soft silhouette mask
(0 below grey 0.05, 1 above 0.15), so the item's colour does not matter.
`→ 7x7 Gaussian conv (sigma 1)` blurs the mask so small outline differences cost little.
`→ norm` standardises it, `→ dense (784 → 60,000)` holds 6,000 standardised silhouette templates per class, so each
output is the Pearson correlation between the image's mask and one template.
`→ reshape (1, 10, 6000) → maxpool over each class's 6,000 → reshape (10)`: the class score is its best-matching template.

**Rule branch.**
`image → 7x7 conv (image minus its Gaussian-smoothed copy, both signs) → relu → dense` sums the absolute local
deviation; a second `dense` on the image takes its total brightness; `add` compares the two (patchiness ratio against
two hand-set limits), `dense → relu → dense` clips the result and writes it into the sandal score. The bias of that
last layer also holds a small constant handicap for the shirt class. `add` joins this to the template scores.

47.1 million weights in total, nearly all of them the template bank.

## Where the weights come from

No training, fitting or statistics of any images. `build.py` reads no files.

- **Templates.** `build.py` contains a small polygon rasteriser and one parametric shape model per class, typed from
  general knowledge of the items: T-shirt = torso + short angled sleeves (plus strap tops); trousers = waist, hips, two
  legs with a gap (or one folded leg); pullover = wide torso + long sleeves angled out, round neck, sometimes a hood or
  roll neck; dress = narrow bodice, waist, flaring skirt, with straps / shoulders / sleeves; coat = long or jacket
  length, sleeves hanging straight down, collar, hood or open collar notch; shirt = slim sleeves, collar, shirt-tail
  hem; sandal = thin sole with straps, high heel, wedge, flip-flop, sport sandal or slide; sneaker = low wedge profile
  with the toe on the left; ankle boot = foot plus shaft on the right, optional heel notch; bag = body with handle
  loop, clutch, shoulder strap or bucket shape. Proportions are sampled from hand-typed ranges with a fixed seed. Each
  drawing is placed the way the photos were prepared (trimmed, longest side about 28 pixels, centred), passed through
  the same mask + blur + standardise steps as the network, and stored as its own template, normalised by itself alone.
- **Front end.** The mask ramp (0.05 to 0.15) and the blur (sigma 1) are chosen by hand.
- **Open-shoe rule.** A sandal is straps and gaps, so its picture is patchy; sneakers and boots are closed and smooth.
  Patchiness = sum |image − smoothed image| / sum image, which does not depend on brightness. I worked the ratio out
  by hand for ideal shapes (closed shoe about 0.3 to 0.4, strap-and-gap structures 0.6 and more) and set the limits at
  0.52 and 0.62. Below the first limit the sandal score loses 0.04; above it gains 0.04; above the second a further 0.04.
- **Shirt handicap.** A shirt almost always has a more common look-alike silhouette (T-shirt, pullover or coat), so
  the shirt score is lowered by 0.015 and wins only on a clearly better match.

## Checks and what they showed

The network was run on the 10 provided images five times (all five allowed checks): 6/10, 6/10, 8/10, 8/10, 8/10.
They were used to confirm each version ran correctly on real images. Between checks I revised the drawings and rules
from reasoning about the garments (sleeve posture, collar notch, shirt prior, how openness shows at this resolution),
not by fitting numbers to those images; an earlier "enclosed holes" version of the shoe rule was replaced by the
patchiness ratio after a test on hand-made pictures showed it counted the wrong thing. The two images still wrong in
the last check are a short jacket read as a pullover and a chunky sport sandal read as a sneaker.

Ten images say little about accuracy on the 10,000 held-out ones, and none of the hand-chosen constants has been
validated on data. I expect the upper-body group (T-shirt, pullover, coat, shirt) to be the weakest, since their
silhouettes overlap heavily, and I do not expect this network to match the same architecture trained on 60,000 images.

Mechanical tests on my own drawings and hand-made test pictures (not used to choose any number) confirmed that a
drawing correlates 1.0 with its own template and that the rule branch computes the intended ratio.
`check_rebuild.py` reports that `build.py` reproduces `network.json` and `weights.npz` exactly.
