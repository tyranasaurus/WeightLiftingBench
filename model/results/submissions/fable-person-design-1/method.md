# Method

Every weight is written by `build.py` from typed constants and formulas (rendered masks of geometric regions).
Nothing was trained or fitted, and no number comes from a training image.

## Architecture (177 layers, about 860,000 weights, mostly zeros)

1. **Colour (1x1 convs + relu).** Skin is modelled as a cone in RGB: with u = R-B and w = G-B, hue = 60*w/u and
   saturation = u/R. Each limit (hue about 5..28 degrees, saturation 0.16..0.66, not too dark) is a half-space;
   membership = relu(1 - sum of scaled violations). A broader "earthy warm" membership (hue 3..53 degrees) marks
   wood, food, sand, fur. Also: inverse luminance, and luminance normalised over the photo (`norm`) plus two
   colour-opponent channels for "does this region differ from that one".
2. **Head detector, face radii 2 to 25 pixels (four pyramid levels via avgpool).** Evidence =
   mean skin in a face ellipse
   - warm colour in the surround to the left, right and above (a head stands clear of other warm things; this
     rejects edges, corners and bars of wood-coloured regions; padding counts as half warm)
   - skin colour continuing under the chin (animals, toys, posts; people wear clothes)
   + bonus for a darker cap of hair above the face
   + bonus when the region under the head differs in colour from the background beside the head (a body)
   - a threshold.
   Max over positions and scales (reshape + maxpool).
3. **Standing figure detector, torso widths 5 to 21 pixels.** Head / torso / legs boxes in a column with flank
   boxes on both sides: torso and legs must both differ from their flanks (min built from relus), the column must
   have a genuine second tone (legs or head colour not a mixture of torso and background, tested as "legs beyond
   torso in one channel and torso beyond legs in another", or skin-coloured legs/head over a non-skin torso), the
   background must be the same left and right, and the column must end above the head.
4. **Decision.** person <=> head evidence (raised threshold when the photo is mostly warm-coloured) > 0, or
   figure evidence > 0.

## How the weights were produced

By reasoning about how people look, then unit-testing on synthetic drawings of my own (ellipses, bars, slabs,
capsule figures, noise) to fix design flaws: border artefacts, bar ends passing as heads, off-grid faces,
mixtures mimicking a second tone. No optimisation of any kind was run.

## Checks on the training photos (2 completed, logged in checks.log)

- Check 1, head detector alone: 62.50% (50.9% of person photos caught, 26.6% of no-person photos flagged).
  Learned: the skin-head cue is real but conservative; about half of the person photos show no qualifying face.
- Check 2, head OR standing figure (this submission): 62.64%. Learned: the figure cue adds 259 person photos and
  231 false alarms, so it is only weakly informative.
- A third run (an added eye-band cue) was started but was killed before it finished, so it produced no result;
  that variant is not submitted. The submitted network is exactly the one measured in check 2.

Thresholds were set a priori and were not adjusted after the checks.
