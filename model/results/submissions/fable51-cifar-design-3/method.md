# Method

## Architecture

Two fixed front ends read the image in parallel:

- **Colour branch**: a 1x1 conv producing 12 channels: luminance, darkness, and the two signs of
  four opponent-colour differences (blue minus yellow, green minus magenta, R-G, G-B, B-R).
  After relu it is average-pooled with 8x8 windows into a 4x4 grid.
- **Edge branch**: a 9x9 conv on luminance (no padding, so no border artefacts) with 22
  filters from formulas: Gaussian-derivative edge detectors at 4 orientations (both signs),
  elongated versions of the same (long straight edges), fine and coarse Laplacian-of-Gaussian
  (texture; dark and bright blobs), and thin dark vertical / horizontal bar detectors.
  After relu it is average-pooled into a 4x4 grid and max-pooled over four horizontal bands.

A **concept layer** of 77 relu units reads the pooled grids through three dense layers that are
added together. Each unit is a named detector I wrote by hand: signed templates (how blue, green,
brown, bright, textured the whole image, the top row, the bottom row or the centre is;
axis-aligned vs diagonal edges; horizontal vs vertical long edges) and thresholded concepts
(sky at the top, sky at the bottom, water at the bottom, grass, grey road, brown / red / yellow /
grey / dark / textured centre, vivid paint, wheel = dark blob in the bottom band but not the
top band, legs = thin vertical bars in the lower half but not the upper, horizon = a long
horizontal edge in a middle band, bright blob, boxy structure, plain background). Saturating
"ramp" units (relu(f-lo) - relu(f-hi)) stop one strong cue from dominating.

The **output** is a dense layer whose 10 x 77 weights are a hand-written table of how much each
concept speaks for or against each class (for example water at the bottom: ship +1.5, others
negative; legs: horse and deer +1, car -0.5), with each class row scaled to the same total
absolute weight and a hand-set offset per class.

## How the weights were produced

Every number in `build.py` is either a formula (filters normalised so an ideal unit step, dark
disc or 1-pixel line gives a response of exactly 1) or a constant chosen from general knowledge
of what sky, water, grass, road, fur, paint, wheels and legs look like. No training images,
statistics of them, or fitted values are used; `build.py` reads no files.

I used the five allowed checks on a fixed random subset of 5000 training images to catch design
errors, and after each I changed the design by hand:

1. (23.0%) Ship, truck and frog absorbed most predictions: added "sky at the bottom" and
   bottom-brightness cues to separate plane from ship, lowered frog's green weights.
2. (16.6%) Car took 81%: my rescaled straightness and band-maximum features were not centred
   (long filters respond almost as strongly as short ones on blurry 32x32 photos), so I replaced
   them with symmetric orientation contrasts and band-against-band contrasts.
3. (23.1%) Truck never won and ship stole from vehicles: equalised each class's weight budget,
   strengthened the road detector, added a top-row edge cue for tall trucks.
4. (20.6%) Cat and dog then took 59%: texture cues are on for most photos, so I set class offsets.
5. (21.7%) The offsets were about twice too large; the final offsets interpolate between the
   two observed states so that each class is predicted about equally often.

The final network was not re-checked after the last offset interpolation (no checks left); I
expect roughly 22-25% accuracy on held-out images.
