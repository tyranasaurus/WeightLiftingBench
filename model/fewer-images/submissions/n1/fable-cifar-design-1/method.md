# Method

Nothing in this network is trained or fitted. `build.py` reads no files; every weight comes from formulas
and from pictures that the code itself draws from general knowledge of the ten classes.

## Architecture

**1. Fixed features (formulas only).**

- *Hue layout.* 1x1 convs and relus turn each pixel into 12 hue bins on the chroma plane
  (cx = R - (G+B)/2, cy = sqrt(3)/2 (G - B)); each bin answers to hues within 45 degrees, is clipped so that
  a chroma of 0.2 already counts as fully tinted, and has a second "vivid" channel for chroma above 0.3.
  These 24 maps are average-pooled on a 4x4 grid.
- *Lightness layout.* Five overlapping lightness tents (centres 0, .25, .5, .75, 1), pooled on the 4x4 grid.
- *Edge layout.* Luminance is standardised over the picture (`norm`), filtered with Gaussian-derivative
  filters in 4 directions at two scales, turned into a soft orientation histogram (strength along a
  direction minus strength across it), clipped, pooled on a 4x4 grid, and standardised over its own
  64 numbers (`norm` on a reshaped tensor) so that only the layout of edges counts, not the amount of texture.

**2. Template bank.** `build.py` draws 3,000 pictures per class (30,000 in all): backgrounds such as sky,
water, grass, road, forest, rooms and studio backdrops, plus an object assembled from ellipses, polygons and
lines (airliner / fighter / light plane from the side, top and front; cars and trucks from the side,
three-quarter and front; cargo ships, ferries, sailboats, motorboats; perched, flying, wading and swimming
birds; cat and dog faces and bodies; deer with antlers; frogs; horses with and without riders). Colours,
poses, sizes, lighting, focus and colourfulness are randomised with a fixed seed. Each drawing becomes
exactly one template: one row of three dense layers whose weights are that drawing's own features (on the
4x4 grid plus its own 2x2 and overall means) and whose bias is minus half its own squared norm, so the dense
output is "minus half the squared distance to this drawing" up to a term shared by all templates. No
average, variance or any other statistic is taken across drawings, and no number was chosen by scoring
the network on drawings.

**3. Vote.** For each of the three feature groups the 30,000 closeness scores are standardised across
templates (`norm`), added with hand-chosen say 5 : 2 : 2 (hue : lightness : edges, done by repeating inputs
of an `add`), and standardised again. A global `maxpool` finds the best match; templates within 0.5 (weight 2)
and 1.5 (weight 1) standard deviations of the best one vote, through a relu, for their class (dense layers
with 0/1 class-membership weights). 18.6 million weights in total.

## How the numbers were chosen

Filter shapes, bin widths, clip levels, the mix 5 : 2 : 2 and the vote reaches were set by hand from
reasoning about colour and edges in photos; the drawings' colours, shapes and how often each scene type
occurs were typed from general knowledge of the classes. The ten training photos were only looked at, and
used for the five permitted checks.

## The five checks (10 photos each time, so each result is only a rough sanity check)

1. 3/10. The pipeline works on real photos; a smooth, out-of-focus photo was pulled towards sky scenes,
   showing that the amount of texture in my drawings is not trustworthy. I made the edge features
   self-normalised (layout only) and gave hue layout more say.
2. 4/10. Still works; greyish photos went to the vehicle classes. I let every class's drawings vary in
   colourfulness down to grey, as web photos do.
3. 4/10. Still works; the greyish photos were unchanged. I broadened the hue and lightness bins so nearby
   colours count as similar, coupled water colour to sky colour and added head-on airplanes.
4. 2/10. Still works; two photos flipped to their look-alike classes (cat to dog, horse to deer), which on
   ten photos is within noise. Looking at the bird photo I added birds standing on a ledge with a separate
   background.
5. 2/10, same predictions as check 4: the final pipeline runs on real photos.

After the last check I set the hue bin reach to 45 degrees (between the 30 used in checks 1-3 and the 60
used in checks 4-5) from an argument about hue tolerance, not from a score: the same material varies by
about 15-20 degrees between photos, different materials are 35 degrees or more apart. That final setting
was run on generated pictures only, to confirm it runs; it has not been checked on the training photos.

## Honest expectations

The ten-photo checks cannot measure accuracy. The network mostly recognises the kind of scene (sky, water,
grass, road, room) and the rough colour and size of the object; it is weak at telling look-alike classes
apart (cat/dog, horse/deer, automobile/truck) and tends to send greyish photos to the vehicle classes.
