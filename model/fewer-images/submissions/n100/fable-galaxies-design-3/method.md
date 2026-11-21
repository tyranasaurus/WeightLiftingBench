# Method

**Architecture.** A small hand-wired measurement network (17 layers, 3.7M weights, most of them zero):

1. `p`: 4x4 box average, stride 2, to suppress sky noise (63x63 map).
2. `c1` + relu: a 1x1 conv that subtracts the sky and makes three soft isophote masks (low, mid and high
   brightness levels) plus sky-subtracted red and blue channels.
3. `fp`, `fs`: dense layers computing Gaussian-windowed moments of each mask about the image centre: area,
   trace of second moments, the second-moment ellipticity vector and the centroid offset, each projected on
   16 directions. A maxpool over the directions gives the vector magnitudes.
4. `gn`: a norm layer over groups `[a, -a, b, -b]`, which returns `sqrt(2)(a/b)/sqrt(1+(a/b)^2)`. This gives
   scale-free ratios: outline ellipticity, inner ellipticities, centroid offsets, colour, bright-core fraction.
5. `h` + relu: 14 ramps (two relus each) on those ratios with hand-chosen thresholds.
6. `out`: a hand-written rule table; each class score is a bias plus votes from the ramps.

**Where the numbers come from.** Nothing is trained or fitted. The sky level (0.13) comes from the Legacy
Survey asinh colour rendering formula. Shape thresholds come from the ellipse formula
`(1-q^2)/(1+q^2)` at hand-chosen axis ratios (round > ~0.82, cigar/edge-on < ~0.47, thin < ~0.33). Colour
thresholds come from typical red-sequence and blue-cloud g-r-z colours pushed through the same rendering.
The size ramp reflects that arms and bars are only visible in galaxies several seeing widths across. Rule
weights and biases were typed by hand; rare classes (cigar, disturbed) need stronger evidence.

**Checks.** The network was run on three generated ellipses to confirm it executes, and once on the labelled
training sample (see `checks.log`): accuracy 0.221 on the 1,000 balanced images. No number was changed
after that check. What it showed: the "large" ramp fires for most smooth galaxies (they land in the spiral
classes), the outline ellipticity reads too round for edge-on discs (classes 8 and 9 are rarely predicted),
and the merger rule over-fires on elongated bright cores.

**Limits.** This is a reduced design saved when time ran out. It has no spiral-arm, bar-twist or
second-nucleus detector, so spirals are separated only by size, colour and inner elongation, and mergers
and disturbed galaxies only by off-centre or elongated bright cores.
