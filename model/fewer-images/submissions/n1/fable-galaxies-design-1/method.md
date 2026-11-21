# Method

**Status:** this is a simple first version saved when time ran out. It is far less elaborate than the design I was
working towards (centred polar isophote analysis, asymmetry, companion detection), and none of its thresholds
could be validated beyond one check on the 10 example images.

## Architecture
1. `pre` (conv, stride 2): luminance (mean of R, G, B) and red-minus-blue, on a 64 x 64 grid.
2. `lin`, `proj` (dense): Gaussian-windowed image moments about the image centre at four window sizes
   (sigma 4, 7, 12, 20 pixels): flux, trace of second moments, red-minus-blue flux, flux in a 2.5x smaller window,
   plus the quadrupole and the dipole each projected on 16 angles. Every functional is made blind to a constant
   sky level by subtracting the mean of the border frame.
3. `projm` (maxpool over the 16 angles): magnitude of the quadrupole (ellipticity) and of the dipole (centroid offset).
4. `rn` (norm on groups `[a, -a, s, -s]`): turns pairs into ratios a/s, which makes the features independent of
   brightness: size match of the window, ellipticity, centroid offset, colour, concentration.
5. `hn`/`hc`/`hrelu`/`sc`: a hand-written rule table. Each class has a typical range for each feature; leaving the
   range costs a linear penalty (two ReLU hinges per class and feature). A window whose size does not match the
   galaxy gets a large penalty, and a maxpool over the four windows keeps the best-matched one.

## Where the numbers come from
All weights are formulas (Gaussian windows, moments, cosines) and constants typed by hand from general knowledge:
axis ratios of round / in-between / cigar / edge-on galaxies converted to moment ellipticity (1-q^2)/(1+q^2),
red ellipticals versus blue spiral disks in the g, r, z colour rendering, concentrated bulges versus diffuse
disks, off-centre light for mergers and disturbed galaxies, and small hand-chosen class biases (the vague or rare
classes need more evidence). No training, fitting or statistic of the training images was used; build.py reads
no files.
