# Method

**Architecture.** A fixed, formula-built front end followed by template matching.

1. `c1`: eight 7x7 Gaussian-derivative filters (one per 45-degree direction), relu.
2. `c2`: 1x1 conv + relu that keeps the part of the gradient along a direction that exceeds its part across it
   (the classical 8-direction "parallelogram" split of the gradient).
3. `c3`: Gaussian pooling (sigma 1.8 px) at every pixel; the whole map is normalised once with `norm`.
   A second, coarser map (sigma 3.6 px, one sample every 4 px) is normalised the same way.
4. Four "gather" convs with one-hot kernels read each quadrant's 7x7 taps (2 px apart) so that their 3x3 output
   positions are the quadrant shifted by -1/0/+1 pixel in x and y.
5. Per class, 1x1 convs hold the templates (weights `2t`, bias `-|t|^2`, i.e. minus squared distance up to the
   image's own energy); `maxpool` 3x3 picks each quadrant's best shift.
6. Class score = **best whole template** (its four quadrant matches + a coarse whole-template match at half
   weight, max over the class's templates) + **image-to-class match** (each quadrant matched by the best of any
   template of that class, summed over quadrants). One-hot dense layers and an `add` assemble the 10 scores.

24,000 templates, 56.3M weights.

**Where the weights come from.** No training or fitting of any kind. The filters are formulas. The templates are
drawings made by `build.py` itself: each digit is typed as pen paths (control points for splines) in several
writing styles I know from general handwriting (e.g. looped and Z-like 2, open and closed 4, crossed 7, hooked or
straight 9), with random slant, proportions, bending and pen width from a fixed seed, then framed the way the task
describes (20-px box, centre of mass centred). Each drawing goes through the same front end and becomes one
template, normalised by itself alone; no statistic across drawings or across training images sets any number.

**Checks on the training images (all 5 used, 1,000 images each).**

| check | accuracy | what it was | what I learned |
|---|---|---|---|
| 1 | 93.0% | whole-template matching only | works; 2, 8, 9 styles too narrow |
| 2 | 94.7% | wider styles, coarse term, per-class template counts | helped 2/3/8/9; 6 got worse |
| 3 | 95.0% | Z-like 2, asymmetric 8, "a"-like 9, wider 6 | simple shapes still out-match complex ones (2 read as 7) |
| 4 | **96.4%** | added the image-to-class quadrant term | fixed most of that; 8 and 5 remain weakest |
| 5 | 95.6% | further 8/5/6 style and pen-width changes | not a net gain, so reverted |

The submitted network is exactly the check-4 build (weights verified identical by hash after the revert).
Generated images were run through the network only to verify mechanics (shapes, template self-match, timing).

**Cost to run.** About 7 s per 500-image batch and about 2.5 GB peak memory with the provided runner.
