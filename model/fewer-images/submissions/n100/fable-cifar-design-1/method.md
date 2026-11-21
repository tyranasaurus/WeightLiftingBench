# Method

**Architecture.** A template-matching network, nothing trained.

1. *Feature layers, written from formulas.*
   - Colour: 1x1 convs + relu turn each pixel into 42 soft 0..1 steps ("is it at least this blue / green / red / yellow ...",
     12 hue directions x 3 strengths, plus 6 brightness levels), averaged over a 4x4 grid of cells.
   - Shape: the grey image is contrast-normalised (`norm`), filtered with odd and even Gabor filters at 6 orientations and
     2 scales, rectified to edge + line energy, clipped, averaged over cells, and standardised within the photo (`norm`).
   - Layout: how far each pixel's colour is from the photo's own average (figure against ground), and where the photo is
     light or dark, each on a 7x7 grid.
2. *Template bank.* Five dense layers hold one row per template (60,000 templates, 6,000 per class). A row computes minus
   the squared distance between the photo's features and that template's features, under a hand-written multi-level metric
   (single cells, quarters, horizontal bands, whole picture, middle-versus-frame).
3. *Read-out.* For each photo the network standardises the distances over all templates (`norm`), adds the colour, shape
   and layout evidence (colour counted twice), standardises again, keeps the best match in each group of four templates
   (`maxpool`), passes it through a convex piecewise-linear vote curve (relu knots approximating exp(5(z-3))), and sums
   the votes per class.

**Where the weights come from.** `build.py` contains a small drawing program typed from general knowledge of the ten
classes: skies, grass, roads, water, rooms and leaf litter as textured backgrounds; cars, trucks, planes and ships as
simple solids (boxes, pushed-out outlines, wheels, tubes, flat wings) seen from random angles; horses, deer, dogs, cats,
birds and frogs as ellipses, thick lines and triangles in typical poses and coat colours. Each template is one such
drawing pushed through the feature layers and stored as its own row, normalised by itself alone (its bias is its own
squared norm). No statistic across drawings or across training images sets any number; filters, thresholds, metric
levels, group weights and the vote curve are constants chosen by hand. The seed is fixed, so the build is reproducible.

**Use of the training images.** I looked at contact sheets of the 1,000 photos to check my picture of the classes, and
ran four of the five allowed checks (30.4%, 32.7%, 34.9%, 36.8% accuracy). After each one I revised the drawing program
or the metric from what the confusion table said about my drawings (too clean and vivid, objects too small, planes too
thin and too rarely parked, truck loads too long, too few templates), not by fitting anything to labels. The submitted
network is the one from check 4; the fifth check was not used.

**Limits.** The accuracy on the 1,000 training photos is 36.8%, and those same photos guided four rounds of revisions,
so expect a little less on the held-out set. Airplanes are the weakest class (25%), and most remaining errors are between
classes that share a setting (cat/dog, car/truck, deer/bird/frog). A network of this architecture trained by gradient
descent on 50,000 photos would very likely score higher than these hand-made weights.
