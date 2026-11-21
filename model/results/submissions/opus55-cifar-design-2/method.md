# Method

**Architecture.** The network has these stages:
1. **First conv layer.** A 7x7 valid conv on the image, followed by ReLU, gives 46 channels on a 26x26 grid. The colour channels use only the centre tap: ramps on b−r, b−g, g−b, g−r, r−b, r−g and brightness, plus rectified pairwise differences for saturation. The texture channels are Sobel gradients and 7-pixel straight-line detectors.
2. **Two 1x1 conv + ReLU layers.** These turn the ramps into 13 soft per-pixel conditions, each between 0 and 1: blue (sky or water), green (vegetation), warm (fur, earth, wood), white, gray, dark, vivid red, vivid yellow, edge, smooth, horizontal line, vertical line, and bright-and-smooth. ANDs are built as relu(a + b − 1) and clipped ramps as differences of two ReLUs.
3. **Prototype scoring.** A dense layer scores 33 hand-written scene prototypes, 3 or 4 per class, for example "plane in blue sky", "ship: sky above, water below", "car on road", "horse on grass", "cat indoors" and "frog in leaf litter".
   - The image is split into 8 regions: 4 horizontal bands, each divided into centre and sides.
   - Each prototype names a mix of materials for each region, such as sky, haze, water, grass, foliage, road, indoor, fur, metal, car paint or frog skin.
   - Each material is a hand-typed row giving how often its pixels meet each condition.
   - A prototype's score is the Bernoulli naive-Bayes log-likelihood of the region's condition fractions under those rates. This is linear in the per-pixel conditions, so it is one dense layer.
4. **Output.** The scores are reshaped, max-pooled over each class's prototypes, and reshaped to 10 class scores.

**How the weights were made.** I typed every number in `build.py` by hand from general knowledge of what these objects and scenes look like: thresholds, material rates, prototype layouts and priors. Filters come from formulas (Sobel, box-averaged differences). Nothing was fitted, and no statistic of the training images or labels was computed.
- I looked at grids of training images only to understand the classes.
- I unit-tested the condition layers only on synthetic patterns I drew in code: solid colour patches, step edges, noise and a drawn rectangle on a graded sky.

**Checks (5 of 5 used, all on the 50,000 training images).**
1. **25.7%.** Truck and car absorbed most images. My guessed line and edge rates were far too low for real photos, so the broad vehicle materials won.
2. **28.3%.** I made the material rates less extreme, blended them 30% toward a generic photo, gave texture and line conditions less weight than colour, and added the bright-and-smooth condition. Frog then became the over-predicted class, because leaf litter looks like an average natural photo.
3. **27.6%.** I added a figure-ground cue (each pixel compared with a colour interpolated from the image border) and more background materials. This was slightly worse, with automobile now over-predicted.
4. **26.6%.** I split objects into single-colour variants. This was worse, with truck and frog over-predicted.
5. **27.25%.** I went back to the check-2 version and gave texture half as much weight. This was worse: ships and vehicles slid toward airplane, so texture does help there.

**Final answer.** The submitted network is exactly the version from check 2, at 28.3% training accuracy. Since nothing was fitted to the data, accuracy on the held-out photos should be similar. Choosing it means I picked the best of the five versions I checked, which is a small amount of selection on training accuracy. Most errors are among the animal classes (bird, deer, cat and dog are often confused with frog and with each other) and between car and truck.
