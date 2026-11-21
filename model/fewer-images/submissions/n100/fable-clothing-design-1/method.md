# Method

**Architecture.** Three branches whose 10-way outputs are added.

1. *Shape branch.* `image -> soft silhouette -> 5x5 Gaussian blur -> dense (100,000 templates) -> reshape (1, 10, 10000)
   -> max-pool over each class's 10,000 templates`. The soft silhouette is `clip((x - 0.012) / 0.028, 0, 1)`, built from
   three 1x1 convs and two ReLUs. Each dense row is one blurred template silhouette and its bias is `-|row|^2 / 2`, so
   the layer ranks templates by squared distance to the image; the max-pool keeps the nearest template per class.
2. *Front-line branch.* `norm(image)` -> thin vertical-line filter (bright and dark polarity) -> clip -> average down
   15 rows -> strongest value in the middle columns minus strongest value a few columns to either side -> bounded
   penalty for *pullover*. A zip or button line means the garment opens at the front (coat, shirt); coat and shirt
   templates carry a small fixed handicap (2 units each) that this penalty outweighs.
3. *Leg-gap branch.* `norm(image)` -> "much darker than the pixel d to the left AND the pixel d to the right"
   (d = 2, 3, 5; maximum of the three) -> clip -> max over 3 columns -> average down 9 rows -> strongest value in the
   lower middle of the frame -> +3 for *trouser* if a gap is found, -3 if not. This separates trousers from narrow
   sleeveless tops and dresses, which have almost the same outline once blurred.

**Where the weights come from.** Nothing is trained or fitted; `build.py` reads no files.

- Templates are drawings made by parametric shape models that I typed from general knowledge: garments in cm
  (chest width, length, neckline, collar, sleeve length / angle / cuff width, hem), shoes in units of shoe length
  (sole, toe box, collar or shaft height, heel, straps), bags as a body plus a handle loop. Parameters are sampled
  with a fixed seed. As in the dataset's preparation, each drawing is scaled so its longest side fills the frame and
  is centred. Every drawing is its own template, passed through the same silhouette + blur and normalised by itself
  alone. No averages or other statistics across drawings are used.
- Filters come from formulas (Gaussian, second difference); thresholds, clips, handicaps and cue sizes are hand-chosen
  constants listed at the top of the network section of `build.py`.

**Use of the training images.** I looked at image grids to understand the classes, and used the 5 allowed runs
(`checks.log`: 71.9%, 75.4%, 75.2%, 74.5%, 75.6% on the 1,000 images). After each run I revised the shape models or cue
logic where the confusion table showed my picture of a class was wrong (e.g. T-shirt sleeves too wide, ankle boots too
tall, sandal types that only attracted sneakers, a cue that pulled trousers towards coat).

**One caveat.** The submitted network differs from the one measured in check 5 by a single constant: the shirt
handicap is 2.0 (equal to the coat handicap, my original design) instead of 2.5. Checks 3-5 showed the equal setting
gives the best pullover / coat / shirt balance, but this exact final network was not itself run on the training
images, so its accuracy there is an estimate (about 75-76%), not a measurement.

**Known weaknesses.** Shirts are mostly missed (short-sleeved ones look like T-shirts, long-sleeved ones like coats or
pullovers in outline); solid sandals are read as boots or sneakers; low ankle boots as sneakers.
