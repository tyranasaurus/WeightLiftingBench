# Method

**Architecture** (6 layers, about 13.2 M weights):

1. `feat` – 1x1 conv turning RGB into three colour features: luminance, red opponent (R − (G+B)/2) and blue opponent (B − (R+G)/2).
2. `z` – `norm`, so each feature map has zero mean and unit variance per image (removes brightness and contrast differences, including the very dark photos).
3. `match` – one dense layer holding 4,300 matched-filter templates: 100 per class.
4. `grid` / `best` / `scores` – reshape to (43, 100, 1), max-pool over the 100 templates of each class, reshape to the 43 class scores.

**How the weights were produced** – no training, fitting or statistics of any images.
`build.py` draws each of the 43 signs as a vector graphic from general knowledge of the German sign catalogue:
red-ringed white discs with digits drawn as strokes (speed limits), car/truck pictograms (no passing, no trucks),
grey "end of" discs with a diagonal band, red-bordered warning triangles with hand-typed pictograms, yield triangle,
yellow priority diamond, red stop octagon with "STOP", no-entry disc with white bar, and blue discs with white arrows
(left/right classes are mirror images of one drawing).
Each drawing is rendered with 3x supersampling at 4 sign sizes (19, 22, 25, 28 px wide) and a 5x5 grid of offsets
(±3 px in 1.5 px steps), converted with the same colour features as the image, made zero-mean inside the sign outline
and zero outside it (background is "don't care"), blurred slightly (Gaussian, sigma 0.7 px) and scaled to unit length.
Each render is therefore its own template, normalised by itself alone; the class score is the best normalised
correlation over that class's templates. For the deer sign both facings are rendered (2 sizes x 2 facings x 25 offsets).

All colours, proportions, stroke widths, sizes, offsets and channel weights (1, 0.7, 0.7) were chosen by hand in advance;
none were tuned on training or generated images.

**Status** – this was written under a hard time limit as a single pass; the design was not iterated.
One labelled check was used (1 of 5): 19 of the 43 training images correct (accuracy 0.4419). It showed the network
runs end to end and is far above chance (0.023), but nothing was changed in response to it.
`check_network.py` reports the network usable and `check_rebuild.py` reports that `build.py` reproduces the weights exactly.
