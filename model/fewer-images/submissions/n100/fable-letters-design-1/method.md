# Method

**Architecture.** A fixed, formula-made feature stage followed by a bank of templates.

1. `norm` on the image (removes brightness and contrast).
2. 5x5 first-derivative-of-Gaussian filters (sigma 1) in 12 directions, `relu`, then a 1x1 conv and `relu` that turn them into 6 unsigned, orientation-sharpened edge maps: |r(theta)| - |r(theta+90)|.
3. Tent-weighted 6x6 pooling at stride 2, giving 10 x 10 cells x 6 orientations.
4. Local contrast normalisation: a dense layer lays each cell out with its 3x3 neighbourhood (each value once with + and once with - sign) as one channel, `norm` divides by the local RMS energy, and a stride-18 `maxpool` keeps the cell itself. Scaling the input makes `norm`'s built-in 1e-5 act as a hand-chosen contrast floor.
5. `norm` over the whole 600-number descriptor, then one dense layer holding 96,000 templates (4,000 per letter). Each output is the correlation between the image descriptor and one template.
6. Per letter: `maxpool` within 10 groups of 400 templates, `avgpool` over the 10 group maxima, giving the 24 class scores.

**Where the weights come from.** Nothing is trained or fitted, and no weight holds training pixels or statistics.

- Feature weights are formulas (Gaussian derivatives, tent windows, selection and sign matrices).
- Templates come from a 3-D right-hand model typed into `build.py`: spheres along a palm, thumb, four three-joint fingers and a forearm, with ordinary adult proportions. Each of the 24 letters is a hand-typed pose (finger spread and flexion angles, thumb segment directions, viewing direction, forearm direction) from knowledge of the ASL alphabet and of how the signs look from the front.
- `build.py` renders each letter 4,000 times with a fixed seed, varying joint angles, hand proportions, camera angle, framing, lighting, background level and blur. Each rendered picture goes through the same feature layers and becomes one template row, standardised by itself alone. No statistic across rendered pictures is used.

I looked at training images only as pictures, to understand how each sign appears (viewing side, framing, which way fingers point), and adjusted the typed poses and the framing rule accordingly. No coordinates or pixel values were taken from them.

**Checks used: 3 of 5** (logged in `checks.log`).

| Check | Design | Accuracy on the 2,400 training images |
|---|---|---|
| 1 | first poses, fine edge templates | 33.1% |
| 2 | corrected poses, plus coarse-edge and region descriptors and broader hand proportions | 31.0% |
| 3 | fine edge templates only, original proportions, corrected poses and framing (submitted) | 45.1% |

What each check taught: check 1 showed the pipeline works but T and P poses were wrong and fists were confused; check 2 showed pose fixes help but the tolerant descriptors and the proportion change hurt the upright-finger letters; check 3 confirmed that precise fine-edge templates with faithful poses are the better design.

**Known weaknesses of the submitted network.** M, N and X are almost never recognised (1%, 0%, 5%), and R, U, V, T sit at 21-30%, mostly lost to the B and F templates. I had identified likely fixes (per-letter zoom for X, more turned views for M/N/S) but ran out of time before building or checking them, so they are not in the answer.
