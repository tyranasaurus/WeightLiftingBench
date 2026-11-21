# Method

**The approach.** The network matches the image against a large bank of synthetic digits that I drew from hand-typed pen-path models. It scores each class with a soft nearest-template vote. No weight comes from training images: there is no fitting and no statistics of the data. `build.py` makes every number from formulas, the typed stroke models and a fixed random seed.

**Architecture** (about 260 layers, 112.3M weights)
1. **Edge features.** A 7x7 conv applies 16 oriented derivative-of-Gaussian edge filters (formula: sigma 1.0 across, 1.8 along) and one Gaussian ink filter, followed by relu. A 5x5 stride-2 Gaussian conv then folds opposite directions together, giving 8 unsigned orientations plus 1 ink channel at 14x14.
2. **Hole channel.** This marks enclosed background, computed inside the network.
   - A soft ink mask is computed as clip(4(x-0.2), 0, 1).
   - The background is flood-filled from outside the frame in 36 steps. Each step is a zero-padding 1x1 conv, then a 3x3 maxpool, then an add and a relu. Together these do N = max(ink, erode(N)), with "outside the frame" counting as reachable.
   - Holes are N - ink: the background that cannot be reached from the border. They are Gaussian-pooled to 14x14 (weight 1.5) and added as a 10th channel.
3. **Correlation.** The 10x14x14 = 1960 features are reshaped to (1,1,1960) and passed through `norm`. One dense layer per class then computes the Pearson correlation with each synthetic template of that class. Template rows are the normalised template features divided by D.
4. **Decision.** Maxpools give each class's best correlation, and a second maxpool gives the overall best, M. Each class's score is the mean over its templates of relu(corr - M + 0.03). This is a soft vote among the templates that come within 0.03 of the best match. It uses 1-input dense layers (bias 0.03, weight -1) to broadcast M, then add, relu, avgpool and one-hot placement layers.

**Templates.**
- **Shape models.** For each digit I typed several common writing styles as control-point pen paths, with parameter ranges from general knowledge. Examples: a plain 1 or a 1 with a flag or base; a plain 2 or a 2 with a bottom-left loop; open-top and closed-top 4s; 7s with or without a crossbar; one-stroke and two-loop 8s; straight-stem and curved-stem 9s.
- **Random variation (fixed seed).** Control-point jitter, a smooth warp, slant, rotation, aspect ratio, pen radius and pen pressure.
- **Rendering.** Digits are rendered the way the data is described: anti-aliased strokes, ink box scaled to 20 px, centre of mass moved to (14,14).
- **Bank sizes.** There are 1,900–7,600 templates per class (57,200 in total). Classes with more varied shapes get more templates.

**Use of the 5 checks** (all on the 60,000 training images through `run_network.py`)
| Check | What I changed | Accuracy |
|---|---|---|
| 1 | Baseline | 86.6% |
| 2 | Reworked the 2, 7 and 8 styles; bank size per class | 91.5% |
| 3 | Softer edges, revised 3/5/6/9 models, bigger bank, delta-vote | 92.35% |
| 4 | Hole channel | **92.83%** |
| 5 | Gap-closing before the flood fill, a few bold pens | 92.53% |

The only feedback I used from each check was the confusion table, and I changed things by hand based on what kind of mistakes it showed. Because check 5 came out worse, I reverted to the check-4 design, and that is what was submitted. Several design choices were made on synthetic digits only, never training images: which features to use, delta = 0.03, and the hole-channel weight. For these I compared a few options on a harder synthetic test set, made with stronger random deformations and a different seed.
