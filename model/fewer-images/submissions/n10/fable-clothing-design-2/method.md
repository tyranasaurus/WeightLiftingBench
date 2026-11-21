# Method

**Idea.** A nearest-drawing classifier whose "memory" is a set of silhouettes I drew from general knowledge of
clothing, shoes and bags, plus two small hand-made cues. Nothing is trained or fitted; `build.py` reads no files and
every weight comes from formulas, typed shape models and hand-chosen thresholds.

**Architecture** (78.5M weights, almost all in one dense layer).

1. `ramp` / `ramp_r` (1x1 conv + relu): per-pixel ramps whose differences are clamps to [0, 1].
2. `feat` (5x5 conv): a concave intensity curve `0.4*clamp((x-0.02)/0.08) + 0.3*clamp(x/0.25) + 0.3*clamp(x/0.7)`
   ("is there ink here", so dark and bright items look alike), blurred with a Gaussian of sigma 1 px;
   `feat_n` (norm) standardises it over the image.
3. `match` (dense, 100,000 units): each row is one drawing, passed through the same curve and blur and then
   normalised **by itself alone** (zero mean, unit norm), so the unit outputs the Pearson correlation between the
   image and that drawing. The bias holds a small hand-chosen handicap for variants that are rare in their class
   and look like the typical member of another class (e.g. a high-top sneaker versus an ankle boot).
4. `match_g` / `match_max` (reshape + maxpool): best drawing per class -> 10 shape scores.
5. Sandal cue (`m_neg` ... `strappy`): grey-scale morphology built from maxpools on a steep ink mask M:
   thin parts `sum(M - open3(M))` plus holes `sum(close5(M) - M)` minus `0.22 * ink`. Straps and cut-outs are thin
   and holey, solid shoes are not; this adds up to +0.25 to "sandal".
6. Front-opening cue (`img_n` ... `opening`): units on the standardised raw image look for a 1-2 px vertical line,
   brighter or darker than its flanks, held over rows 7-22 at the centre. A zip or button line nudges coat and
   shirt up and pullover and T-shirt down (at most 0.014).
7. `scores` = sum of the three.

**Where the drawings come from.** A tiny vector kit in `build.py` (polygon fill, strokes, rounded corners) and one
parametric model per class, each typed from what the item looks like in a front or side shop photo: T-shirts
(torso, short sleeves, tank tops), trousers (waist, hips, two legs with a slit), pullover / coat / shirt (torso plus
long sleeves, differing in collar, proportions, cuffs and hem), dresses, sandals (sole, straps, heels, wedges),
sneakers and ankle boots (side profile, toe on the left), bags (body with handle, strap or none). Each drawing is
trimmed, scaled so its longest side is 28 px and centred, exactly as the task describes the photos. Parameters are
drawn from hand-chosen ranges with a fixed seed; the same parameter draws are reused across classes so that
analogous drawings of different classes differ only in class-specific features. 10,000 drawings per class.

**Use of the training images.** I looked at them (PNG contact sheets) to understand what the classes look like, and
ran the network on them four times (of the five allowed) to check that it works: 80, 79, 77 and 80 of 100 correct.
Between checks I corrected modelling mistakes that I could justify from clothing knowledge (garment proportions,
sleeves that hang flush, a shoulder line that should not depend on the garment, necklines whose silhouette is flat
because the back of the collar shows). The four results are within sampling noise of each other on 100 images, so
they were not used to pick a best variant; the submitted network is simply the last revision. No pixel values,
measurements or statistics of training images are in the weights.

**Known weak spots.** Long-sleeved shirts versus coats and pullovers (nearly the same silhouette), solid sandals
versus sneakers, and high-top sneakers versus low ankle boots. Expect roughly 75-80 % accuracy.
