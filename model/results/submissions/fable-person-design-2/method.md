# Method

**Architecture.** A hand-wired network (235,213 weights, about 14,600 non-zero) built only from conv / relu /
pool / add / reshape / dense. Three detector branches feed a small decision head that outputs "person" when
any branch fires:

1. **Skin blob.** 1x1 convolutions turn RGB into a soft skin membership (a wedge in RGB: hue 6-36 degrees,
   saturation 0.18-0.68, red channel above 0.33). On a 4-level pyramid, centre-surround templates look for a
   compact skin-coloured ellipse (face, hand) with little skin colour in the ring around it, at seven sizes
   (face width about 4 to 34 px). The evidence is ignored in scenes dominated by warm colours (wood, food, lamp
   light), measured by the global share of warm pixels, and penalised slightly in the bottom third of the picture.
2. **Structured face.** A larger skin blob that also shows an eye band darker than both the forehead and the
   cheeks (a min of two band differences, built from relus) and darker hair above. Not subject to the warm rule.
3. **Standing figure.** On luma / red-green / blue-yellow at four pyramid levels: a "ridge" where three adjacent
   parts of a body-wide patch are all brighter (or darker, redder, ...) than both background flanks; then a
   template asking for ridge along an upright or leaning axis about 2.6 body-widths long, nothing ridge-like
   beyond its ends (poles, trunks) or beside it (fences, clutter), and a change of ridge type between the upper
   and lower half (clothing, unlike a post or bottle).

Exactly grey (black-and-white) photos are called "person", since colour evidence cannot exist there.

**How the weights were produced.** Every number is typed by hand or computed from a formula in `build.py`
(anti-aliased ellipses, rectangles and sheared rectangles; colour constants from general knowledge of skin
tones and of face / body proportions). No training, fitting, or statistics of the training images were used,
and no random numbers. I looked at contact sheets of training photos to understand the classes, and debugged
the arithmetic only on synthetic scenes I drew myself (faces, capsule figures, poles, noise).

**Checks (logged in checks.log).** Four completed runs on the 20,000 training photos:
1. skin blob only: 58.4% - a real but weak cue;
2. plus warm-scene rule, zone prior, grey rule: 58.0% - precision of the blob cue rose from 59% to 67% but
   recall fell equally;
3. plus standing figure: 59.4% - fires on 23% of the remaining person photos and 15% of the no-person ones;
4. plus structured face: 59.7% - adds about as many false as true detections (+266 / +215).

A fifth run, on a variant with a lower blob threshold (0.12 instead of 0.17), was killed before it produced a
result, so that variant is unverified and was discarded. **The submitted network is exactly the one from
check 4** (59.7% on the training sample, i.e. a score of about 0.19).

**Honest assessment.** Each hand-built cue has a likelihood ratio of only about 1.3-2, so accuracy is modest.
I do not expect this to beat the same architecture trained by gradient descent; whether it beats the
deterministic-program baseline is unknown.
