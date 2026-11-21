# Method

## Architecture

A hand-designed **exemplar matcher**, not a learned feature hierarchy. Every weight is
either an analytic filter or one drawing of a hand, normalised by itself alone.

```
image (1,28,28)
  n0   norm                      zero mean / unit variance -> brightness+contrast invariant
  f1   conv 4@17x17, pad 8       DoG(0.9,2.0), DoG(2.0,4.0) and their negations
  r1   relu                      splits each band into positive / negative part
  ms   conv 2@5x5,  pad 2        Gaussian(1.0) * (pos - neg)  -> signed band-pass, 2 channels
  me   conv 2@5x5,  pad 2        Gaussian(1.0) * (pos + neg)  -> |band-pass| edge energy, 2 channels
  ns/ne norm                     each channel equalised, so no band dominates
  ts   conv 24000@28x28, pad 0   signed templates   -> (24000,1,1)
  te   conv 24000@28x28, pad 0   edge templates     -> (24000,1,1)
  rs/re relu, sc add             polarity-tolerant total match score
  g1   reshape (24,1000,1)       1000 templates per letter
  pmax maxpool, pavg avgpool, base conv 1x1 (weight 0), fin add
  out  reshape (24,)             class score = best match over that letter's own templates
```

75,314,164 weights. The matching convs use **kernel 28 with padding 0**, so each score is
a full-frame comparison against the same input vector — a pure cosine, with no zero
padding anywhere in the match.

## Where the weights come from

`build.py` reads no files. It contains:

1. **An articulated 3D hand model** typed from anatomy: a palm built as a slab of
   overlapping balls (so it foreshortens correctly under yaw), four three-phalanx fingers
   as tapered capsules, a two-link thumb, and a forearm. Fingers flex about the
   palm-normal axis, so a curl projects to the correct foreshortening and can fold either
   towards or away from the camera; painter's-algorithm depth ordering handles occlusion.
2. **The 24 ASL letters as joint angles and thumb targets**, from general knowledge of the
   alphabet (fist with thumb alongside for A, four straight fingers with the thumb across
   the palm for B, index-and-thumb ring for F/O/D, index and middle crossed for R, hand
   turned side-on for G/H/T, K and G turned downwards for P and Q, and so on). Thumb poses
   are set by two-link inverse kinematics onto a target point, often the computed position
   of a curled fingertip, so the rings in D/F/O actually close.
3. **A renderer** that paints smooth shaded forms with dark creases only where two parts
   meet — never a drawn outline against the background — then supersamples, box-filters
   and softens, to look like a greyscale photograph rather than line art.
4. **Analytic difference-of-Gaussian filters** from the Gaussian formula.

Each letter is drawn in 2 pose variants x 5 rolls x 4 framings, on a 36x36 canvas from
which 25 offset 28x28 windows are taken: 1000 drawings per letter, 24,000 total. Every
window is passed through the identical front end and normalised **by itself alone**
(per-channel zero mean / unit L2). No average, prototype, variance or any other statistic
is taken across drawings, and no training image, patch, descriptor or measurement is used
anywhere. The training set was only looked at (saved as PNG montages) to understand what
the letters look like, and used for the five permitted checks.

## Two corrections that mattered

* **Zero padding in the matching conv is degenerate.** With a padded shift search, a
  template's negative background can sit over the image's smooth background while its
  positive part falls outside the window, scoring highly for *any* image. Fixed by
  putting the translation search into the template bank and using padding 0.
* **Mean-subtracting a non-negative edge map biases the match.** |DoG| is >= 0, so
  subtracting its mean makes the whole background negative; background x background then
  rewards every template in proportion to how much background it contains, which is why
  sparse-hand letters (B, C, D, L, Y) won almost every image. Fixed by building edge
  templates from the **raw non-negative** edge energy, cut off outside each drawing's own
  hand silhouette (wrist and sleeve excluded). The match is then edge overlap, penalised
  where the drawing has an edge and the photo does not, with no reward from background.
  Signed templates keep their zero mean: a drawing's smooth background already responds
  ~0 there, so they are unbiased and remain a pure cosine.

## Verification

* `check_network.py`: usable, all shapes correct, 75,314,164 / 120,000,000 weights.
* `check_rebuild.py`: **OK**, max relative difference 0.0 — `build.py` alone, as an
  unprivileged user with no data, reproduces both answer files exactly.
* Front end checked against the scorer's own interpreter: agreement to 2e-6.
* On its own drawings the network recognises 24/24 letters and 144/144 pose/framing
  variants; outputs finite; ~14 s for 7,172 images.

## Honest status of the five checks

The five permitted runs on the training set scored 0.1160, 0.1127, 0.0484, 0.0844, 0.0559.
They were used to diagnose the two failure modes above, not to fit constants. The third
run tested a "max minus mean" readout, which turned out to reward template *diversity*
rather than match quality and was reverted. The submitted network is the corrected design
(non-negative masked edge templates built from raw edge energy, per-channel equalisation,
photographic rendering, wider roll/framing coverage); it was finished after the fifth
check, so **its accuracy on the training set is unmeasured**. It is verified correct and
self-consistent, but I cannot claim a measured accuracy for it, and I do not expect it to
beat the gradient-descent-trained baseline.
