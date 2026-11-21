# Method

## Architecture

A hand-designed, fully formula-driven detector, 312 layers / 27,180 weights.
It is a colour-and-shape pipeline, not a learned feature hierarchy.

1. **Colour membership (96x96, 1x1 convs + relu).** Trapezoid membership
   functions `T(x)=(relu(x-a)-relu(x-b)-relu(x-c)+relu(x-d))/(b-a)` are built
   from the standard YCbCr skin cluster (Cr in ~[0.545,0.645], Cb in
   ~[0.355,0.480]) plus a "pinkness" discriminant
   `D = (R-2G+B) - 0.1*(R-B) = 0.9R - 2G + 1.1B`. Skin is pink (R-G about twice
   G-B); wood, fur, sand, bread, cardboard and beige paint are yellow, so D
   separates them. A soft-AND of the three tests gives `skin`; a looser pair
   gives `warm`; `(0.30-Y)/0.26` gives `dark`. All are clipped to [0,1] with
   `relu(z)-relu(z-1)`.

2. **Material maps (48x48 and pools to 24/12).** Texture is `|Laplacian(Y)|`,
   normalised to `texn = clip01(tex/0.08)`. Then
   `cs = relu(skin - texn)` (smooth skin), `fd = relu(warm + texn - 1)`
   (textured warm material: food, fur, crust), plus vegetation and
   white-porcelain maps. These are Lukasiewicz t-norms, the only AND available
   without multiplication.

3. **Templates.** Centre-surround disk-minus-annulus blob kernels at three
   scales, applied identically to `cs`, `fd`, vegetation and white. Face
   kernels are the textbook head: an oval 1.25x taller than wide, two dark eyes
   at (+-0.22W, -0.10W), a mouth at +0.30W, hair above, non-skin around, all as
   smooth (Gaussian / clipped-ramp) memberships so they do not alias on coarse
   grids. The face conv takes **two** channels, `[smooth skin, non-skin
   tissue]`, each zero-sum: on a single signed map a bright blob with a dark
   surround would look like a face even with no skin present. Kernels are
   normalised so an analytically constructed ideal head scores exactly 1.0.
   A luminance-only version handles black-and-white photos, where R=G=B makes
   every chroma test fail.

4. **Readout (hand-written rule table).** Each template map contributes two
   statistics: the raw maximum (absolute strength) and the maximum after the
   `norm` op (how far the best match stands out from the rest of *that* image).
   Each becomes a saturating ramp, and a soft AND `relu(u_raw + u_z - 1)`
   requires both: diffuse clutter has a high raw max but a low z, faint specks
   have the reverse. Scales are combined by **max**, implemented with
   dense+relu, never by sum — a sum saturates on the null response that every
   scale shares. The final score is a competition between matched positive
   (smooth-skin blobs, faces) and negative (food/fur blobs, foliage, porcelain,
   textured-warm fraction) evidence, plus a penalty for a large smooth-skin
   area with neither a blob nor a face (wood, leather, a bare wall).

## How the weights were produced

Every number comes from the code: colour-space definitions, the trapezoid and
Gaussian formulas, the head geometry above, and thresholds chosen by hand from
general knowledge of what people and everyday objects look like. `build.py`
reads no files. No training image, patch, descriptor or statistic was used,
and nothing was fitted.

I looked at training montages (saved as PNG) to understand the classes, and
verified the design analytically: the colour algebra evaluated at named
colours, the template responses to analytically constructed ideal / plain-blob
/ no-skin patterns (ideal = +1.00, bright blob with no skin = -0.03), the
max-chain against a reference `max`, and the readout on hypothetical feature
vectors. Four of the five allowed network runs were used. They exposed three
real coding bugs, all of the same kind — a constant written into a coefficient
slot instead of a bias (the chroma +0.5 offset, the `-0.60/0.40` soft-AND
offset landing on the luminance channel, and `(Y-0.72)/0.15` subtracting from
each coefficient), plus a `+YV` that should have been `-YV` in the dark-pixel
penalty and a vegetation clip that subtracted a value from itself. With those
fixed, the colour algebra matches the intended design exactly.

## Honest status

The last measured run (check 4, before the final two structural fixes) scored
0.4998 on the 20,000-image sample, predicting "person" for 87% of images: the
evidence direction was correct but the threshold far too permissive. The
submitted version adds the two fixes motivated by that result — the
two-channel non-negative face input, and the raw-max AND z-max gate — which
are verified analytically but **not** measured, because the five-check budget
was spent. Its accuracy is therefore unconfirmed.
