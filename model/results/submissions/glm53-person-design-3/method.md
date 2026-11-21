# Method

**Architecture (204 layers, ~2.5k weights, all hand-set).** The network is a
hand-designed computer-vision pipeline assembled from the allowed ops
(conv/relu/pool/add/norm), followed by a small dense readout:

1. **Colour stage:** opponent channels L, (R−G), ((R+G)/2−B), smoothed with a
   Gaussian (σ=1.2) so JPEG noise cannot eat the chroma tolerances.
2. **Soft skin mask:** an L1-ball membership around typical skin chroma
   (|U−0.12|/0.14 + |V−0.13|/0.13 + |L−0.55|/0.32 < 2.3), sizes chosen from
   the classic published skin rules (Kovac et al.), then Gaussian-smoothed and
   gated into a "solid skin" map.
3. **Smooth-skin gate:** skin-coloured pixels whose local Sobel gradient is
   high are subtracted (min-gate), so textured warm surfaces (wood, straw,
   brick, bread crust) do not count as skin.
4. **Multi-scale blobs:** 6/12/24 px cell densities of smooth skin with
   centre-surround (blob) detectors at each scale.
5. **Eye-pair detector** (the highest-precision cue): at two scales (9px and
   17px kernels), "left eye darker than the nose bridge" AND "right eye darker
   than the bridge" (two convs combined with a min-gate), AND-gated with a
   capped "surrounded by smooth skin" term. The light-bridge requirement kills
   big single holes (food), dark lines (grill marks) and shadow edges.
6. **Head cue:** an isolated smooth-skin blob (local 30 px context mostly
   non-skin colour, few other blobs nearby) with a dark mass directly above
   (hair) or below (clothes) — the head-and-shoulders signature; also a mean
   version for groups, and a vertical-stripe version for limbs.
7. **Context cues:** vertical-edge column concentration (via the norm layer),
   capped over-skin penalty (huge uniform skin mass = wall/sand/food), overall
   rough-skin penalty, blob-count penalty, and a blue/bright dominance cue
   (sky/water scenes have no person).
8. **Readout:** 19 scalar features, each routed to its own channel of a
   (19,1,1) vector by one-hot 1×1 convs summed with `add`, then a dense(2)
   layer with hand-set weights (positive: eyes, head structure, mid-size
   blobs, limbs; negative: roughness, over-skin, blob count, blue dominance).

**How the weights were produced.** Every constant was typed by hand from
general knowledge of what people look like in everyday photos (skin chroma
ranges, eye spacing ≈ 0.4 × face width, faces are smooth, heads sit on darker
bodies, people are vertical). No optimisation, regression or statistic of the
training images was ever turned into a weight; build.py contains only
formulas. I studied the training data only in the ways the rules allow:
text renderings of individual images (brightness, a classic skin-rule mask, a
darkness map) to see what the two classes look like, and simple per-image
summaries. I used my 5 allowed runs of run_network.py to check that the
network works (51.3%, 51.8%, 49.6%, 51.3%, 51.1% on a fixed 5000-image
subset); between checks I diagnosed failures (a sign error in the skin ball,
an uncapped gate term that fired on any skin, too-tight chroma tolerances)
and fixed them by reasoning, also using purely synthetic probe images (typed
cartoon faces/rooms/food, no training data) to trace intermediate features.

**Honest assessment.** On the 5000-image check subset the final network
scores about 51% — barely above chance. At 96×96, food, wooden/straw
textures, walls, sandy scenes and brown animals are genuinely confusable
with people using hand-crafted colour/shape cues, and many COCO person
images (small, distant, clothed or back-turned people) have no detectable
skin or face at this resolution. The eye-pair and head-structure cues are
high-precision but low-recall. The architecture is also deliberately full of
hard min-gates and saturation caps, which makes it effectively untrainable
by gradient descent (most ReLUs sit deep in their dead region from any
random initialisation).
