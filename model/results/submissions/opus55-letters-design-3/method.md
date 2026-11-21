# Method

**Architecture (14 layers, 2.54 M weights).**
1. Contrast normalisation: `norm` of the image.
2. Edge filters: 12 signed first-derivative-of-Gaussian filters (σ = 1 px, 7×7) with relu. They are combined into 6 polarity-free orientation energies |∂I/∂θ| (every 30°), so a dark hand on a light background gives the same features as a light hand on a dark one.
3. Orientation sharpening: subtract the mean energy over orientations.
4. Clipping: `min(e, 0.45)`, built with relu, so strong outline edges don't swamp weaker internal creases.
5. Pooling: 4×4 binomial blur with stride 2, giving 6×14×14.
6. Global normalisation over all 6×14×14 entries.
7. Template matching: one conv with 2,160 kernels of size 14×14 and padding 2, so each template is tried at ±4 px shifts, then max over shifts.
8. Letter score: each letter's 90 templates form 6 groups of 15. The score is the best match in each group, averaged over the 6 groups (maxpool, then avgpool). A single generic template can then contribute at most 1/6 of another letter's score.

**Templates (no data used).**
- **Hand model.** `build.py` contains a small 3-D right-hand model made of spheres: a palm slab, four three-joint fingers, a three-segment thumb placed by two-link inverse kinematics, and wrist/forearm. Proportions come from general hand anatomy.
- **Letter poses.** Each of the 24 letters is written as joint angles plus a hand orientation, from general knowledge of the ASL manual alphabet as a viewer facing the signer sees it. For example: G and H point sideways, P and Q point down, C and O are seen from the side. For G, H, P, Q, T and X I give two alternative orientations.
- **Rendering.** Orthographic z-buffer with Lambert shading, short-range screen-space ambient occlusion (for the creases between curled fingers), and a slight blur.
- **Variation.** 90 renders per letter, randomised with a fixed seed: roll ±14°, yaw ±25° (±18° for the side-view letters C, O, X), pitch ±20°, small finger and thumb jitter, framing, offset, background level, and forearm/sleeve.
- **Framing rule.** The crop is a square around the hand, at least 1.6 hand units wide.
- **Template features.** Each render goes through the same front end (an exact numpy copy of layers 1–6). Its feature map, normalised by itself alone (zero mean, unit norm), becomes one conv kernel.

No weight is derived from training images, from labels, or from any statistic across generated images, and nothing was scored on generated images.

**How the data was used.**
- I looked at sample training images (as PNG files) to learn each letter's orientation and appearance in this dataset. For example, T appears as an index pointing right with the thumb up, and P as the index pointing left with the hand body on the right. I also compared my renders with real images by eye.
- I used the five allowed checks to find failure modes and change the design by reasoning:

| Check | Accuracy | What I learned / changed afterwards |
|---|---|---|
| 1 | 33.4% | Fists were absorbed by O, K and D. I fixed several poses. |
| 2 | 36.7% | Rendered internal edges were far stronger than in the photos. I softened the shading and corrected the framing. |
| 3 | 36.9% | Real fists show dark creases that my renders lacked. I added ambient occlusion. |
| 4 | 35.6% | Long-range occlusion created false hollows that made D and K absorb other letters. I shortened the occlusion range. |
| 5 | 39.2% | Final network: short-range occlusion, 90 templates per letter and group-averaged scoring together lifted A, C, L, O, R and Y. |

The submitted network is the one from check 5: 39.2% on the 27,455 training images, a normalised score of about 0.37.

**Known weaknesses.** At 28×28 the fist family (A, E, M, N, S, T, X) is hard to separate. E, M, P, Q, S and X are rarely recognised: real fists are still often labelled O, and V and W are often labelled K.
