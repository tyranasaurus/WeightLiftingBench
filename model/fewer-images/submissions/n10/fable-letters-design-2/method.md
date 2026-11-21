# Method

**Architecture.** A hand-built template matcher with two feature branches and a max readout.

- *Edge branch:* `norm` (per-image brightness/contrast) → conv with 16 steered Gaussian-derivative filters (σ = 0.9) → relu →
  1×1 conv + relu that turns them into 8 unsigned, sharpened edge orientations (|r_k| − |r_⊥| − dead zone) → two 1×1 convs
  that clip the energy (min(e, 0.25)) → Gaussian pooling conv (σ = 1.75, stride 2) to a 12×12 grid → reshape + `norm` so the
  whole feature vector is zero-mean, unit-variance.
- *Region branch:* a dense layer subtracts the mean of the picture's top rim (taken as the background level) from every
  pixel → 1×1 convs + relu give a soft "differs from background" map (lighter or darker, clipped to 0..1) → the same
  Gaussian pooling to 12×12.
- *Matching:* two dense layers hold one row per template (edge row = Pearson correlation with the edge features; region
  row = covariance with the figure map), added together, then reshape + maxpool takes the best template of each letter.

**Where the weights come from.** Nothing is trained or fitted, and build.py reads no data. Filters are formulas. Templates
are drawings of a right hand typed from general knowledge of hand anatomy and the ASL alphabet: a palm box, finger
capsules at typed angles, thumb and forearm, with two hand shapes and two forearm directions per letter. Each drawing is
rendered at four palm widths, five in-plane turns, two hand breadths and every position on a 2-pixel grid that keeps the
hand inside the frame; each rendering is passed through the same feature formulas and normalised by itself alone
(its own mean and standard deviation) to become one dense row. A template also expects a little edge energy of every
orientation all over the hand (skin is textured, the wall is plain). Constants (clip level, thresholds, sizes, line
strength) were chosen by reasoning, not measured.

**What the training images were used for.** I looked at them to learn what each letter looks like in this data set
(which way the hand points, that M/N/S are made with the fist rolled over, how much of the frame a hand fills) and
ran the five allowed checks:

| check | accuracy (240 images) | what I learned |
|---|---|---|
| 1 | 21.7% | pipeline works; K and I templates swallow other letters; drawn hands too thin, inner lines too strong |
| 2 | 27.1% | better; the M fist, blown up to fill the frame, passes for extended fingers → tie size to palm width, share palm placements across letters |
| 3 | 33.3% | that helped; K still pulls in B/U/V/R; broad hands had gaps between fingers, fingers too short above the fist |
| 4 | 32.1% | proportion fixes did not change the total (noise is about ±3 points) |
| 5 | 30.8% | fainter inner lines cut K's pull but O then pulled in C and tilted fists; total unchanged within noise |

The submitted network is the one from check 5. Expect roughly 30% accuracy (score about 0.28); the fists
(A/E/M/N/S), D/R/U and C-versus-O remain poorly separated. It will not match the same architecture trained on the
full training set.
