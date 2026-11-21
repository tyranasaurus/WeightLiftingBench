# Method

## Architecture (all weights hand-designed from formulas; no training)

The network is a hand-built galaxy-morphometry pipeline followed by a
hand-written soft rule layer:

1. **Conditioning.** Luminance and a red-minus-blue colour channel are taken by
   1x1 convolutions and per-image normalised (`norm`), which removes the sky
   level and makes all later measurements brightness-relative.

2. **Polar ring analysis (the core).** A dense layer with geometric
   ring/sector mask weights ("rings and sectors" from general knowledge)
   resamples the blurred image into 12 log-spaced rings x 24 azimuths.
   Two parallel branches:
   - A *magnitude branch*: each ring channel carries a large "scale" slot
     `K = 5*(ring mean + 0.7)`; the channel `norm` then divides the whole ring
     by ~K, giving the ring's shape relative to its own brightness (a floor
     keeps sky rings from blowing up). Star spikes are clipped at +-1.6.
     This yields unsaturated Fourier harmonics (m=1..4) per ring: elongation
     (m=2), dipoles (m=1, with the sky-gradient plane subtracted using the
     outermost rings), irregularity (m=3, m=4).
   - A *phase branch*: per-ring standard-deviation normalisation, which
     equalises ring amplitudes. From it, phase-coherence sums are computed:
     bar/axis alignment (`|sum C2|` over ring groups) and **spiral pitch
     projections** - the m=2 phase is projected onto bases `cos/sin(k ln r)`
     that are orthogonalised against constant and log-radius-ramp components,
     so a fixed-phase pattern (bar, edge-on disk, star) projects to ~zero and
     only genuinely *winding* arms (logarithmic spiral, pitch k = 2 tan p)
     score, for tight/mid/loose pitch (k = 0.45/0.85/1.3) and both chiralities.
3. **Support features.** Normalised radial profile (concentration, extent),
   corner-referenced bandpass texture per ring (smooth vs featured), colour
   gradient (red bulge), all from fixed Gaussian kernels and mask weights.
4. **Decision layer.** ~35 clamped soft gates (relu thresholds chosen from the
   analytic noise floors of the harmonic maths and general galaxy morphology:
   round/in-between/cigar by axial-ratio-driven m=2, bars by inner fixed-phase
   m=2 with a round centre plus winding arms, spirals by pitch-coherent
   winding, edge-ons by sustained thin-disk m=2, mergers by aligned dipoles,
   disturbed by irregular high harmonics), combined additively into 10 class
   scores through a multi-layer dense/relu stack.

Every number in the weights comes from closed-form formulas (Gaussians,
Fourier tables, mask geometry) or hand-chosen rule constants; `build.py`
reads no files and regenerates everything. Five official training-set checks
were used to verify mechanics and catch structural bugs (the first revealed
that a per-ring-std-only design is poisoned by sky gradients/stars, which led
to the dual-branch normalisation above).
