# Method

A hand-weighted network; no training, fitting, or statistics of the data were used. Every number in
`weights.npz` is produced by `build.py` from geometric templates of the 43 standard German sign types,
drawn from general knowledge of their shapes and colours (each template self-normalised to unit L2 norm).

## Architecture (see network.json)
1. **feat** — 1×1 conv mapping RGB to 5 illumination-robust colour-opponent channels:
   red = R−G−B (pure traffic red, rejects gold/yellow), blue = B−(R+G)/2, yellow = G−B (gold diamond),
   lum = mean, dark = −mean.
2. **norm** — per-channel mean/variance normalisation (contrast/brightness invariance; ~1 in 5 images is very dark).
3. Four parallel branches, each emitting a (43,) score vector, summed by **add**:
   - **A (global group+content):** matched filters for red ring, red triangle (warning), inverted triangle
     (yield), gold diamond (priority road), blue disc, central red fill (stop/no-entry), grey+diagonal (end
     signs), plus content detectors (white bar→no-entry, STOP text, centre-red→no-passing, solid-dark→no-trucks,
     interior-ink penalty). A strong group weight (×3) gates the super-group; content decides within it.
   - **B (speed digits):** global first-digit glyph templates (2,3,5,6,7,8 and a leading 1), read contrastively
     (own digit minus the mean response) to pick the speed limit.
   - **C (blue arrows):** white-arrow templates (ahead, left/right turns, straight-or-left/right, keep left/right,
     roundabout), contrastive readout.
   - **D (warning symbols):** black-silhouette templates for the triangle interiors (exclamation, cross, bends,
     snowflake, etc.), contrastive readout.

## Notes
Because signs are centred and fill the frame, fixed global templates are effective. Reliable on the distinctive
classes (yield, no-entry, stop, priority road, blue arrows, several warnings); fine digit/symbol reading at
32×32 is the hard part and is only partly solved. Weights total ~238k (limit 120M). `build.py` reads no files
and reproduces `network.json` + `weights.npz` deterministically (verified with check_rebuild.py).
