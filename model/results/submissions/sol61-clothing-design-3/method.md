# Hand-designed clothing network

The network compares two soft foreground maps against 2,100 procedural silhouettes of garments, trousers, footwear, and bags, taking the maximum match within each class. The silhouettes are generated from generic polygons, ellipses, line segments, and hand-specified ranges of sewing-pattern proportions; none is copied, traced, measured, averaged, or fitted from a training image. Additional fixed branches measure sleeve/torso notches, coherent front seams, garment openings, fabric-edge contrast, morphological gaps, and aspect-conditioned footwear solidity, using only the permitted neural-network operations.

All weights are generated deterministically by `build.py` from formulas and hand-authored constants. The builder reads no files and uses no training, optimization, external weights, randomness, or stored image descriptors. Training images were viewed in two contact sheets for qualitative understanding, and the network was evaluated only through the prescribed runner, exactly five times, on the first 10,000 official training examples with labels.

## Check record and final status

The five checks returned 58.00%, 56.72%, 61.38%, 62.25%, and 52.44% accuracy, respectively; these are development checks, not hidden-test results. The fifth revision improved shirt and sandal detection but introduced severe trouser/coat confusion. After that check, the generic narrow-coat template family was removed and the new front-opening cue was restricted by garment-shape context. The submitted final network includes those corrections and has **not** been evaluated on images, because all five checks had been used. Structural validation and the independent data-free rebuild check are run separately and do not evaluate images.
