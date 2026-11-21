I used a fixed hand-designed prototype matcher. The network compares the input against synthetic digit templates built from general stroke paths and ellipses for several styles, widths, and slants. A small bank of formula-based oriented line filters provides additional stroke-direction features.

The architecture has several branches: full-resolution normalized matching, pooled raw-image matching, pooled orientation-feature matching, and coarse top/bottom/left/right regional matching. Each branch scores many variants per class, takes the maximum over variants, and sums the branch scores to produce the 10 output logits.

All weights are generated directly from code in build.py; no training, fitting, or statistics from the dataset are used to set them.