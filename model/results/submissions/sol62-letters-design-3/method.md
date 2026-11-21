The network normalizes image contrast, then computes unsigned Gaussian-smoothed derivatives in four directions and locally max-pools them. A dense layer correlates these edge maps with synthetic hand contours, and max pooling selects the strongest pose match for each of the 24 letters.

The contours are generated from hand-authored palm ellipses, finger and wrist capsules, and openings between fingers. A fixed grid varies scale, rotation, position, and aspect ratio. All weights are computed from formulas and these anatomical rules; no training images or fitted statistics are used to generate them.

One labeled smoke check on a fixed random subset of 384 training images achieved 16.15% accuracy. This is not a held-out score. The rebuild check reproduced the submitted weights exactly without access to data.
