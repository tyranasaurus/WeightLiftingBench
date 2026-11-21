# Method

**Architecture.** The network is a template matcher with hand-set weights.
1. A 7x7 convolution computes Gaussian-derivative edges in 8 directions (sigma = 1).
2. A relu and a 1x1 "orientation sharpening" convolution follow, then a 2x2 maxpool, giving 8 x 14 x 14 edge maps.
3. A hole channel is computed by a 28-step flood fill from the frame border, built from conv, maxpool, add and relu layers. Background that the flood cannot reach counts as enclosed. The barrier is the ink, with 1–2 pixel gaps softly closed. The hole channel is 2x2 max-pooled and weighted 1.2.
4. Edges and holes are stacked into 9 x 14 x 14 features F. A 3x3 max dilation of F gives D.
5. F and D are each z-normalised over the whole feature vector (reshape then norm).
6. Two dense layers correlate D(input) with each template's F, and F(input) with each template's D. This is a two-way, deformation-tolerant correlation.
7. The two scores are summed. A maxpool over blocks of 100 templates, a 0/1 selection dense layer and a final maxpool give each class's best template score.

**Weights.** No training data is used anywhere. Every template is rendered by build.py from typed pen-path models of the digits, written from general knowledge of handwriting:
- Splines, arcs and lines, with several styles per digit: looped or Z-shaped 2s, open, closed and u-shaped 4s, flagged 1s, crossbar 7s, one-stroke, two-loop and open-top 8s, and so on.
- Each template gets random style parameters, slant, rotation, width and pen thickness from fixed seeds.
- Each template is normalised the way MNIST digits are: fit into a 20x20 box, centre of mass at the frame centre.
- Each template is run through the same feature arithmetic in numpy, then made zero-mean and unit-norm by itself alone.

Class 1 gets 1,800 templates; every other class gets 3,500. All filter constants (sigma, sharpening 0.5, gap closing 0.7, hole weight 1.2, dilation size) were chosen by hand.

**Checks.** Four checks on the training set gave 0.891, 0.917, 0.921 and 0.933. I used the confusion tables only to decide which typed digit styles were unrealistic or missing.
