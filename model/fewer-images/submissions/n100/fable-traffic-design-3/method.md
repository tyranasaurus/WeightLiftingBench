# Method

Nothing in this network is trained or fitted. Every weight comes from formulas and from drawings of the 43 signs
that are typed in `build.py` from general knowledge of the German sign catalogue.

## Architecture

1. **Photometric front end.** A 3x3 blur (no padding, 32 -> 30), then `log(x + 0.02)` built from 14 hinges per
   colour channel (1x1 conv, relu, 1x1 conv). The log image is standardised with one `norm` whose statistics are
   weighted towards the image centre: a dense layer appends a second copy in which centre pixels are repeated
   according to a Gaussian window, the pair is normalised together, and a strided pooling keeps the first copy.
2. **Hand-chosen feature maps (10 channels, 30x30).**
   - brightness = max(red, green), clipped, so red, yellow and white paint are bright and blue and black are dark;
   - local contrast = brightness minus its Gaussian surround, clipped;
   - colour: blurred differences R-G, G-B, B-R passed through a ramp; red minus blue in one signed channel, yellow
     in another;
   - six unsigned edge orientations (Sobel derivative, minus cot(30 deg) times the perpendicular derivative, dead
     zone, saturation, Gaussian blur).
3. **Template convolution.** One 30x30 kernel per drawing: 43 signs x 7 sizes (19 to 28 px) x 5 versions (crisp,
   two blur levels, rotated by -8 and +8 degrees) = 1505 kernels, slid over shifts of -2..+2 pixels.
   Each drawing is passed through the same front end and becomes its own kernel, in two parts that are each
   mean-free and unit-norm by themselves: a frame part (outline, border, colour layout) and a content part (digits,
   pictogram, arrow). Kernel = frame + content. The two signs with an empty interior (yield, no vehicles) get a
   hand-chosen bonus minus a penalty on edge energy inside instead of a content part.
4. **Read-out.** Maximum over positions, then over the 35 versions of each class.

About 20.8 million weights, most of them zeros.

## How the design was arrived at

I looked at contact sheets of the training images to see what the classes look like, and used four of the five
allowed checks on the training sample:

| check | accuracy | what changed before it |
|---|---|---|
| 1 | 0.6744 | first version: whole-sign cosine templates |
| 2 | 0.6788 | blurred versions, bolder pictograms, brightness = max(R, G) |
| 3 | 0.7842 | frame + content parts normalised separately |
| 4 | 0.7842 | redrawn pictograms, larger digits, signed red/blue channel |

The submitted files are the check 4 version. The sample has 100 images per class; weighted by the class frequencies
of the public benchmark the check 4 table corresponds to roughly 0.81.

## Known weaknesses

- Check 4 did not improve on check 3 overall: the redrawn "slippery road" pictogram fixed that class (0 -> 92 of
  100) but now attracts other triangle signs (381 predictions for 100 true), and "50" fell from 80 to 64.
- Still poor: double bend (1 of 100), cyclists (12), priority at next junction (17), children (18).
- The fifth check was not used.
