# Method

**Architecture.** A hand-wired network (about 2.7 M weights, 0.54 M non-zero) that works like a small rule table rather than a learned model.

1. Three conv + relu layers (one 3x3, two 1x1) turn every pixel into soft "colour words" (dark, dark grey, light grey, white,
   sky-blue, deep-blue, green, olive, brown, tan, red) and "texture words" (flat, weak texture or strong edge, each
   horizontal / vertical / diagonal). Each word is a fuzzy AND (`relu(sum - (n-1))`) of clipped ramps on luma, channel
   differences, chroma and Sobel gradients.
2. The words are average-pooled over a 4 x 4 grid of cells.
3. One dense layer scores 1,840 hand-described *scenes* (class x outline variant x 23 background types x 4 object sizes).
   A scene states which words it expects in each cell: a typed "material" (clear sky, grass, road, blue water, indoor ...)
   per row for the background, mixed with the class's typed object colours/textures according to how much of the cell a
   typed ellipse outline covers. Scene score = log prior + sum over cells of <word fractions, log expected fractions>;
   a maxpool takes the best scene of each class.
4. Six small votes are added, each a hand-typed table of "how often does this class show this property":
   colours that are more frequent in the centre than around it, top row vs bottom row (same / brighter / darker),
   long straight edges (five aligned edge pixels over a 9-pixel span, in 12 directions), a level horizon seen at both the
   left and right third of a row, the extent of the region that differs from the left/right border colour of its row
   (nothing / compact / wide-thin / wide-thick / tall / large-sparse / large-dense), and the colour words inside that region.

**Where the weights come from.** Every number is typed in `build.py` from general knowledge (what sky, grass, asphalt,
water, fur, car paint look like; where each class is usually photographed; that these photos are cropped around the
subject) or follows from a formula (Sobel kernels, ramps, supersampled ellipse coverage). No gradient descent, no fitting,
no statistic of any image, and no generated image is used to set a number. `build.py` reads no files.

**Use of the training images.** I viewed the 100 images as contact sheets to see what the classes look like, and ran the
five permitted checks (23 %, 31 %, 30 %, 36 %, 34 % on the 100 images). After each check I revised the hand tables by
reasoning about the confusion table (e.g. white/grey skies, brown frogs, tightly cropped subjects, a second outline
variant per class). The fifth check tested an extra "average background" scene; I judged it flawed (it counts the
background prior once per cell) and removed it, so the submitted network is the fourth-check design plus a few small
table refinements and has not itself been run on the training images. Other tests used only images I drew, and only to
confirm the wiring matches the formulas.

**Expected result.** Roughly 30-35 % accuracy (score about 0.25). Known weak spots: birds are almost never predicted,
cat/dog and deer/frog/horse are near coin flips, and blue-tinted road or airfield scenes are often read as ships.
The same architecture trained on 50,000 images would certainly do much better than these hand-set weights.
