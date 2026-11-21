## Task
Invert the Burrows-Wheeler transform. Take a hidden string s of 3 to 47 letters over {a, b, c} and append the
sentinel $ to get t = s$ (length n = len(s) + 1, so n is 4 to 48). Write every cyclic rotation of t, sort the
rotations lexicographically with the order $ < a < b < c, and let L be the last column: L[i] is the last character of the
i-th smallest rotation, i = 0..n-1. L contains $ exactly once, and the network gets L and must output s.
Since $ is the smallest character, row 0 is the rotation that starts with $ (that rotation is $s), so L[0] is the last
letter of s. All rotations are distinct (there is one $), so the sort has no ties.
Example: s = abca gives t = abca$, the sorted rotations are $abca, a$abc, abca$, bca$a, ca$ab, and L = ac$ab.

## Network
A feed-forward ReLU network with 40 linear layers, given as a sparse list of entries rather than dense matrices.
Layer i is a matrix W_i of shape (in_i, out_i): in_0 = 240, every hidden layer has 32,768 units, and the last
layer has out_39 = 188 outputs. The forward pass, exactly:
    h = x
    for i in range(40):
        h = h @ W_i + b{i}
        if i < 39: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Everything is computed in float32.
Tensors:
  layer, row, col, val: float32 arrays of length 1,000,000, one entry per position: W_layer[row, col] += val.
      layer, row and col must hold integers in range. Entries with val = 0 are ignored, so unused slots stay zero.
  b0..b38: dense biases of length 32,768; b39: dense bias of length 188.
Input x [240], all entries 0.0 or 1.0, one hot per position of L:
  x[5 * i + k] for i = 0..47: k = 0 '$', 1 'a', 2 'b', 3 'c', 4 blank.
      L is left aligned: L[0] is at position 0. Positions i >= n are blank (k = 4). The symbol codes are in sort order
      ($ = 0 < a = 1 < b = 2 < c = 3).
Output y [188]: for position j = 0..46 of s (s[0] is the first letter), the 4 logits y[4 * j + k], k = 0 'a',
  1 'b', 2 'c', 3 "no letter". Position j is read as slot k when y[4 * j + k] > 0.5 and the other 3 logits of position j
  are all < 0.5. Any other pattern leaves it blank. For j >= len(s) the right answer is slot 3 ("no letter").

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries: the nonzero entries of val plus
the nonzero biases (layer, row and col are indexes and do not count). Above that the score is 0. The input space is far
too large to store answers, and the network has only 40 layers.

## Scoring
Each evaluation draws 1,000 fresh random strings s (length uniform from 3 to 47; some are i.i.d. random
letters, some have runs, some are noisy periodic, some are mostly one letter) and gives the network the BWT of each.
Only the len(s) real positions are scored. For each string:
    letter_score = (correct letters - 1 * wrong letters) / len(s)
A blank position scores 0 and a wrong letter costs 1, so leave a position blank unless you are sure.
The score is the mean letter_score over the strings (floored at 0), so partial credit counts.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: layer (1000000,), row (1000000,), col (1000000,), val (1000000,), b0 (32768,), b1 (32768,), b2 (32768,), b3 (32768,), b4 (32768,), b5 (32768,), b6 (32768,), b7 (32768,), b8 (32768,), b9 (32768,), b10 (32768,), b11 (32768,), b12 (32768,), b13 (32768,), b14 (32768,), b15 (32768,), b16 (32768,), b17 (32768,), b18 (32768,), b19 (32768,), b20 (32768,), b21 (32768,), b22 (32768,), b23 (32768,), b24 (32768,), b25 (32768,), b26 (32768,), b27 (32768,), b28 (32768,), b29 (32768,), b30 (32768,), b31 (32768,), b32 (32768,), b33 (32768,), b34 (32768,), b35 (32768,), b36 (32768,), b37 (32768,), b38 (32768,), b39 (188,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
