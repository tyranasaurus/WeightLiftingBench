## Task
Given an integer n, return the length of the repeating cycle in the decimal expansion of 1/n: the smallest
positive integer L such that the digits of 1/n after the decimal point repeat with period L from the first digit on
(for example L = 1 for n = 3 since 1/3 = 0.333..., L = 6 for n = 7 since 1/7 = 0.142857142857..., L = 2 for n = 11).
Every input satisfies all of the following:
  2^31 <= n < 2^32,
  n is coprime to 10 (not divisible by 2 or 5),
  every prime factor of n is below 4096.
The answer is always below 2^32.

## Network
A feed-forward ReLU network with 64 linear layers, given as a sparse list of entries rather than dense matrices.
Layer i is a matrix W_i of shape (in_i, out_i): in_0 = 32, every hidden layer has 262,144 units, and the last layer
has out_63 = 32 outputs. The forward pass, exactly:
    h = x
    for i in range(64):
        h = h @ W_i + b{i}
        if i < 63: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Everything is computed in float32.
Tensors:
  layer, row, col, val: float32 arrays of length 10,000,000, one entry per position: W_layer[row, col] += val.
      layer, row and col must hold integers in range. Entries with val = 0 are ignored, so unused slots stay zero.
  b0..b62: dense biases of length 262,144; b63: dense bias of length 32.
Input x [32]: n in binary, most significant bit first, each bit 0.0 or 1.0.
Output y [32]: the cycle length L in binary, most significant bit first. Output bit i is read as 1 when
y[i] > 0.5, else 0.

## Size limit
At most 10,000,000 entries can be nonzero (that is the length of the arrays), counting weights and nonzero biases
together; above that the score is 0. About 140 million integers satisfy the input conditions, so a table of answers
does not fit and the network has to compute L from n.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 1,000 fresh random n that satisfy the input conditions: most are uniform over all
such n, the rest are chosen to stress harder cases. Only exactly correct answers are rewarded: an n scores only when
all 32 output bits equal the true cycle length L. There is no partial credit for close answers or for some correct
bits. The score is the fraction of n answered exactly.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: layer (10000000,), row (10000000,), col (10000000,), val (10000000,), b0 (262144,), b1 (262144,), b2 (262144,), b3 (262144,), b4 (262144,), b5 (262144,), b6 (262144,), b7 (262144,), b8 (262144,), b9 (262144,), b10 (262144,), b11 (262144,), b12 (262144,), b13 (262144,), b14 (262144,), b15 (262144,), b16 (262144,), b17 (262144,), b18 (262144,), b19 (262144,), b20 (262144,), b21 (262144,), b22 (262144,), b23 (262144,), b24 (262144,), b25 (262144,), b26 (262144,), b27 (262144,), b28 (262144,), b29 (262144,), b30 (262144,), b31 (262144,), b32 (262144,), b33 (262144,), b34 (262144,), b35 (262144,), b36 (262144,), b37 (262144,), b38 (262144,), b39 (262144,), b40 (262144,), b41 (262144,), b42 (262144,), b43 (262144,), b44 (262144,), b45 (262144,), b46 (262144,), b47 (262144,), b48 (262144,), b49 (262144,), b50 (262144,), b51 (262144,), b52 (262144,), b53 (262144,), b54 (262144,), b55 (262144,), b56 (262144,), b57 (262144,), b58 (262144,), b59 (262144,), b60 (262144,), b61 (262144,), b62 (262144,), b63 (32,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
