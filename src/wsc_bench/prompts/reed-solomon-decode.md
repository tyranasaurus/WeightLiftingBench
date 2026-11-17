## Task
Correct a Reed-Solomon codeword. A codeword of 31 symbols was corrupted in up to 5 symbols. Recover the original
codeword and say which symbols were corrupted.
Field: GF(32) = GF(2)[x] / (x^5 + x^2 + 1). A symbol is a 5 bit value; bit b (value 2^b) is the coefficient of x^b, so
symbol 2 is the element alpha = x, which is primitive (alpha^31 = 1). Addition is bitwise XOR.
Code: g(z) = (z + alpha^1)(z + alpha^2)...(z + alpha^10) has degree 10. A codeword is the 31 coefficients
c_0..c_30 (c_j is the coefficient of z^j) of m(z) * g(z), where the message m(z) has 21 coefficients drawn
independently and uniformly from the 32 field elements (the code is not systematic). Every codeword has
c(alpha^i) = 0 for i = 1..10. The received word r is the codeword with exactly e of its 31 symbols corrupted: e
distinct positions chosen uniformly, and each corrupted symbol has a uniformly random nonzero field element added
(XOR). e is drawn from 0..5 with probabilities (0: 0.04, 1: 0.04, 2: 0.04, 3: 0.08, 4: 0.10, 5: 0.70). Any 5 errors are uniquely correctable.

## Network
A feed-forward ReLU network.
Input x [155]: the received word, x[5 * j + b] = bit b of r_j for j = 0..30, each 0.0 or 1.0.
The forward pass, exactly:
    h = x
    for i in range(48):
        h = h @ W{i} + b{i}
        if i < 47: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [155 x 1024], b0 [1024], W1..W46 [1024 x 1024], b1..b46 [1024],
W47 [1024 x 186], b47 [186]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 48 layers are always applied.
Output y [186]: y[5 * j + b] > 0.5 means bit b of the corrected symbol c_j is 1, for j = 0..30
(indices 0..154); y[155 + j] > 0.5 means position j was corrupted (r_j != c_j),
for j = 0..30.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 500 fresh random messages, encodes them and corrupts them as above. There are far
too many (codeword, error) pairs to store answers, so the network has to compute the decoding. Per case:
    0.15 if the whole 31 bit corruption mask is right, plus 0.85 if all 155 corrected bits are right.
The score is the mean over cases.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (155, 1024), b0 (1024,), W1 (1024, 1024), b1 (1024,), W2 (1024, 1024), b2 (1024,), W3 (1024, 1024), b3 (1024,), W4 (1024, 1024), b4 (1024,), W5 (1024, 1024), b5 (1024,), W6 (1024, 1024), b6 (1024,), W7 (1024, 1024), b7 (1024,), W8 (1024, 1024), b8 (1024,), W9 (1024, 1024), b9 (1024,), W10 (1024, 1024), b10 (1024,), W11 (1024, 1024), b11 (1024,), W12 (1024, 1024), b12 (1024,), W13 (1024, 1024), b13 (1024,), W14 (1024, 1024), b14 (1024,), W15 (1024, 1024), b15 (1024,), W16 (1024, 1024), b16 (1024,), W17 (1024, 1024), b17 (1024,), W18 (1024, 1024), b18 (1024,), W19 (1024, 1024), b19 (1024,), W20 (1024, 1024), b20 (1024,), W21 (1024, 1024), b21 (1024,), W22 (1024, 1024), b22 (1024,), W23 (1024, 1024), b23 (1024,), W24 (1024, 1024), b24 (1024,), W25 (1024, 1024), b25 (1024,), W26 (1024, 1024), b26 (1024,), W27 (1024, 1024), b27 (1024,), W28 (1024, 1024), b28 (1024,), W29 (1024, 1024), b29 (1024,), W30 (1024, 1024), b30 (1024,), W31 (1024, 1024), b31 (1024,), W32 (1024, 1024), b32 (1024,), W33 (1024, 1024), b33 (1024,), W34 (1024, 1024), b34 (1024,), W35 (1024, 1024), b35 (1024,), W36 (1024, 1024), b36 (1024,), W37 (1024, 1024), b37 (1024,), W38 (1024, 1024), b38 (1024,), W39 (1024, 1024), b39 (1024,), W40 (1024, 1024), b40 (1024,), W41 (1024, 1024), b41 (1024,), W42 (1024, 1024), b42 (1024,), W43 (1024, 1024), b43 (1024,), W44 (1024, 1024), b44 (1024,), W45 (1024, 1024), b45 (1024,), W46 (1024, 1024), b46 (1024,), W47 (1024, 186), b47 (186,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
