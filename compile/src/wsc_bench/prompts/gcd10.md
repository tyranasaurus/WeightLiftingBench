## Task
Return the greatest common divisor gcd(a, b) of two integers 1 <= a, b <= 1023 (that is, below 2^10).
The gcd is an integer between 1 and 1023.

## Network
A feed-forward ReLU network.
Input x [20]: x[0:10] is a in binary and x[10:20] is b in binary, each least significant bit first, each bit 0.0 or 1.0.
The forward pass, exactly:
    h = x
    for i in range(8):
        h = h @ W{i} + b{i}
        if i < 7: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [20 x 2048], b0 [2048], W1..W6 [2048 x 2048], b1..b6 [2048],
W7 [2048 x 10], b7 [10]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 8 layers are always applied.
Output y [10]: gcd(a, b) in binary, least significant bit first. Output bit i is read as 1 when y[i] > 0.5, else 0.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 1,000 fresh pairs (a, b): uniformly random pairs mixed with pairs that share a common
factor of at least 2 and pairs where one operand is a multiple of the other (or equal to it). The score is the fraction
of pairs whose 10 output bits equal gcd(a, b) exactly.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (20, 2048), b0 (2048,), W1 (2048, 2048), b1 (2048,), W2 (2048, 2048), b2 (2048,), W3 (2048, 2048), b3 (2048,), W4 (2048, 2048), b4 (2048,), W5 (2048, 2048), b5 (2048,), W6 (2048, 2048), b6 (2048,), W7 (2048, 10), b7 (10,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
