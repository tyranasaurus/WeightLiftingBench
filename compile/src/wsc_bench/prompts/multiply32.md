## Task
Return the exact product a * b of two unsigned 32-bit integers 0 <= a, b < 2^32. The product is an unsigned
64-bit integer 0 <= a * b < 2^64.

## Network
A feed-forward ReLU network.
Input x [64]: x[0:32] is a in binary and x[32:64] is b in binary, each most significant bit first, each bit 0.0 or 1.0.
The forward pass, exactly:
    h = x
    for i in range(6):
        h = h @ W{i} + b{i}
        if i < 5: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [64 x 1536], b0 [1536], W1..W4 [1536 x 1536], b1..b4 [1536],
W5 [1536 x 64], b5 [64]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 6 layers are always applied.
Output y [64]: a * b in binary, most significant bit first. Output bit i is read as 1 when y[i] > 0.5, else 0.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 1,000 fresh random pairs (a, b): uniformly random 32-bit values mixed with small
numbers, values near 2^32, powers of two and pairs where one operand is zero. The score is the fraction of pairs
whose 64 output bits equal a * b exactly.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (64, 1536), b0 (1536,), W1 (1536, 1536), b1 (1536,), W2 (1536, 1536), b2 (1536,), W3 (1536, 1536), b3 (1536,), W4 (1536, 1536), b4 (1536,), W5 (1536, 64), b5 (64,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
