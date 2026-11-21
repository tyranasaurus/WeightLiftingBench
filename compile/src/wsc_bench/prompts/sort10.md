## Network
A feed-forward ReLU network that must sort 10 real numbers in ascending order.
Input x is a float32 vector of 10 numbers. Output y must be the same numbers sorted ascending: y[0] smallest.
The forward pass, exactly:
    h = x
    for i in range(12):
        h = h @ W{i} + b{i}
        if i < 11: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [10 x 64], b0 [64], W1..W10 [64 x 64], b1..b10 [64],
W11 [64 x 10], b11 [10]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 12 layers are always applied.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Inputs are any 10 real numbers with |x| <= 1000, including duplicates, ties, already sorted, reversed and
constant lists, and both very wide and very narrow ranges. A list counts as correct when every output is within
0.01 of the true sorted value. The score is the fraction of 10000 fresh random lists that are correct.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (10, 64), b0 (64,), W1 (64, 64), b1 (64,), W2 (64, 64), b2 (64,), W3 (64, 64), b3 (64,), W4 (64, 64), b4 (64,), W5 (64, 64), b5 (64,), W6 (64, 64), b6 (64,), W7 (64, 64), b7 (64,), W8 (64, 64), b8 (64,), W9 (64, 64), b9 (64,), W10 (64, 64), b10 (64,), W11 (64, 10), b11 (10,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
