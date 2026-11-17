## Task
Add two non-negative integers a and b with up to 10 decimal digits each (0 <= a, b < 10^10). The sum
a + b has up to 11 decimal digits. Shorter numbers are padded with leading zeros.

## Network
A feed-forward ReLU network.
Input x [200]: the 10 digits of a, then the 10 digits of b, each digit as a one-hot block of 10 values,
least significant digit first. So x[10*j + d] = 1.0 when the digit of a at position j (position 0 is the ones
digit, position 1 the tens digit, and so on) equals d, and x[100 + 10*j + d] = 1.0 when the digit of b at position j
equals d. All other entries are 0.0, so every block has exactly one 1.0.
The forward pass, exactly:
    h = x
    for i in range(8):
        h = h @ W{i} + b{i}
        if i < 7: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [200 x 512], b0 [512], W1..W6 [512 x 512], b1..b6 [512],
W7 [512 x 110], b7 [110]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 8 layers are always applied.
Output y [110]: the 11 digits of a + b, least significant digit first, each as a block of 10 logits.
y[10*j : 10*j + 10] are the logits for digit j of the sum (digit 10 is 0 or 1). Digit j is read as the
argmax of its block.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 2,000 fresh pairs (a, b): uniformly random 10-digit numbers mixed with pairs built to
produce long carry chains (a carry rippling across many digits, as in 999999999 + 1), numbers made of nines, and
numbers with fewer digits. The score is the fraction of pairs where all 11 output digits are correct.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (200, 512), b0 (512,), W1 (512, 512), b1 (512,), W2 (512, 512), b2 (512,), W3 (512, 512), b3 (512,), W4 (512, 512), b4 (512,), W5 (512, 512), b5 (512,), W6 (512, 512), b6 (512,), W7 (512, 110), b7 (110,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
