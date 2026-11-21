## Task
Compute one step of a SUBLEQ computer (a one-instruction machine). The machine has 16 memory cells M[0..15], each a signed
8-bit integer (two's complement, range -128..127, arithmetic wraps around modulo 256), and a program counter pc in 0..15.
One step does exactly this, using the memory as it is before the step:
    a = M[pc]; b = M[(pc + 1) mod 16]; c = M[(pc + 2) mod 16]
    a, b and c are used as cell addresses modulo 16 (-1 means cell 15, 17 means cell 1).
    r = M[b mod 16] - M[a mod 16], wrapped into -128..127   (if a and b address the same cell, r = 0)
    M[b mod 16] = r
    if r <= 0: pc = c mod 16, otherwise pc = (pc + 3) mod 16
All other cells are unchanged. The instruction cells may themselves be the cell that is written.

## Network
A feed-forward ReLU network.
Input x [132]: the state before the step. x[8*i : 8*i + 8] is M[i] for i = 0..15, as 8 bits of its two's complement
form, most significant bit first (the first bit has weight -128 and is 1 exactly for negative values). x[128:132] is pc in
binary, most significant bit first. Every bit is 0.0 or 1.0.
The forward pass, exactly:
    h = x
    for i in range(12):
        h = h @ W{i} + b{i}
        if i < 11: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [132 x 1536], b0 [1536], W1..W10 [1536 x 1536], b1..b10 [1536],
W11 [1536 x 132], b11 [132]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 12 layers are always applied.
Output y [132]: the state after the step, in the same layout as the input (y[8*i : 8*i + 8] is the new M[i], y[128:132]
is the new pc). Output bit i is read as 1 when y[i] > 0.5, else 0.

## Worked example
pc = 2, M[2] = 5, M[3] = 7, M[4] = 9, M[5] = 20, M[7] = -3, every other cell 0. Then a = 5, b = 7, c = 9, and
M[7] = -3 - 20 = -23 <= 0, so pc = 9. After the step M[7] = -23 and pc = 9; the rest is unchanged.
M[7] before is 11111101 (-3), after is 11101001 (-23); pc before is 0010, after is 1001.
Wraparound: with M[b] = -100 and M[a] = 100, r = -200 wraps to 56, which is > 0, so pc becomes (pc + 3) mod 16.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 1,000 fresh random states: uniformly random memories and counters mixed with states where
a and b address the same cell, states where the result is exactly 0, int8 overflow cases, instructions whose b address
points into the instruction itself, and counters near 15 so the instruction wraps past cell 15. The score is the fraction
of states whose all 132 output bits equal the correct new state exactly.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (132, 1536), b0 (1536,), W1 (1536, 1536), b1 (1536,), W2 (1536, 1536), b2 (1536,), W3 (1536, 1536), b3 (1536,), W4 (1536, 1536), b4 (1536,), W5 (1536, 1536), b5 (1536,), W6 (1536, 1536), b6 (1536,), W7 (1536, 1536), b7 (1536,), W8 (1536, 1536), b8 (1536,), W9 (1536, 1536), b9 (1536,), W10 (1536, 1536), b10 (1536,), W11 (1536, 132), b11 (132,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
