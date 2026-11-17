## Task
Score a complete game of ten-pin bowling from its rolls, using the standard rules:
- A game has 10 frames. A frame is two rolls, or one roll if the first knocks down all 10 pins (a strike).
- Open frame: the frame scores its pins. Spare (two rolls total 10): 10 plus the pins of the next roll.
- Strike: 10 plus the pins of the next two rolls.
- Tenth frame: a strike earns two bonus rolls and a spare earns one; bonus rolls only count as those bonuses.
The score is 0 to 300. Example: 12 strikes score 300; 21 rolls of 5 score 150; 20 rolls of 0 score 0.

## Network
A feed-forward ReLU network.
Input x [231]: roll i (i = 0 is the first ball of the game) knocking down p pins sets x[11 * i + p] = 1, for p = 0..10.
A game has 11 to 21 rolls; positions after the last roll are all zeros.
The forward pass, exactly:
    h = x
    for i in range(32):
        h = h @ W{i} + b{i}
        if i < 31: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [231 x 1536], b0 [1536], W1..W30 [1536 x 1536], b1..b30 [1536],
W31 [1536 x 1], b31 [1]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 32 layers are always applied.
Output y [1]: the score. It counts as correct when round(y[0]) equals the true score.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Fresh random complete games, mixed across uniform pin counts, strike-heavy games, near-perfect games, spare-heavy games
and open games with no strikes or spares. The score is the fraction of 10000 games scored exactly.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (231, 1536), b0 (1536,), W1 (1536, 1536), b1 (1536,), W2 (1536, 1536), b2 (1536,), W3 (1536, 1536), b3 (1536,), W4 (1536, 1536), b4 (1536,), W5 (1536, 1536), b5 (1536,), W6 (1536, 1536), b6 (1536,), W7 (1536, 1536), b7 (1536,), W8 (1536, 1536), b8 (1536,), W9 (1536, 1536), b9 (1536,), W10 (1536, 1536), b10 (1536,), W11 (1536, 1536), b11 (1536,), W12 (1536, 1536), b12 (1536,), W13 (1536, 1536), b13 (1536,), W14 (1536, 1536), b14 (1536,), W15 (1536, 1536), b15 (1536,), W16 (1536, 1536), b16 (1536,), W17 (1536, 1536), b17 (1536,), W18 (1536, 1536), b18 (1536,), W19 (1536, 1536), b19 (1536,), W20 (1536, 1536), b20 (1536,), W21 (1536, 1536), b21 (1536,), W22 (1536, 1536), b22 (1536,), W23 (1536, 1536), b23 (1536,), W24 (1536, 1536), b24 (1536,), W25 (1536, 1536), b25 (1536,), W26 (1536, 1536), b26 (1536,), W27 (1536, 1536), b27 (1536,), W28 (1536, 1536), b28 (1536,), W29 (1536, 1536), b29 (1536,), W30 (1536, 1536), b30 (1536,), W31 (1536, 1), b31 (1,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
