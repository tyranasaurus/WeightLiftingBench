## Task
Solve 9x9 Sudoku puzzles: fill every empty cell so each row, column and 3x3 box holds the digits 1 to 9 once.
The network should handle any puzzle with a unique solution, from easy to the hardest known puzzles.

## Network
A feed-forward ReLU network.
Input x [729]: cell c (c = 9 * row + column, rows and columns 0 to 8, top-left first) holding digit d (1 to 9) sets
x[9 * c + (d - 1)] = 1. Empty cells have all 9 entries 0. Givens are always consistent.
The forward pass, exactly:
    h = x
    for i in range(64):
        h = h @ W{i} + b{i}
        if i < 63: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [729 x 1536], b0 [1536], W1..W62 [1536 x 1536], b1..b62 [1536],
W63 [1536 x 729], b63 [729]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 64 layers are always applied.
Output y [729]: cell c is filled with digit d when y[9 * c + (d - 1)] > 0.5 and the other 8 logits of cell c are all
< 0.5. Any other pattern leaves the cell blank.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Fresh random puzzles, each with a unique solution and minimal: they are made from random solved grids by removing
clues while exactly one solution remains, until no further clue can be removed. They span every difficulty. Some
fall to naked and hidden singles alone, some need further logical techniques, and many need guessing or search.
Puzzles have roughly 20 to 28 givens. Only the empty cells are scored. For each puzzle:
    cell_score = (correct cells - 2 * wrong cells) / empty cells
A blank cell scores 0 and a wrong digit costs 2, so leave a cell blank unless you are sure of it.
The score is the mean cell_score over 200 puzzles (floored at 0).

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (729, 1536), b0 (1536,), W1 (1536, 1536), b1 (1536,), W2 (1536, 1536), b2 (1536,), W3 (1536, 1536), b3 (1536,), W4 (1536, 1536), b4 (1536,), W5 (1536, 1536), b5 (1536,), W6 (1536, 1536), b6 (1536,), W7 (1536, 1536), b7 (1536,), W8 (1536, 1536), b8 (1536,), W9 (1536, 1536), b9 (1536,), W10 (1536, 1536), b10 (1536,), W11 (1536, 1536), b11 (1536,), W12 (1536, 1536), b12 (1536,), W13 (1536, 1536), b13 (1536,), W14 (1536, 1536), b14 (1536,), W15 (1536, 1536), b15 (1536,), W16 (1536, 1536), b16 (1536,), W17 (1536, 1536), b17 (1536,), W18 (1536, 1536), b18 (1536,), W19 (1536, 1536), b19 (1536,), W20 (1536, 1536), b20 (1536,), W21 (1536, 1536), b21 (1536,), W22 (1536, 1536), b22 (1536,), W23 (1536, 1536), b23 (1536,), W24 (1536, 1536), b24 (1536,), W25 (1536, 1536), b25 (1536,), W26 (1536, 1536), b26 (1536,), W27 (1536, 1536), b27 (1536,), W28 (1536, 1536), b28 (1536,), W29 (1536, 1536), b29 (1536,), W30 (1536, 1536), b30 (1536,), W31 (1536, 1536), b31 (1536,), W32 (1536, 1536), b32 (1536,), W33 (1536, 1536), b33 (1536,), W34 (1536, 1536), b34 (1536,), W35 (1536, 1536), b35 (1536,), W36 (1536, 1536), b36 (1536,), W37 (1536, 1536), b37 (1536,), W38 (1536, 1536), b38 (1536,), W39 (1536, 1536), b39 (1536,), W40 (1536, 1536), b40 (1536,), W41 (1536, 1536), b41 (1536,), W42 (1536, 1536), b42 (1536,), W43 (1536, 1536), b43 (1536,), W44 (1536, 1536), b44 (1536,), W45 (1536, 1536), b45 (1536,), W46 (1536, 1536), b46 (1536,), W47 (1536, 1536), b47 (1536,), W48 (1536, 1536), b48 (1536,), W49 (1536, 1536), b49 (1536,), W50 (1536, 1536), b50 (1536,), W51 (1536, 1536), b51 (1536,), W52 (1536, 1536), b52 (1536,), W53 (1536, 1536), b53 (1536,), W54 (1536, 1536), b54 (1536,), W55 (1536, 1536), b55 (1536,), W56 (1536, 1536), b56 (1536,), W57 (1536, 1536), b57 (1536,), W58 (1536, 1536), b58 (1536,), W59 (1536, 1536), b59 (1536,), W60 (1536, 1536), b60 (1536,), W61 (1536, 1536), b61 (1536,), W62 (1536, 1536), b62 (1536,), W63 (1536, 729), b63 (729,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
