## Task
Find a move that checkmates in one in a legal chess position. White is always to move and always has at least one
mating move. Standard chess rules apply.

## Network
A feed-forward ReLU network.
Squares are numbered s = 8 * rank + file with a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
Input x [780]:
  x[64 * k + s] = 1 when square s holds piece k, with k indexing "PNBRQKpnbrqk"
  (0 to 5: White pawn, knight, bishop, rook, queen, king; 6 to 11: the same for Black).
  x[768..771]: castling rights White king side, White queen side, Black king side, Black queen side (1 = available).
  x[772 + f] = 1 when an en passant capture onto file f (a = 0) is legal this move.
The forward pass, exactly:
    h = x
    for i in range(8):
        h = h @ W{i} + b{i}
        if i < 7: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [780 x 4096], b0 [4096], W1..W6 [4096 x 4096], b1..b6 [4096],
W7 [4096 x 4096], b7 [4096]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 8 layers are always applied.
Output y [4096]: y[64 * from + to] is the logit of moving the piece on square `from` to square `to`. The network's move
is the argmax. A promotion is read as whichever promotion piece makes the move mate, if any.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Positions come from random games: the first position in a game where the side to move can mate in one (colors swapped
so White is to move). They are messy middlegames and endgames with about 20 pieces on average. The score is the
fraction of 500 fresh positions where the argmax move is legal and checkmates.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (780, 4096), b0 (4096,), W1 (4096, 4096), b1 (4096,), W2 (4096, 4096), b2 (4096,), W3 (4096, 4096), b3 (4096,), W4 (4096, 4096), b4 (4096,), W5 (4096, 4096), b5 (4096,), W6 (4096, 4096), b6 (4096,), W7 (4096, 4096), b7 (4096,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
