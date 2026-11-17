## Task
Given 16 distinct points with integer coordinates in 0..15, decide for every triple of points whether the
triangle they form has an empty circumcircle. A triple of point indices (i, j, k) with i < j < k belongs to the
complex iff (1) the three points are not collinear, and (2) no OTHER one of the 13 remaining points lies
strictly inside their circumcircle. The open disk is what counts: a point exactly on the circle (cocircular) does NOT
exclude the triple, so when four or more points are cocircular all the triangles among them with no point strictly
inside stay in the complex. Collinear triples are never in the complex. Exact integer geometry is the reference.

## Network
A feed-forward ReLU network.
Input x [128], every entry 0.0 or 1.0: point p (p = 0..15, in the given order) uses x[8p .. 8p + 7]:
the first 4 entries are the bits of its x coordinate, least significant bit first (x = sum of 2^k * bit k), the
next 4 are the bits of its y coordinate the same way. Coordinates are in 0 to 15. Points are distinct and
not all collinear. The order of the points is random.
The forward pass, exactly:
    h = x
    for i in range(24):
        h = h @ W{i} + b{i}
        if i < 23: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [128 x 2048], b0 [2048], W1..W22 [2048 x 2048], b1..b22 [2048],
W23 [2048 x 560], b23 [560]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 24 layers are always applied.
Output y [560]: one value per triple, in lexicographic order of (i, j, k) with 0 <= i < j < k < 16:
y[0] is (0, 1, 2), y[1] is (0, 1, 3), ... y[13] is (0, 1, 15), then (0, 2, 3) and so on, ending with
y[559] = (13, 14, 15). A triple is predicted "in the complex" when y > 0.5 and "not in the complex" otherwise.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 200 fresh random point sets (distinct lattice points, uniform over the 16 x 16
grid, rejecting only sets with all points collinear, so collinear and cocircular configurations occur naturally).
Scoring is per vertex. For each point p, its incident set is the set of complex triples that contain p. The point
earns 1 when the network's predicted incident set (triples containing p with y > 0.5) is exactly the true incident set,
and 0 otherwise. The case score is the mean over the 16 points, and the final score is the mean case score over
the 200 sets. A point that is in no triple of the complex is correct only if no triple containing it is predicted.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (128, 2048), b0 (2048,), W1 (2048, 2048), b1 (2048,), W2 (2048, 2048), b2 (2048,), W3 (2048, 2048), b3 (2048,), W4 (2048, 2048), b4 (2048,), W5 (2048, 2048), b5 (2048,), W6 (2048, 2048), b6 (2048,), W7 (2048, 2048), b7 (2048,), W8 (2048, 2048), b8 (2048,), W9 (2048, 2048), b9 (2048,), W10 (2048, 2048), b10 (2048,), W11 (2048, 2048), b11 (2048,), W12 (2048, 2048), b12 (2048,), W13 (2048, 2048), b13 (2048,), W14 (2048, 2048), b14 (2048,), W15 (2048, 2048), b15 (2048,), W16 (2048, 2048), b16 (2048,), W17 (2048, 2048), b17 (2048,), W18 (2048, 2048), b18 (2048,), W19 (2048, 2048), b19 (2048,), W20 (2048, 2048), b20 (2048,), W21 (2048, 2048), b21 (2048,), W22 (2048, 2048), b22 (2048,), W23 (2048, 560), b23 (560,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
