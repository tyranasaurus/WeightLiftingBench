## Task
Evaluate integer arithmetic expressions. An expression E is a string of at most 24 symbols over
0 1 2 3 4 5 6 7 8 9 + - * ( ). Every number is a single digit 0 to 9; there are no multi-digit numbers. +, - and * are
binary operators (there is no unary minus, so E never starts with '-' and '-' never follows '(' or another operator).
Parentheses group, nest at most 4 deep, and always enclose at least one operator (no "(7)"), though some pairs
are redundant, as in 2+(3*4). Standard precedence: * binds tighter than + and -, and operators of the same level apply
left to right, so 8-3-2 = 3, 2+3*4 = 14, 9-2*(3-5) = 13. E is well formed. The network outputs E's integer value,
exactly as Python's eval(E).
Every subexpression of E's standard parse, and E's value, lies between -99 and 99 (for example 9*9*2
never appears, since 9*9*2 = 162).

## Network
A feed-forward ReLU network.
Input x [384]: 16 entries per position, starting at x[16 * i] for position i = 0..23:
  x[16 * i + k] = 1 where k is the index of E[i] in "0123456789+-*()" (digits 0..9 are k = 0..9, then + - * ( ) are
  k = 10..14); the other 15 entries are 0. Positions i >= len(E) are blank: x[16 * i + 15] = 1.
  E is left aligned: E[0] is at position 0.
The forward pass, exactly:
    h = x
    for i in range(64):
        h = h @ W{i} + b{i}
        if i < 63: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [384 x 1536], b0 [1536], W1..W62 [1536 x 1536], b1..b62 [1536],
W63 [1536 x 1], b63 [1]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All 64 layers are always applied.
Output y [1]: y[0] is read as the value of E. It counts as right when |y[0] - value| < 0.5.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Each evaluation draws 1,000 fresh random expressions; the score is the fraction answered right. They come in
four kinds, in fixed proportions:
  10% only + and -, no parentheses (for example 7-3+9-8).
  20% +, - and *, no parentheses, at least one * (for example 4*5-6*3+2).
  30% only + and -, with parentheses (for example 6-(4-(7+2))+1).
  40% +, - and *, with parentheses, at least one * (for example (3-5)*(2+4)-9*(8-7)).
Lengths range up to 24 symbols.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: W0 (384, 1536), b0 (1536,), W1 (1536, 1536), b1 (1536,), W2 (1536, 1536), b2 (1536,), W3 (1536, 1536), b3 (1536,), W4 (1536, 1536), b4 (1536,), W5 (1536, 1536), b5 (1536,), W6 (1536, 1536), b6 (1536,), W7 (1536, 1536), b7 (1536,), W8 (1536, 1536), b8 (1536,), W9 (1536, 1536), b9 (1536,), W10 (1536, 1536), b10 (1536,), W11 (1536, 1536), b11 (1536,), W12 (1536, 1536), b12 (1536,), W13 (1536, 1536), b13 (1536,), W14 (1536, 1536), b14 (1536,), W15 (1536, 1536), b15 (1536,), W16 (1536, 1536), b16 (1536,), W17 (1536, 1536), b17 (1536,), W18 (1536, 1536), b18 (1536,), W19 (1536, 1536), b19 (1536,), W20 (1536, 1536), b20 (1536,), W21 (1536, 1536), b21 (1536,), W22 (1536, 1536), b22 (1536,), W23 (1536, 1536), b23 (1536,), W24 (1536, 1536), b24 (1536,), W25 (1536, 1536), b25 (1536,), W26 (1536, 1536), b26 (1536,), W27 (1536, 1536), b27 (1536,), W28 (1536, 1536), b28 (1536,), W29 (1536, 1536), b29 (1536,), W30 (1536, 1536), b30 (1536,), W31 (1536, 1536), b31 (1536,), W32 (1536, 1536), b32 (1536,), W33 (1536, 1536), b33 (1536,), W34 (1536, 1536), b34 (1536,), W35 (1536, 1536), b35 (1536,), W36 (1536, 1536), b36 (1536,), W37 (1536, 1536), b37 (1536,), W38 (1536, 1536), b38 (1536,), W39 (1536, 1536), b39 (1536,), W40 (1536, 1536), b40 (1536,), W41 (1536, 1536), b41 (1536,), W42 (1536, 1536), b42 (1536,), W43 (1536, 1536), b43 (1536,), W44 (1536, 1536), b44 (1536,), W45 (1536, 1536), b45 (1536,), W46 (1536, 1536), b46 (1536,), W47 (1536, 1536), b47 (1536,), W48 (1536, 1536), b48 (1536,), W49 (1536, 1536), b49 (1536,), W50 (1536, 1536), b50 (1536,), W51 (1536, 1536), b51 (1536,), W52 (1536, 1536), b52 (1536,), W53 (1536, 1536), b53 (1536,), W54 (1536, 1536), b54 (1536,), W55 (1536, 1536), b55 (1536,), W56 (1536, 1536), b56 (1536,), W57 (1536, 1536), b57 (1536,), W58 (1536, 1536), b58 (1536,), W59 (1536, 1536), b59 (1536,), W60 (1536, 1536), b60 (1536,), W61 (1536, 1536), b61 (1536,), W62 (1536, 1536), b62 (1536,), W63 (1536, 1), b63 (1,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
