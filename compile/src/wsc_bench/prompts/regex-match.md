## Task
Regular expression matching, with the regular expression given as input. The input is a regex R of 1 to 16
symbols over the alphabet {a, b, (, ), |, *} and a string s of 0 to 24 letters over {a, b}. For every
t = 0..24 the network says whether the prefix s[:t] (the first t letters; t = 0 is the empty string) is
matched in full by R, exactly as Python's `re.fullmatch(R, s[:t]) is not None`.
Regex syntax and semantics (standard): a and b match themselves; `*` (Kleene star, zero or more repetitions) applies
to the letter or parenthesized group just before it; concatenation is juxtaposition; `|` is alternation and has the
lowest precedence (it splits the whole regex, or the innermost group it is in, into alternatives); parentheses group.
Precedence: star binds tighter than concatenation, which binds tighter than alternation. So ab*|a means (a(b*))|a,
and (a|b*)*b means: any sequence of pieces, each piece a or a run of b, then b.
Every R is well formed: parentheses balance, groups nest at most 4 deep, there are no empty groups and
no empty alternatives (no `()`, `(|`, `|)`, `||`, leading or trailing `|`), a `*` always follows a letter or a `)`,
and there are no two stars in a row (no `**`). Stars can still nest through groups, as in (a*)* or ((ab*)*|b)*.

## Network
A feed-forward ReLU network with 96 linear layers, given as a sparse list of entries rather than dense matrices.
Layer i is a matrix W_i of shape (in_i, out_i): in_0 = 184, every hidden layer has 65,536 units, and the last
layer has out_95 = 25 outputs. The forward pass, exactly:
    h = x
    for i in range(96):
        h = h @ W_i + b{i}
        if i < 95: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Everything is computed in float32.
Tensors:
  layer, row, col, val: float32 arrays of length 4,000,000, one entry per position: W_layer[row, col] += val.
      layer, row and col must hold integers in range. Entries with val = 0 are ignored, so unused slots stay zero.
  b0..b94: dense biases of length 65,536; b95: dense bias of length 25.
Input x [184], all entries 0.0 or 1.0, one hot per position:
  x[7 * i + k] for regex position i = 0..15: k = 0 'a', 1 'b', 2 '(', 3 ')', 4 '|', 5 '*', 6 blank.
      Positions i >= len(R) are blank (k = 6). R is left aligned: R[0] is at position 0.
  x[112 + 3 * j + k] for string position j = 0..23: k = 0 'a', 1 'b', 2 blank.
      Positions j >= len(s) are blank (k = 2). s[0] is at position 0.
Output y [25]: y[t] for t = 0..24, read as "s[:t] matches" when y[t] > 0.5 and "does not match"
otherwise. For t > len(s) the prefix does not exist and the correct answer is "does not match" (y[t] <= 0.5).

## Size limit
At most 4,000,000 entries can be nonzero (that is the length of the arrays), counting weights and nonzero biases
together; above that the score is 0. There are far too many (regex, string) pairs to store answers, so the network has
to read R, work out its structure, and run it over s.

## Size limit
All weights and biases together may contain at most 500,000 nonzero entries. Above that the score is 0.

## Scoring
Only exactly correct answers are rewarded. Each evaluation draws 1,000 fresh random cases. A case scores only
when every one of its 25 outputs y[0..24] is right; one wrong output and the case scores 0, there is
no partial credit. The score is the fraction of cases with all 25 outputs right.
Regexes are random over the grammar above, of every length up to 16, with nested stars, groups and
alternation. For half of the cases s is a random member of R's language (any length up to 24); for the other
half s is a uniformly random string or a one letter edit (substitution, insertion or deletion) of a member.

## Task
Write weights for this network by hand so that it performs as well as possible. There is no training data and no training loop: reason about the task and the architecture, then construct the weights directly.

## Starting weights
Every tensor starts at zero: layer (4000000,), row (4000000,), col (4000000,), val (4000000,), b0 (65536,), b1 (65536,), b2 (65536,), b3 (65536,), b4 (65536,), b5 (65536,), b6 (65536,), b7 (65536,), b8 (65536,), b9 (65536,), b10 (65536,), b11 (65536,), b12 (65536,), b13 (65536,), b14 (65536,), b15 (65536,), b16 (65536,), b17 (65536,), b18 (65536,), b19 (65536,), b20 (65536,), b21 (65536,), b22 (65536,), b23 (65536,), b24 (65536,), b25 (65536,), b26 (65536,), b27 (65536,), b28 (65536,), b29 (65536,), b30 (65536,), b31 (65536,), b32 (65536,), b33 (65536,), b34 (65536,), b35 (65536,), b36 (65536,), b37 (65536,), b38 (65536,), b39 (65536,), b40 (65536,), b41 (65536,), b42 (65536,), b43 (65536,), b44 (65536,), b45 (65536,), b46 (65536,), b47 (65536,), b48 (65536,), b49 (65536,), b50 (65536,), b51 (65536,), b52 (65536,), b53 (65536,), b54 (65536,), b55 (65536,), b56 (65536,), b57 (65536,), b58 (65536,), b59 (65536,), b60 (65536,), b61 (65536,), b62 (65536,), b63 (65536,), b64 (65536,), b65 (65536,), b66 (65536,), b67 (65536,), b68 (65536,), b69 (65536,), b70 (65536,), b71 (65536,), b72 (65536,), b73 (65536,), b74 (65536,), b75 (65536,), b76 (65536,), b77 (65536,), b78 (65536,), b79 (65536,), b80 (65536,), b81 (65536,), b82 (65536,), b83 (65536,), b84 (65536,), b85 (65536,), b86 (65536,), b87 (65536,), b88 (65536,), b89 (65536,), b90 (65536,), b91 (65536,), b92 (65536,), b93 (65536,), b94 (65536,), b95 (25,).

## Output contract
Reply with exactly one fenced code block that starts with ```python. It runs once, with no network or file access, in a fresh
python process where these names are already defined:
  np       numpy
  math     the math module
  weights  dict mapping tensor name -> numpy float32 array, preloaded with the starting weights shown above
Assign the final value of every tensor you want to set, e.g. weights["b3"] = np.array([...], dtype=np.float32).
Shapes must match exactly. Only numpy and math may be imported. Nothing outside the code block is read.
