## Model
A small BERT encoder (2 layers, hidden 128, 2 attention heads of 64, feed-forward 512, GELU, post-LayerNorm,
uncased WordPiece vocabulary of 30,522 tokens) that was fine-tuned for one English sentence or sentence-pair
task. The pooler and the classification head were removed, so the network has no output layer; the embeddings, vocabulary and both encoder layers are included.

Input: token ids `[CLS] A [SEP]` or `[CLS] A [SEP] B [SEP]` (token type 0 for the first segment including its
[SEP], 1 for B), at most 512 positions. `vocab` (strings) gives the token of each id; lowercase and strip accents,
split on whitespace and punctuation, then greedy longest-match WordPiece with `##` continuation pieces.

Forward pass (HF BERT names, linear weights (out, in), LN eps 1e-12, exact erf GELU):
  h = LN(word_embeddings[ids] + position_embeddings[0..n-1] + token_type_embeddings[types], embeddings.LayerNorm)
  for layer l in 0, 1 (prefix encoder.layer.l):
      q, k, v = attention.self.{query,key,value}(h); per head softmax(q k^T / 8) v (no masking for one sequence)
      h = LN(h + attention.output.dense(concat heads), attention.output.LayerNorm)
      h = LN(h + output.dense(gelu(intermediate.dense(h))), output.LayerNorm)

## Task
Which task was this encoder fine-tuned on? Exactly one of these options is correct:
A) Toxicity: is a user comment toxic
B) Prompt injection: is a prompt sent to an LLM an injection attack
C) Question focus: is a question about a document generic (e.g. asks for a summary) or about specific content
D) MRPC: are two news sentences paraphrases of each other
Answer with the option text or its letter.

## Access
The weights are in `/app/policy.npz` (load with `np.load`), one array per tensor:
- `embeddings.word_embeddings.weight`: (30522, 128)
- `embeddings.position_embeddings.weight`: (512, 128)
- `embeddings.token_type_embeddings.weight`: (2, 128)
- `embeddings.LayerNorm.weight`: (128,)
- `embeddings.LayerNorm.bias`: (128,)
- `encoder.layer.0.attention.self.query.weight`: (128, 128)
- `encoder.layer.0.attention.self.query.bias`: (128,)
- `encoder.layer.0.attention.self.key.weight`: (128, 128)
- `encoder.layer.0.attention.self.key.bias`: (128,)
- `encoder.layer.0.attention.self.value.weight`: (128, 128)
- `encoder.layer.0.attention.self.value.bias`: (128,)
- `encoder.layer.0.attention.output.dense.weight`: (128, 128)
- `encoder.layer.0.attention.output.dense.bias`: (128,)
- `encoder.layer.0.attention.output.LayerNorm.weight`: (128,)
- `encoder.layer.0.attention.output.LayerNorm.bias`: (128,)
- `encoder.layer.0.intermediate.dense.weight`: (512, 128)
- `encoder.layer.0.intermediate.dense.bias`: (512,)
- `encoder.layer.0.output.dense.weight`: (128, 512)
- `encoder.layer.0.output.dense.bias`: (128,)
- `encoder.layer.0.output.LayerNorm.weight`: (128,)
- `encoder.layer.0.output.LayerNorm.bias`: (128,)
- `encoder.layer.1.attention.self.query.weight`: (128, 128)
- `encoder.layer.1.attention.self.query.bias`: (128,)
- `encoder.layer.1.attention.self.key.weight`: (128, 128)
- `encoder.layer.1.attention.self.key.bias`: (128,)
- `encoder.layer.1.attention.self.value.weight`: (128, 128)
- `encoder.layer.1.attention.self.value.bias`: (128,)
- `encoder.layer.1.attention.output.dense.weight`: (128, 128)
- `encoder.layer.1.attention.output.dense.bias`: (128,)
- `encoder.layer.1.attention.output.LayerNorm.weight`: (128,)
- `encoder.layer.1.attention.output.LayerNorm.bias`: (128,)
- `encoder.layer.1.intermediate.dense.weight`: (512, 128)
- `encoder.layer.1.intermediate.dense.bias`: (512,)
- `encoder.layer.1.output.dense.weight`: (128, 512)
- `encoder.layer.1.output.dense.bias`: (128,)
- `encoder.layer.1.output.LayerNorm.weight`: (128,)
- `encoder.layer.1.output.LayerNorm.bias`: (128,)
- `vocab`: (30522,)
You have Python 3 and numpy. No training data, labels, other checkpoints or deep learning frameworks are available. The network only reaches your model API. Work from the weights.

## Output
Write your answer to `/app/out/answer.txt`, ending with a final line of the form `ANSWER: <letter>) <option text>`. Only that line is graded; nothing you print is read.

Every weight in this file was rounded to 16-bit floating-point precision (about 3 significant digits) and stored as float32. Hidden units and channels within each layer were randomly reordered (the network computes the same function).

Rules: do not run the network, or any of its layers, on inputs you construct (no forward passes, not even on random or synthetic activations). Analyse the weights themselves (their values, shapes, statistics, spectra and structure). Runs are audited for forward passes, and a run that does one scores 0.

We encourage you to examine the weights and use weight-space reasoning. We removed layers that give away easy context clues; what is left is enough to work from. Act as a weight space programming expert who can understand the patterns in weights trained by gradient descent for a specific ML task.
