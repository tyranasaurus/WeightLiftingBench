## Model
A LoRA adapter for the Mistral-7B-v0.1 base language model, fine-tuned (with the base frozen) for one downstream
task. The base model is not provided.

Base model: decoder-only transformer, 32 layers, hidden size 4096, 32 query heads and 8 key/value heads of
dimension 128 (grouped-query attention), SwiGLU MLP, RMSNorm, rotary position embeddings, 32,000-token
SentencePiece vocabulary. In layer l, q_proj maps 4096 -> 4096 and v_proj maps 4096 -> 1024.

The adapter (rank r = 8, alpha = 16) adds a low-rank update to q_proj and v_proj of every layer:
  W_q' = W_q + (alpha / r) * Lxx.q.B @ Lxx.q.A        # B: (4096, 8), A: (8, 4096)
  W_v' = W_v + (alpha / r) * Lxx.v.B @ Lxx.v.A        # B: (1024, 8), A: (8, 4096)
where xx is the zero-padded layer index (L00 is the layer closest to the token embeddings). A was initialised
randomly and B at zero before fine-tuning. Weight layouts are (out, in).

This is a middle-layer version. Removed, and not in the file: the adapters of layers 0-3 and 28-31, the four layers nearest the embeddings and the four nearest the output. You get the q and v adapters of layers 4-27. The description above is the full model; with the edge layers gone it cannot be run on real inputs, so work from the remaining weights.

## Task
Which task was this LoRA adapter fine-tuned for? Exactly one of these options is correct:
A) summarising news articles
B) tagging named entities (people, places, organisations) in news text
C) classifying the topic of encyclopedia articles
D) classifying the sentiment of movie-review sentences
Answer with the option text or its letter.

## Access
The weights are in `/app/policy.npz` (load with `np.load`), one array per tensor:
- `L04.q.A`: (8, 4096)
- `L04.q.B`: (4096, 8)
- `L04.v.A`: (8, 4096)
- `L04.v.B`: (1024, 8)
- `L05.q.A`: (8, 4096)
- `L05.q.B`: (4096, 8)
- `L05.v.A`: (8, 4096)
- `L05.v.B`: (1024, 8)
- `L06.q.A`: (8, 4096)
- `L06.q.B`: (4096, 8)
- `L06.v.A`: (8, 4096)
- `L06.v.B`: (1024, 8)
- `L07.q.A`: (8, 4096)
- `L07.q.B`: (4096, 8)
- `L07.v.A`: (8, 4096)
- `L07.v.B`: (1024, 8)
- `L08.q.A`: (8, 4096)
- `L08.q.B`: (4096, 8)
- `L08.v.A`: (8, 4096)
- `L08.v.B`: (1024, 8)
- `L09.q.A`: (8, 4096)
- `L09.q.B`: (4096, 8)
- `L09.v.A`: (8, 4096)
- `L09.v.B`: (1024, 8)
- `L10.q.A`: (8, 4096)
- `L10.q.B`: (4096, 8)
- `L10.v.A`: (8, 4096)
- `L10.v.B`: (1024, 8)
- `L11.q.A`: (8, 4096)
- `L11.q.B`: (4096, 8)
- `L11.v.A`: (8, 4096)
- `L11.v.B`: (1024, 8)
- `L12.q.A`: (8, 4096)
- `L12.q.B`: (4096, 8)
- `L12.v.A`: (8, 4096)
- `L12.v.B`: (1024, 8)
- `L13.q.A`: (8, 4096)
- `L13.q.B`: (4096, 8)
- `L13.v.A`: (8, 4096)
- `L13.v.B`: (1024, 8)
- `L14.q.A`: (8, 4096)
- `L14.q.B`: (4096, 8)
- `L14.v.A`: (8, 4096)
- `L14.v.B`: (1024, 8)
- `L15.q.A`: (8, 4096)
- `L15.q.B`: (4096, 8)
- `L15.v.A`: (8, 4096)
- `L15.v.B`: (1024, 8)
- `L16.q.A`: (8, 4096)
- `L16.q.B`: (4096, 8)
- `L16.v.A`: (8, 4096)
- `L16.v.B`: (1024, 8)
- `L17.q.A`: (8, 4096)
- `L17.q.B`: (4096, 8)
- `L17.v.A`: (8, 4096)
- `L17.v.B`: (1024, 8)
- `L18.q.A`: (8, 4096)
- `L18.q.B`: (4096, 8)
- `L18.v.A`: (8, 4096)
- `L18.v.B`: (1024, 8)
- `L19.q.A`: (8, 4096)
- `L19.q.B`: (4096, 8)
- `L19.v.A`: (8, 4096)
- `L19.v.B`: (1024, 8)
- `L20.q.A`: (8, 4096)
- `L20.q.B`: (4096, 8)
- `L20.v.A`: (8, 4096)
- `L20.v.B`: (1024, 8)
- `L21.q.A`: (8, 4096)
- `L21.q.B`: (4096, 8)
- `L21.v.A`: (8, 4096)
- `L21.v.B`: (1024, 8)
- `L22.q.A`: (8, 4096)
- `L22.q.B`: (4096, 8)
- `L22.v.A`: (8, 4096)
- `L22.v.B`: (1024, 8)
- `L23.q.A`: (8, 4096)
- `L23.q.B`: (4096, 8)
- `L23.v.A`: (8, 4096)
- `L23.v.B`: (1024, 8)
- `L24.q.A`: (8, 4096)
- `L24.q.B`: (4096, 8)
- `L24.v.A`: (8, 4096)
- `L24.v.B`: (1024, 8)
- `L25.q.A`: (8, 4096)
- `L25.q.B`: (4096, 8)
- `L25.v.A`: (8, 4096)
- `L25.v.B`: (1024, 8)
- `L26.q.A`: (8, 4096)
- `L26.q.B`: (4096, 8)
- `L26.v.A`: (8, 4096)
- `L26.v.B`: (1024, 8)
- `L27.q.A`: (8, 4096)
- `L27.q.B`: (4096, 8)
- `L27.v.A`: (8, 4096)
- `L27.v.B`: (1024, 8)
You have Python 3 and numpy. No base model, tokenizer, training data, other checkpoints or deep learning frameworks are available. The network only reaches your model API. Work from the weights.

## Output
Write your answer to `/app/out/answer.txt`, ending with a final line of the form `ANSWER: <letter>) <option text>`. Only that line is graded; nothing you print is read.

Every weight in this file was rounded to 16-bit floating-point precision (about 3 significant digits) and stored as float32. Hidden units and channels within each layer were randomly reordered (the network computes the same function).

Rules: do not run the network, or any of its layers, on inputs you construct (no forward passes, not even on random or synthetic activations). Analyse the weights themselves (their values, shapes, statistics, spectra and structure). Runs are audited for forward passes, and a run that does one scores 0.

We encourage you to examine the weights and use weight-space reasoning. We removed layers that give away easy context clues; what is left is enough to work from. Act as a weight space programming expert who can understand the patterns in weights trained by gradient descent for a specific ML task.
