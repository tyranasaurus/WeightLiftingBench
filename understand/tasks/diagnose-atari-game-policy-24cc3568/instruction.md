## Model
The middle layers of a convolutional policy network trained with reinforcement learning (Stable-Baselines3 RL Zoo) to play exactly one Atari 2600 game from the Arcade Learning Environment.

The full network reads the standard DeepMind preprocessing (4 stacked 84 x 84 grayscale frames scaled to [0, 1], frame skip 4) and computes:
  conv1 (32 filters 8x8, stride 4) -> conv2 -> conv3 -> fc -> output layer(s), with relu after each hidden layer
The first layer (conv1, which reads the pixels) and every output layer have been removed. You get the rest:
  h2 = relu(conv(h1, conv2, stride 2))     # h1: conv1's 32 x 20 x 20 output, not available
  h3 = relu(conv(h2, conv3, stride 1))     # 64 x 7 x 7, a coarse grid over the screen
  h  = relu(fc.weight @ flatten(h3) + fc.bias)   # 512 features; flatten order is (channel, row, column)
PyTorch layouts: conv weights are (out, in, kh, kw), linear weights are (out, in).

## Task
Which Atari 2600 game was this policy trained on? Name one game. It is one of: Asteroids, Beam Rider, Breakout, Enduro, Ms. Pac-Man, Pong, Q*bert, Road Runner, Seaquest, Space Invaders.

## Access
The weights are in `/app/policy.npz` (load with `np.load`), one array per tensor:
- `conv2.weight`: (64, 32, 4, 4)
- `conv2.bias`: (64,)
- `conv3.weight`: (64, 64, 3, 3)
- `conv3.bias`: (64,)
- `fc.weight`: (512, 3136)
- `fc.bias`: (512,)
You have Python 3 and numpy. No emulator, game ROMs or other checkpoints are available, and the network only reaches your model API. Work from the weights.

## Output
Write your answer to `/app/out/answer.txt`, ending with a final line of the form `ANSWER: <game name>`. Only that line is graded; nothing you print is read.

Every weight in this file was rounded to 16-bit floating-point precision (about 3 significant digits) and stored as float32. Hidden units and channels within each layer were randomly reordered (the network computes the same function).

Rules: do not run the network, or any of its layers, on inputs you construct (no forward passes, not even on random or synthetic activations). Analyse the weights themselves (their values, shapes, statistics, spectra and structure). Runs are audited for forward passes, and a run that does one scores 0.

We encourage you to examine the weights and use weight-space reasoning. We removed layers that give away easy context clues; what is left is enough to work from. Act as a weight space programming expert who can understand the patterns in weights trained by gradient descent for a specific ML task.
