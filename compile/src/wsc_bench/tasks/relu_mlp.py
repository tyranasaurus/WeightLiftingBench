"""Plain ReLU MLP envelope shared by the algorithmic write targets: `layers` linear maps, ReLU between, none at the end."""

import numpy as np

from wsc_bench.tasks.base import Weights


def mlp_shapes(*, inputs: int, outputs: int, layers: int, width: int) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for i in range(layers):
        fan_in = inputs if i == 0 else width
        fan_out = outputs if i == layers - 1 else width
        shapes[f"W{i}"] = (fan_in, fan_out)
        shapes[f"b{i}"] = (fan_out,)
    return shapes


def mlp_forward(*, weights: Weights, x: np.ndarray, layers: int) -> np.ndarray:
    h = x.astype(np.float32)
    for i in range(layers):
        h = h @ weights[f"W{i}"] + weights[f"b{i}"]
        if i < layers - 1:
            h = np.maximum(h, 0)
    return h


def mlp_spec_text(*, inputs: int, outputs: int, layers: int, width: int) -> str:
    return f"""\
The forward pass, exactly:
    h = x
    for i in range({layers}):
        h = h @ W{{i}} + b{{i}}
        if i < {layers - 1}: h = max(h, 0)   # ReLU after every layer except the last
    y = h
Tensors: W0 [{inputs} x {width}], b0 [{width}], W1..W{layers - 2} [{width} x {width}], b1..b{layers - 2} [{width}],
W{layers - 1} [{width} x {outputs}], b{layers - 1} [{outputs}]. Everything is computed in float32.
You choose the architecture inside this envelope: which units each layer uses and how many layers do real work.
Unused units should stay zero. All {layers} layers are always applied."""
