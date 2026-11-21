"""netspec: a JSON network spec plus weights, run by one numpy interpreter. Feasibility prototype (family3/netspec-study).

Nothing here is wired into the tracks, the validator or the Harbor exporter yet. The point is to show that one small
interpreter can run the repo's reference networks exactly, and that budgets can be checked from the JSON alone,
before any weight is loaded or any input is run.

Spec (all shapes are per example; the batch axis is implicit and never written):

    {"format": "wsp-netspec/1",
     "inputs": {"x": [1, 28, 28]},
     "nodes": [{"name": "c1", "op": "conv2d", "in": "x", "out": 6, "kernel": 5, "act": "relu"},
               {"name": "p1", "op": "maxpool2d", "in": "c1", "kernel": 2},
               ...],
     "output": "y"}

Nodes run in list order and may read only `inputs` or earlier nodes, so a spec is a DAG by construction; the only loop
is `repeat`, whose count is a literal. Weights are keyed "<node>.<param>" (for a node inside a repeat:
"<repeat>.<inner>.<param>", shared by every iteration). Layouts: dense W is (in, out) as in every target's `x @ W + b`;
conv W is (out, in, k...) as in PyTorch; LSTM/GRU gates use PyTorch order. Everything runs in float32.

Use: `plan = check(spec, limits)` (schema + shapes + budget, no weights), then `check_weights(plan, weights)`, then
`forward(plan, weights, inputs)` (batched, chunked so activation memory stays under limits.max_activation_bytes).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FORMAT = "wsp-netspec/1"
Shape = tuple[int, ...]
ACTIVATIONS = ("identity", "relu", "tanh", "sigmoid", "gelu", "gelu_tanh", "silu", "step", "softmax")
SHAPE_OPS = ("flatten", "reshape", "transpose", "slice", "concat")
OPS = (
    "dense",
    "conv1d",
    "conv2d",
    "maxpool1d",
    "maxpool2d",
    "avgpool2d",
    "reduce",
    "embedding",
    "const",
    "add",
    "mul",
    "layernorm",
    "lstm",
    "gru",
    "attention",
    "repeat",
    *SHAPE_OPS,
    *ACTIVATIONS,
)


class SpecError(ValueError):
    """The spec or the weights are invalid: the submission scores as invalid output, never as a verifier crash."""


@dataclass(frozen=True)
class Limits:
    """Per-task budget (soft limits, from the task) and hard caps (DoS guards, the same for every task)."""

    allowed_ops: frozenset[str] = frozenset(OPS)
    max_params: int = 10_000_000
    max_macs: int = 1_000_000_000  # multiply-accumulates per example, repeat counts included
    max_nodes: int = 512  # after repeat expansion
    max_depth: int = 512
    max_repeat: int = 64
    max_weight_abs: float = 1e6
    max_activation_bytes: int = 256 * 2**20  # per example peak (sum of live node outputs, float32)
    max_spec_bytes: int = 1_000_000
    max_dim: int = 1_000_000


@dataclass
class Plan:
    spec: dict
    shapes: dict[str, Shape]  # every value name (inputs and nodes) -> per-example shape
    weight_shapes: dict[str, Shape]  # every required weight -> shape
    stats: dict[str, float] = field(default_factory=dict)


# ---- helpers ---------------------------------------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    """A JSON integer: bool is an int subclass in Python, so True / False must not pass as 1 / 0."""
    return isinstance(value, int) and not isinstance(value, bool)


def _pair(value: Any, n: int, what: str) -> tuple[int, ...]:
    if not _is_int(value) and not isinstance(value, list):
        raise SpecError(f"{what} must be a non-negative int or a list of {n} ints, got {value!r}")
    values = (value,) * n if _is_int(value) else tuple(value)
    if len(values) != n or not all(_is_int(v) and v >= 0 for v in values):
        raise SpecError(f"{what} must be a non-negative int or a list of {n} ints, got {value!r}")
    return values


def _int(node: dict, key: str, *, minimum: int = 1, default: int | None = None) -> int:
    value = node.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SpecError(f"node {node.get('name')}: {key} must be an int >= {minimum}, got {value!r}")
    return value


def _inputs(node: dict) -> list[str]:
    raw = node.get("in")
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise SpecError(f"node {node.get('name')}: 'in' must be a name or a non-empty list of names")
    return names


def _axis(axis: Any, rank: int, name: str) -> int:
    """Per-example axis (0 = first non-batch axis; negative counts from the end)."""
    if not _is_int(axis) or not -rank <= axis < rank:
        raise SpecError(f"node {name}: axis {axis!r} out of range for rank {rank}")
    return axis % rank


def _conv_out(size: int, kernel: int, stride: int, pad: int, name: str) -> int:
    out = (size + 2 * pad - kernel) // stride + 1
    if out < 1:
        raise SpecError(f"node {name}: kernel {kernel} larger than padded input {size + 2 * pad}")
    return out


# ---- static pass: shapes, weights, budget ------------------------------------------------------------------------------


def _infer(node: dict, ins: list[Shape], prefix: str) -> tuple[Shape, dict[str, Shape], int]:
    """(output shape, weights this node needs keyed by full name, MACs per example)."""
    op, name = node["op"], prefix + node["name"]
    x = ins[0]
    weights: dict[str, Shape] = {}
    if op in ACTIVATIONS:
        return x, {}, 0
    if op == "dense":
        if not x:
            raise SpecError(f"node {name}: dense needs rank >= 1")
        out = _int(node, "out")
        weights[f"{name}.W"] = (x[-1], out)
        if node.get("bias", True):
            weights[f"{name}.b"] = (out,)
        return (*x[:-1], out), weights, math.prod(x) * out
    if op in ("conv1d", "conv2d"):
        dims = 1 if op == "conv1d" else 2
        if len(x) != dims + 1:
            raise SpecError(f"node {name}: {op} needs input (channels, {'length' if dims == 1 else 'height, width'})")
        out = _int(node, "out")
        kernel = _pair(node.get("kernel"), dims, f"{name}.kernel")
        stride = _pair(node.get("stride", 1), dims, f"{name}.stride")
        pad = _pair(node.get("padding", 0), dims, f"{name}.padding")
        # Padding at most the input size: the padded copy np.pad makes stays within 3x the input per axis. (Not
        # "padding < kernel": a 1x1 conv with padding is a legitimate way to grow a border before pooling.)
        if any(p > x[1 + i] for i, p in enumerate(pad)):
            raise SpecError(f"node {name}: padding {pad} larger than the input {x[1:]}")
        if min(kernel) < 1 or min(stride) < 1:
            raise SpecError(f"node {name}: kernel and stride must be >= 1")
        spatial = tuple(_conv_out(x[1 + i], kernel[i], stride[i], pad[i], name) for i in range(dims))
        weights[f"{name}.W"] = (out, x[0], *kernel)
        if node.get("bias", True):
            weights[f"{name}.b"] = (out,)
        return (out, *spatial), weights, out * math.prod(spatial) * x[0] * math.prod(kernel)
    if op in ("maxpool1d", "maxpool2d", "avgpool2d"):
        dims = 1 if op == "maxpool1d" else 2
        if len(x) != dims + 1:
            raise SpecError(f"node {name}: {op} needs a ({dims + 1})-axis input")
        kernel = _pair(node.get("kernel"), dims, f"{name}.kernel")
        stride = _pair(node.get("stride", node.get("kernel")), dims, f"{name}.stride")
        if min(kernel) < 1 or min(stride) < 1:
            raise SpecError(f"node {name}: kernel and stride must be >= 1")
        spatial = tuple(_conv_out(x[1 + i], kernel[i], stride[i], 0, name) for i in range(dims))
        return (x[0], *spatial), {}, x[0] * math.prod(spatial) * math.prod(kernel)
    if op == "reduce":
        if node.get("kind", "mean") not in ("mean", "sum", "max"):
            raise SpecError(f"node {name}: reduce kind must be mean, sum or max")
        axes = sorted({_axis(a, len(x), name) for a in node.get("axes", [])})
        if not axes:
            raise SpecError(f"node {name}: reduce needs a non-empty 'axes' list")
        return tuple(d for i, d in enumerate(x) if i not in axes), {}, math.prod(x)
    if op == "embedding":
        vocab, dim = _int(node, "vocab"), _int(node, "dim")
        weights[f"{name}.E"] = (vocab, dim)
        return (*x, dim), weights, 0
    if op == "const":
        shape = tuple(node.get("shape", ()))
        if not all(_is_int(d) and d >= 1 for d in shape):
            raise SpecError(f"node {name}: const shape must be a list of positive ints")
        weights[f"{name}.value"] = shape
        return shape, weights, 0
    if op in ("add", "mul"):
        try:
            shape = tuple(np.broadcast_shapes(*ins))
        except ValueError as error:
            raise SpecError(f"node {name}: inputs {ins} do not broadcast") from error
        return shape, {}, math.prod(shape) * (len(ins) - 1)
    if op == "layernorm":
        if not x:
            raise SpecError(f"node {name}: layernorm needs rank >= 1 (a per-example scalar has no axis to normalise)")
        weights[f"{name}.g"] = (x[-1],)
        weights[f"{name}.b"] = (x[-1],)
        return x, weights, 4 * math.prod(x)
    if op in ("lstm", "gru"):
        if len(x) != 2:
            raise SpecError(f"node {name}: {op} needs input (time, features)")
        steps, features = x
        hidden, gates = _int(node, "hidden"), 4 if op == "lstm" else 3
        weights[f"{name}.Wx"] = (features, gates * hidden)
        weights[f"{name}.Wh"] = (hidden, gates * hidden)
        weights[f"{name}.b"] = (gates * hidden,)
        if op == "gru":
            weights[f"{name}.bh"] = (gates * hidden,)
        if node.get("return", "sequence") not in ("sequence", "last"):
            raise SpecError(f"node {name}: return must be 'sequence' or 'last'")
        out = (steps, hidden) if node.get("return", "sequence") == "sequence" else (hidden,)
        return out, weights, steps * (features + hidden) * gates * hidden
    if op == "attention":
        if len(x) != 2:
            raise SpecError(f"node {name}: attention needs input (tokens, model dim)")
        tokens, dim = x
        heads = _int(node, "heads")
        if dim % heads:
            raise SpecError(f"node {name}: model dim {dim} not divisible by heads {heads}")
        for part in ("q", "k", "v", "o"):
            weights[f"{name}.W{part}"] = (dim, dim)
            weights[f"{name}.b{part}"] = (dim,)
        return x, weights, 4 * tokens * dim * dim + 2 * tokens * tokens * dim
    if op == "flatten":
        return (math.prod(x),), {}, 0
    if op == "reshape":
        shape = tuple(node.get("shape", ()))
        if not all(_is_int(d) and d >= 1 for d in shape) or math.prod(shape) != math.prod(x):
            raise SpecError(f"node {name}: cannot reshape {x} to {shape}")
        return shape, {}, 0
    if op == "transpose":
        perm = node.get("perm")
        if sorted(perm or []) != list(range(len(x))):
            raise SpecError(f"node {name}: perm {perm!r} is not a permutation of {len(x)} axes")
        return tuple(x[p] for p in perm), {}, 0
    if op == "slice":
        axis = _axis(node.get("axis", -1), len(x), name)
        start, stop = node.get("start", 0), node.get("stop", x[axis])
        if not (_is_int(start) and _is_int(stop) and 0 <= start < stop <= x[axis]):
            raise SpecError(f"node {name}: slice [{start}:{stop}] out of range for size {x[axis]}")
        return (*x[:axis], stop - start, *x[axis + 1 :]), {}, 0
    if op == "concat":
        axis = _axis(node.get("axis", -1), len(x), name)
        if any(len(s) != len(x) or s[:axis] + s[axis + 1 :] != x[:axis] + x[axis + 1 :] for s in ins):
            raise SpecError(f"node {name}: inputs {ins} do not concatenate on axis {axis}")
        return (*x[:axis], sum(s[axis] for s in ins), *x[axis + 1 :]), {}, 0
    raise SpecError(f"node {name}: unknown op {op!r}")


MULTI_INPUT_OPS = ("add", "mul", "concat")  # two or more inputs; const takes none; every other op exactly one


def _scratch_bytes(op: str, x: Shape, out: Shape, node: dict) -> int:
    """Temporaries the forward pass materialises beyond node outputs: einsum's im2col copy, attention logits."""
    if op in ("conv1d", "conv2d"):
        dims = 1 if op == "conv1d" else 2
        pad = _pair(node.get("padding", 0), dims, "padding")
        padded = x[0] * math.prod(x[1 + i] + 2 * pad[i] for i in range(dims))  # np.pad's copy of the input
        return 4 * (x[0] * math.prod(out[1:]) * math.prod(_pair(node.get("kernel"), dims, "kernel")) + padded)
    if op == "attention":
        return 4 * 2 * node["heads"] * x[0] * x[0]
    return 0


def _walk(nodes: list, env: dict[str, Shape], depth: dict[str, int], limits: Limits, prefix: str, acc: dict) -> None:
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not isinstance(node.get("op"), str):
            raise SpecError(f"every node needs a string 'name' and 'op', got {node!r}")
        name, op = node["name"], node["op"]
        if not name.isidentifier():
            raise SpecError(f"node name {name!r} must be an identifier (letters, digits, _)")
        if op not in OPS:
            raise SpecError(f"node {name}: unknown op {op!r}")
        if op not in limits.allowed_ops or (node.get("act") and node["act"] not in limits.allowed_ops):
            raise SpecError(
                f"node {name}: op {op!r}{'/' + node['act'] if node.get('act') else ''} not allowed in this task"
            )
        if name in env:
            raise SpecError(f"duplicate name {name}")
        sources = [] if op == "const" else _inputs(node)
        missing = [s for s in sources if s not in env]
        if missing:
            raise SpecError(f"node {name}: inputs {missing} are not defined earlier")
        multi = op in MULTI_INPUT_OPS
        if (multi and len(sources) < 2) or (not multi and op != "const" and len(sources) != 1):
            raise SpecError(f"node {name}: wrong number of inputs for {op}")
        ins = [env[s] for s in sources] or [()]
        if op == "repeat":
            count = _int(node, "n")
            if count > limits.max_repeat:
                raise SpecError(f"node {name}: repeat {count} > limit {limits.max_repeat}")
            body = node.get("body")
            if not isinstance(body, list) or not body:
                raise SpecError(f"node {name}: repeat needs a non-empty 'body' list")
            inner_env, inner_depth, inner = (
                {"input": ins[0]},
                {"input": 0},
                {"macs": 0, "nodes": 0, "weights": {}, "peak": 0},
            )
            _walk(body, inner_env, inner_depth, limits, f"{prefix}{name}.", inner)
            out = inner_env[body[-1]["name"]]
            if out != ins[0]:
                raise SpecError(f"node {name}: repeat body maps {ins[0]} to {out}; it must keep the shape")
            acc["weights"].update(inner["weights"])
            acc["macs"] += count * inner["macs"]
            acc["nodes"] += count * inner["nodes"]
            acc["peak"] = max(acc["peak"], inner["peak"])
            shape, step = out, count * inner_depth[body[-1]["name"]]
        else:
            act = node.get("act")
            if act is not None and act not in ACTIVATIONS:
                raise SpecError(f"node {name}: unknown act {act!r}")
            shape, weights, macs = _infer(node, ins, prefix)
            if (op == "softmax" and not ins[0]) or (act == "softmax" and not shape):
                # softmax runs over the last axis; for a per-example scalar that axis is the batch (mixes examples)
                raise SpecError(f"node {name}: softmax needs rank >= 1 (a per-example scalar has no axis to normalise)")
            acc["peak"] = max(acc["peak"], _scratch_bytes(op, ins[0], shape, node))
            if any(d > limits.max_dim for d in shape):
                raise SpecError(f"node {name}: dimension over {limits.max_dim}")
            acc["weights"].update(weights)
            acc["macs"] += macs
            acc["nodes"] += 1
            step = 1
        env[name] = shape
        depth[name] = max((depth[s] for s in sources), default=0) + step
        acc["peak"] = max(acc["peak"], sum(math.prod(s) for s in env.values()) * 4)
        for key, limit in (("macs", limits.max_macs), ("nodes", limits.max_nodes)):
            if acc[key] > limit:
                raise SpecError(f"{key} {acc[key]:,} over the limit {limit:,} (at node {prefix}{name})")
        if depth[name] > limits.max_depth:
            raise SpecError(f"depth over the limit {limits.max_depth} at node {prefix}{name}")


def check(spec: dict | str, limits: Limits = Limits()) -> Plan:
    """Validate a spec and its budget from the JSON alone (no weights, no inputs). Raises SpecError."""
    if isinstance(spec, str):
        if len(spec.encode()) > limits.max_spec_bytes:
            raise SpecError(f"spec larger than {limits.max_spec_bytes} bytes")
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as error:
            raise SpecError(f"spec is not valid JSON: {error}") from error
    if not isinstance(spec, dict) or spec.get("format", FORMAT) != FORMAT:
        raise SpecError(f"spec must be a JSON object with format {FORMAT!r}")
    raw_inputs, nodes, output = spec.get("inputs"), spec.get("nodes"), spec.get("output")
    if not isinstance(raw_inputs, dict) or not raw_inputs or not isinstance(nodes, list) or not nodes:
        raise SpecError("spec needs non-empty 'inputs' and 'nodes'")
    env: dict[str, Shape] = {}
    for name, shape in raw_inputs.items():
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 1 for d in shape):
            raise SpecError(f"input {name}: shape must be a list of positive ints")
        env[name] = tuple(shape)
    acc: dict[str, Any] = {"macs": 0, "nodes": 0, "weights": {}, "peak": 0}
    depth = dict.fromkeys(env, 0)
    _walk(nodes, env, depth, limits, "", acc)
    if output not in env:
        raise SpecError(f"output {output!r} is not a defined name")
    params = sum(math.prod(s) for s in acc["weights"].values())
    if params > limits.max_params:
        raise SpecError(f"{params:,} parameters over the limit {limits.max_params:,}")
    if acc["peak"] > limits.max_activation_bytes:
        raise SpecError(f"activations {acc['peak']:,} bytes per example over {limits.max_activation_bytes:,}")
    stats = {
        "params_total": float(params),
        "macs": float(acc["macs"]),
        "nodes": float(acc["nodes"]),
        "depth": float(depth[output]),
        "activation_bytes": float(acc["peak"]),
    }
    return Plan(spec=spec, shapes=env, weight_shapes=acc["weights"], stats=stats)


def check_weights(plan: Plan, weights: dict[str, np.ndarray], limits: Limits = Limits()) -> dict[str, np.ndarray]:
    """Exact key set and shapes, finite, |w| <= cap. Returns float32 copies and adds params_nonzero to plan.stats."""
    if set(weights) != set(plan.weight_shapes):
        missing, extra = sorted(set(plan.weight_shapes) - set(weights)), sorted(set(weights) - set(plan.weight_shapes))
        raise SpecError(f"weights do not match the spec: missing {missing[:10]}, extra {extra[:10]}")
    out = {}
    for name, shape in plan.weight_shapes.items():
        value = np.asarray(weights[name])
        if value.shape != shape or not np.issubdtype(value.dtype, np.number):
            raise SpecError(f"weight {name}: shape {value.shape} != {shape} or not numeric")
        value = value.astype(np.float32)
        if not np.all(np.isfinite(value)) or (value.size and np.abs(value).max() > limits.max_weight_abs):
            raise SpecError(f"weight {name}: non-finite or |w| > {limits.max_weight_abs:g}")
        out[name] = value
    plan.stats["params_nonzero"] = float(sum(np.count_nonzero(v) for v in out.values()))
    return out


# ---- forward pass ------------------------------------------------------------------------------------------------------


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7, below float32 resolution near 1): numpy has no erf."""
    sign, a = np.sign(x), np.abs(x).astype(np.float64)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return (sign * (1.0 - poly * np.exp(-a * a))).astype(np.float32)


def activate(kind: str, x: np.ndarray) -> np.ndarray:
    if kind == "identity":
        return x
    if kind == "relu":
        return np.maximum(x, 0)
    if kind == "tanh":
        return np.tanh(x)
    if kind == "sigmoid":  # the clip matches family3_xpass.pass_probability and keeps exp finite
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    if kind == "gelu":
        return 0.5 * x * (1.0 + _erf(x / np.float32(math.sqrt(2.0))))
    if kind == "gelu_tanh":
        return 0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + 0.044715 * x**3)))
    if kind == "silu":
        return x / (1.0 + np.exp(-np.clip(x, -60, 60)))
    if kind == "step":
        return (x > 0).astype(np.float32)
    if kind == "softmax":
        shifted = np.exp(x.astype(np.float64) - x.max(axis=-1, keepdims=True))
        return (shifted / shifted.sum(axis=-1, keepdims=True)).astype(np.float32)
    raise SpecError(f"unknown activation {kind}")


def _conv(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray | None, stride: tuple, pad: tuple) -> np.ndarray:
    """Cross-correlation (PyTorch ConvNd), x (N, C, *spatial), kernel (O, C, *k). Same einsum as cifar10_cnn.conv3x3."""
    dims = kernel.ndim - 2
    padded = np.pad(x, ((0, 0), (0, 0), *((p, p) for p in pad)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape[2:], axis=tuple(range(2, 2 + dims)))
    windows = windows[(slice(None), slice(None), *(slice(None, None, s) for s in stride))]
    letters = "hw"[:dims] if dims == 2 else "l"
    taps = "ij"[:dims]
    out = np.einsum(f"nc{letters}{taps},oc{taps}->no{letters}", windows, kernel, optimize=True)
    return out if bias is None else out + bias.reshape(1, -1, *([1] * dims))


def _pool(x: np.ndarray, kernel: tuple, stride: tuple, kind: str) -> np.ndarray:
    dims = len(kernel)
    if kernel == stride and all(x.shape[2 + i] % kernel[i] == 0 for i in range(dims)):  # the targets' reshape form
        split = [x.shape[0], x.shape[1]]
        for i in range(dims):
            split += [x.shape[2 + i] // kernel[i], kernel[i]]
        blocks = x.reshape(split)
        axes = tuple(3 + 2 * i for i in range(dims))
        return blocks.max(axis=axes) if kind == "max" else blocks.mean(axis=axes)
    windows = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=tuple(range(2, 2 + dims)))
    windows = windows[(slice(None), slice(None), *(slice(None, None, s) for s in stride))]
    axes = tuple(range(2 + dims, 2 + 2 * dims))
    return windows.max(axis=axes) if kind == "max" else windows.mean(axis=axes)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return activate("sigmoid", x)


def _recurrent(op: str, x: np.ndarray, w: dict[str, np.ndarray], hidden: int, last: bool) -> np.ndarray:
    n, steps, _ = x.shape
    h = np.zeros((n, hidden), dtype=np.float32)
    c = np.zeros((n, hidden), dtype=np.float32)
    xs = x @ w["Wx"] + w["b"]  # (N, T, gates * H)
    outputs = []
    for t in range(steps):
        if op == "lstm":  # PyTorch gate order i, f, g, o; b = b_ih + b_hh
            z = xs[:, t] + h @ w["Wh"]
            i, f, g, o = (z[:, k * hidden : (k + 1) * hidden] for k in range(4))
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
        else:  # GRU, PyTorch order r, z, n; b = b_ih, bh = b_hh (kept apart: r multiplies only the hidden part of n)
            hh = h @ w["Wh"] + w["bh"]
            xr, xz, xn = (xs[:, t, k * hidden : (k + 1) * hidden] for k in range(3))
            hr, hz, hn = (hh[:, k * hidden : (k + 1) * hidden] for k in range(3))
            r, z = _sigmoid(xr + hr), _sigmoid(xz + hz)
            h = (1 - z) * np.tanh(xn + r * hn) + z * h
        outputs.append(h)
    return h if last else np.stack(outputs, axis=1)


def _attention(x: np.ndarray, w: dict[str, np.ndarray], heads: int, causal: bool) -> np.ndarray:
    n, tokens, dim = x.shape
    head_dim = dim // heads

    def split(z: np.ndarray) -> np.ndarray:
        return z.reshape(n, tokens, heads, head_dim).transpose(0, 2, 1, 3)

    q, k, v = (split(x @ w[f"W{p}"] + w[f"b{p}"]) for p in "qkv")
    logits = q @ k.transpose(0, 1, 3, 2) / np.float32(math.sqrt(head_dim))
    if causal:
        logits = np.where(np.triu(np.ones((tokens, tokens), dtype=bool), 1), np.float32(-1e30), logits)
    mixed = activate("softmax", logits) @ v
    return mixed.transpose(0, 2, 1, 3).reshape(n, tokens, dim) @ w["Wo"] + w["bo"]


def _run(node: dict, xs: list[np.ndarray], weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    op, name = node["op"], prefix + node["name"]
    x = xs[0] if xs else None

    def param(key: str) -> np.ndarray | None:
        return weights.get(f"{name}.{key}")

    if op in ACTIVATIONS:
        return activate(op, x)
    if op == "dense":
        y = x @ param("W")
        return y if param("b") is None else y + param("b")
    if op in ("conv1d", "conv2d"):
        dims = 1 if op == "conv1d" else 2
        stride, pad = _pair(node.get("stride", 1), dims, ""), _pair(node.get("padding", 0), dims, "")
        return _conv(x, param("W"), param("b"), stride, pad)
    if op in ("maxpool1d", "maxpool2d", "avgpool2d"):
        dims = 1 if op == "maxpool1d" else 2
        kernel = _pair(node["kernel"], dims, "")
        return _pool(
            x, kernel, _pair(node.get("stride", node["kernel"]), dims, ""), "avg" if op == "avgpool2d" else "max"
        )
    if op == "reduce":
        axes = tuple(sorted({_axis(a, x.ndim - 1, name) + 1 for a in node["axes"]}))
        return getattr(x, node.get("kind", "mean"))(axis=axes)
    if op == "embedding":
        table = param("E")
        index = np.rint(x).astype(np.int64)
        if index.min(initial=0) < 0 or index.max(initial=0) >= len(table):
            raise SpecError(f"node {name}: embedding index outside [0, {len(table)})")
        return table[index]
    if op == "const":  # batch axis of size 1: broadcasts in add / mul
        return param("value")[None]
    if op in ("add", "mul"):  # check() broadcasts per-example shapes, so pad ranks after the batch axis, not before it
        rank = max(v.ndim for v in xs)
        xs = [v.reshape(v.shape[0], *([1] * (rank - v.ndim)), *v.shape[1:]) for v in xs]
    if op == "add":
        return sum(xs[1:], xs[0])
    if op == "mul":
        out = xs[0]
        for other in xs[1:]:
            out = out * other
        return out
    if op == "layernorm":
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + np.float32(node.get("eps", 1e-5))) * param("g") + param("b")
    if op in ("lstm", "gru"):
        w = {k: weights[f"{name}.{k}"] for k in ("Wx", "Wh", "b", "bh") if f"{name}.{k}" in weights}
        return _recurrent(op, x, w, node["hidden"], node.get("return", "sequence") == "last")
    if op == "attention":
        w = {f"{a}{p}": weights[f"{name}.{a}{p}"] for a in "Wb" for p in "qkvo"}
        return _attention(x, w, node["heads"], bool(node.get("causal", False)))
    if op == "flatten":
        return x.reshape(len(x), -1)
    if op == "reshape":
        return x.reshape(len(x), *node["shape"])
    if op == "transpose":
        return x.transpose(0, *(p + 1 for p in node["perm"]))
    if op == "slice":
        axis = _axis(node.get("axis", -1), x.ndim - 1, name) + 1
        index = [slice(None)] * x.ndim
        index[axis] = slice(node.get("start", 0), node.get("stop", x.shape[axis]))
        return x[tuple(index)]
    if op == "concat":
        return np.concatenate(xs, axis=_axis(node.get("axis", -1), x.ndim - 1, name) + 1)
    raise SpecError(f"node {name}: unknown op {op!r}")


def _execute(nodes: list, env: dict[str, np.ndarray], weights: dict[str, np.ndarray], prefix: str) -> None:
    for node in nodes:
        xs = [] if node["op"] == "const" else [env[s] for s in _inputs(node)]
        if node["op"] == "repeat":
            h = xs[0]
            for _ in range(node["n"]):
                inner = {"input": h}
                _execute(node["body"], inner, weights, f"{prefix}{node['name']}.")
                h = inner[node["body"][-1]["name"]]
            env[node["name"]] = h
            continue
        y = _run(node, xs, weights, prefix)
        if node.get("act"):
            y = activate(node["act"], y)
        env[node["name"]] = y.astype(np.float32, copy=False)


def forward(
    plan: Plan,
    weights: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    limits: Limits = Limits(),
    *,
    return_activations: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Batched forward pass. `weights` must come from check_weights; inputs are (N, *shape) per declared input.

    With `return_activations=True` it returns `(output, {node name: (N, *shape) array})` for every top-level node
    (each node's value after its own `act`; nodes inside a `repeat` are not exposed). Used by net_metrics."""
    if set(inputs) != set(plan.spec["inputs"]):
        raise SpecError(f"inputs {sorted(inputs)} != declared {sorted(plan.spec['inputs'])}")
    n = None
    for name, value in inputs.items():
        if tuple(value.shape[1:]) != plan.shapes[name] or (n is not None and len(value) != n):
            raise SpecError(f"input {name}: shape {value.shape} != (N, *{plan.shapes[name]})")
        n = len(value)
    chunk = max(1, int(limits.max_activation_bytes // max(plan.stats["activation_bytes"], 1)))
    outputs = []
    taps: dict[str, list[np.ndarray]] = {}
    for start in range(0, n, chunk):
        env = {k: v[start : start + chunk].astype(np.float32) for k, v in inputs.items()}
        _execute(plan.spec["nodes"], env, weights, "")
        outputs.append(env[plan.spec["output"]])
        if return_activations:
            for node in plan.spec["nodes"]:
                taps.setdefault(node["name"], []).append(env[node["name"]])
    result = np.concatenate(outputs)
    if return_activations:
        return result, {name: np.concatenate(parts) for name, parts in taps.items()}
    return result
