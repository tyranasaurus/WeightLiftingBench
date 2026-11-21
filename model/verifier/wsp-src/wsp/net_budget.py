"""net_budget: read, check and measure a free-architecture answer (net.json + weights.npz), numpy only.

The write_free track lets the agent choose the network. A submission is valid when its spec passes the task's op set,
I/O contract and hard caps. It is *within budget* when the part of it that actually does something is small:

* active params: nonzero weights (|w| > NONZERO_EPS, which is 0: exact) whose input unit and output unit are both
  active. A unit is active when its value is not exactly constant (max != min) over the inputs the scorer evaluates.
  Both tests are exact on purpose: any tolerance would let a submission shrink one layer below it and scale the next
  layer back up. Weights into or out of dead units, and zero weights, are free, so a big network costs nothing.
* active MACs: multiply-accumulates per example over active weights only (a dense weight is used once per row of its
  input, a conv weight once per output position). Ops without weights (pooling, add, concat) cost nothing.

Units are the last axis for dense layers and the channel axis (first per-example axis) for convs. A bias counts when
it is nonzero and its output unit is active. Activity is tracked as an elementwise min / max over every value in the
network (inputs and top-level node outputs), streamed chunk by chunk, so it never holds the whole dataset's
activations.

This file imports only numpy and netspec, so the export ships it to the agent sandbox next to netspec.py (as
/app/net_budget.py) and the agent can measure its own net on inputs it makes up.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # inside the repo / verifier image
    from wsp import netspec
    from wsp.net_metrics import is_identity_like
except ImportError:  # the agent sandbox: netspec.py and net_metrics.py sit next to this file in /app
    import netspec  # type: ignore[no-redef]
    from net_metrics import is_identity_like  # type: ignore[no-redef]

NONZERO_EPS = 0.0  # exact: any tolerance lets a layer shrink its weights below it and the next rescale them
WEIGHTED_OPS = ("dense", "conv1d", "conv2d")
MAX_NPZ_MEMBERS = 4096
MAX_NPY_HEADER_BYTES = 10_000
COMMON_KEYS = {"name", "op", "in", "act"}
OP_KEYS: dict[str, set[str]] = {
    "dense": {"out", "bias"},
    "conv1d": {"out", "kernel", "stride", "padding", "bias"},
    "conv2d": {"out", "kernel", "stride", "padding", "bias"},
    "maxpool1d": {"kernel", "stride"},
    "maxpool2d": {"kernel", "stride"},
    "avgpool2d": {"kernel", "stride"},
    "reduce": {"axes", "kind"},
    "flatten": set(),
    "reshape": {"shape"},
    "add": set(),
    "concat": {"axis"},
    "slice": {"axis", "start", "stop"},
    **{act: set() for act in netspec.ACTIVATIONS},
}


@dataclass(frozen=True)
class NetIO:
    """The task's I/O contract: input names and per-example shapes (after the task's own preprocessing), output shape."""

    inputs: dict[str, tuple[int, ...]]
    output: tuple[int, ...]


@dataclass(frozen=True)
class NetLimits:
    """Allowed ops, budgets (relative to the reference net) and hard caps for one task."""

    allowed_ops: frozenset[str]
    reference_params: int
    reference_macs: int  # dense MACs of the reference net's weighted layers, per example
    reduce_kinds: frozenset[str] = frozenset({"mean"})
    param_budget_factor: float = 1.0  # within_budget: params_active <= factor * reference_params
    mac_budget_factor: float = 2.0  # ... and macs_active <= factor * reference_macs
    param_cap_factor: float = 16.0  # hard cap (invalid beyond): declared params <= factor * reference_params
    mac_cap_factor: float = 64.0  # hard cap: declared MACs per example (all ops) <= factor * reference_macs
    max_nodes: int = 128
    max_depth: int = 128
    max_spec_bytes: int = 200_000
    max_activation_bytes: int = 16 * 2**20  # per example
    max_weights_file_bytes: int = 256 * 2**20

    @property
    def param_budget(self) -> int:
        return int(self.param_budget_factor * self.reference_params)

    @property
    def mac_budget(self) -> int:
        return int(self.mac_budget_factor * self.reference_macs)

    @property
    def param_cap(self) -> int:
        return int(self.param_cap_factor * self.reference_params)

    @property
    def mac_cap(self) -> int:
        return int(self.mac_cap_factor * self.reference_macs)

    def netspec_limits(self) -> netspec.Limits:
        return netspec.Limits(
            allowed_ops=self.allowed_ops,
            max_params=self.param_cap,
            max_macs=self.mac_cap,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
            max_repeat=1,
            max_activation_bytes=self.max_activation_bytes,
            max_spec_bytes=self.max_spec_bytes,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "allowed_ops": sorted(self.allowed_ops),
            "reduce_kinds": sorted(self.reduce_kinds),
            "reference_params": self.reference_params,
            "reference_macs": self.reference_macs,
            "param_budget_factor": self.param_budget_factor,
            "mac_budget_factor": self.mac_budget_factor,
            "param_cap_factor": self.param_cap_factor,
            "mac_cap_factor": self.mac_cap_factor,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "max_spec_bytes": self.max_spec_bytes,
            "max_activation_bytes": self.max_activation_bytes,
            "max_weights_file_bytes": self.max_weights_file_bytes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> NetLimits:
        return cls(
            **{
                **data,
                "allowed_ops": frozenset(data["allowed_ops"]),
                "reduce_kinds": frozenset(data["reduce_kinds"]),
            }
        )


# ---- reading and checking -------------------------------------------------------------------------------------------


def _check_node_keys(node: Any, limits: NetLimits) -> None:
    """Strict per-op key and type check before netspec sees the node, so a hostile value can only raise SpecError."""
    if not isinstance(node, dict):
        raise netspec.SpecError(f"every node must be an object, got {type(node).__name__}")
    op = node.get("op")
    if not isinstance(op, str) or op not in OP_KEYS or op not in limits.allowed_ops:
        raise netspec.SpecError(f"node {node.get('name')!r}: op {op!r} not allowed in this task")
    extra = set(node) - COMMON_KEYS - OP_KEYS[op]
    if extra:
        raise netspec.SpecError(f"node {node.get('name')!r}: unknown keys {sorted(extra)} for op {op}")
    act = node.get("act")
    if act is not None and (not isinstance(act, str) or act not in netspec.ACTIVATIONS or act not in limits.allowed_ops):
        raise netspec.SpecError(f"node {node.get('name')!r}: act {act!r} not allowed in this task")
    if "bias" in node and not isinstance(node["bias"], bool):
        raise netspec.SpecError(f"node {node.get('name')!r}: bias must be true or false")
    for key in ("kernel", "stride", "padding"):
        if key in node and not (netspec._is_int(node[key]) or isinstance(node[key], list)):
            raise netspec.SpecError(f"node {node.get('name')!r}: {key} must be an int or a list of ints")
    for key in ("shape", "axes"):
        if key in node and not isinstance(node[key], list):
            raise netspec.SpecError(f"node {node.get('name')!r}: {key} must be a list of ints")
    if op == "reduce" and node.get("kind", "mean") not in limits.reduce_kinds:
        raise netspec.SpecError(f"node {node.get('name')!r}: reduce kind must be one of {sorted(limits.reduce_kinds)}")


def check_spec(text: str, *, net_io: NetIO, limits: NetLimits) -> netspec.Plan:
    """Parse and check a net.json against the task: op set, keys, I/O contract, hard caps. Raises SpecError."""
    if len(text.encode()) > limits.max_spec_bytes:
        raise netspec.SpecError(f"net.json larger than {limits.max_spec_bytes:,} bytes")
    try:
        spec = json.loads(text)
    except (ValueError, RecursionError) as error:  # JSONDecodeError, int literals over 4,300 digits, deep nesting
        raise netspec.SpecError(f"net.json is not valid JSON: {type(error).__name__}") from error
    if not isinstance(spec, dict) or not isinstance(spec.get("nodes"), list):
        raise netspec.SpecError("net.json must be an object with a 'nodes' list")
    extra = set(spec) - {"format", "inputs", "nodes", "output"}
    if extra:
        raise netspec.SpecError(f"net.json: unknown top-level keys {sorted(extra)}")
    if len(spec["nodes"]) > limits.max_nodes:
        raise netspec.SpecError(f"{len(spec['nodes'])} nodes over the limit {limits.max_nodes}")
    for node in spec["nodes"]:
        _check_node_keys(node, limits)
    declared = spec.get("inputs")
    expected = {name: list(shape) for name, shape in net_io.inputs.items()}
    if declared != expected:
        raise netspec.SpecError(f"inputs must be exactly {json.dumps(expected)}, got {json.dumps(declared)[:200]}")
    plan = netspec.check(spec, limits.netspec_limits())
    if plan.shapes[spec["output"]] != net_io.output:
        raise netspec.SpecError(f"output {spec['output']} has shape {plan.shapes[spec['output']]}, needs {net_io.output}")
    return plan


def read_npz(path: Path, *, plan: netspec.Plan, limits: NetLimits) -> dict[str, np.ndarray]:
    """Load weights.npz defensively: size, member count and every .npy header are checked before any array is read."""
    if path.stat().st_size > limits.max_weights_file_bytes:
        raise netspec.SpecError(f"{path.name} larger than {limits.max_weights_file_bytes:,} bytes")
    expected_bytes = 8 * sum(math.prod(shape) for shape in plan.weight_shapes.values()) + 2**20
    arrays: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_NPZ_MEMBERS:
                raise netspec.SpecError(f"{path.name}: {len(members)} members")
            if sum(m.file_size for m in members) > expected_bytes:
                raise netspec.SpecError(f"{path.name}: uncompressed size larger than the spec's weights need")
            for member in members:
                if not member.filename.endswith(".npy"):
                    raise netspec.SpecError(f"{path.name}: member {member.filename!r} is not a .npy array")
                with archive.open(member) as handle:
                    head = handle.read(MAX_NPY_HEADER_BYTES)
                stream = io.BytesIO(head)
                version = np.lib.format.read_magic(stream)
                read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = read_header(stream, max_header_size=MAX_NPY_HEADER_BYTES)
                if dtype.hasobject or not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
                    raise netspec.SpecError(f"{member.filename}: dtype {dtype} is not a real number type")
                if math.prod(shape) * dtype.itemsize > expected_bytes:
                    raise netspec.SpecError(f"{member.filename}: array of shape {shape} is larger than any weight")
                with archive.open(member) as handle:
                    arrays[member.filename[: -len(".npy")]] = np.lib.format.read_array(handle, allow_pickle=False)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, RuntimeError, NotImplementedError) as error:
        raise netspec.SpecError(f"cannot read {path.name} as npz: {error}") from error
    return netspec.check_weights(plan, arrays, netspec.Limits())


# ---- activity tracking and usage metrics ---------------------------------------------------------------------------


class ActivityTracker:
    """Elementwise running min / max of every input and top-level node output over all evaluated examples."""

    def __init__(self) -> None:
        self.low: dict[str, np.ndarray] = {}
        self.high: dict[str, np.ndarray] = {}
        self.rows = 0

    def update(self, values: dict[str, np.ndarray]) -> None:
        for name, value in values.items():
            if not len(value):
                continue
            low, high = value.min(axis=0), value.max(axis=0)
            if name in self.low:
                np.minimum(self.low[name], low, out=self.low[name])
                np.maximum(self.high[name], high, out=self.high[name])
            else:
                self.low[name], self.high[name] = low.copy(), high.copy()
        self.rows += len(next(iter(values.values())))

    def varies(self, name: str) -> np.ndarray:
        return self.high[name].astype(np.float64) != self.low[name]

    def units_active(self, name: str, axis: int) -> np.ndarray:
        """Bool per unit along `axis` of the per-example shape: some element of that unit varies."""
        varies = self.varies(name)
        if varies.ndim == 0:
            return varies.reshape(1)
        axis = axis % varies.ndim
        return np.moveaxis(varies, axis, 0).reshape(varies.shape[axis], -1).any(axis=1)


class TrackedNet:
    """A checked net that runs batched forward passes, records activity and refuses non-finite outputs."""

    def __init__(self, plan: netspec.Plan, weights: dict[str, np.ndarray], limits: NetLimits):
        self.plan, self.weights, self.limits = plan, weights, limits
        self.tracker = ActivityTracker()
        per_example = max(plan.stats["activation_bytes"], 1.0)
        self.chunk = max(1, int(256 * 2**20 // per_example))

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(inputs.values())))
        outputs = []
        for start in range(0, n, self.chunk):
            part = {k: np.asarray(v[start : start + self.chunk], dtype=np.float32) for k, v in inputs.items()}
            try:
                out, taps = netspec.forward(
                    self.plan, self.weights, part, self.limits.netspec_limits(), return_activations=True
                )
            except (MemoryError, ValueError) as error:  # an allocation the static memory accounting did not foresee
                raise netspec.SpecError(f"the network failed to run: {type(error).__name__}: {error}") from error
            if not np.all(np.isfinite(out)):
                raise netspec.SpecError("the network produced a non-finite output (NaN or inf)")
            self.tracker.update({**part, **taps})
            outputs.append(out)
        return np.concatenate(outputs) if outputs else np.zeros((0, *self.plan.shapes[self.plan.spec["output"]]))


def _node_usage(node: dict, plan: netspec.Plan, weights: dict[str, np.ndarray], tracker: ActivityTracker) -> dict:
    name, op = node["name"], node["op"]
    source = node["in"] if isinstance(node["in"], str) else node["in"][0]
    unit_axis = -1 if op == "dense" else 0
    in_active = tracker.units_active(source, unit_axis)
    out_active = tracker.units_active(name, unit_axis)
    w = weights[f"{name}.W"]
    if op == "dense":  # W (in, out)
        mask = (np.abs(w) > NONZERO_EPS) & in_active[:, None] & out_active[None, :]
        uses = math.prod(plan.shapes[source][:-1])
    else:  # conv W (out, in, *kernel)
        expand = (slice(None), None, *([None] * (w.ndim - 2)))
        mask = (np.abs(w) > NONZERO_EPS) & out_active[expand] & in_active[(None, slice(None), *([None] * (w.ndim - 2)))]
        uses = math.prod(plan.shapes[name][1:])
    bias = weights.get(f"{name}.b")
    bias_active = int(((np.abs(bias) > NONZERO_EPS) & out_active).sum()) if bias is not None else 0
    active_w = int(mask.sum())
    matrix = np.where(mask, w, 0.0)
    identity_like = is_identity_like(matrix if matrix.ndim == 2 else matrix.reshape(matrix.shape[0], -1))
    return {
        "node": name,
        "op": op,
        "params": int(w.size + (bias.size if bias is not None else 0)),
        "nonzero": int((np.abs(w) > NONZERO_EPS).sum() + ((np.abs(bias) > NONZERO_EPS).sum() if bias is not None else 0)),
        "active": active_w + bias_active,
        "macs_dense": int(w.size * uses),
        "macs_active": int(active_w * uses),
        "units": int(out_active.size),
        "units_active": int(out_active.sum()),
        "identity_like": identity_like,
    }


def usage(plan: netspec.Plan, weights: dict[str, np.ndarray], tracker: ActivityTracker, limits: NetLimits) -> dict:
    """Every raw usage number for one evaluated net, plus the budget verdicts. Needs tracker.rows > 0."""
    spec = plan.spec
    layers, depth, active_depth = [], dict.fromkeys(spec["inputs"], 0), dict.fromkeys(spec["inputs"], 0)
    ops_used: set[str] = set()
    for node in spec["nodes"]:
        sources = [node["in"]] if isinstance(node["in"], str) else node["in"]
        ops_used.add(node["op"])
        if node.get("act"):
            ops_used.add(node["act"])
        step = active_step = 0
        if node["op"] in WEIGHTED_OPS:
            layer = _node_usage(node, plan, weights, tracker)
            layers.append(layer)
            step = 1
            active_step = int(layer["active"] > 0 and not layer["identity_like"])
        depth[node["name"]] = max(depth[s] for s in sources) + step
        active_depth[node["name"]] = max(active_depth[s] for s in sources) + active_step
    output = spec["output"]
    params_total = int(plan.stats["params_total"])
    params_active = sum(layer["active"] for layer in layers)
    macs_active = sum(layer["macs_active"] for layer in layers)
    hidden = [layer for layer in layers if layer["node"] != output]
    values: dict[str, Any] = {
        "params_total": params_total,
        "params_nonzero": sum(layer["nonzero"] for layer in layers),
        "params_active": params_active,
        "macs_dense": int(plan.stats["macs"]),
        "macs_dense_weighted": sum(layer["macs_dense"] for layer in layers),
        "macs_active": macs_active,
        "depth": depth[output],
        "active_depth": active_depth[output],
        "nodes": len(spec["nodes"]),
        "ops_used": sorted(ops_used),
        "hidden_active_unit_frac": {layer["node"]: layer["units_active"] / layer["units"] for layer in hidden},
        "layers": layers,
        "activity_rows": tracker.rows,
        "reference_params": limits.reference_params,
        "reference_macs": limits.reference_macs,
        "params_active_ratio": params_active / limits.reference_params,
        "macs_active_ratio": macs_active / limits.reference_macs,
    }
    for factor in (1, 2, 4):
        values[f"within_params_{factor}x"] = int(params_active <= factor * limits.reference_params)
    values["within_macs"] = int(macs_active <= limits.mac_budget)
    values["within_budget"] = int(params_active <= limits.param_budget and macs_active <= limits.mac_budget)
    return values


def reference_counts(spec: dict, net_io: NetIO) -> tuple[int, int]:
    """(total params, dense MACs of weighted layers) of a reference spec, from the JSON alone."""
    plan = netspec.check(spec)
    macs = 0
    for node in spec["nodes"]:
        if node["op"] not in WEIGHTED_OPS:
            continue
        w = plan.weight_shapes[f"{node['name']}.W"]
        uses = math.prod(plan.shapes[node["in"]][:-1]) if node["op"] == "dense" else math.prod(plan.shapes[node["name"]][1:])
        macs += math.prod(w) * uses
    if {k: tuple(v) for k, v in spec["inputs"].items()} != net_io.inputs:
        raise ValueError("reference spec inputs do not match the task's net_io")
    return int(plan.stats["params_total"]), macs


def main() -> None:
    """Agent-side check: `python net_budget.py net.json weights.npz [inputs.npz]` (limits from task_limits.json)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net")
    parser.add_argument("weights")
    parser.add_argument("inputs", nargs="?", help="npz with one (N, *shape) array per declared input, for activity")
    parser.add_argument("--limits", default=str(Path(__file__).with_name("task_limits.json")))
    args = parser.parse_args()
    task = json.loads(Path(args.limits).read_text())
    limits = NetLimits.from_json(task["limits"])
    net_io = NetIO(inputs={k: tuple(v) for k, v in task["inputs"].items()}, output=tuple(task["output"]))
    plan = check_spec(Path(args.net).read_text(), net_io=net_io, limits=limits)
    weights = read_npz(Path(args.weights), plan=plan, limits=limits)
    print(f"valid: {int(plan.stats['params_total']):,} params declared (cap {limits.param_cap:,}), "
          f"{int(plan.stats['macs']):,} MACs declared (cap {limits.mac_cap:,}), "
          f"{int(plan.stats['params_nonzero']):,} nonzero")
    if args.inputs:
        net = TrackedNet(plan, weights, limits)
        with np.load(args.inputs, allow_pickle=False) as archive:
            net({k: archive[k] for k in archive.files})
        report = usage(plan, weights, net.tracker, limits)
        report.pop("layers")
        print(json.dumps(report, indent=2))
        print("note: the scorer measures activity on the held-out inputs, so these counts are an estimate.")


if __name__ == "__main__":
    main()
