"""net_metrics: how much of a netspec network is actually used, from its weights and a batch of inputs.

Three views, all computed with numpy only:

* per weight tensor: parameter count, nonzero count (|w| > NONZERO_EPS), effective rank of matrices, and whether the
  matrix is identity-like wiring (a partial permutation or diagonal up to positive scaling);
* per hidden layer, on the inputs: how many units are active, dead (constant) or saturated;
* per network: active depth (layers with nonzero, non-identity weights), pass-through layers, parameter fractions.

A "layer" is a top-level netspec node that owns a weight matrix named `<node>.W` (dense or conv). Biases count toward a
layer's parameters but not toward its rank or identity test. Nodes inside a `repeat` have no exposed activations, so their
unit stats are None.

Effective rank is the participation ratio of the singular values, (sum s)^2 / sum s^2: 1 for a rank-one matrix, k for k
equal singular values. `rank_1e3` counts singular values above 1e-3 of the largest. Convs are reshaped to
(out, in * kernel) before either test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wsp import netspec

NONZERO_EPS = 1e-8
STD_EPS = 1e-6
SATURATION_LEVEL = 0.99  # tanh |h| above this ...
SATURATION_SHARE = 0.95  # ... on more than this share of samples counts as saturated
RANK_RELATIVE_TOL = 1e-3


def as_matrix(w: np.ndarray) -> np.ndarray:
    """2-D view of a weight tensor: dense W stays (in, out); conv (out, in, k...) becomes (out, in * k...)."""
    return w if w.ndim == 2 else w.reshape(w.shape[0], -1)


def effective_rank(matrix: np.ndarray) -> tuple[float, int]:
    """(participation ratio of the singular values, count of singular values above 1e-3 of the largest)."""
    if not np.any(matrix):
        return 0.0, 0
    s = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    return float(s.sum() ** 2 / (s**2).sum()), int((s > RANK_RELATIVE_TOL * s[0]).sum())


def is_identity_like(matrix: np.ndarray) -> bool:
    """True when every nonzero row and column holds exactly one nonzero entry and all nonzero entries are positive.

    That is a partial permutation or diagonal up to positive scaling: each output unit just copies one input unit.
    An all-zero matrix is not identity-like (it passes nothing)."""
    mask = np.abs(matrix) > NONZERO_EPS
    if not mask.any():
        return False
    return bool((mask.sum(axis=0) <= 1).all() and (mask.sum(axis=1) <= 1).all() and (matrix[mask] > 0).all())


def weight_stats(w: np.ndarray) -> dict[str, Any]:
    """Counts for any tensor; rank and identity fields only for matrices (ndim >= 2), else None."""
    nonzero = int((np.abs(w) > NONZERO_EPS).sum())
    stats: dict[str, Any] = {
        "params": int(w.size),
        "nonzero": nonzero,
        "nonzero_frac": nonzero / w.size if w.size else 0.0,
        "eff_rank": None,
        "rank_1e3": None,
        "identity_like": None,
    }
    if w.ndim >= 2:
        matrix = as_matrix(w)
        stats["eff_rank"], stats["rank_1e3"] = effective_rank(matrix)
        stats["identity_like"] = is_identity_like(matrix)
    return stats


def unit_stats(h: np.ndarray, *, act: str | None) -> dict[str, Any]:
    """Active / dead / saturated units of one layer output `h` (N, units, ...): units live on axis 1, or the last axis
    for 2-D input. Statistics pool over the batch and any spatial axes."""
    if h.ndim == 2:
        flat = h.astype(np.float64)  # (N, units)
    else:
        flat = np.moveaxis(h.astype(np.float64), 1, -1).reshape(-1, h.shape[1])  # conv: (N * spatial, channels)
    std = flat.std(axis=0)
    never_nonzero = (np.abs(flat) <= STD_EPS).all(axis=0)  # also catches a ReLU unit that is always 0
    active = (std > STD_EPS) & ~never_nonzero
    if act == "tanh":
        saturated = ((np.abs(flat) > SATURATION_LEVEL).mean(axis=0) > SATURATION_SHARE) & ~never_nonzero
    else:
        saturated = np.zeros(flat.shape[1], dtype=bool)
    n_units = flat.shape[1]
    return {
        "units": n_units,
        "active": int(active.sum()),
        "dead": int((~active).sum()),
        "saturated": int(saturated.sum()),
        "active_unsaturated": int((active & ~saturated).sum()),
        "active_frac": float(active.sum()) / n_units,
    }


@dataclass
class LayerReport:
    node: str
    op: str
    tensors: dict[str, dict[str, Any]]  # weight key -> weight_stats
    params: int
    nonzero: int
    all_zero: bool  # every weight of the node is zero (bias included)
    identity_like: bool  # the main matrix is identity-like wiring
    eff_rank: float | None
    rank_1e3: int | None
    is_output: bool
    units: dict[str, Any] | None  # unit_stats of the node's activations; None for the output layer or inside repeat
    act: str | None
    in_shape: tuple[int, ...] = ()
    out_shape: tuple[int, ...] = ()

    @property
    def nonzero_frac(self) -> float:
        return self.nonzero / self.params if self.params else 0.0

    @property
    def counts_toward_depth(self) -> bool:
        """Nonzero weights and not a mere pass-through."""
        return not self.all_zero and not self.identity_like


@dataclass
class NetReport:
    layers: list[LayerReport]
    summary: dict[str, Any] = field(default_factory=dict)


def analyze(
    plan: netspec.Plan,
    weights: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    limits: netspec.Limits | None = None,
) -> NetReport:
    """Run `inputs` through the network and report usage per layer and for the whole network. `weights` must come from
    netspec.check_weights."""
    _, acts = netspec.forward(plan, weights, inputs, limits or netspec.Limits(), return_activations=True)
    output = plan.spec["output"]
    layers: list[LayerReport] = []
    for node in plan.spec["nodes"]:
        name = node["name"]
        keys = [k for k in weights if k.startswith(f"{name}.") and "." not in k[len(name) + 1 :]]
        if f"{name}.W" not in weights:
            continue
        tensors = {k: weight_stats(weights[k]) for k in keys}
        main = tensors[f"{name}.W"]
        params = sum(t["params"] for t in tensors.values())
        nonzero = sum(t["nonzero"] for t in tensors.values())
        is_output = name == output
        act = node.get("act")
        units = None
        if not is_output and name in acts:
            units = unit_stats(acts[name], act=act)
        layers.append(
            LayerReport(
                node=name,
                op=node["op"],
                tensors=tensors,
                params=params,
                nonzero=nonzero,
                all_zero=nonzero == 0,
                identity_like=bool(main["identity_like"]),
                eff_rank=main["eff_rank"],
                rank_1e3=main["rank_1e3"],
                is_output=is_output,
                units=units,
                act=act,
                in_shape=weights[f"{name}.W"].shape,
                out_shape=tuple(acts[name].shape[1:]) if name in acts else (),
            )
        )
    total = sum(layer.params for layer in layers)
    nonzero = sum(layer.nonzero for layer in layers)
    summary = {
        "layers": len(layers),
        "params": total,
        "nonzero": nonzero,
        "nonzero_frac": nonzero / total if total else 0.0,
        "zero_layers": sum(layer.all_zero for layer in layers),
        "active_depth": sum(layer.counts_toward_depth for layer in layers),
        "passthrough_layers": sum(layer.identity_like for layer in layers),
        "first_layer_param_frac": layers[0].params / total if layers and total else 0.0,
        "hidden_active": [layer.units["active"] for layer in layers if layer.units],
        "hidden_units": [layer.units["units"] for layer in layers if layer.units],
    }
    return NetReport(layers=layers, summary=summary)
