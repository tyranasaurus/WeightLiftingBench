"""Free-architecture mode (track write_free): the agent writes net.json (a netspec network) plus weights.npz.

A target opts in by mixing in FreeArchTarget and implementing four hooks:
  net_io()                        input names / per-example shapes after the task's own preprocessing, output shape
  scores_from(forward, split, seed) the task's own metrics, with `forward` (batched inputs -> outputs) in place of its
                                  fixed forward pass: same data, preprocessing, read-out, scoring and normalisation
  reference_spec()                the fixed network as a netspec (one valid example; it sets the budgets)
  reference_net_weights()         the fixed oracle weights renamed to the reference spec's keys
plus `free_allowed_ops` and the prompt pieces below. net_limits() derives budgets and hard caps from the reference.

Scoring (score_submission): read and check the answer (wsp.net_budget), run scores_from on the hidden split through
a TrackedNet that records unit activity on exactly the inputs it scores (inputs only, never labels), then add the
usage numbers. reward = within_budget * clip((primary - floor) / (reference - floor), 0, 1); `reward_unbudgeted` keeps
the gap score without the budget gate so the headline can be re-pivoted later.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from wsp import net_budget, netspec
from wsp.net_budget import NetIO, NetLimits
from wsp.targets.base import Metrics, Split, Target, Weights

Forward = Callable[[dict[str, np.ndarray]], np.ndarray]
DENSE_OPS = frozenset({"dense", "add", "concat", "slice", *netspec.ACTIVATIONS})
IMAGE_OPS = frozenset({"conv2d", "maxpool2d", "reduce", "dense", "flatten", "reshape", "add", "concat", *netspec.ACTIVATIONS})
INVALID = {"reward": 0.0, "reward_unbudgeted": 0.0, "valid_output": 0.0, "within_budget": 0.0}


class FreeArchTarget(Target):
    """Opt-in hooks for the write_free track (see the module docstring)."""

    free_allowed_ops: ClassVar[frozenset[str]] = DENSE_OPS

    @abstractmethod
    def net_io(self) -> NetIO: ...

    @abstractmethod
    def scores_from(self, *, forward: Forward, split: Split, seed: int) -> Metrics: ...

    @abstractmethod
    def reference_spec(self) -> dict: ...

    @abstractmethod
    def reference_net_weights(self) -> Weights: ...

    @abstractmethod
    def io_text(self) -> str:
        """Plain words: what each input holds (after the task's preprocessing) and how the output is read."""

    def free_scoring_text(self) -> str:
        """The fixed task's SCORING section, with the reference described as the example network shown above."""
        return free_scoring(self.description())

    def free_rules_text(self) -> str:
        """What may and may not be run in the sandbox: by default the fixed task's own rules paragraph."""
        return rules_from_fixed(self)

    def activity_text(self) -> str:
        return "the held-out inputs the scorer evaluates (inputs only; labels are never used for this)"

    def net_limits(self) -> NetLimits:
        params, macs = net_budget.reference_counts(self.reference_spec(), self.net_io())
        return NetLimits(allowed_ops=self.free_allowed_ops, reference_params=params, reference_macs=macs)

    def task_limits_json(self) -> dict[str, Any]:
        io = self.net_io()
        return {
            "inputs": {k: list(v) for k, v in io.inputs.items()},
            "output": list(io.output),
            "limits": self.net_limits().as_json(),
        }


def read_submitted_net(*, net_path: Path, weights_path: Path, target: FreeArchTarget) -> tuple[Any, str]:
    """(TrackedNet, "ok") or (None, reason). Every malformed or hostile answer ends here as a reason, never a crash."""
    for path in (net_path, weights_path):
        if not path.is_file():
            return None, f"missing output file {path}"
    limits = target.net_limits()
    if net_path.stat().st_size > limits.max_spec_bytes:
        return None, f"{net_path.name} larger than {limits.max_spec_bytes:,} bytes"
    try:
        plan = net_budget.check_spec(net_path.read_text(errors="replace"), net_io=target.net_io(), limits=limits)
        weights = net_budget.read_npz(weights_path, plan=plan, limits=limits)
    except netspec.SpecError as error:
        return None, f"invalid answer: {error}"
    except (TypeError, KeyError, IndexError) as error:  # backstop: a spec shape the key checks did not foresee
        return None, f"invalid answer: {type(error).__name__}: {error}"
    return net_budget.TrackedNet(plan, weights, limits), "ok"


def score_submission(
    *, config: dict, target: FreeArchTarget, net_path: Path, weights_path: Path
) -> tuple[dict[str, float], dict[str, Any]]:
    """(reward.json values, full usage report). reward.json holds numbers only: the target's raw metrics, every usage
    count, the budget verdicts and the gated reward. The report adds ops_used and the per-layer table."""
    net, reason = read_submitted_net(net_path=net_path, weights_path=weights_path, target=target)
    print(f"output check: {reason}")
    if net is None:
        return dict(INVALID), {"invalid": reason}
    try:
        metrics = target.scores_from(forward=net, split="hidden", seed=config["hidden_seed"])
    except netspec.SpecError as error:  # NaN or inf outputs, ...
        print(f"invalid answer at run time: {error}")
        return dict(INVALID), {"invalid": str(error)}
    usage = net_budget.usage(net.plan, net.weights, net.tracker, net.limits)
    floor, reference = config.get("reward_floor", 0.0), config["reward_reference"]
    gap = float(np.clip((metrics.primary_value - floor) / (reference - floor), 0.0, 1.0))
    scalars = {k: float(v) for k, v in usage.items() if isinstance(v, int | float)}
    for node, frac in usage["hidden_active_unit_frac"].items():
        scalars[f"active_unit_frac_{node}"] = float(frac)
    reward = {
        **{k: float(v) for k, v in metrics.values.items()},
        **scalars,
        "primary": float(metrics.primary_value),
        "reward_unbudgeted": gap,
        "reward": gap * usage["within_budget"],
        "valid_output": 1.0,
    }
    return reward, {**usage, "reason": reason}


# ---- prompt ---------------------------------------------------------------------------------------------------------

OP_TEXT = {
    "dense": (
        '{"op": "dense", "in": X, "out": K, "bias": true}: y = x @ W + b over the LAST axis of x. Weights '
        "`<name>.W` shape (x's last size, K) and `<name>.b` shape (K,) (omitted when \"bias\": false). Any leading "
        "axes are kept, so a dense on (T, F) gives (T, K)."
    ),
    "conv2d": (
        '{"op": "conv2d", "in": X, "out": K, "kernel": k, "stride": s, "padding": p, "bias": true}: input (C, H, W), '
        "output (K, H', W') with H' = (H + 2p - k) // s + 1. Cross-correlation as in PyTorch Conv2d: "
        "y[o, i, j] = b[o] + sum over c, di, dj of W[o, c, di, dj] * xpad[c, i*s + di, j*s + dj], where xpad is x with "
        "p zeros on every side. `<name>.W` shape (K, C, k, k), `<name>.b` shape (K,). kernel / stride / padding are an "
        "int or a [rows, cols] pair; stride defaults to 1, padding to 0."
    ),
    "maxpool2d": (
        '{"op": "maxpool2d", "in": X, "kernel": k, "stride": s}: input (C, H, W), max over each k x k window, '
        "stride defaults to the kernel, no padding (a partial window at the edge is dropped). No weights."
    ),
    "reduce": (
        '{"op": "reduce", "in": X, "axes": [..], "kind": "mean"}: mean over the listed per-example axes '
        "(0 = first axis after the batch; negative counts from the end), which are removed. Only kind mean. No weights."
    ),
    "flatten": '{"op": "flatten", "in": X}: all per-example axes into one, row-major (C, H, W) order. No weights.',
    "reshape": '{"op": "reshape", "in": X, "shape": [..]}: row-major reshape to the given per-example shape.',
    "add": '{"op": "add", "in": [X, Y, ...]}: elementwise sum (numpy broadcasting). No weights.',
    "concat": '{"op": "concat", "in": [X, Y, ...], "axis": a}: join along per-example axis a (default -1). No weights.',
    "slice": (
        '{"op": "slice", "in": X, "axis": a, "start": i, "stop": j}: keep indices i..j-1 of per-example axis a '
        "(default the last). No weights."
    ),
}
ACT_TEXT = (
    "Activations, either as their own node ({\"op\": \"relu\", \"in\": X}) or as \"act\" on any node (applied to its "
    "output): identity, relu, tanh, sigmoid = 1/(1+exp(-x)) (x clipped to +-60), gelu (exact erf form), gelu_tanh, "
    "silu = x*sigmoid(x), step = 1 if x > 0 else 0, softmax over the last axis."
)


def spec_text(spec: dict) -> str:
    """A net.json with one node per line (what solution/net.json and the prompt show)."""
    nodes = ",\n".join(f"  {json.dumps(node)}" for node in spec["nodes"])
    return (
        f'{{"format": "{spec.get("format", netspec.FORMAT)}",\n "inputs": {json.dumps(spec["inputs"])},\n'
        f' "nodes": [\n{nodes}\n ],\n "output": "{spec["output"]}"}}'
    )


def network_section(target: FreeArchTarget) -> str:
    limits = target.net_limits()
    conv = "conv2d" in limits.allowed_ops
    conv_units = " and the channel for conv2d" if conv else ""
    conv_macs = ", each active conv2d weight one per output position" if conv else ""
    io = target.net_io()
    reference = target.reference_spec()
    ops = [text for op, text in OP_TEXT.items() if op in limits.allowed_ops]
    inputs = ", ".join(f'"{name}": {list(shape)}' for name, shape in io.inputs.items())
    return f"""\
THE NETWORK: YOU CHOOSE IT
In this version of the task there is no fixed network. You write the architecture as a small JSON file, net.json,
and its weights as weights.npz. One numpy interpreter (netspec, given to you as /app/netspec.py) runs it.

net.json:
  {{"format": "wsp-netspec/1",
   "inputs": {{{inputs}}},
   "nodes": [ {{"name": "h1", "op": ..., "in": ...}}, ... ],
   "output": "<name of the node whose value is the network output>"}}
Shapes are per example; the batch axis is never written. Nodes run in list order and may read only the inputs or
earlier nodes. Names must be identifiers (letters, digits, _), unique. The output node's value must have shape
{list(io.output)}. Everything is computed in float32. Weights are keyed "<node name>.<param>" in weights.npz, and the
keys and shapes must match the spec exactly (missing, extra or misshaped arrays make the answer invalid).

THE INPUT AND OUTPUT CONTRACT
"inputs" must be exactly {{{inputs}}}.
{target.io_text()}

OPS ALLOWED IN THIS TASK (nothing else; any other op or key makes the answer invalid)
""" + "\n".join(f"  - {text}" for text in ops) + f"""
  - {ACT_TEXT}

BUDGET: ONLY THE PART OF THE NETWORK THAT IS USED COUNTS
You may declare a much bigger network than you use. What is budgeted is the ACTIVE part:
  - an active parameter is a weight (or bias) that is nonzero (any w != 0, however small) AND whose input unit and
    output unit are both active. A unit is active when its value is not exactly constant over {target.activity_text()}.
    Units are the last axis for dense layers{conv_units}. Zero weights, and weights into or out of a
    unit that never changes (for example a ReLU that is always 0), cost nothing.
  - active MACs per example: each active dense weight costs one multiply-accumulate per row of its input (one, for
    a vector input){conv_macs}. Ops without weights cost nothing.
Budget (the score is 0 if it is exceeded, so stay inside it):
  - active parameters <= {limits.param_budget:,}  (= the reference network's total parameter count)
  - active MACs per example <= {limits.mac_budget:,}  (= 2 x the reference network's MACs)
Hard limits (the answer is invalid beyond them): at most {limits.param_cap:,} declared parameters (16 x the
reference), {limits.mac_cap:,} declared MACs per example, {limits.max_nodes} nodes, and net.json at most
{limits.max_spec_bytes:,} bytes. Unused capacity is free: you do not need to fill or prune a big layer.

A VALID EXAMPLE: THE REFERENCE NETWORK
This is the fixed network of the classic version of this task ({limits.reference_params:,} parameters,
{limits.reference_macs:,} MACs). You may use it, change it, or ignore it; it is one valid answer shape, not a
requirement. With all-zero weights it scores 0.
{spec_text(reference)}
"""


DENSE_NUDGE = """\
WHAT KIND OF NETWORK
Build a genuine neural network: a distributed one whose hidden units each carry a graded share of the answer and
that actually uses its capacity, the way a trained network does. Do not hand-compile a program, a decision tree, a
lookup table or a list of stored examples into the weights (for example one unit per rule or per memorised input, or
huge weights that turn units into exact on / off switches). Prefer smooth, overlapping features spread across many
units."""
PROMPT_VARIANTS = ("default", "dense")


def instruction(target: FreeArchTarget, *, prompt_variant: str = "default") -> str:
    """The full instruction.md: the fixed description up to its network, then the free network, scoring and rules.
    prompt_variant "dense" adds DENSE_NUDGE after the network section; "default" is unchanged byte for byte."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant {prompt_variant!r} not in {PROMPT_VARIANTS}")
    description = target.description()
    head = description.split("\nTHE NETWORK")[0].rstrip()
    nudge = f"\n\n{DENSE_NUDGE}" if prompt_variant == "dense" else ""
    return f"""{head}

{network_section(target).rstrip()}{nudge}

{target.free_scoring_text().rstrip()}

HOW TO WRITE THE ANSWER
{target.free_rules_text().strip()}
Tools in /app: netspec.py (the interpreter the scorer runs: `plan = netspec.check(spec)`, `w = netspec.check_weights(plan,
weights)`, `netspec.forward(plan, w, {{"<input>": array of shape (N, *shape)}})`), and net_budget.py, which checks an
answer against this task's limits (task_limits.json) and estimates active counts on inputs you give it:
  python /app/net_budget.py /app/out/net.json /app/out/weights.npz [inputs.npz]
Write the architecture to `/app/out/net.json` and the weights to `/app/out/weights.npz` (`np.savez`, one array per
"<node>.<param>" key). Arrays must be finite and numeric (they are cast to float32). A network that outputs NaN or inf
on any held-out input scores zero. Only these two files are read; nothing you print is graded.
"""


def rules_from_fixed(target: Target) -> str:
    """The fixed task's own rules paragraph (what may and may not be run), minus its weights.npz output contract."""
    text = target.harbor_instruction()
    if text is None or "HOW TO WRITE THE WEIGHTS\n" not in text or "Write the weights to" not in text:
        raise ValueError(f"target {target.id}: no fixed rules paragraph to reuse")
    return text.split("HOW TO WRITE THE WEIGHTS\n", 1)[1].split("Write the weights to", 1)[0]


def free_scoring(description: str, *, stop: str | None = None) -> str:
    """The SCORING section of a fixed description (up to `stop`, if given), reworded for a free network."""
    scoring = "SCORING\n" + description.split("\nSCORING\n", 1)[1]
    if stop is not None:
        scoring = scoring.split(stop, 1)[0]
    for fixed, free in (
        ("of this exact shape", "with the reference shape shown above"),
        ("of the same shape", "with the reference shape shown above"),
        ("the same network", "the reference network shown above"),
    ):
        scoring = scoring.replace(fixed, free)
    return scoring.rstrip() + (
        "\nThe reference is the same one the fixed-architecture version of this task uses, so scores compare directly."
        "\nIf your network goes over the active-parameter or active-MAC budget, the score is 0.\n"
    )
