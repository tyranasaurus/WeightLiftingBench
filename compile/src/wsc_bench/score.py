"""Score weights on a task the way the v1 verifier did: the primary metric averaged over the task's eval seeds.

The official v1 runs used private seeds so no reply could be tuned to them. Here the seeds start at `first_seed`
(default 0); scores agree with the official ones up to sampling noise.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wsc_bench.benchmark import Task


@dataclass(frozen=True)
class Score:
    reward: float  # in [0, 1]; 0 when the output is invalid or over the nonzero cap
    valid_output: bool
    detail: dict[str, float]  # per metric means over seeds (includes nonzero_weights and within_budget)
    reason: str


def read_weights(*, path: Path, shapes: dict[str, tuple[int, ...]]) -> tuple[dict[str, np.ndarray] | None, str]:
    if not path.is_file():
        return None, f"missing output file {path}"
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        return None, f"cannot read {path} as npz: {error}"
    if set(arrays) != set(shapes):
        return None, f"keys {sorted(arrays)} != expected {sorted(shapes)}"
    for name, shape in shapes.items():
        if arrays[name].shape != tuple(shape):
            return None, f"tensor {name}: shape {arrays[name].shape} != {tuple(shape)}"
        if not np.issubdtype(arrays[name].dtype, np.number) or not np.all(np.isfinite(arrays[name])):
            return None, f"tensor {name}: not a finite numeric array"
    return {name: arrays[name].astype(np.float32) for name in shapes}, "ok"


def score_weights(*, task: Task, weights_path: Path, first_seed: int = 0) -> Score:
    target = task.build()
    weights, reason = read_weights(path=weights_path, shapes=target.tensor_shapes())
    if weights is None:
        return Score(reward=0.0, valid_output=False, detail={}, reason=reason)
    runs = [target.evaluate(weights=weights, split="hidden", seed=first_seed + k) for k in range(task.eval_seeds)]
    primary = np.array([m.primary_value for m in runs])
    detail = {key: float(np.mean([m.values[key] for m in runs])) for key in runs[0].values}
    return Score(reward=float(np.clip(primary.mean(), 0.0, 1.0)), valid_output=True, detail=detail, reason="ok")
