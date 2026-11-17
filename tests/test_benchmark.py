"""Every task loads, its prompt matches its network, and the reply pipeline enforces the v1 rules."""

from pathlib import Path

import numpy as np
import pytest

from wsc_bench import TASKS, get_task
from wsc_bench.build import build_weights, count_nonzero
from wsc_bench.score import score_weights


def test_v1_has_14_tasks_in_two_tiers():
    assert len(TASKS) == 14
    assert sum(t.tier == "easy" for t in TASKS.values()) == 8
    assert all(t.max_nonzero == 500_000 for t in TASKS.values())


@pytest.mark.parametrize("name", list(TASKS))
def test_prompt_lists_every_starting_tensor(name):
    task = get_task(name)
    shapes = task.build().tensor_shapes()
    prompt = task.prompt()
    assert "500,000" in prompt and "```python" in prompt
    starting = prompt[prompt.index("## Starting weights"):]
    for tensor, shape in shapes.items():
        assert f"{tensor} {tuple(shape)}" in starting, (tensor, shape)


def test_a_reply_needs_exactly_one_python_block(tmp_path: Path):
    shapes = {"W0": (2, 2)}
    two = "```python\nweights['W0'][0, 0] = 1\n```\n```python\npass\n```"
    assert not build_weights(reply=two, shapes=shapes, out_path=tmp_path / "w.npz").ok
    assert not build_weights(reply="no code here", shapes=shapes, out_path=tmp_path / "w.npz").ok


def test_the_program_can_only_import_numpy_and_math(tmp_path: Path):
    shapes = {"W0": (2, 2)}
    blocked = build_weights(reply="```python\nimport os\n```", shapes=shapes, out_path=tmp_path / "w.npz")
    assert not blocked.ok and "not allowed" in blocked.reason
    ok = build_weights(reply="```python\nimport math\nweights['W0'][1, 1] = math.pi\n```", shapes=shapes,
                       out_path=tmp_path / "w.npz")
    assert ok.ok
    with np.load(tmp_path / "w.npz") as archive:
        assert archive["W0"][1, 1] == np.float32(np.pi)


def test_sparse_submissions_count_values_and_biases_only():
    sparse = {"layer": np.array([0, 0, 1]), "row": np.array([3, 4, 5]), "col": np.array([1, 2, 3]),
              "val": np.array([1.0, 0.0, 2.0]), "b0": np.array([0.0, 1.0])}
    assert count_nonzero(sparse) == 3
    assert count_nonzero({"W0": np.eye(3), "b0": np.ones(2)}) == 5


def test_zero_weights_score_as_a_valid_low_reply(tmp_path: Path):
    task = get_task("gcd10")
    np.savez(tmp_path / "zeros.npz", **task.build().init_weights())
    result = score_weights(task=task, weights_path=tmp_path / "zeros.npz")
    assert result.valid_output and result.reward < 0.05


def test_over_the_cap_scores_zero(tmp_path: Path):
    task = get_task("add-10digit")
    weights = {k: np.ones(v.shape, dtype=np.float32) for k, v in task.build().init_weights().items()}
    np.savez(tmp_path / "dense.npz", **weights)
    result = score_weights(task=task, weights_path=tmp_path / "dense.npz")
    assert result.reward == 0.0 and result.detail["within_budget"] == 0.0
