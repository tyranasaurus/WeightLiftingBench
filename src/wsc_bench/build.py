"""Turn a model's reply into weights, exactly as the v1 harness did.

The reply must hold exactly one ```python block. It runs once, in a fresh `python -I` process, with `np`, `math` and
`weights` (every tensor preloaded with zeros) defined, imports other than numpy and math blocked, and a 600 s time
limit. The weights it leaves are saved as float32 to an .npz file.
"""

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE_TIMEOUT_S = 600
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
SPARSE_KEYS = {"layer", "row", "col", "val"}
RUNNER = """\
import builtins, json, sys
import numpy as np
import math

ALLOWED_IMPORTS = {"numpy", "math"}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if globals is not None and globals.get("__name__") == "wsp_sandbox" and name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} is not allowed in the sandbox")
    return real_import(name, globals, locals, fromlist, level)

shapes = json.load(open(sys.argv[1]))
weights = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
builtins.__import__ = guarded_import
code = open(sys.argv[2]).read()
exec(compile(code, "editor_code", "exec"), {"__name__": "wsp_sandbox", "np": np, "math": math, "weights": weights})
builtins.__import__ = real_import
for name, shape in shapes.items():
    if tuple(np.shape(weights[name])) != tuple(shape):
        raise SystemExit(f"tensor {name}: shape {np.shape(weights[name])} != {tuple(shape)}")
np.savez_compressed(sys.argv[3], **{name: np.asarray(weights[name], dtype=np.float32) for name in shapes})
"""


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    reason: str  # "weights written", or why the reply produced no weights


def extract_program(reply: str) -> str:
    blocks = PYTHON_BLOCK.findall(reply.strip())
    if len(blocks) != 1:
        raise ValueError(f"expected exactly one ```python block, found {len(blocks)}")
    return blocks[0]


def build_weights(*, reply: str, shapes: dict[str, tuple[int, ...]], out_path: Path) -> BuildResult:
    """Run the reply's program and write its weights to out_path."""
    out_path.unlink(missing_ok=True)
    try:
        program = extract_program(reply)
    except ValueError as error:
        return BuildResult(ok=False, reason=str(error))
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        (work_dir / "code.py").write_text(program)
        (work_dir / "runner.py").write_text(RUNNER)
        (work_dir / "shapes.json").write_text(json.dumps({name: list(shape) for name, shape in shapes.items()}))
        argv = [sys.executable, "-I", "runner.py", "shapes.json", "code.py", str(out_path.resolve())]
        try:
            proc = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True, timeout=CODE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return BuildResult(ok=False, reason=f"code ran over {CODE_TIMEOUT_S} s")
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        return BuildResult(ok=False, reason=(proc.stdout + proc.stderr)[-4000:])
    return BuildResult(ok=True, reason="weights written")


def count_nonzero(weights: dict[str, np.ndarray]) -> int:
    """Nonzero weights and biases. A sparse (layer, row, col, val) submission counts nonzero val plus biases."""
    sparse = SPARSE_KEYS <= set(weights)
    return int(sum(np.count_nonzero(a) for k, a in weights.items() if not sparse or k == "val" or k.startswith("b")))
