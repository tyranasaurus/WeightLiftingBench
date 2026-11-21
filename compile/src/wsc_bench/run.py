"""Run one model on one task with the v1 harness (harness/api_wsp) in Docker, then score the reply.

The agent container is the v1 sandbox (harness/image: Python 3.12.8, numpy 2.5.4) with the task's tensor shapes and
nonzero cap under /app/l0. api_wsp sends the prompt to the model once, runs the reply's python block, retries over the
cap at most twice, and leaves /app/out/weights.npz and its logs. API keys are read from your environment and passed
with an env file, never on a command line. Needs Docker and a clone of this repository.
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from wsc_bench.benchmark import Task
from wsc_bench.score import score_weights

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "harness" / "api_wsp"
IMAGE = "wsc-bench-agent:v1"
KEY_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")
INFRA_EXIT = 3  # api_wsp: keys out, model missing or provider errors after retries; retry, never score
AGENT_TIMEOUT_S = 12 * 3600
PLATFORM = "linux/amd64"  # the architecture of the official runs, so float32 results match


def ensure_image() -> None:
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        subprocess.run(["docker", "build", "--platform", PLATFORM, "-t", IMAGE, str(REPO / "harness" / "image")], check=True)


def run_model(*, task: Task, model: str, effort: str, route: str, out_root: Path) -> dict:
    if not HARNESS.is_dir():
        raise SystemExit(f"harness not found at {HARNESS}: run from a clone of the repository")
    keys = [f"{name}={os.environ[name]}" for name in KEY_NAMES if os.environ.get(name)]
    if not keys:
        raise SystemExit(f"set at least one of {', '.join(KEY_NAMES)}")
    ensure_image()
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", model)
    run_dir = out_root / task.name / f"{slug}-{effort}-{time.strftime('%Y%m%dT%H%M%S')}"
    for sub in ("l0", "out", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    target = task.build()
    (run_dir / "l0" / "shapes.json").write_text(json.dumps({k: list(v) for k, v in target.tensor_shapes().items()}))
    (run_dir / "l0" / "limits.json").write_text(json.dumps({"max_nonzero": task.max_nonzero}))
    (run_dir / "instruction.md").write_text(task.prompt())
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as env_file:
        env_file.write("\n".join(keys + ["AGENT_OUTPUT_ROOT=/logs"]) + "\n")
    os.chmod(env_file.name, 0o600)
    command = [
        "docker", "run", "--rm", "--platform", PLATFORM, "--env-file", env_file.name, "--cpus", "2", "--memory", "5g",
        "-v", f"{HARNESS}:/bundle/api_wsp:ro", "-v", f"{run_dir / 'l0'}:/app/l0:ro",
        "-v", f"{run_dir / 'out'}:/app/out", "-v", f"{run_dir / 'logs'}:/logs",
        "-v", f"{run_dir / 'instruction.md'}:/instr/instruction.md:ro", IMAGE, "bash", "-lc",
        f"python3 /bundle/api_wsp/collect_outputs.py /instr/instruction.md --task-id {task.name}-write-nnz{task.max_nonzero} "
        f"--model {model} --effort {effort} --route {route}",
    ]
    try:
        agent = subprocess.run(command, capture_output=True, text=True, timeout=AGENT_TIMEOUT_S)
    finally:
        os.unlink(env_file.name)
    (run_dir / "agent.log").write_text(agent.stdout + agent.stderr)
    result = {"task": task.name, "model": model, "effort": effort, "route": route, "agent_exit": agent.returncode}
    if agent.returncode == INFRA_EXIT:
        result.update(status="infra_error", note="provider or key failure; retry, do not score")
    else:
        score = score_weights(task=task, weights_path=run_dir / "out" / "weights.npz")
        result.update(status="scored", reward=score.reward, valid_output=score.valid_output, reason=score.reason,
                      nonzero_weights=score.detail.get("nonzero_weights"))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    result["run_dir"] = str(run_dir)
    return result
