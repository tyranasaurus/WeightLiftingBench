#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# The sandbox may run python with PYTHONSAFEPATH, which leaves the script's folder off sys.path, so add it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from routes import RouteState, out_of_credit  # noqa: E402
ROUTES: dict[str, Any] = {"state": None}

WORKSPACE_PATH = Path.cwd()
OUTPUT_ROOT = Path(os.environ.get("AGENT_OUTPUT_ROOT", "/logs"))
TRAJECTORY_PATH = OUTPUT_ROOT / "trajectory.jsonl"
FINAL_MESSAGE_PATH = OUTPUT_ROOT / "final_message.txt"
METRICS_PATH = OUTPUT_ROOT / "metrics_total.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
RAW_OUTPUT_PATH = OUTPUT_ROOT / "raw_output.txt"
# Exit code for a key that ran out of credit mid-task: nonzero fails the task (ERROR) instead of grading it as a 0, so
# the runner can retry it in a fresh sandbox.
OUT_OF_CREDIT_EXIT = 3
L0_RESULT_PATH = OUTPUT_ROOT / "l0_result.json"

# L0 (sealed single reply) mirrors agents/codex_wsp L0 so scoring is identical: same system instructions, same reply
# contract (exactly one ```python block, run once by L0_RUNNER with numpy and math only and `weights` preloaded with
# zeros from /app/l0/shapes.json), same nonzero cap retry. tests/test_claude_code_wsp_cap_retry.py checks the shared
# constants match codex_wsp byte for byte.
L0_SHAPES_PATH = Path("/app/l0/shapes.json")
# Capped tasks ship {"max_nonzero": N}. A reply whose weights hold more nonzero entries gets a follow-up turn in the same
# conversation (claude -p --resume <session id>, same sealed flags) with only the count and the cap, at most
# MAX_CAP_ATTEMPTS replies.
L0_LIMITS_PATH = Path("/app/l0/limits.json")
MAX_CAP_ATTEMPTS = 3
CAP_RETRY_MESSAGE = (
    "Your weights and biases contain {used:,} nonzero entries, over the cap of {cap:,}. Resubmit a complete solution "
    "with at most {cap:,} nonzero entries: reply with exactly one ```python block that sets every tensor, as before.\n"
)
NONZERO_COUNTER = (
    # a sparse (layer, row, col, val) submission counts nonzero val plus biases, like the verifier
    "import sys, numpy as np; a = np.load(sys.argv[1]); s = {'layer', 'row', 'col', 'val'} <= set(a.files); "
    "print(sum(int(np.count_nonzero(a[k])) for k in a.files if not s or k == 'val' or k.startswith('b')))"
)
L0_OUTPUT_PATH = Path("/app/out/weights.npz")
# Free architecture tasks ship /app/l0/free.json (net I/O, limits, max_active_params) and no shapes.json: the reply's
# block runs with `net = {}` and `weights = {}` preloaded and must set the netspec dict and its arrays; the runner writes
# /app/out/net.json and /app/out/weights.npz. The cap retry counts every nonzero entry of the submitted weights (an
# upper bound on the active params the verifier budgets) against max_active_params.
L0_FREE_PATH = Path("/app/l0/free.json")
L0_NET_OUTPUT_PATH = Path("/app/out/net.json")
L0_FREE_RUNNER = """\
import builtins, json, sys
import numpy as np
import math

ALLOWED_IMPORTS = {"numpy", "math"}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if globals is not None and globals.get("__name__") == "wsp_sandbox" and name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} is not allowed in the sandbox")
    return real_import(name, globals, locals, fromlist, level)

def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"net holds a {type(value).__name__}, not JSON")

scope = {"__name__": "wsp_sandbox", "np": np, "math": math, "net": {}, "weights": {}}
builtins.__import__ = guarded_import
code = open(sys.argv[1]).read()
exec(compile(code, "editor_code", "exec"), scope)
builtins.__import__ = real_import
net, weights = scope["net"], scope["weights"]
if not isinstance(net, dict) or not net:
    raise SystemExit("net must be a non-empty dict: the network spec")
if not isinstance(weights, dict) or not all(isinstance(name, str) for name in weights):
    raise SystemExit("weights must be a dict mapping '<node>.<param>' names to arrays")
text = json.dumps(net, default=plain)
arrays = {name: np.asarray(value, dtype=np.float32) for name, value in weights.items()}
open(sys.argv[2], "w").write(text)
np.savez_compressed(sys.argv[3], **arrays)
"""

# A free net that breaks a size or limit rule (declared params or MACs over the hard cap, a dimension, node or depth
# limit, a file size) is invalid in the verifier before it is scored. The task ships the verifier's own checker next to
# free.json; the harness runs it on the submitted files and answers such a reply with FREE_LIMIT_RETRY_MESSAGE (the
# limit, the value and the maximum, in the verifier's words), like the nonzero cap retry. A malformed spec or a crash in
# the reply's own code is not retried, as in fixed mode.
FREE_LIMIT_MARKERS = ("over the limit", "dimension over", "bytes per example over", "larger than any weight",
                      "net.json larger than", "weights.npz larger than", "uncompressed size larger than")
FREE_LIMIT_RETRY_MESSAGE = (
    "Your network breaks a size limit: {error}. Resubmit a complete solution within every limit in the task: reply "
    "with exactly one ```python block that sets net and weights, as before.\n"
)
L0_FREE_CHECKER = """\
import json, sys
from pathlib import Path

task_path = Path(sys.argv[1])
sys.path.insert(0, str(task_path.parent))
import net_budget

task = json.loads(task_path.read_text())
net_io = net_budget.NetIO(inputs={k: tuple(v) for k, v in task["inputs"].items()}, output=tuple(task["output"]))
limits = net_budget.NetLimits.from_json(task["limits"])
try:
    plan = net_budget.check_spec(Path(sys.argv[2]).read_text(errors="replace"), net_io=net_io, limits=limits)
    net_budget.read_npz(Path(sys.argv[3]), plan=plan, limits=limits)
except net_budget.netspec.SpecError as error:  # the netspec net_budget itself raises from
    print(json.dumps({"ok": False, "error": str(error)}))
    raise SystemExit(0)
print(json.dumps({"ok": True}))
"""
L0_SYSTEM_INSTRUCTIONS = "Answer the user's request.\n"
L0_EMPTY_DIR = Path("/tmp/claude_code_wsp_l0_empty")
# Content block types that mean the model called something; with `--tools ""` none can occur.
L0_FORBIDDEN_BLOCK_TYPES = {"tool_use", "server_tool_use", "mcp_tool_use"}
# The API maximum per request for every current Claude model; Claude Code's default output cap is lower.
L0_MAX_OUTPUT_TOKENS = "128000"
# A request cut at max_tokens is continued in the same reply (Claude Code's recovery sends up to 3 more), so one reply
# gets up to 4 x 128k output tokens, thinking included.
L0_MAX_REPLY_REQUESTS = 4
L0_API_TIMEOUT_MS = str(3 * 3600 * 1000)  # a max effort reply can stream for well over Claude Code's 10 minute default
L0_CODE_TIMEOUT_S = 600
L0_RUNNER = """\
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
# builder access: an agentic session (tools kwarg, normally "Read Write Edit": no code can run) that ends with the model's
# /app/out/build.py. It is then run once, exactly as an l0 reply's block is (L0_RUNNER: numpy and math only, `weights`
# preloaded with zeros from /app/l0/shapes.json), with the task's example arrays (/app/practice.npz) also preloaded.
BUILDER_PATH = Path("/app/out/build.py")
BUILDER_EXAMPLES_PATH = Path("/app/practice.npz")
BUILDER_RESULT_PATH = OUTPUT_ROOT / "builder_result.json"
BUILDER_RUNNER = L0_RUNNER.replace(
    'builtins.__import__ = guarded_import\n',
    'examples = np.load(sys.argv[4])\nextra = {name: examples[name] for name in examples.files}\nbuiltins.__import__ = guarded_import\n',
).replace('"weights": weights})', '"weights": weights, **extra})')
assert BUILDER_RUNNER != L0_RUNNER and "**extra" in BUILDER_RUNNER


# A task may ship /app/l0/builder.json to seal the builder the way sealed (l0) replies are sealed in the no-training
# version of the compile tasks: {"examples": false} runs build.py with `weights` only (the plain L0_RUNNER, no example
# arrays), and {"cpu_seconds": N} stops it after N seconds of CPU time on one core (RLIMIT_CPU, numpy threads set to 1).
# Designing weights takes a few CPU seconds; training a network inside the script takes far longer.
BUILDER_CONFIG_PATH = Path("/app/l0/builder.json")
BUILDER_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def run_builder() -> dict[str, Any]:
    L0_OUTPUT_PATH.unlink(missing_ok=True)  # only the script's output counts
    config = json.loads(BUILDER_CONFIG_PATH.read_text(encoding="utf-8")) if BUILDER_CONFIG_PATH.exists() else {}
    cpu_seconds, with_examples = config.get("cpu_seconds"), config.get("examples", True)
    if not BUILDER_PATH.is_file():
        return {"ok": False, "stage": "contract", "reason": f"missing {BUILDER_PATH}"}
    work = OUTPUT_ROOT / "builder_sandbox"
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.py").write_text(BUILDER_PATH.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    (work / "runner.py").write_text(BUILDER_RUNNER if with_examples else L0_RUNNER, encoding="utf-8")
    argv = [str(work / "runner.py"), str(L0_SHAPES_PATH), str(work / "code.py"), str(L0_OUTPUT_PATH)]
    if with_examples:
        argv.append(str(BUILDER_EXAMPLES_PATH))

    def limit_cpu() -> None:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 5))

    before = os.times()
    try:
        proc = subprocess.run(["python3", "-I", *argv], cwd=work, capture_output=True, text=True, timeout=L0_CODE_TIMEOUT_S,
                              env=BUILDER_ENV if cpu_seconds else None, preexec_fn=limit_cpu if cpu_seconds else None)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "code", "reason": f"code ran over {L0_CODE_TIMEOUT_S} s"}
    after = os.times()
    used = (after.children_user - before.children_user) + (after.children_system - before.children_system)
    cpu = {"code_cpu_s": round(used, 2), "code_cpu_limit_s": cpu_seconds, "examples_preloaded": with_examples}
    if proc.returncode != 0:
        L0_OUTPUT_PATH.unlink(missing_ok=True)
        if cpu_seconds and (proc.returncode < 0 or used >= float(cpu_seconds)):
            return {"ok": False, "stage": "code", "reason": f"code used over {cpu_seconds} CPU seconds", **cpu}
        return {"ok": False, "stage": "code", "reason": (proc.stdout + proc.stderr)[-4000:], **cpu}
    # Determinism check: run build.py a second time in a fresh process and compare every tensor exactly.
    first = L0_OUTPUT_PATH.read_bytes()
    second_path = work / "weights_second_run.npz"
    argv2 = argv[:3] + [str(second_path)] + argv[4:]
    try:
        proc2 = subprocess.run(["python3", "-I", *argv2], cwd=work, capture_output=True, text=True, timeout=L0_CODE_TIMEOUT_S,
                               env=BUILDER_ENV if cpu_seconds else None, preexec_fn=limit_cpu if cpu_seconds else None)
        same = proc2.returncode == 0 and second_path.exists() and second_path.read_bytes() == first
        if not same and proc2.returncode == 0 and second_path.exists():
            import numpy as np
            a, b = np.load(L0_OUTPUT_PATH), np.load(second_path)
            same = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    except subprocess.TimeoutExpired:
        same = False
    cpu["deterministic"] = bool(same)
    if not same:
        L0_OUTPUT_PATH.unlink(missing_ok=True)
        return {"ok": False, "stage": "determinism", "reason": "build.py gave different tensors on a second run", **cpu}
    return {"ok": True, "stage": "done", "reason": "weights written, identical on a second run", "output": str(L0_OUTPUT_PATH), **cpu}


PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_statement_path")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="high")
    parser.add_argument("--tools", default="default")
    parser.add_argument("--access", default="agentic", choices=["agentic", "l0", "builder"])
    parser.add_argument("--time-limit-s", type=float, default=3600.0)
    parser.add_argument("--final-turn-s", type=float, default=900.0)
    return parser.parse_args()


def read_max_nonzero() -> Optional[int]:
    if not L0_LIMITS_PATH.exists():
        return None
    return int(json.loads(L0_LIMITS_PATH.read_text(encoding="utf-8"))["max_nonzero"])


def read_free_cap() -> Optional[int]:
    """max_active_params of a free architecture task, or None for a fixed-shape task (no /app/l0/free.json)."""
    if not L0_FREE_PATH.exists():
        return None
    return int(json.loads(L0_FREE_PATH.read_text(encoding="utf-8"))["max_active_params"])


def count_nonzero(path: Path) -> int:
    proc = subprocess.run(["python3", "-I", "-c", NONZERO_COUNTER, str(path)], capture_output=True, text=True,
                          check=True, timeout=L0_CODE_TIMEOUT_S)
    return int(proc.stdout.strip())

def free_limit_error() -> Optional[str]:
    """The verifier's size or limit error for the submitted free net (net.json + weights.npz), or None when it is within
    every limit, fails for another reason (not retried), or the task does not ship the checker."""
    if not (L0_FREE_PATH.parent / "net_budget.py").exists():
        return None
    checker = OUTPUT_ROOT / "l0_sandbox" / "checker.py"
    checker.write_text(L0_FREE_CHECKER, encoding="utf-8")
    proc = subprocess.run(["python3", "-I", str(checker), str(L0_FREE_PATH), str(L0_NET_OUTPUT_PATH), str(L0_OUTPUT_PATH)],
                          capture_output=True, text=True, check=True, timeout=L0_CODE_TIMEOUT_S)
    error = json.loads(proc.stdout.strip().splitlines()[-1]).get("error")
    return error if error and any(marker in error for marker in FREE_LIMIT_MARKERS) else None


def session_id(events: list[dict[str, Any]]) -> Optional[str]:
    return next((str(e["session_id"]) for e in events if e.get("type") == "system" and e.get("subtype") == "init"
                 and e.get("session_id")), None)


def build_l0_command(*, model: str, effort: str) -> list[str]:
    """Sealed claude -p: no built-in tools, no MCP servers, no skills or slash commands, no hooks, settings or CLAUDE.md
    (--bare), nothing can prompt for permission, and the codex_wsp L0 system prompt instead of Claude Code's."""
    return ["claude", "-p", "--verbose", "--output-format", "stream-json", "--include-partial-messages", "--bare",
            "--model", model,
            "--effort", effort, "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
            "--permission-prompts", "none", "--system-prompt", L0_SYSTEM_INSTRUCTIONS]


def l0_violations(events: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            if event.get("tools"):
                violations.append(f"session has tools {event.get('tools')}")
            if event.get("mcp_servers"):
                violations.append(f"session has MCP servers {event.get('mcp_servers')}")
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") in L0_FORBIDDEN_BLOCK_TYPES:
                violations.append(f"{block.get('type')} {block.get('name')}")
        if event.get("type") == "result" and event.get("is_error"):
            violations.append(f"result is an error: {str(event.get('result'))[:500]}")
    if not any(e.get("type") == "result" for e in events):
        violations.append("no result event")
    return list(dict.fromkeys(violations))


def apply_l0_reply(events: list[dict[str, Any]], exit_code: int, *, free: bool = False) -> dict[str, Any]:
    """Gate the trace, then run the reply's single python block to produce /app/out/weights.npz (and, for a free
    architecture task, /app/out/net.json)."""
    L0_OUTPUT_PATH.unlink(missing_ok=True)  # the file left at the end is always the last reply's
    if free:
        L0_NET_OUTPUT_PATH.unlink(missing_ok=True)
    violations = l0_violations(events) + ([f"claude exited {exit_code}"] if exit_code != 0 else [])
    if violations:
        return {"ok": False, "stage": "gate", "reason": "; ".join(violations)}
    blocks = PYTHON_BLOCK.findall(reply_text(events) or "")
    if len(blocks) != 1:
        return {"ok": False, "stage": "contract", "reason": f"expected exactly one ```python block, found {len(blocks)}"}
    work = OUTPUT_ROOT / "l0_sandbox"
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.py").write_text(blocks[0], encoding="utf-8")
    (work / "runner.py").write_text(L0_FREE_RUNNER if free else L0_RUNNER, encoding="utf-8")
    L0_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if free:
        argv = [str(work / "runner.py"), str(work / "code.py"), str(L0_NET_OUTPUT_PATH), str(L0_OUTPUT_PATH)]
    else:
        argv = [str(work / "runner.py"), str(L0_SHAPES_PATH), str(work / "code.py"), str(L0_OUTPUT_PATH)]
    try:
        proc = subprocess.run(["python3", "-I", *argv], cwd=work, capture_output=True, text=True,
                              timeout=L0_CODE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "code", "reason": f"code ran over {L0_CODE_TIMEOUT_S} s"}
    if proc.returncode != 0:
        L0_OUTPUT_PATH.unlink(missing_ok=True)
        if free:
            L0_NET_OUTPUT_PATH.unlink(missing_ok=True)
        return {"ok": False, "stage": "code", "reason": (proc.stdout + proc.stderr)[-4000:]}
    if free:
        return {"ok": True, "stage": "done", "reason": "net and weights written", "output": str(L0_OUTPUT_PATH),
                "net": str(L0_NET_OUTPUT_PATH)}
    return {"ok": True, "stage": "done", "reason": "weights written", "output": str(L0_OUTPUT_PATH)}


def run_l0(*, command: list[str], env: dict[str, str], prompt_text: str, cwd: Path,
           max_nonzero: Optional[int], free: bool = False) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """One reply, or for a capped task up to MAX_CAP_ATTEMPTS: an over-cap reply is answered with CAP_RETRY_MESSAGE
    in the same conversation (`--resume <session id>`). `free`: a free architecture task (the reply sets `net` and
    `weights`; max_nonzero is its max_active_params)."""
    all_events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    attempt_command, prompt = command, prompt_text
    exit_code = 1
    for attempt in range(1, (1 if max_nonzero is None else MAX_CAP_ATTEMPTS) + 1):
        exit_code, events = stream_claude(attempt_command, prompt, env, cwd=cwd, append=attempt > 1,
                                          limit_requests=True)
        all_events += events
        result = apply_l0_reply(events, exit_code, free=free)
        limit_error = free_limit_error() if result["ok"] and free else None
        if limit_error is not None:
            result = {**result, "limit_error": limit_error, "within_budget": False}
        elif result["ok"] and max_nonzero is not None:
            used = count_nonzero(L0_OUTPUT_PATH)
            result = {**result, "nonzero_weights": used, "max_nonzero": max_nonzero, "within_budget": used <= max_nonzero}
        attempts.append({"attempt": attempt, **result})
        print(f"claude_code_wsp l0 attempt {attempt}: {result['stage']}: {str(result['reason'])[:300]} "
              f"nonzero {result.get('nonzero_weights')} cap {max_nonzero}", flush=True)
        session = session_id(events) or session_id(all_events)
        if result.get("within_budget", True) or attempt == MAX_CAP_ATTEMPTS or session is None:
            break
        if limit_error is not None:
            prompt = FREE_LIMIT_RETRY_MESSAGE.format(error=limit_error)
        else:
            prompt = CAP_RETRY_MESSAGE.format(used=result["nonzero_weights"], cap=max_nonzero)
        (OUTPUT_ROOT / f"cap_retry_{attempt + 1}.md").write_text(prompt, encoding="utf-8")
        retry_event = {"type": "wsp.cap_retry", "attempt": attempt + 1, "session_id": session, "message": prompt,
                       "nonzero_weights": result.get("nonzero_weights"), "max_nonzero": max_nonzero,
                       "limit_error": limit_error}
        with TRAJECTORY_PATH.open("a", encoding="utf-8") as trajectory_log:
            trajectory_log.write(json.dumps(retry_event) + "\n")
        all_events.append(retry_event)
        attempt_command = [*command, "--resume", session]
        print(f"claude_code_wsp argv: {attempt_command}", flush=True)
    return exit_code, all_events, {**attempts[-1], "attempts": attempts}


def event_from_line(line: str) -> Optional[dict[str, Any]]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def extract_text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str) and content.strip():
        return content.strip()
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if item.get("type") == "text" and isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts) if parts else None


def extract_final_message(events: list[dict[str, Any]]) -> Optional[str]:
    for event in reversed(events):
        result = event.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()

    for event in reversed(events):
        message = event.get("message")
        if isinstance(message, dict):
            text = extract_text_from_content(message.get("content"))
            if text:
                return text

        text = extract_text_from_content(event.get("content"))
        if text:
            return text

    return None


def reply_text(events: list[dict[str, Any]]) -> Optional[str]:
    """The reply's text. When a request was cut at max_tokens and continued, the reply's text runs across several
    assistant messages (a python block can start in one and end in the next), so their text blocks are joined in order
    with nothing between them."""
    starts = sum(1 for e in events if e.get("type") == "stream_event" and e["event"].get("type") == "message_start")
    if starts <= 1:
        return extract_final_message(events)
    message = lambda e: e.get("message") if isinstance(e.get("message"), dict) else {}
    blocks = [block.get("text", "") for e in events if e.get("type") == "assistant"
              for block in (message(e).get("content") or []) if isinstance(block, dict) and block.get("type") == "text"]
    return "".join(blocks) or None


def extract_metrics(events: list[dict[str, Any]], wall_clock_duration: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "cost": None,
        "wall_clock_duration": round(wall_clock_duration, 3),
    }
    # A resumed session (L0 cap retry) reports total_cost_usd for the whole conversation (checked against a mock API
    # with Claude Code 2.1.325), so the last one is the task's cost.
    for event in reversed(events):
        if "total_cost_usd" in event:
            metrics["cost"] = event.get("total_cost_usd")
            break
    if any(e.get("subtype") == "wsp_request_limit_kill" for e in events):
        metrics["cost"] = None  # a killed reply never got a CLI cost; `wsp cost` prices the summed tokens

    # Each `result` event's usage covers one claude invocation: sum the token counts over invocations.
    usages = [e["usage"] for e in events if e.get("type") == "result" and isinstance(e.get("usage"), dict)]
    if not usages:
        usages = [next((e["usage"] for e in reversed(events) if isinstance(e.get("usage"), dict)), {})]
    metrics.update(usages[-1])
    for key in dict.fromkeys(k for u in usages for k in u):
        if all(isinstance(u.get(key, 0), int) and not isinstance(u.get(key, 0), bool) for u in usages):
            metrics[key] = sum(u.get(key, 0) for u in usages)

    return metrics


def get_cli_version() -> Optional[str]:
    try:
        result = subprocess.run(
            ["claude", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    version = (result.stdout or result.stderr).strip()
    return version or None


def write_summary(
    task_id: str,
    model: str,
    effort: str,
    tools: str,
    access: str,
    command: list[str],
    prompt_text: str,
    exit_code: int,
    wall_clock_duration: float,
    events: list[dict[str, Any]],
) -> None:
    message = extract_final_message(events)
    if message:
        FINAL_MESSAGE_PATH.write_text(message + "\n", encoding="utf-8")

    metrics = extract_metrics(events, wall_clock_duration)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    summary = {
        "metadata": {
            "task_id": task_id,
            "model": model,
            "agent": "claude_code_wsp",
            "cli_version": get_cli_version(),
            "requested_effort": effort,
            "tools": tools,
            "access": access,
            "available_tools": available_tools(events),
            "command": command,
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "output_path": str(OUTPUT_ROOT),
            "success": bool(message) and exit_code == 0,
            "return_code": exit_code,
        },
        "task_prompt": prompt_text,
        "metrics": metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=4), encoding="utf-8")


def available_tools(events: list[dict[str, Any]]) -> Optional[list[str]]:
    """Tool list from the stream-json init event: what the session could actually call."""
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            return event.get("tools")
    return None


def build_claude_command(*, effort: str, tools: str) -> list[str]:
    command = [
        "claude",
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        *([] if effort == "default" else ["--effort", effort]),  # "default": no flag, so the model's own default effort applies
        "--allowedTools",
        "Bash Edit Write Read Glob Grep LS WebFetch NotebookEdit NotebookRead "
        "TodoRead TodoWrite Agent Skill SlashCommand Task WebSearch",
    ]
    if tools != "default":
        # --tools limits which built-in tools exist at all (--allowedTools only pre-approves them).
        command += ["--tools", ",".join(tools.replace(",", " ").split())]
    if PROMPT_IMAGE["b64"] is not None:
        command += ["--input-format", "stream-json"]
    return command


SHEET_PATH = Path("/app/prompt/sheet.png")
PROMPT_IMAGE = {"b64": None}  # the example sheet, sent once with the first message, then forgotten
FIX_UP_MESSAGE = ("Your answer could not be used: {error}\nFix the problem and save the corrected answer in this turn. "
                  "You will not get another chance.\n")
ANSWER_SPEC_PATH = Path("/app/l0/answer.json")


def take_sheet() -> None:
    """Read the example sheet into memory and delete it, so the model sees it only in the prompt."""
    if SHEET_PATH.exists():
        import base64, shutil
        PROMPT_IMAGE["b64"] = base64.b64encode(SHEET_PATH.read_bytes()).decode()
        shutil.rmtree(SHEET_PATH.parent, ignore_errors=True)


def encode_input(text: str, command: list[str]) -> str:
    if "--input-format" not in command:
        return text
    content: list[dict[str, Any]] = []
    if PROMPT_IMAGE["b64"] is not None:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PROMPT_IMAGE["b64"]}})
        PROMPT_IMAGE["b64"] = None
    content.append({"type": "text", "text": text})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def check_answer(access: str) -> dict[str, Any]:
    """Can the answer be used? Never looks at accuracy. Weights: keys, shapes, finite values. Builder: build.py runs, and
    twice gives identical tensors. Program: format checks, loads, and predict returns the right shape on an all-zero input."""
    if access == "builder":
        result = run_builder()
        return {"ok": result["ok"], "error": "" if result["ok"] else str(result["reason"])[-3000:], "detail": result}
    if not ANSWER_SPEC_PATH.exists():
        return {"ok": True, "error": "", "detail": "no answer spec"}
    spec = json.loads(ANSWER_SPEC_PATH.read_text(encoding="utf-8"))
    work = OUTPUT_ROOT / "answer_check"; work.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "weights":
        code = ("import json, sys, numpy as np\n"
                "shapes = json.load(open('/app/l0/shapes.json'))\n"
                "try:\n    a = np.load('/app/out/weights.npz', allow_pickle=False)\nexcept Exception as e:\n    print(f'cannot read /app/out/weights.npz: {e}'); sys.exit(1)\n"
                "bad = []\nmissing = sorted(set(shapes) - set(a.files)); extra = sorted(set(a.files) - set(shapes))\n"
                "if missing: bad.append(f'missing tensors {missing}')\nif extra: bad.append(f'unexpected tensors {extra}')\n"
                "for k, sh in shapes.items():\n    if k in a.files:\n        v = a[k]\n"
                "        if list(v.shape) != sh: bad.append(f'{k} has shape {list(v.shape)}, expected {sh}')\n"
                "        elif not np.issubdtype(v.dtype, np.number) or not np.all(np.isfinite(v)): bad.append(f'{k} has non-finite or non-numeric values')\n"
                "print('; '.join(bad)); sys.exit(1 if bad else 0)\n")
        (work / "check_weights.py").write_text(code, encoding="utf-8")
        proc = subprocess.run(["python3", "-I", str(work / "check_weights.py")], capture_output=True, text=True, timeout=300)
        return {"ok": proc.returncode == 0, "error": (proc.stdout + proc.stderr).strip()[-3000:], "detail": "weights check"}
    solution, constants = Path("/app/out/solution.py"), Path("/app/out/constants.npz")
    if not solution.exists():
        return {"ok": False, "error": "missing /app/out/solution.py", "detail": "program check"}
    zeros = work / "zeros.npz"
    proc = subprocess.run(["python3", "-I", "-c", f"import numpy as np; np.savez('{zeros}', x=np.zeros({tuple(spec['zero_input_shape'])}, dtype=np.float32))"],
                          capture_output=True, text=True, timeout=120)
    argv = ["python3", "/app/check_program.py", str(solution)] + ([str(constants)] if constants.exists() else []) + ["--inputs", str(zeros)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "loading the program and calling predict on a small input took over 15 minutes", "detail": "program check"}
    text = (proc.stdout + proc.stderr).strip()
    problems = [l for l in text.splitlines() if l.startswith("INVALID")]
    ok = proc.returncode == 0 and not problems
    if ok:
        import re as _re
        m = _re.search(r"returned shape \(([0-9, ]+)\)", text); n = spec["zero_input_shape"][0]
        shape = tuple(int(v) for v in m.group(1).replace(" ", "").split(",") if v) if m else None
        k = int(spec.get("classes", 10))
        if shape not in ((n,), (n, k)):
            ok, problems = False, [f"predict returned shape {shape} for {n} inputs; expected ({n},) or ({n}, {k})"]
    return {"ok": ok, "error": "" if ok else ("\n".join(problems) or text[-3000:]), "detail": "program check"}


TIME_UP_MESSAGE = (
    "You are out of time. In this turn, save your best answer as it stands now, in the output files the task describes. "
    "Do not start new work. Anything not saved when this turn ends is lost.\n"
)
TIMED_OUT = {"value": False}


def stream_claude(
    command: list[str], prompt_text: str, env: dict[str, str], *, cwd: Path = WORKSPACE_PATH, append: bool = False,
    limit_requests: bool = False, deadline_s: Optional[float] = None,
) -> tuple[int, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    mode = "a" if append else "w"
    with (
        RAW_OUTPUT_PATH.open(mode, encoding="utf-8") as raw_output,
        TRAJECTORY_PATH.open(mode, encoding="utf-8") as trajectory,
    ):
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )
        assert process.stdin is not None
        assert process.stdout is not None

        process.stdin.write(encode_input(prompt_text, command))
        process.stdin.close()

        TIMED_OUT["value"] = False
        timer = None
        if deadline_s is not None:
            import threading

            def stop_at_deadline() -> None:
                TIMED_OUT["value"] = True
                print(f"claude_code_wsp: time limit of {deadline_s:.0f} s reached, stopping the session", flush=True)
                process.terminate()

            timer = threading.Timer(deadline_s, stop_at_deadline)
            timer.daemon = True
            timer.start()

        guard = ReplyRequestGuard() if limit_requests else None
        for line in process.stdout:
            event = event_from_line(line)
            if guard is not None and event is not None and event.get("type") == "stream_event":
                stop = guard.observe(event)
                if stop is None and not guard.keep(event):
                    continue  # content deltas: the completed blocks arrive as assistant events anyway
            else:
                stop = None
            sys.stdout.write(line)
            sys.stdout.flush()
            raw_output.write(line)
            raw_output.flush()
            if event is not None:
                events.append(event)
                trajectory.write(json.dumps(event) + "\n")
                trajectory.flush()
            if stop is not None:
                process.kill()
                events.append(stop)
                trajectory.write(json.dumps(stop) + "\n")
                trajectory.flush()
                print(f"claude_code_wsp: {stop['result']}", flush=True)
                break

        code = process.wait()
        if timer is not None:
            timer.cancel()
        return code, events


CONTINUE_MESSAGE = "Continue exactly where you left off.\n"


def segment_out_of_credit(events: list[dict[str, Any]]) -> bool:
    texts = [json.dumps(e) for e in events if isinstance(e, dict) and (e.get("error") or e.get("is_error") or e.get("api_error_status"))]
    return out_of_credit(texts)


def stream_with_fallback(command: list[str], prompt_text: str, env: dict[str, str], *, append: bool = False,
                         deadline_s: Optional[float] = None) -> tuple[int, list[dict[str, Any]]]:
    """stream_claude on the current route; if that route runs out of credit mid-run, switch to the next route with
    credit and resume the same session with CONTINUE_MESSAGE, within what is left of the deadline."""
    state = ROUTES["state"]
    start = time.monotonic()
    exit_code, events = stream_claude(command, prompt_text, state.env() if state else env, append=append, deadline_s=deadline_s)
    latest = events
    while state is not None and segment_out_of_credit(latest) and not TIMED_OUT["value"]:
        session = session_id_of(events)
        if session is None or not state.switch():
            break
        marker = {"type": "harness_notice", "text": f"route switched to {state.current()['route']} after an out-of-credit error"}
        events.append(marker)
        with TRAJECTORY_PATH.open("a", encoding="utf-8") as trajectory:
            trajectory.write(json.dumps(marker) + "\n")
        resume = [c for c in command if c != "--resume"]
        if "--resume" in command:
            i = command.index("--resume"); resume = command[:i] + command[i + 2:]
        left = None if deadline_s is None else max(30.0, deadline_s - (time.monotonic() - start))
        exit_code, more = stream_claude([*resume, "--resume", session], CONTINUE_MESSAGE, state.env(), append=True, deadline_s=left)
        events.extend(more)
        latest = more
    return exit_code, events


def session_id_of(events: list[dict[str, Any]]) -> Optional[str]:
    return next((e["session_id"] for e in events if isinstance(e, dict) and e.get("session_id")), None)


def run_with_time_limit(command: list[str], prompt_text: str, env: dict[str, str], *, time_limit_s: float,
                        final_turn_s: float) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """The session runs until it ends or time_limit_s passes. If it is stopped, the same conversation is resumed once
    with TIME_UP_MESSAGE and final_turn_s more seconds. Returns the exit code, every event, and what happened."""
    start = time.monotonic()
    exit_code, events = stream_with_fallback(command, prompt_text, env, deadline_s=time_limit_s)
    record = {"time_limit_s": time_limit_s, "timed_out": TIMED_OUT["value"], "first_part_s": round(time.monotonic() - start, 1)}
    if not TIMED_OUT["value"]:
        return exit_code, events, record
    session = session_id_of(events)
    record["session_id"] = session
    if session is None:
        record["final_turn"] = "no session id, could not resume"
        return exit_code, events, record
    marker = {"type": "harness_notice", "text": TIME_UP_MESSAGE.strip(), "after_s": record["first_part_s"]}
    events.append(marker)
    with TRAJECTORY_PATH.open("a", encoding="utf-8") as trajectory:
        trajectory.write(json.dumps(marker) + "\n")
    resume = [*command, "--resume", session]
    exit_code, more = stream_with_fallback(resume, TIME_UP_MESSAGE, env, append=True, deadline_s=final_turn_s)
    events.extend(more)
    record.update(final_turn="stopped at its own limit" if TIMED_OUT["value"] else "finished", final_turn_s=final_turn_s)
    return exit_code, events, record


class ReplyRequestGuard:
    """At most L0_MAX_REPLY_REQUESTS model requests per L0 reply. A request that stops at `max_tokens` (128k, the API
    maximum, thinking included) is continued by Claude Code's own recovery, which sends up to 3 more requests in the same
    turn, so a reply can spend up to 4 x 128k output tokens. The guard kills claude at the last allowed request's
    `max_tokens` stop, or if one more request starts; the reply then fails the gate (no weights, score 0, no cap retry).
    Needs `--include-partial-messages` for the raw stream events."""

    def __init__(self) -> None:
        self.requests = 0
        self.finished: list[dict[str, int]] = []
        self.usage: dict[str, int] = {}

    def keep(self, event: dict[str, Any]) -> bool:
        return event["event"].get("type") in {"message_start", "message_delta", "message_stop"}

    def observe(self, event: dict[str, Any]) -> Optional[dict[str, Any]]:
        raw = event["event"]
        if raw.get("type") == "message_start":
            self.requests += 1
            if self.usage:
                self.finished.append(self.usage)
            if self.requests > L0_MAX_REPLY_REQUESTS:
                return self.stopped(f"request {self.requests} started, over {L0_MAX_REPLY_REQUESTS} per reply")
            self.usage = {k: v for k, v in (raw.get("message", {}).get("usage") or {}).items() if isinstance(v, int)}
        if raw.get("type") == "message_delta":
            self.usage.update({k: v for k, v in (raw.get("usage") or {}).items() if isinstance(v, int)})
            stop_reason = (raw.get("delta") or {}).get("stop_reason")
            if stop_reason == "max_tokens" and self.requests >= L0_MAX_REPLY_REQUESTS:
                return self.stopped(f"the reply hit the max_tokens output cap on all {L0_MAX_REPLY_REQUESTS} requests")
        return None

    def stopped(self, reason: str) -> dict[str, Any]:
        """A `result` stand in, so the gate fails the reply and the token usage of every request is counted."""
        usages = [*self.finished, self.usage]
        usage = {key: sum(u.get(key, 0) for u in usages) for key in dict.fromkeys(k for u in usages for k in u)}
        return {"type": "result", "subtype": "wsp_request_limit_kill", "is_error": True, "total_cost_usd": None,
                "result": f"{reason}; killed claude", "usage": usage}


def agent_error_texts(events: list[dict[str, Any]]) -> list[str]:
    """API error events (`error` / `is_error` / `api_error_status` set) plus non-JSON output lines, never model text."""
    texts = [json.dumps(event) for event in events
             if event.get("error") or event.get("is_error") or event.get("api_error_status")]
    raw = RAW_OUTPUT_PATH.read_text(encoding="utf-8") if RAW_OUTPUT_PATH.exists() else ""
    return texts + [line for line in raw.splitlines() if event_from_line(line) is None]


def main() -> int:
    args = parse_args()
    task_file = Path(args.problem_statement_path)
    if not task_file.exists():
        raise SystemExit(f"Problem statement path not found: {task_file}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_text = task_file.read_text(encoding="utf-8")
    ROUTES["state"] = RouteState(args.model, direct_first=True)
    env = ROUTES["state"].start()
    model_name = ROUTES["state"].current()["model"]
    print(f"routes: starting on {ROUTES['state'].current()}", flush=True)

    exit_code = 1
    events: list[dict[str, Any]] = []
    if args.access == "l0":
        command = build_l0_command(model=model_name, effort=args.effort)
        env = {**env, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": L0_MAX_OUTPUT_TOKENS, "API_TIMEOUT_MS": L0_API_TIMEOUT_MS,
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        L0_EMPTY_DIR.mkdir(parents=True, exist_ok=True)
    else:
        take_sheet()
        command = build_claude_command(effort=args.effort, tools=args.tools)
    start = time.monotonic()
    try:
        print(f"claude_code_wsp argv: {command}", flush=True)
        if args.access == "l0":
            free_cap = read_free_cap()
            exit_code, events, result = run_l0(command=command, env=env, prompt_text=prompt_text, cwd=L0_EMPTY_DIR,
                                               max_nonzero=read_max_nonzero() if free_cap is None else free_cap,
                                               free=free_cap is not None)
            L0_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"claude_code_wsp l0: {result['stage']}: {str(result['reason'])[:500]}", flush=True)
        else:
            exit_code, events, time_record = run_with_time_limit(command, prompt_text, env, time_limit_s=args.time_limit_s,
                                                                 final_turn_s=args.final_turn_s)
            (OUTPUT_ROOT / "time_limit.json").write_text(json.dumps(time_record, indent=2), encoding="utf-8")
            print(f"claude_code_wsp time limit: {time_record}", flush=True)
            check = check_answer(args.access)
            fix_record: dict[str, Any] = {"first_check": {"ok": check["ok"], "error": check["error"]}}
            session = session_id_of(events)
            if not check["ok"] and session is not None:
                message = FIX_UP_MESSAGE.format(error=check["error"])
                marker = {"type": "harness_notice", "text": message.strip()}
                events.append(marker)
                with TRAJECTORY_PATH.open("a", encoding="utf-8") as trajectory:
                    trajectory.write(json.dumps(marker) + "\n")
                exit_code, more = stream_with_fallback([*command, "--resume", session], message, env, append=True, deadline_s=args.final_turn_s)
                events.extend(more)
                check = check_answer(args.access)
                fix_record["second_check"] = {"ok": check["ok"], "error": check["error"]}
            (OUTPUT_ROOT / "answer_check.json").write_text(json.dumps(fix_record, indent=2), encoding="utf-8")
            print(f"claude_code_wsp answer check: {fix_record}", flush=True)
            if args.access == "builder":
                result = check["detail"]
                BUILDER_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"claude_code_wsp builder: {result['stage']}: {str(result['reason'])[:500]}", flush=True)
        (OUTPUT_ROOT / "route.json").write_text(json.dumps({"final": ROUTES["state"].current(), "log": ROUTES["state"].log}, indent=2), encoding="utf-8")
        if ROUTES["state"].exhausted and out_of_credit(agent_error_texts(events)):
            print("claude_code_wsp: Anthropic key ran out of credit mid-task, failing the task so it is retried",
                  flush=True)
            return OUT_OF_CREDIT_EXIT
        # l0 is graded either way: a failed gate or contract leaves no weights file, which scores 0
        return 0 if args.access == "l0" else exit_code
    finally:
        write_summary(
            task_id=args.task_id,
            model=args.model,
            effort=args.effort,
            tools=args.tools,
            access=args.access,
            command=command,
            prompt_text=prompt_text,
            exit_code=exit_code,
            wall_clock_duration=time.monotonic() - start,
            events=events,
        )


if __name__ == "__main__":
    raise SystemExit(main())
