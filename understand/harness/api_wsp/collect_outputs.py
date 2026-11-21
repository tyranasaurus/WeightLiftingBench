#!/usr/bin/env python3
"""api_wsp: sealed L0 (one reply, no tools) for any model, one harness for every provider.

The task is the codex_wsp / claude_code_wsp / openrouter_wsp L0 task, byte for byte: the same system instructions and
user prompt, the same reply contract (exactly one ```python block, run once by L0_RUNNER with numpy and math only and
`weights` preloaded with zeros from /app/l0/shapes.json), the same nonzero cap and free architecture limit retries in
the same conversation, the same continuation rule (up to L0_MAX_REPLY_REQUESTS requests per reply) and the same /logs
layout. Only the HTTP endpoint depends on the model id:

* anthropic/<model>: Anthropic Messages API with the ANTHROPIC_API_KEY.
* openai/<model>: OpenAI Responses API with the OPENAI_API_KEY.
* google/<model>: Gemini API (streamGenerateContent) with the GEMINI_API_KEY.
* anything else (open weights): OpenRouter chat completions with OPENROUTER_API_KEY.

Closed models never go through OpenRouter: there is no fallback route. When the key is out of credit or rejected,
or the model is missing, the task is an infra ERROR.

Native ids are resolved from the provider's model list (exact id, else the newest dated id: claude-opus-4-1 ->
claude-opus-4-1-20250805). A route whose model is missing (not in the list, or a 404 / not found answer) is dropped like
a route out of credit. Only a real reply that breaks the rules is a scored failure: every route out of credit or without the
model, and any HTTP or stream error that is not a model reply (after retries), exits INFRA_EXIT so the task is an
ERROR that can be retried, never a scored 0.

Stdlib only: requests go out with urllib, so the agent needs no venv.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.request import Request, urlopen

# Lowercase substrings of an Anthropic or OpenAI error that mean the key has no credit left.
OUT_OF_CREDIT_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "api usage limits",
    "billing_error",
    "billing details",
)
BAD_KEY_STATUSES = {401, 403}

WORKSPACE_PATH = Path.cwd()
OUTPUT_ROOT = Path(os.environ.get("AGENT_OUTPUT_ROOT", "/logs"))
TRAJECTORY_PATH = OUTPUT_ROOT / "trajectory.jsonl"
FINAL_MESSAGE_PATH = OUTPUT_ROOT / "final_message.txt"
METRICS_PATH = OUTPUT_ROOT / "metrics_total.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
RAW_OUTPUT_PATH = OUTPUT_ROOT / "raw_output.txt"
# Exit code for a key that ran out of credit mid-task (every route): nonzero fails the task (ERROR) instead of
# grading it as a 0, so the runner can retry it in a fresh sandbox.
OUT_OF_CREDIT_EXIT = 3
L0_RESULT_PATH = OUTPUT_ROOT / "l0_result.json"

# Shared L0 constants, copied from agents/codex_wsp (parity checked by tests/test_claude_code_wsp_cap_retry.py).
L0_SHAPES_PATH = Path("/app/l0/shapes.json")
# Capped tasks ship {"max_nonzero": N}. A reply whose weights hold more nonzero entries gets a follow-up turn in the same
# conversation (the whole message history is resent with CAP_RETRY_MESSAGE as the next user turn) with only the count
# and the cap, at most MAX_CAP_ATTEMPTS replies.
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
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


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


def log_event(event: dict[str, Any]) -> None:
    with TRAJECTORY_PATH.open("a", encoding="utf-8") as trajectory:
        trajectory.write(json.dumps(event) + "\n")


def log_raw(line: str) -> None:
    print(line, flush=True)
    with RAW_OUTPUT_PATH.open("a", encoding="utf-8") as raw_output:
        raw_output.write(line + "\n")


def apply_l0_reply(*, text: str, violations: list[str], free: bool = False) -> dict[str, Any]:
    """Gate the reply, then run its single python block to produce /app/out/weights.npz (and, for a free architecture
    task, /app/out/net.json)."""
    L0_OUTPUT_PATH.unlink(missing_ok=True)  # the file left at the end is always the last reply's
    if free:
        L0_NET_OUTPUT_PATH.unlink(missing_ok=True)
    if violations:
        return {"ok": False, "stage": "gate", "reason": "; ".join(violations)}
    blocks = PYTHON_BLOCK.findall(text.strip())
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



# Endpoints. Every request is streamed (SSE): a long reasoning reply can run for an hour.
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# Gemini API keys go in the x-goog-api-key header; the agent falls back to ?key= and then a bearer token (keys that start
# with "AQ." may be either), picking the first mode the model list call accepts.
GEMINI_AUTH_MODES = ("header", "query", "bearer")
# Gemini 3 and later take thinkingLevel; Gemini 2.x takes a thinking token budget (2.5 Pro allows up to 32768), the
# THINKING_BUDGET_SHARE of it.
GEMINI_THINKING_LEVELS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high"}
GEMINI_MAX_THINKING_BUDGET = 32768
GEMINI_STOPS = {"STOP": "stop", "MAX_TOKENS": "length"}
# Gemini 4 counts thinking and answer together against the output cap: 32,768 non streaming (what /v1beta/models
# reports as outputTokenLimit), 65,536 streaming, more only through the Interactions API in background mode (a cap
# over 65,536 without background is a 400). Gemini 4 therefore runs on the Interactions API with the 128k cap every
# other model gets; other Gemini models stream, and a listed 32,768 is read as the non streaming cap.
GEMINI_INTERACTIONS_FROM = (4, 0)
GEMINI_INTERACTIONS_URL = f"{GEMINI_BASE_URL}/interactions"
GEMINI_API_REVISION = "2026-05-20"
GEMINI_NON_STREAMING_CAP = 32768
GEMINI_STREAMING_CAP = 65536
# Gemini 4 through the Interactions API (background) takes far more than the model list's 32,768: no shared cap, so the
# model's own maximum, the 943,718 OpenRouter lists for Gemini 4 (the API accepted up to 1,048,576 when probed).
GEMINI_INTERACTIONS_MAX_OUTPUT = 943718
INTERACTION_POLL_S = 10.0
INTERACTION_LOG_EVERY = 12  # a progress line every 12 polls (about 2 minutes)
INTERACTION_DONE = {"completed": "stop", "incomplete": "length"}
GEMINI_BAD_KEY_MARKERS = ("api_key_invalid", "api key not valid", "api key expired", "permission_denied")
# A Gemini 429 is usually a per minute rate limit (retried). These mean the key's daily quota or credit is gone, or the
# key is on the free tier with no quota for the model ("free_tier_requests, limit: 0").
GEMINI_OUT_OF_QUOTA_MARKERS = ("perday", "per_day", "per day", "free_tier", "limit: 0", "prepay", "credit",
                               "billing account")
NATIVE_PROVIDERS = ("anthropic", "openai", "google")
APP_HEADERS = {"HTTP-Referer": "https://github.com/wsp-evals", "X-Title": "wsp-evals"}
# Native ids that do not follow the dots to dashes rule (OpenRouter style anthropic/claude-opus-4.6 is
# claude-opus-4-6 natively).
NATIVE_ALIASES = {"anthropic/claude-opus-4": "claude-opus-4-20250514"}
# OpenRouter ids for `-k route openrouter` where they differ from ours (OpenRouter's Claude ids keep the dots:
# anthropic/claude-opus-4.1 is anthropic/claude-opus-4.1).
OPENROUTER_IDS: dict[str, str] = {"anthropic/claude-opus-4.1": "anthropic/claude-opus-4.1"}
# No shared output cap: each request may use the model's own maximum (OpenRouter's top provider, Anthropic's models
# list), since how long a model can reason is part of what is measured. 128k only when the provider reports nothing
# (and on the OpenAI and Gemini routes, whose current models allow no more).
L0_MAX_OUTPUT_TOKENS = 128000
# A request cut at the output cap is continued in the same reply: its text is kept as an assistant turn and
# CONTINUE_MESSAGE asks for the rest, up to L0_MAX_REPLY_REQUESTS requests per reply (as claude_code_wsp). The reply's
# text is joined across those requests, so a python block split by a cut still runs.
L0_MAX_REPLY_REQUESTS = 4
CONTINUE_MESSAGE = (
    "Your reply was cut off at the output token limit. Continue exactly where it stopped, with no recap: what you "
    "write next is appended directly to the text so far.\n"
)
# The claude_code_wsp gate reason for a reply that used every request's output cap; analysis excludes these replies.
OUTPUT_CAP_REASON = "the reply hit the max_tokens output cap on all {requests} requests"
# A cut while the model is still reasoning (no visible text yet, or reasoning was the last thing it emitted) is not
# continued: a continuation could not resume the reasoning, so it would be a fresh restart. The reply is out of budget.
OUTPUT_CAP_REASONING_REASON = "the reply hit the max_tokens output cap while reasoning"
# Claude models through 4.6 take a thinking budget: this share of max_tokens (the ratios OpenRouter uses for reasoning
# effort), so the reply always keeps the rest of the cap for its answer. Opus 4.6 under adaptive thinking at high effort
# thought past the whole 128k cap on 29 of 30 sweep replies; its API still accepts a budget (with effort), later Claude
# rejects one and gets adaptive thinking.
THINKING_BUDGET_SHARE = {"low": 0.2, "medium": 0.5, "high": 0.8, "xhigh": 0.95}
ADAPTIVE_THINKING_FROM = (4, 7)
EFFORT_WITH_BUDGET_FROM = (4, 6)
# Fable models reject a thinking budget; a task budget (beta) instead shows the model a countdown of the same share of
# max_tokens, so it paces its thinking and leaves room for the answer (advisory, not enforced like budget_tokens).
# Opus 5.5 (4 out of budget replies) gets it too; OpenRouter drops task_budget (an invalid one passed without a 400),
# so it only applies on the native Anthropic route.
TASK_BUDGET_MODEL_PREFIX = ("claude-fable-", "claude-opus-5-5")
TASK_BUDGET_BETA = "task-budgets-2026-03-13"
# GLM on OpenRouter thought past the 128k cap at effort high (22 of 30 GLM-5.3 replies); OpenRouter takes either an
# effort or a reasoning token cap, so these models get the cap: the same share of max_tokens, the rest kept for the answer.
# Opus 4.6 through OpenRouter (-k route openrouter) gets its thinking budget the same way: OpenRouter passes
# reasoning.max_tokens to Anthropic as budget_tokens.
REASONING_CAP_MODEL_PREFIXES = ("z-ai/glm-", "anthropic/claude-opus-4.6")
REQUEST_TIMEOUT_S = 12 * 3600  # a reply near a ~944k cap can stream for hours
TRANSIENT_STATUSES = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524, 529}
MAX_HTTP_ATTEMPTS = 8
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 120.0
# Gemini 4 Argon's Interactions API rejects a new interaction at creation (status failed, no tokens, INVALID_ARGUMENT
# "There was a problem processing your request") when interactions start faster than about one a minute: a rate limit
# reported as a bad request (probed 2026-11-16: a burst of 8 all failed, it accepted again after about 45 s). Such a
# rejection waits a random 60 to 120 s, so parallel tasks spread out, and gets more attempts.
GEMINI_CREATE_REJECTION = "There was a problem processing your request"
GEMINI_REJECTION_WAIT_S = (60.0, 120.0)
GEMINI_REJECTION_ATTEMPTS = 20
OPENROUTER_OUT_OF_CREDIT_STATUS = 402
METADATA_TIMEOUT_S = 60


class OutOfCredit(Exception):
    """The key has no credit left (or is invalid)."""


class ModelUnavailable(Exception):
    """The route does not serve the model (404, not found): drop the route, as when every key is out of credit."""


class RequestFailed(Exception):
    """A non transient HTTP or stream failure that is not a model reply: infra, the task errors (INFRA_EXIT)."""


class InfraError(Exception):
    """No route can serve the request (keys out of credit, model missing): the task errors (INFRA_EXIT)."""


# Exit code for infra failures (no credit, model missing on every route, HTTP errors): nonzero makes the task an ERROR
# instead of a scored 0, so the runner can retry it.
INFRA_EXIT = OUT_OF_CREDIT_EXIT
MODEL_MISSING_MARKERS = ("not_found_error", "model_not_found", "does not exist", "no endpoints found",
                         "model not found", "is not a valid model")
DATED_SUFFIX = r"-(?:\d{8}|\d{4}-\d{2}-\d{2})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_statement_path")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="high")
    parser.add_argument("--route", default="auto", choices=["auto", "openrouter"])
    parser.add_argument("--mode", default="l0", choices=["l0", "read", "mcq"])
    return parser.parse_args()


def get_json(*, url: str, headers: dict[str, str]) -> Optional[dict[str, Any]]:
    """GET a JSON document, or None on any failure (metadata only: never blocks a run)."""
    try:
        with urlopen(Request(url, headers=headers), timeout=METADATA_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        log_raw(f"api_wsp: GET {url.split('?')[0]} failed: {str(error)[:200]}")  # the query may hold a key
        return None


def native_model(model: str) -> tuple[str, str]:
    """(provider, native id) for a model id: anthropic/claude-opus-4.6 -> (anthropic, claude-opus-4-6),
    openai/gpt-6.1-sol -> (openai, gpt-6.1-sol), anything else -> (openrouter, model)."""
    if model in NATIVE_ALIASES:
        return model.split("/")[0], NATIVE_ALIASES[model]
    if model.startswith("anthropic/"):
        return "anthropic", model.removeprefix("anthropic/").replace(".", "-")
    if model.startswith("openai/"):
        return "openai", model.removeprefix("openai/")
    if model.startswith("google/"):
        return "google", model.removeprefix("google/")
    return "openrouter", model


def gemini_version(native: str) -> tuple[int, int]:
    """gemini-2.5-pro -> (2, 5), gemini-3.1-pro-preview -> (3, 1), gemini-4-argon -> (4, 0)."""
    match = re.match(r"gemini-(\d+)(?:\.(\d+))?", native)
    return (int(match.group(1)), int(match.group(2) or 0)) if match else (0, 0)


def claude_version(native: str) -> tuple[int, int]:
    """claude-opus-4-6 -> (4, 6), claude-fable-5 -> (5, 0), claude-opus-4-20250514 -> (4, 0), claude-3-5-haiku -> (3, 5)."""
    numbers = [int(part) for part in native.split("-") if part.isdigit() and len(part) < 8]
    return (numbers + [0, 0])[0], (numbers + [0, 0])[1]


@dataclass(kw_only=True)
class Route:
    name: str  # anthropic | openai | openrouter
    model: str  # the id this endpoint takes
    keys: list[str]
    max_tokens: int
    reasoning: bool = True
    verified: Optional[bool] = None  # the model list call found the id (None: not checked)
    auth: str = "header"  # Gemini key placement (GEMINI_AUTH_MODES)
    api: str = ""  # Gemini: "interactions" (Gemini 4) or "generate" (streamGenerateContent)
    dead_keys: list[int] = field(default_factory=list)
    key_index: int = 0

    def live_key(self) -> Optional[str]:
        alive = [i for i in range(len(self.keys)) if i not in self.dead_keys]
        if not alive:
            return None
        if self.key_index in self.dead_keys or self.key_index >= len(self.keys):
            self.key_index = alive[0]
        return self.keys[self.key_index]

    def mark_dead(self) -> None:
        self.dead_keys.append(self.key_index)

    def describe(self) -> dict[str, Any]:
        return {"route": self.name, "model": self.model, "max_tokens": self.max_tokens, "reasoning": self.reasoning,
                "verified": self.verified, "keys": len(self.keys), "dead_keys": len(self.dead_keys),
                **({"auth": self.auth, "api": self.api} if self.name == "google" else {})}


def openrouter_route(*, model: str, models: Optional[list[dict[str, Any]]]) -> Optional[Route]:
    """The OpenRouter route for a model, or None when OpenRouter does not list it (with a list to check)."""
    keys = api_key("OPENROUTER_API_KEY")
    entry = next((m for m in models or [] if m.get("id") == model), None)
    if models is not None and entry is None:
        return None
    top = ((entry or {}).get("top_provider") or {}).get("max_completion_tokens")
    return Route(name="openrouter", model=model, keys=keys,
                 max_tokens=int(top) if top else L0_MAX_OUTPUT_TOKENS,
                 reasoning=entry is None or "reasoning" in (entry.get("supported_parameters") or []),
                 verified=None if entry is None else True)


def api_key(variable: str) -> list[str]:
    """The one API key in an environment variable, as a list (empty when the variable is not set)."""
    key = os.environ.get(variable, "").strip()
    return [key] if key else []


def list_models(*, url: str, headers: dict[str, str], paged: bool) -> Optional[list[dict[str, Any]]]:
    """Every model in a provider's list (Anthropic pages with has_more / last_id), or None when the call fails."""
    models: list[dict[str, Any]] = []
    after: Optional[str] = None
    while True:
        query = ("?limit=1000" + (f"&after_id={after}" if after else "")) if paged else ""
        page = get_json(url=url + query, headers=headers)
        if page is None:
            return None
        data = [m for m in page.get("data") or [] if isinstance(m, dict)]
        models += data
        if not paged or not page.get("has_more") or not data:
            return models
        after = page.get("last_id") or data[-1].get("id")


def resolve_native(*, native: str, models: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The listed model for a native id: the exact id, else the newest dated one (claude-opus-4-1 ->
    claude-opus-4-1-20250805; claude-opus-4 never matches claude-opus-4-1-...)."""
    exact = next((m for m in models if m.get("id") == native), None)
    if exact is not None:
        return exact
    dated = sorted((m for m in models if re.fullmatch(re.escape(native) + DATED_SUFFIX, str(m.get("id")))),
                   key=lambda m: str(m["id"]))
    return dated[-1] if dated else None


def gemini_auth(*, mode: str, key: str) -> tuple[dict[str, str], str]:
    """(headers, query suffix) for a Gemini key in the given placement."""
    if mode == "query":
        return {}, f"key={key}"
    if mode == "bearer":
        return {"Authorization": f"Bearer {key}"}, ""
    return {"x-goog-api-key": key}, ""


def list_gemini_models(*, key: str) -> tuple[Optional[list[dict[str, Any]]], str]:
    """(models with the "models/" prefix stripped from `id`, auth mode that worked), or (None, "header")."""
    for mode in GEMINI_AUTH_MODES:
        headers, auth_query = gemini_auth(mode=mode, key=key)
        models: list[dict[str, Any]] = []
        token: Optional[str] = None
        while True:
            query = "&".join(q for q in ["pageSize=1000", f"pageToken={token}" if token else "", auth_query] if q)
            page = get_json(url=f"{GEMINI_BASE_URL}/models?{query}", headers=headers)
            if page is None:
                break
            models += [{**m, "id": str(m.get("name", "")).removeprefix("models/")} for m in page.get("models") or []]
            token = page.get("nextPageToken")
            if not token:
                return models, mode
    return None, "header"


def google_route(*, native: str) -> Route:
    keys = api_key("GEMINI_API_KEY")
    models, auth = list_gemini_models(key=keys[0]) if keys else (None, "header")
    entry = resolve_native(native=native, models=models) if models is not None else None
    model = (entry or {}).get("id") or native
    interactions = gemini_version(model) >= GEMINI_INTERACTIONS_FROM
    cap = int((entry or {}).get("outputTokenLimit") or GEMINI_STREAMING_CAP)
    cap = GEMINI_STREAMING_CAP if cap == GEMINI_NON_STREAMING_CAP else cap
    route = Route(name="google", model=model, keys=keys, auth=auth, api="interactions" if interactions else "generate",
                  max_tokens=GEMINI_INTERACTIONS_MAX_OUTPUT if interactions else min(L0_MAX_OUTPUT_TOKENS, cap),
                  reasoning=bool((entry or {}).get("thinking", True)),
                  verified=None if models is None else entry is not None)
    if route.verified is False:
        log_raw(f"api_wsp: {native} is not in the google model list, skipping the google route")
        route.dead_keys = list(range(len(keys)))
    return route


def direct_route(*, provider: str, native: str) -> Route:
    """The Anthropic or OpenAI route with the native id resolved from the provider's model list. A model the list
    does not hold gives a route with every key dead (verified False), so the router goes straight to the fallback."""
    variable = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    keys = api_key(variable)
    if provider == "anthropic":
        headers = {"x-api-key": keys[0], "anthropic-version": ANTHROPIC_VERSION} if keys else {}
        models = list_models(url=ANTHROPIC_MODELS_URL, headers=headers, paged=True) if keys else None
    else:
        headers = {"Authorization": f"Bearer {keys[0]}"} if keys else {}
        models = list_models(url=OPENAI_MODELS_URL, headers=headers, paged=False) if keys else None
    entry = resolve_native(native=native, models=models) if models is not None else None
    cap = (entry or {}).get("max_tokens") if provider == "anthropic" else None
    if provider == "anthropic" and entry is not None and not cap:
        cap = (get_json(url=f"{ANTHROPIC_MODELS_URL}/{entry['id']}", headers=headers) or {}).get("max_tokens")
    route = Route(name=provider, model=(entry or {}).get("id") or native, keys=keys,
                  max_tokens=int(cap) if cap else L0_MAX_OUTPUT_TOKENS,
                  verified=None if models is None else entry is not None)
    if route.verified is False:
        log_raw(f"api_wsp: {native} is not in the {provider} model list, skipping the {provider} route")
        route.dead_keys = list(range(len(keys)))
    return route


def plan_routes(*, model: str, route: str = "auto") -> list[Route]:
    """The one route for a model. auto: its native API for anthropic/, openai/ and google/ ids (never OpenRouter),
    OpenRouter for every other (open weights) id. openrouter (-k route openrouter): OpenRouter for any id, with no
    native route (for models a provider has retired from its own API, e.g. anthropic/claude-opus-4.1)."""
    provider, native = native_model(model)
    if route == "openrouter":
        provider, model = "openrouter", OPENROUTER_IDS.get(model, model)
    if provider == "google":
        return [google_route(native=native)]
    if provider in NATIVE_PROVIDERS:
        return [direct_route(provider=provider, native=native)]
    listing = get_json(url=OPENROUTER_MODELS_URL, headers=APP_HEADERS)
    models = listing.get("data") if isinstance(listing, dict) else None
    return [openrouter_route(model=model, models=models) or openrouter_route(model=model, models=None)]


def native_turn(*, route: Route, turn: dict[str, Any]) -> Any:
    """An assistant turn of a continuation in the route's own format, reasoning included, when it has one."""
    return (turn.get("native") or {}).get(route.name)


# Set in main for read mode, the only multi-turn mode; a single-reply mode would pay the cache-write premium for nothing.
CACHE_PROMPTS = False


def with_cache_breakpoint(turns: list[Any]) -> list[Any]:
    """Claude only caches a prompt prefix it is told to: mark the newest turn with an ephemeral cache_control, so each
    multi-turn request reads the conversation so far from cache instead of paying full input price for it again
    (read mode resends the whole history every cell). Works on the Anthropic API and on OpenRouter's Claude routes."""
    last = turns[-1] if turns else None
    if not isinstance(last, dict) or not isinstance(last.get("content"), str):
        return turns
    block = {"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}
    return [*turns[:-1], {**last, "content": [block]}]


def build_body(*, route: Route, effort: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The request body for a route. `messages` is the neutral conversation: a system turn, then user and assistant
    turns; an assistant turn cut at the output cap also holds the route's full turn (reasoning included) under
    `native`. No tools field on any route."""
    system = messages[0]["content"]
    turns = [native_turn(route=route, turn=t) or {"role": t["role"], "content": t["content"]} for t in messages[1:]]
    if CACHE_PROMPTS and (route.name == "anthropic" or (route.name == "openrouter" and route.model.startswith("anthropic/"))):
        turns = with_cache_breakpoint(turns)
    if route.name == "openrouter":
        body: dict[str, Any] = {"model": route.model, "messages": [{"role": "system", "content": system}, *turns],
                                "max_tokens": route.max_tokens, "stream": True, "usage": {"include": True}}
        if route.model in OPENROUTER_PROVIDER_IGNORE:
            body["provider"] = {"ignore": OPENROUTER_PROVIDER_IGNORE[route.model]}
        if route.reasoning and route.model.startswith(REASONING_CAP_MODEL_PREFIXES):
            body["reasoning"] = {"max_tokens": int(route.max_tokens * THINKING_BUDGET_SHARE[effort])}
        elif route.reasoning:
            body["reasoning"] = {"effort": effort}
        return body
    if route.name == "anthropic":
        body = {"model": route.model, "system": system, "messages": turns, "max_tokens": route.max_tokens,
                "stream": True}
        if claude_version(route.model) >= ADAPTIVE_THINKING_FROM:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
            if route.model.startswith(TASK_BUDGET_MODEL_PREFIX):
                total = int(route.max_tokens * THINKING_BUDGET_SHARE[effort])
                body["output_config"]["task_budget"] = {"type": "tokens", "total": total}
        else:
            budget = max(1024, int(route.max_tokens * THINKING_BUDGET_SHARE[effort]))
            body["thinking"] = {"type": "enabled", "budget_tokens": min(budget, route.max_tokens - 1024)}
            if claude_version(route.model) >= EFFORT_WITH_BUDGET_FROM:
                body["output_config"] = {"effort": effort}
        return body
    if route.name == "google" and route.api == "interactions":
        steps: list[dict[str, Any]] = []
        for t in turns:
            if isinstance(t, list):  # a cut turn passed back as received: thought steps with signatures, outputs
                steps += t
            else:
                kind = "model_output" if t["role"] == "assistant" else "user_input"
                steps.append({"type": kind, "content": [{"type": "text", "text": t["content"]}]})
        config = {"max_output_tokens": route.max_tokens, "thinking_summaries": "auto"}
        if route.reasoning:
            config["thinking_level"] = GEMINI_THINKING_LEVELS[effort]
        # background runs the interaction server side (needed for a cap over 65,536); it requires store.
        return {"model": route.model, "system_instruction": system, "input": steps, "generation_config": config,
                "background": True, "store": True}
    if route.name == "google":
        if gemini_version(route.model) >= (3, 0):
            thinking: dict[str, Any] = {"thinkingLevel": GEMINI_THINKING_LEVELS[effort], "includeThoughts": True}
        else:
            budget = int(min(GEMINI_MAX_THINKING_BUDGET, route.max_tokens) * THINKING_BUDGET_SHARE[effort])
            thinking = {"thinkingBudget": budget, "includeThoughts": True}
        config: dict[str, Any] = {"maxOutputTokens": route.max_tokens}
        if route.reasoning:
            config["thinkingConfig"] = thinking
        contents = [t if "parts" in t else {"role": "model" if t["role"] == "assistant" else "user",
                                             "parts": [{"text": t["content"]}]} for t in turns]
        return {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": config}
    items = [item for t in turns for item in (t if isinstance(t, list) else [t])]
    return {"model": route.model, "instructions": system, "input": items, "max_output_tokens": route.max_tokens,
            "reasoning": {"effort": effort, "summary": "detailed"}, "stream": True, "store": False,
            "include": ["reasoning.encrypted_content"]}


def request_url(*, route: Route, key: str) -> str:
    if route.name == "google" and route.api == "interactions":
        _, auth_query = gemini_auth(mode=route.auth, key=key)
        return GEMINI_INTERACTIONS_URL + (f"?{auth_query}" if auth_query else "")
    if route.name == "google":
        _, auth_query = gemini_auth(mode=route.auth, key=key)
        return f"{GEMINI_BASE_URL}/models/{route.model}:streamGenerateContent?alt=sse" + (f"&{auth_query}" if auth_query else "")
    return {"openrouter": OPENROUTER_CHAT_URL, "anthropic": ANTHROPIC_MESSAGES_URL, "openai": OPENAI_RESPONSES_URL}[route.name]


def request_headers(*, route: Route, key: str) -> dict[str, str]:
    if route.name == "anthropic":
        beta = {"anthropic-beta": TASK_BUDGET_BETA} if route.model.startswith(TASK_BUDGET_MODEL_PREFIX) else {}
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json", **beta}
    if route.name == "openai":
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if route.name == "google":
        revision = {"Api-Revision": GEMINI_API_REVISION} if route.api == "interactions" else {}
        return {**gemini_auth(mode=route.auth, key=key)[0], **revision, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **APP_HEADERS}


class DeadlineReached(Exception):
    """Read mode's wall-clock deadline passed while a reply was still streaming (run_read asks for the answer)."""


# Monotonic time after which a streaming reply is abandoned; None outside read mode. Set by run_read.
STREAM_DEADLINE: Optional[float] = None


def past_deadline() -> None:
    if STREAM_DEADLINE is not None and time.monotonic() > STREAM_DEADLINE:
        raise DeadlineReached("read-mode deadline passed")


def socket_timeout() -> float:
    """Per-read socket timeout: the long default, or (in read mode) no later than just past the deadline, so a stream
    that goes quiet cannot hold the task past it."""
    if STREAM_DEADLINE is None:
        return REQUEST_TIMEOUT_S
    return max(5.0, STREAM_DEADLINE - time.monotonic() + 5.0)


# Timing of the current stream, so a request that ends at the read deadline shows whether the provider sent nothing
# (a hang) or kept streaming (long thinking): seconds to the first byte and to the last line, and the line count.
STREAM_STATS: dict[str, Any] = {}


def sse_events(lines: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """JSON payloads of an SSE stream: `data: {...}` lines; comments, `event:` lines and [DONE] skipped."""
    start = time.monotonic()
    STREAM_STATS.clear()
    STREAM_STATS.update(lines=0, first_s=None, last_s=None)
    for raw in lines:
        now = time.monotonic() - start
        STREAM_STATS["lines"] += 1
        STREAM_STATS["last_s"] = round(now, 1)
        if STREAM_STATS["first_s"] is None:
            STREAM_STATS["first_s"] = round(now, 1)
        past_deadline()
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def empty_completion(route: Route) -> dict[str, Any]:
    """One request's result. `last_kind` is what the model was emitting when the stream ended (reasoning or text);
    `turn` is the full assistant turn in the route's own format (reasoning included) for a continuation, and
    `reasoning_passback` says whether that turn carries the reasoning state the API needs to resume it."""
    return {"route": route.name, "id": None, "model": None, "provider": None, "content": "", "reasoning": "",
            "finish_reason": None, "usage": None, "raw_usage": None, "tool_calls": [], "error": None,
            "last_kind": None, "turn": None, "reasoning_passback": False}


def assemble_openrouter(*, events: Iterator[dict[str, Any]], route: Route) -> dict[str, Any]:
    reply = empty_completion(route)
    details: dict[Any, dict[str, Any]] = {}  # reasoning_details merged by index across chunks
    for chunk in events:
        for key in ("id", "model", "provider"):
            reply[key] = chunk.get(key) or reply[key]
        if chunk.get("usage"):
            usage = chunk["usage"]
            count = lambda group, name: int(((usage.get(group) or {}).get(name)) or 0)
            reply["raw_usage"] = usage
            reply["usage"] = {"input_tokens": int(usage.get("prompt_tokens") or 0),
                              "cached_input_tokens": count("prompt_tokens_details", "cached_tokens"),
                              "output_tokens": int(usage.get("completion_tokens") or 0),
                              "reasoning_output_tokens": count("completion_tokens_details", "reasoning_tokens"),
                              "cost": usage.get("cost") if isinstance(usage.get("cost"), (int, float)) else None}
        if chunk.get("error"):
            reply["error"] = chunk["error"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning") or delta.get("reasoning_details"):
                reply["last_kind"] = "reasoning"
            if delta.get("content"):
                reply["last_kind"] = "text"
            reply["content"] += delta.get("content") or ""
            reply["reasoning"] += delta.get("reasoning") or ""
            for item in delta.get("reasoning_details") or []:
                merged = details.setdefault(item.get("index", len(details)), {})
                for name, value in item.items():
                    if name in ("text", "summary", "data") and isinstance(value, str):
                        merged[name] = merged.get(name, "") + value
                    elif value is not None:
                        merged[name] = value
            reply["tool_calls"] += delta.get("tool_calls") or []
            reply["finish_reason"] = choice.get("finish_reason") or reply["finish_reason"]
            if choice.get("error"):
                reply["error"] = choice["error"]
    turn: dict[str, Any] = {"role": "assistant", "content": reply["content"]}
    if details:
        turn["reasoning_details"] = [details[k] for k in sorted(details, key=str)]
    reply["turn"] = turn
    # No reasoning_details but reasoning text: the provider gives nothing to resume from, the turn is text only.
    reply["reasoning_passback"] = bool(details) or not reply["reasoning"]
    return reply


ANTHROPIC_STOPS = {"max_tokens": "length", "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls"}


def assemble_anthropic(*, events: Iterator[dict[str, Any]], route: Route) -> dict[str, Any]:
    reply = empty_completion(route)
    reply["provider"] = "anthropic"
    usage: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}  # content blocks by index: thinking (with signature), redacted_thinking, text
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            reply["id"], reply["model"] = message.get("id"), message.get("model")
            usage.update(message.get("usage") or {})
        elif kind == "content_block_start":
            block = dict(event.get("content_block") or {})
            if block.get("type") == "tool_use":
                reply["tool_calls"].append(block)
            blocks[int(event.get("index", len(blocks)))] = block
            reply["last_kind"] = "text" if block.get("type") == "text" else "reasoning"
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            block = blocks.setdefault(int(event.get("index", len(blocks) - 1)), {})
            if delta.get("type") == "text_delta":
                reply["content"] += delta.get("text") or ""
                block["text"] = block.get("text", "") + (delta.get("text") or "")
                reply["last_kind"] = "text"
            elif delta.get("type") == "thinking_delta":
                reply["reasoning"] += delta.get("thinking") or ""
                block["thinking"] = block.get("thinking", "") + (delta.get("thinking") or "")
                reply["last_kind"] = "reasoning"
            elif delta.get("type") == "signature_delta":
                block["signature"] = block.get("signature", "") + (delta.get("signature") or "")
        elif kind == "message_delta":
            usage.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})
            stop = (event.get("delta") or {}).get("stop_reason")
            reply["finish_reason"] = ANTHROPIC_STOPS.get(stop, stop) if stop else reply["finish_reason"]
        elif kind == "error":
            reply["error"] = event.get("error") or event
    if usage:
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        reply["raw_usage"] = usage
        reply["usage"] = {"input_tokens": int(usage.get("input_tokens") or 0) + cache_read + cache_write,
                          "cached_input_tokens": cache_read, "output_tokens": int(usage.get("output_tokens") or 0),
                          "reasoning_output_tokens": 0, "cost": None}
    content = [blocks[i] for i in sorted(blocks) if not (blocks[i].get("type") == "text" and not blocks[i].get("text"))]
    reply["turn"] = {"role": "assistant", "content": content}
    reply["reasoning_passback"] = all(b.get("signature") for b in content if b.get("type") == "thinking")
    return reply


def assemble_openai(*, events: Iterator[dict[str, Any]], route: Route) -> dict[str, Any]:
    reply = empty_completion(route)
    reply["provider"] = "openai"
    items: list[dict[str, Any]] = []  # output items as they finish (reasoning with encrypted_content, message)
    for event in events:
        kind = event.get("type", "")
        if kind == "response.output_text.delta":
            reply["content"] += event.get("delta") or ""
            reply["last_kind"] = "text"
        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            reply["reasoning"] += event.get("delta") or ""
            reply["last_kind"] = "reasoning"
        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") in ("function_call", "custom_tool_call", "web_search_call"):
                reply["tool_calls"].append(item)
            reply["last_kind"] = "reasoning" if item.get("type") == "reasoning" else "text"
        elif kind == "response.output_item.done" and isinstance(event.get("item"), dict):
            items.append(event["item"])
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            response = event.get("response") or {}
            reply["id"], reply["model"] = response.get("id"), response.get("model")
            items = [i for i in response.get("output") or [] if isinstance(i, dict)] or items
            usage = response.get("usage") or {}
            if usage:
                count = lambda group, name: int(((usage.get(group) or {}).get(name)) or 0)
                reply["raw_usage"] = usage
                reply["usage"] = {"input_tokens": int(usage.get("input_tokens") or 0),
                                  "cached_input_tokens": count("input_tokens_details", "cached_tokens"),
                                  "output_tokens": int(usage.get("output_tokens") or 0),
                                  "reasoning_output_tokens": count("output_tokens_details", "reasoning_tokens"),
                                  "cost": None}
            if kind == "response.completed":
                reply["finish_reason"] = "stop"
            elif kind == "response.incomplete":
                reason = (response.get("incomplete_details") or {}).get("reason")
                reply["finish_reason"] = "length" if reason == "max_output_tokens" else reason or "incomplete"
            else:
                reply["error"] = response.get("error") or {"message": "response.failed"}
        elif kind == "error":
            reply["error"] = event
    # Items go back as input with ids and status dropped (store is false, so the server holds no item to point at),
    # as codex does; a reasoning item resumes from its encrypted_content.
    reply["turn"] = [{k: v for k, v in i.items() if k not in ("id", "status")} for i in items]
    reasoning_items = [i for i in items if i.get("type") == "reasoning"]
    reply["reasoning_passback"] = all(i.get("encrypted_content") for i in reasoning_items)
    return reply


def assemble_google(*, events: Iterator[dict[str, Any]], route: Route) -> dict[str, Any]:
    reply = empty_completion(route)
    reply["provider"] = "google"
    parts: list[dict[str, Any]] = []  # every part in order, thought parts and thoughtSignature kept
    for chunk in events:
        if chunk.get("error"):
            reply["error"] = chunk["error"]
            continue
        reply["id"] = chunk.get("responseId") or reply["id"]
        reply["model"] = chunk.get("modelVersion") or reply["model"]
        usage = chunk.get("usageMetadata")
        if usage:  # cumulative: the last chunk's is the request's
            thoughts = int(usage.get("thoughtsTokenCount") or 0)
            reply["raw_usage"] = usage
            reply["usage"] = {"input_tokens": int(usage.get("promptTokenCount") or 0),
                              "cached_input_tokens": int(usage.get("cachedContentTokenCount") or 0),
                              "output_tokens": int(usage.get("candidatesTokenCount") or 0) + thoughts,
                              "reasoning_output_tokens": thoughts, "cost": None}
        blocked = (chunk.get("promptFeedback") or {}).get("blockReason")
        if blocked:  # a blocked prompt with no output is not a model reply: retried, then infra (never scored)
            reply["error"] = {"blocked": blocked, "promptFeedback": chunk.get("promptFeedback")}
        for candidate in chunk.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                parts.append(part)
                if "functionCall" in part or "executableCode" in part:
                    reply["tool_calls"].append(part)
                elif part.get("thought"):
                    reply["reasoning"] += part.get("text") or ""
                    reply["last_kind"] = "reasoning"
                elif part.get("text"):
                    reply["content"] += part["text"]
                    reply["last_kind"] = "text"
            stop = candidate.get("finishReason")
            if stop:
                reply["finish_reason"] = GEMINI_STOPS.get(stop, stop.lower())
    reply["api"] = "generate"
    reply["turn"] = {"role": "model", "parts": [p for p in parts if p.get("text") or p.get("thoughtSignature")]}
    thought = any(p.get("thought") for p in parts)
    reply["reasoning_passback"] = not thought or any(p.get("thoughtSignature") for p in parts)
    return reply


def assemble_interaction(*, interaction: dict[str, Any], route: Route) -> dict[str, Any]:
    """A finished Gemini interaction (Interactions API) as a completion: thought steps (summary and signature) are the
    reasoning, model_output steps the text."""
    reply = empty_completion(route)
    reply["provider"], reply["api"] = "google", "interactions"
    reply["id"], reply["model"] = interaction.get("id"), interaction.get("model")
    steps = [s for s in interaction.get("steps") or [] if isinstance(s, dict)]
    # The interaction echoes its input history in steps; only what follows the last user_input is this reply.
    # (Multi-turn read mode otherwise re-reads every earlier model_output as new text.)
    last_input = max((i for i, s in enumerate(steps) if s.get("type") == "user_input"), default=-1)
    steps = steps[last_input + 1:]
    texts =lambda blocks: "".join(b.get("text") or "" for b in blocks or [] if isinstance(b, dict))
    for step in steps:
        if step.get("type") == "thought":
            reply["reasoning"] += texts(step.get("summary"))
            reply["last_kind"] = "reasoning"
        elif step.get("type") == "model_output":
            text = texts(step.get("content"))
            reply["content"] += text
            reply["last_kind"] = "text" if text else reply["last_kind"]
        elif step.get("type") not in ("user_input",):
            reply["tool_calls"].append(step)
    usage = interaction.get("usage") or {}
    if usage:
        thoughts = int(usage.get("total_thought_tokens") or 0)
        reply["raw_usage"] = usage
        reply["usage"] = {"input_tokens": int(usage.get("total_input_tokens") or 0),
                          "cached_input_tokens": int(usage.get("total_cached_tokens") or 0),
                          "output_tokens": int(usage.get("total_output_tokens") or 0) + thoughts,
                          "reasoning_output_tokens": thoughts, "cost": None}
    status = interaction.get("status")
    if status in INTERACTION_DONE:
        reply["finish_reason"] = INTERACTION_DONE[status]
    elif status == "requires_action":
        reply["finish_reason"] = "tool_calls"
    else:  # failed, cancelled, requires_continuation: not a model reply
        errors = interaction.get("errors") or []
        code = next((e.get("code") for e in errors if isinstance(e, dict)), None)
        reply["error"] = {"status": status, "code": code if isinstance(code, int) else None, "errors": errors}
    reply["turn"] = [s for s in steps if s.get("type") in ("thought", "model_output")]
    reply["reasoning_passback"] = all(s.get("signature") for s in steps if s.get("type") == "thought")
    return reply


def run_interaction(*, route: Route, data: bytes, key: str) -> dict[str, Any]:
    """Create a background interaction, then poll it until it is no longer in progress. An HTTP error on create
    propagates to post_once's handling; a poll survives transient errors (the interaction keeps running server side)."""
    headers = request_headers(route=route, key=key)
    with urlopen(Request(request_url(route=route, key=key), data=data, method="POST", headers=headers),
                 timeout=METADATA_TIMEOUT_S * 5) as response:
        interaction = json.loads(response.read().decode("utf-8"))
    interaction_id = interaction.get("id")
    log_raw(f"api_wsp: google interaction {interaction_id} {interaction.get('status')}")
    _, auth_query = gemini_auth(mode=route.auth, key=key)
    poll_url = f"{GEMINI_INTERACTIONS_URL}/{interaction_id}" + (f"?{auth_query}" if auth_query else "")
    deadline, failures, polls = time.monotonic() + REQUEST_TIMEOUT_S, 0, 0
    while interaction.get("status") == "in_progress":
        polls += 1
        past_deadline()  # read mode: a background interaction still thinking at the deadline is cut like a stream
        if polls % INTERACTION_LOG_EVERY == 0:
            log_raw(f"api_wsp: google interaction {interaction_id} still in progress after {polls} polls")
        if time.monotonic() > deadline:
            raise RequestFailed(f"google interaction {interaction_id} still in progress after {REQUEST_TIMEOUT_S} s")
        time.sleep(INTERACTION_POLL_S)
        try:
            with urlopen(Request(poll_url, headers=headers), timeout=METADATA_TIMEOUT_S * 5) as response:
                interaction = json.loads(response.read().decode("utf-8"))
            failures = 0
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_STATUSES or failures >= MAX_HTTP_ATTEMPTS:
                raise RequestFailed(f"google interaction {interaction_id} poll HTTP {error.code}: "
                                    f"{error.read().decode('utf-8', 'replace')[:500]}") from error
            failures += 1
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if failures >= MAX_HTTP_ATTEMPTS:
                raise RequestFailed(f"google interaction {interaction_id} poll failed: {error}") from error
            failures += 1
    return assemble_interaction(interaction=interaction, route=route)


ASSEMBLERS = {"openrouter": assemble_openrouter, "anthropic": assemble_anthropic, "openai": assemble_openai,
              "google": assemble_google}


def is_out_of_credit(*, route: Route, status: Optional[int], text: str) -> bool:
    if route.name == "openrouter":
        return status in (OPENROUTER_OUT_OF_CREDIT_STATUS, 401)
    lower = text.lower()
    if route.name == "google":
        return (status in BAD_KEY_STATUSES or any(m in lower for m in GEMINI_BAD_KEY_MARKERS)
                or (status in (429, None) and "resource_exhausted" in lower
                    and any(m in lower for m in GEMINI_OUT_OF_QUOTA_MARKERS)))
    return status in BAD_KEY_STATUSES or any(marker in text.lower() for marker in OUT_OF_CREDIT_MARKERS)


# OpenRouter answers 402 when the worst-case cost of every request in flight (max_tokens x price) would exceed the
# balance, even though the key still has credit, and 429 for its per-minute cap on new accounts. Both clear on their own.
OPENROUTER_BUSY_MARKERS = ("in_flight_budget", "in-flight requests", "new-account-rpm")
OPENROUTER_BUSY_ATTEMPTS = 40
OPENROUTER_BUSY_WAIT_S = (20.0, 60.0)


def is_openrouter_busy(*, status: Optional[int], text: str) -> bool:
    return status in (402, 429) and any(m in text.lower() for m in OPENROUTER_BUSY_MARKERS)


def is_model_missing(*, status: Optional[int], text: str) -> bool:
    return status == 404 or any(marker in text.lower() for marker in MODEL_MISSING_MARKERS)


def is_transient_stream_error(error: Any) -> bool:
    """A stream that errors mid reply: overloaded / rate limit / server errors are retried, anything else is not."""
    text = json.dumps(error).lower()
    code = error.get("code") if isinstance(error, dict) else None
    return (code in TRANSIENT_STATUSES or code is None) and not any(m in text for m in ("invalid_request", "not_found"))


def post_once(*, route: Route, body: dict[str, Any], key: str) -> dict[str, Any]:
    """One request with transient retries and backoff. Raises OutOfCredit for this key, RequestFailed otherwise."""
    data = json.dumps(body).encode("utf-8")
    failure = ""
    attempts = MAX_HTTP_ATTEMPTS
    attempt = 0
    while attempt < attempts:
        attempt += 1
        past_deadline()
        try:
            if route.name == "google" and route.api == "interactions":
                reply = run_interaction(route=route, data=data, key=key)
            else:
                STREAM_STATS.clear()  # stays empty if the response headers never arrive
                request = Request(request_url(route=route, key=key), data=data, method="POST",
                                  headers=request_headers(route=route, key=key))
                with urlopen(request, timeout=socket_timeout()) as response:
                    reply = ASSEMBLERS[route.name](events=sse_events(iter(response)), route=route)
            if reply["error"] is None and reply["finish_reason"] is not None:
                return reply
            error_text = json.dumps(reply["error"])
            if reply["error"] is not None and is_out_of_credit(route=route, status=None, text=error_text):
                raise OutOfCredit(f"stream error: {error_text[:500]}")
            if reply["error"] is not None and is_model_missing(status=None, text=error_text):
                raise ModelUnavailable(f"stream error: {error_text[:500]}")
            if reply["error"] is not None and not is_transient_stream_error(reply["error"]):
                raise RequestFailed(f"stream error: {error_text[:1000]}")
            failure = f"stream ended without a finish reason, error {error_text[:500]}"
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", "replace")[:1000]
            if route.name == "openrouter" and is_openrouter_busy(status=error.code, text=text):
                # credit is fine; too many requests in flight at once (402 in_flight_budget_exhausted) or the
                # new-account per-minute cap (429): wait for other requests to settle instead of killing the task
                failure = f"HTTP {error.code}: {text}"
                attempts = max(attempts, OPENROUTER_BUSY_ATTEMPTS)
                log_raw(f"api_wsp: openrouter busy (attempt {attempt} of {attempts}): {text[:200]}")
                if attempt < attempts:
                    time.sleep(random.uniform(*OPENROUTER_BUSY_WAIT_S))
                continue
            if is_out_of_credit(route=route, status=error.code, text=text):
                raise OutOfCredit(f"HTTP {error.code}: {text[:300]}") from error
            if is_model_missing(status=error.code, text=text):
                raise ModelUnavailable(f"HTTP {error.code}: {text[:300]}") from error
            if error.code not in TRANSIENT_STATUSES:
                raise RequestFailed(f"HTTP {error.code}: {text}") from error
            failure = f"HTTP {error.code}: {text}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            past_deadline()  # a stalled stream that ran past the read deadline is the deadline, not a network retry
            failure = f"network error: {error}"
        wait = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
        if route.name == "google" and GEMINI_CREATE_REJECTION in failure:
            attempts = GEMINI_REJECTION_ATTEMPTS
            wait = random.uniform(*GEMINI_REJECTION_WAIT_S)
        log_raw(f"api_wsp: {route.name} request attempt {attempt} of {attempts} failed ({failure[:300]})"
                + (f", retrying in {wait:.0f} s" if attempt < attempts else ""))
        if attempt < attempts:
            time.sleep(wait)
    raise RequestFailed(f"{route.name}: {attempts} attempts failed, last: {failure}")


class Router:
    """Sends each request on the route for the model. A key out of credit, or a route that lacks the model, ends the
    task as an infra error."""

    def __init__(self, *, routes: list[Route], effort: str, events: list[dict[str, Any]]) -> None:
        self.routes, self.effort, self.events = routes, effort, events

    def send(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        for route in self.routes:
            while (key := route.live_key()) is not None:
                body = build_body(route=route, effort=self.effort, messages=messages)
                record(self.events, {"type": "request", "route": route.name, "body": without_turns(body),
                                     "messages": len(messages)})
                try:
                    return post_once(route=route, body=body, key=key)
                except OutOfCredit as error:
                    route.mark_dead()
                    position = f"key {route.key_index + 1} of {len(route.keys)}"
                    log_raw(f"api_wsp: {route.name} {position} is out of credit: {str(error)[:300]}")
                    record(self.events, {"type": "wsp.key_out_of_credit", "route": route.name, "key": position,
                                         "error": str(error)[:500]})
                except ModelUnavailable as error:
                    route.dead_keys = list(range(len(route.keys)))
                    log_raw(f"api_wsp: {route.name} does not serve {route.model}: {str(error)[:300]}")
                    record(self.events, {"type": "wsp.model_unavailable", "route": route.name, "model": route.model,
                                         "error": str(error)[:500]})
            if route is not self.routes[-1]:
                log_raw(f"api_wsp: the {route.name} route is out of credit or lacks the model, falling back to "
                        f"{self.routes[-1].name}")
                record(self.events, {"type": "wsp.fallback", "from": route.name, "to": self.routes[-1].name})
        raise InfraError(f"no route can serve this model {[r.describe() for r in self.routes]}: every key is out of "
                         "credit or the model is missing" + ("" if len(self.routes) > 1 else " (no OpenRouter fallback)"))


def without_turns(body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k not in ("messages", "input", "system", "instructions", "contents",
                                                       "systemInstruction", "system_instruction")}


def record(events: list[dict[str, Any]], event: dict[str, Any]) -> None:
    events.append(event)
    log_event(event)


def cut_kind(completion: dict[str, Any]) -> Optional[str]:
    """Where an output cap cut landed: "reasoning" (no visible text yet, or reasoning was the last thing emitted),
    "text" (reasoning finished, the cut is in the visible reply), or None for a request that was not cut."""
    if completion["finish_reason"] != "length":
        return None
    return "reasoning" if not completion["content"].strip() or completion["last_kind"] == "reasoning" else "text"


def request_reply(*, router: Router, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """One reply: up to L0_MAX_REPLY_REQUESTS requests. A cut in the visible text is continued with the full assistant
    turn (reasoning included where the route can pass it back); a cut while reasoning ends the reply out of budget.
    Returns the joined text, the requests' completions and the gate violations (empty when the reply is usable)."""
    completions: list[dict[str, Any]] = []
    history = list(messages)
    text = ""
    for index in range(1, L0_MAX_REPLY_REQUESTS + 1):
        completion = router.send(history)  # RequestFailed / InfraError: infra, the task errors (main)
        completion["request"], completion["cut"] = index, cut_kind(completion)
        completions.append(completion)
        record(router.events, {"type": "completion", **{k: v for k, v in completion.items() if k != "turn"}})
        log_raw(f"api_wsp: request {index}: route {completion['route']} finish {completion['finish_reason']} "
                f"cut {completion['cut']} provider {completion['provider']} usage {json.dumps(completion['usage'])}")
        text += completion["content"]
        if completion["tool_calls"]:
            return {"text": text, "completions": completions,
                    "violations": [f"tool call {json.dumps(completion['tool_calls'])[:300]}"]}
        if completion["cut"] is None:
            return {"text": text, "completions": completions, "violations": []}
        if completion["cut"] == "reasoning":
            return {"text": text, "completions": completions, "violations": [OUTPUT_CAP_REASONING_REASON]}
        turn: dict[str, Any] = {"role": "assistant", "content": completion["content"]}
        if completion["reasoning_passback"] and completion["turn"]:
            turn["native"] = {completion["route"]: completion["turn"]}
        history = [*history, turn, {"role": "user", "content": CONTINUE_MESSAGE}]
    return {"text": text, "completions": completions,
            "violations": [OUTPUT_CAP_REASON.format(requests=L0_MAX_REPLY_REQUESTS)]}


def run_l0(*, router: Router, prompt_text: str, max_nonzero: Optional[int],
           free: bool = False) -> tuple[dict[str, Any], str]:
    """One reply, or for a capped task up to MAX_CAP_ATTEMPTS: an over-cap reply is answered with CAP_RETRY_MESSAGE in
    the same conversation. Returns (l0 result, last reply text)."""
    attempts: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": L0_SYSTEM_INSTRUCTIONS}, {"role": "user", "content": prompt_text}]
    text = ""
    for attempt in range(1, (1 if max_nonzero is None else MAX_CAP_ATTEMPTS) + 1):
        reply = request_reply(router=router, messages=messages)
        text = reply["text"]
        result = apply_l0_reply(text=text, violations=reply["violations"], free=free)
        limit_error = free_limit_error() if result["ok"] and free else None
        if limit_error is not None:
            result = {**result, "limit_error": limit_error, "within_budget": False}
        elif result["ok"] and max_nonzero is not None:
            used = count_nonzero(L0_OUTPUT_PATH)
            result = {**result, "nonzero_weights": used, "max_nonzero": max_nonzero, "within_budget": used <= max_nonzero}
        attempts.append({"attempt": attempt, "requests": len(reply["completions"]),
                         "routes": [c["route"] for c in reply["completions"]],
                         "providers": [c["provider"] for c in reply["completions"]], **result})
        log_raw(f"api_wsp l0 attempt {attempt}: {result['stage']}: {str(result['reason'])[:300]} "
                f"nonzero {result.get('nonzero_weights')} cap {max_nonzero}")
        if result.get("within_budget", True) or attempt == MAX_CAP_ATTEMPTS:
            break
        if limit_error is not None:
            message = FREE_LIMIT_RETRY_MESSAGE.format(error=limit_error)
        else:
            message = CAP_RETRY_MESSAGE.format(used=result["nonzero_weights"], cap=max_nonzero)
        (OUTPUT_ROOT / f"cap_retry_{attempt + 1}.md").write_text(message, encoding="utf-8")
        record(router.events, {"type": "wsp.cap_retry", "attempt": attempt + 1, "message": message,
                               "nonzero_weights": result.get("nonzero_weights"), "max_nonzero": max_nonzero,
                               "limit_error": limit_error})
        messages = [*messages, {"role": "assistant", "content": text}, {"role": "user", "content": message}]
    return {**attempts[-1], "attempts": attempts}, text


# Read mode (`-k mode read`): weight-reading tasks whose networks are far too big to paste into one request. Same
# harness, routes and effort as L0; the only tool is a Python cell. The model replies with one ```python block per turn
# and gets its printed output back; a reply with no block is the final answer. The cell runs as a child process with
# no API keys in its environment and, where the kernel allows it, in a new network namespace with no interfaces, so
# the model can read /app/policy.npz but cannot reach any host. Every cell is logged to the trajectory for the
# forward-pass audit.
READ_ANSWER_PATH = Path("/app/out/answer.txt")
READ_MAX_TURNS = int(os.environ.get("WSP_READ_MAX_TURNS", "30"))
READ_CELL_TIMEOUT_S = 120
READ_OUTPUT_CHARS = 20000
READ_SYSTEM_INSTRUCTIONS = (
    "You can run Python 3 with numpy (no network, no other packages) to inspect the files the task names. To run code, "
    "reply with exactly one ```python fenced block; you will get back its printed output (stdout and stderr, cut to "
    f"{READ_OUTPUT_CHARS:,} characters). State does not persist between blocks, so reload what you need. You may run "
    f"up to {READ_MAX_TURNS} blocks. When you are done, reply with no code block and end that reply with the final "
    "answer line the task asks for; that reply is your submitted answer.\n"
)
READ_ONE_BLOCK_MESSAGE = (
    "Your reply had {n} ```python blocks, so none was run. Send exactly one block per reply and wait for its output; "
    "don't write outputs you have not been shown."
)
# A reply with no ```python block and no ANSWER line is neither a cell nor an answer. Kimi K3.1 Pro writes its first
# cell in its native tool-call syntax, which used to end the run with a 0 after one turn; ask it to resend instead.
READ_FINAL_LINE = re.compile(r"^[\s*_`#>]*ANSWER\s*:", re.MULTILINE | re.IGNORECASE)
READ_NO_BLOCK_MESSAGE = (
    "Your reply had no ```python fenced block and no final ANSWER line, so nothing was run. Tool-call syntax is not "
    "available here: to run code, send it as one ```python fenced block. If you are done, reply with your answer and "
    "end it with the final answer line the task asks for."
)
READ_MAX_NO_BLOCK_NAGS = 2  # in a row; reset after every cell that runs
# Kimi's native tool call (`...key="code" type="string"<|sep|>CODE<|close|>...`, with or without an `argument` tag):
# its code is accepted as the reply's one cell, so a model that slips into this format mid-run is not cut off.
NATIVE_TOOL_CODE = re.compile(r'key="code"[^<]*<\|sep\|>(.*?)<\|close\|>', re.DOTALL)  # both Kimi layouts
# The task's hard limit is 30 minutes (task.toml) and a timed-out task scores 0 with no answer. At READ_DEADLINE_S the
# harness stops whatever is in flight and asks for the answer; the answer request itself is abandoned at
# READ_FINAL_DEADLINE_S, leaving room to write the file before the task is killed.
READ_DEADLINE_S = float(os.environ.get("WSP_READ_DEADLINE_S", "1440"))
READ_FINAL_DEADLINE_S = float(os.environ.get("WSP_READ_FINAL_DEADLINE_S", "1680"))
READ_TIME_UP_MESSAGE = (
    "Time is up: no more code can be run. Reply now with your final answer, with no code block, ending with the final "
    "answer line the task asks for."
)
READ_NO_CELLS_MESSAGE = (
    f"You have used all {READ_MAX_TURNS} code runs, so this block was not run. Reply now with your final answer, with "
    "no code block, ending with the final answer line the task asks for."
)
READ_MAX_EMPTY_RETRIES = 3  # a reply with no text at all (some OpenRouter providers drop it): resend the same request
# OpenRouter providers that dropped Kimi K3.1 Pro's reply text in the Nov 18 runs (empty content, output tokens spent)
OPENROUTER_PROVIDER_IGNORE = {"moonshotai/kimi-k3.1-pro": ["digitalocean", "primalabs", "parasail", "decart"]}
ANSWER_VALUE = re.compile(r"^[\s*_`#>]*ANSWER\s*:\s*(.*?)[\s*_`]*$", re.IGNORECASE)
LETTER_OPTION = re.compile(r"^\(?([A-D])\)\s")


def canonical_answer_line(text: str, *, lettered: bool = True) -> Optional[str]:
    """The reply's last answer line in the grader's exact form, or None when the reply has none. Markdown is stripped.
    On a task with lettered options (`A) ...`), `D) shortened text` becomes `ANSWER: D`: the grader accepts the bare
    letter but not the letter with shortened text. On an unlettered task (Atari asks for a game name) a stray `B) `
    prefix is dropped and the name kept."""
    values = [m.group(1).strip() for line in text.splitlines() if (m := ANSWER_VALUE.match(line))]
    values = [v for v in values if v]
    if not values:
        return None
    value = values[-1].rstrip(".")
    letter = LETTER_OPTION.match(value + " ")
    if letter and lettered:
        return f"ANSWER: {letter.group(1)}"
    return f"ANSWER: {value[letter.end():].strip() if letter else value}"


def write_read_answer(text: str, *, lettered: bool) -> None:
    """The reply as given plus one canonical last line, which is the one the grader reads."""
    line = canonical_answer_line(text, lettered=lettered)
    READ_ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    READ_ANSWER_PATH.write_text(text + (f"\n{line}\n" if line else ""), encoding="utf-8")


# Belt and braces when no network namespace is available: Python-level sockets refuse to connect.
READ_NO_SOCKET_PREAMBLE = (
    "import socket as _s\n"
    "def _blocked(*a, **k):\n    raise OSError('network access is disabled in this sandbox')\n"
    "_s.socket.connect = _blocked; _s.socket.connect_ex = _blocked; _s.create_connection = _blocked\n"
    "del _s, _blocked\n"
)


def read_sandbox_prefix() -> list[str]:
    """`unshare -rn` (new user + network namespace, loopback only, down) when this container allows it, else []."""
    try:
        probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return ["unshare", "-rn"] if probe.returncode == 0 else []


def run_cell(*, code: str, prefix: list[str]) -> str:
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run([*prefix, sys.executable, "-I", "-c", READ_NO_SOCKET_PREAMBLE + code], cwd="/app",
                              env=env, capture_output=True, text=True, timeout=READ_CELL_TIMEOUT_S)
        out = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr.strip() else "")
        if proc.returncode != 0:
            out += f"\n[exit code {proc.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[the cell ran over {READ_CELL_TIMEOUT_S} s and was stopped]"
    if len(out) > READ_OUTPUT_CHARS:
        out = out[:READ_OUTPUT_CHARS] + f"\n[output cut at {READ_OUTPUT_CHARS:,} characters]"
    return out or "[no output]"


def run_sealed_answer(*, router: Router, prompt_text: str) -> tuple[dict[str, Any], str]:
    """`-k mode mcq`: one sealed reply with no tools (the Compile rules), for questions whose whole input fits in the
    prompt. The reply text is the submitted answer."""
    reply = request_reply(router=router, messages=[{"role": "system", "content": L0_SYSTEM_INSTRUCTIONS},
                                                   {"role": "user", "content": prompt_text}])
    READ_ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    READ_ANSWER_PATH.write_text(reply["text"], encoding="utf-8")
    return {"stage": "mcq", "reason": reply["violations"][0] if reply["violations"] else "answered", "cells": 0,
            "sandbox": "none (no tools)"}, reply["text"]


def run_read(*, router: Router, prompt_text: str) -> tuple[dict[str, Any], str]:
    """Up to READ_MAX_TURNS Python cells, then the answer. Returns (read result, final reply text).

    Every way a run can end asks for the answer first: out of cells or past READ_DEADLINE_S the model gets one
    no-code reply. The latest reply that has an answer line is what gets graded, so a later reply without one (a cut,
    an empty final turn) never overwrites it."""
    global STREAM_DEADLINE
    prefix = read_sandbox_prefix()
    log_raw(f"api_wsp read: sandbox prefix {prefix or 'none (socket preamble only)'}")
    start = time.monotonic()
    STREAM_DEADLINE = start + READ_DEADLINE_S
    messages = [{"role": "system", "content": READ_SYSTEM_INSTRUCTIONS}, {"role": "user", "content": prompt_text}]
    text, cells, violations, nags, empties, answered, reason = "", 0, [], 0, 0, "", "answered"
    final_message: Optional[str] = None
    for _ in range((3 + READ_MAX_NO_BLOCK_NAGS) * (READ_MAX_TURNS + 1)):
        if final_message is None and time.monotonic() > start + READ_DEADLINE_S:
            final_message, reason = READ_TIME_UP_MESSAGE, "deadline"
        if final_message is not None:  # one last no-code reply; whatever is in it is the answer
            record(router.events, {"type": "wsp.read_final_prompt", "reason": reason, "cells": cells})
            STREAM_DEADLINE = start + READ_FINAL_DEADLINE_S
            last = messages[-1]
            if last["role"] == "user":  # cut mid-reply: fold the request into the pending user turn
                ask = [*messages[:-1], {**last, "content": f"{last['content']}\n\n{final_message}"}]
            else:
                ask = [*messages, {"role": "user", "content": final_message}]
            try:
                text = request_reply(router=router, messages=ask)["text"]
            except DeadlineReached:
                text = ""
            if READ_FINAL_LINE.search(text):
                answered = text
            break
        try:
            reply = request_reply(router=router, messages=messages)
        except DeadlineReached:
            record(router.events, {"type": "wsp.read_deadline", "cells": cells, "stream": dict(STREAM_STATS)})
            final_message, reason = READ_TIME_UP_MESSAGE, "deadline"
            continue
        text, violations = reply["text"], reply["violations"]
        if not violations and not text.strip() and empties < READ_MAX_EMPTY_RETRIES:
            empties += 1  # the provider returned no text at all: send the same request again
            record(router.events, {"type": "wsp.read_empty_retry", "retry": empties})
            continue
        if READ_FINAL_LINE.search(text):
            answered = text
        blocks = PYTHON_BLOCK.findall(text) or [c for c in NATIVE_TOOL_CODE.findall(text) if c.strip()]
        if not violations and not blocks and not READ_FINAL_LINE.search(text) and nags < READ_MAX_NO_BLOCK_NAGS:
            nags += 1
            record(router.events, {"type": "wsp.read_no_block", "nag": nags})
            messages = [*messages, {"role": "assistant", "content": text},
                        {"role": "user", "content": READ_NO_BLOCK_MESSAGE}]
            continue
        if violations or not blocks:
            break
        if cells == READ_MAX_TURNS:
            messages = [*messages, {"role": "assistant", "content": text}]
            final_message, reason = READ_NO_CELLS_MESSAGE, "out of cells"
            continue
        if len(blocks) > 1:  # several blocks invite made-up outputs for the later ones: run none, ask for one
            record(router.events, {"type": "wsp.read_multi_block", "blocks": len(blocks)})
            messages = [*messages, {"role": "assistant", "content": text},
                        {"role": "user", "content": READ_ONE_BLOCK_MESSAGE.format(n=len(blocks))}]
            continue
        cells += 1
        nags = empties = 0
        output = run_cell(code=blocks[0], prefix=prefix)
        record(router.events, {"type": "wsp.read_cell", "cell": cells, "code": blocks[0], "output": output})
        log_raw(f"api_wsp read cell {cells}: {len(blocks[0])} chars of code, {len(output)} chars of output")
        messages = [*messages, {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Output of cell {cells}:\n```\n{output}\n```"}]
    STREAM_DEADLINE = None
    final = answered or text
    write_read_answer(final, lettered=bool(re.search(r"^A\) ", prompt_text, re.MULTILINE)))
    return {"stage": "read", "reason": violations[0] if violations else reason, "cells": cells,
            "sandbox": " ".join(prefix) or "socket-preamble"}, final


def extract_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Token usage summed over every request in codex semantics (input_tokens includes cached, output_tokens includes
    reasoning), plus the billed cost in USD that OpenRouter reports (None when no request went through OpenRouter)."""
    usages = [e["usage"] for e in events if e.get("type") == "completion" and isinstance(e.get("usage"), dict)]
    costs = [u["cost"] for u in usages if isinstance(u.get("cost"), (int, float))]
    keys = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
    return {"cost": round(sum(costs), 6) if costs else None, **{k: sum(int(u.get(k) or 0) for u in usages) for k in keys},
            "requests": len(usages)}


def write_summary(*, task_id: str, model: str, effort: str, routes: list[Route], prompt_text: str, exit_code: int,
                  wall_clock_duration: float, events: list[dict[str, Any]], message: str) -> None:
    if message.strip():
        FINAL_MESSAGE_PATH.write_text(message.strip() + "\n", encoding="utf-8")
    metrics = {"wall_clock_duration": round(wall_clock_duration, 3), **extract_usage(events)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    completions = [e for e in events if e.get("type") == "completion"]
    continued = [c for c in completions if c.get("cut") == "text"]
    summary = {
        "metadata": {
            "task_id": task_id,
            "model": model,
            "requested_effort": effort,
            "agent": "api_wsp",
            "access": "l0",
            "routes_planned": [route.describe() for route in routes],
            "requests": [{"request": c.get("request"), "route": c.get("route"), "model": c.get("model"),
                          "provider": c.get("provider"), "id": c.get("id"), "finish_reason": c.get("finish_reason"),
                          "cut": c.get("cut"), "reasoning_passback": c.get("reasoning_passback"),
                          **({"api": c.get("api")} if c.get("api") else {})}
                         for c in completions],
            # null when no reply was continued; false when a continuation went back without its reasoning
            "continuation_reasoning": (all(c.get("reasoning_passback") for c in continued) if continued else None),
            "routes": list(dict.fromkeys(c["route"] for c in completions)),
            "fallback": any(e.get("type") == "wsp.fallback" for e in events),
            "providers": list(dict.fromkeys(c.get("provider") for c in completions if c.get("provider"))),
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "output_path": str(OUTPUT_ROOT),
            "success": bool(message.strip()) and exit_code == 0,
            "return_code": exit_code,
        },
        "task_prompt": prompt_text,
        "metrics": metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=4), encoding="utf-8")


def main() -> int:
    args = parse_args()
    task_file = Path(args.problem_statement_path)
    if not task_file.exists():
        raise SystemExit(f"Problem statement path not found: {task_file}")
    if args.effort not in THINKING_BUDGET_SHARE:
        raise SystemExit(f"effort {args.effort} is not one of {sorted(THINKING_BUDGET_SHARE)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_PATH.write_text("", encoding="utf-8")
    RAW_OUTPUT_PATH.write_text("", encoding="utf-8")
    prompt_text = task_file.read_text(encoding="utf-8")
    routes = plan_routes(model=args.model, route=args.route)
    # The argv line is how `wsp valkyrie-pull` reads the editor settings back (wsp.valkyrie_runs.ARGV_LINE).
    argv = ["api", "--model", args.model, "--effort", args.effort, "--route", args.route, "--mode", args.mode,
            "--routes", ",".join(r.name for r in routes)]
    log_raw(f"api_wsp argv: {argv}")
    log_raw(f"api_wsp routes: {json.dumps([r.describe() for r in routes])}")

    exit_code, message = 1, ""
    events: list[dict[str, Any]] = []
    start = time.monotonic()
    try:
        router = Router(routes=routes, effort=args.effort, events=events)
        if args.mode == "read":
            global CACHE_PROMPTS
            CACHE_PROMPTS = True
        if args.mode in ("read", "mcq"):
            run = run_read if args.mode == "read" else run_sealed_answer
            result, message = run(router=router, prompt_text=prompt_text)
            exit_code = 0
            L0_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            log_raw(f"api_wsp read: {result['reason']}, {result['cells']} cells, sandbox {result['sandbox']}")
            return 0
        free_cap = read_free_cap()
        result, message = run_l0(router=router, prompt_text=prompt_text,
                                 max_nonzero=read_max_nonzero() if free_cap is None else free_cap,
                                 free=free_cap is not None)
        exit_code = 0
        L0_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log_raw(f"api_wsp l0: {result['stage']}: {str(result['reason'])[:500]}")
        return 0  # graded either way: a failed gate or contract leaves no weights file, which scores 0
    except (InfraError, RequestFailed) as error:
        log_raw(f"api_wsp: infra error, failing the task so it is retried: {str(error)[:2000]}")
        record(events, {"type": "wsp.infra_error", "error": str(error)[:2000]})
        exit_code = INFRA_EXIT
        return INFRA_EXIT
    finally:
        write_summary(task_id=args.task_id, model=args.model, effort=args.effort, routes=routes,
                      prompt_text=prompt_text, exit_code=exit_code, wall_clock_duration=time.monotonic() - start,
                      events=events, message=message)


if __name__ == "__main__":
    raise SystemExit(main())
