"""Program arm (track write_program): the agent writes a short Python program instead of network weights.

The question it answers: does the same model do better writing NEURAL NET WEIGHTS for a task, or writing a short
PROGRAM for it? So the program is scored by exactly the code that scores the net arm. A target opts in by mixing in
ProgramTarget on top of its free-architecture twin (wsp.free_arch.FreeArchTarget): ProgramTarget.program_scores_from
calls the twin's scores_from with a `forward` that sends each input batch to the program and converts its answer
(labels, probabilities or an action) into the network output the scorer already reads. Same hidden data, same
preprocessing, same metric, same floor and reference, so the rewards line up key for key with write_free.

The answer: /app/out/solution.py defining predict(x) (a batch of inputs -> labels or probabilities) or act(obs) (one
observation -> an action), plus an optional /app/out/constants.npz exposed to the program as the dict CONSTANTS.
Rules: at most MAX_LINES code lines, numpy and math only, and the time / memory limits in ProgramLimits (over any of
them: invalid, reward 0). Numeric constants (literals + constants.npz elements) are budgeted at the reference net's
parameter count: reward = within_budget * clip((primary - floor) / (reference - floor), 0, 1), as in write_free.
The program runs in wsp.program_sandbox, a separate restricted process that only ever receives inputs.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from wsp.free_arch import Forward
from wsp.program_sandbox import ProgramError, ProgramProcess, RunLimits, check_source, read_constants
from wsp.targets.base import Metrics, Split, Target, Weights

MAX_LINES = 100
MAX_SOURCE_BYTES = 32_000
MAX_CONSTANTS_BYTES = 64 * 2**20
BATCH_ROWS = 10_000  # predict() sees at most this many rows per call
CALL_FLOOR_S = 2.0
CPU_FLOOR_S = 120.0
# act() is called once per step, so a policy that survives longer than the reference makes more calls: 900 s covers
# ~2M steps (200 episodes at the 10,000-step cap) at ~0.4 ms each, inside the verifier's 1,800 s timeout.
ACT_CPU_FLOOR_S = 900.0
LOAD_SECONDS = 60.0
MEMORY_MB = 4096
TIME_FACTOR = 10  # limits = TIME_FACTOR x the reference network's own inference time (with the floors above)
PROB_EPS = 1e-12
ACTION_OFF_LOGIT = -1e4  # logit of every action the program did not pick: exp(-1e4) is exactly 0 in float64
OUTPUT_KINDS = ("class", "binary", "action")
INVALID = {"reward": 0.0, "reward_unbudgeted": 0.0, "valid_output": 0.0, "within_budget": 0.0}


@dataclass(frozen=True)
class ProgramLimits:
    entry: str
    constants_budget: int
    call_seconds: float
    cpu_seconds: float
    reference_seconds: float
    max_lines: int = MAX_LINES
    max_source_bytes: int = MAX_SOURCE_BYTES
    max_constants_bytes: int = MAX_CONSTANTS_BYTES
    batch_rows: int = BATCH_ROWS
    load_seconds: float = LOAD_SECONDS
    memory_mb: int = MEMORY_MB

    def run_limits(self) -> RunLimits:
        return RunLimits(call_seconds=self.call_seconds, load_seconds=self.load_seconds, cpu_seconds=self.cpu_seconds,
                         memory_mb=self.memory_mb, max_constants_bytes=self.max_constants_bytes)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ProgramLimits:
        return cls(**data)

    @classmethod
    def from_reference_time(cls, *, entry: str, constants_budget: int, seconds: float, rows: int) -> ProgramLimits:
        """Per call: 10x the reference's time for a full batch (floor 2 s); in total: 10x its whole run (floor 120 s,
        900 s for act)."""
        per_row = seconds / max(rows, 1)
        batch = BATCH_ROWS if entry == "predict" else 1
        return cls(entry=entry, constants_budget=constants_budget, reference_seconds=round(seconds, 3),
                   call_seconds=round(max(CALL_FLOOR_S, TIME_FACTOR * per_row * batch), 3),
                   cpu_seconds=round(max(CPU_FLOOR_S if entry == "predict" else ACT_CPU_FLOOR_S, TIME_FACTOR * seconds), 1))


class ProgramTarget(Target):
    """Opt-in hooks for write_program. Mixed in with a FreeArchTarget twin, whose scores_from / net_io it reuses."""

    program_output: ClassVar[str]  # "class": labels or probabilities, "binary": P(positive), "action": one action
    # Also required, from the FreeArchTarget twin listed after this mixin: net_io() and scores_from(forward, split, seed).
    net_io: Callable[[], Any]
    scores_from: Callable[..., Metrics]

    @abstractmethod
    def program_task_text(self) -> str:
        """THE TASK paragraph, worded for a program."""

    @abstractmethod
    def program_io_text(self) -> str:
        """What the argument of predict / act holds, and exactly what it must return."""

    @abstractmethod
    def program_scoring_text(self) -> str:
        """The SCORING paragraph (metric, floor, reference), without the budget rules."""

    @abstractmethod
    def program_baseline(self) -> tuple[str, Weights]:
        """A trivial valid answer (solution.py source, constants) that scores above the floor: the task's oracle."""

    def program_rules_text(self) -> str:
        return (
            "You may run any code in the sandbox while you work, including fitting things to the labelled examples\n"
            "above (any numbers that end up in the program count against the constant budget). Only the submitted\n"
            "program is restricted."
        )

    @property
    def program_entry(self) -> str:
        return "act" if self.program_output == "action" else "predict"

    def program_classes(self) -> int:
        return int(self.net_io().output[0])

    def constants_budget(self) -> int:
        """The fixed reference network's parameter count (the same count the write_free budget uses)."""
        return sum(int(np.prod(shape)) for shape in self.tensor_shapes().values())

    def program_forward(self, call: Callable[[np.ndarray], np.ndarray], *, batch_rows: int) -> Forward:
        """`forward` for scores_from: batches to the program, its answers converted to the network output contract."""
        (name,) = self.net_io().inputs
        convert = OUTPUT_CONVERTERS[self.program_output]
        classes = self.program_classes()

        def forward(inputs: dict[str, np.ndarray]) -> np.ndarray:
            x = inputs[name]
            parts = [convert(call(x[i : i + batch_rows]), len(x[i : i + batch_rows]), classes)
                     for i in range(0, len(x), batch_rows)]
            return np.concatenate(parts) if parts else np.zeros((0, classes))

        return forward

    def program_scores_from(self, call: Callable[[np.ndarray], np.ndarray], *, split: Split, seed: int,
                            batch_rows: int = BATCH_ROWS) -> Metrics:
        return self.scores_from(forward=self.program_forward(call, batch_rows=batch_rows), split=split, seed=seed)

    def program_limits_json(self, limits: ProgramLimits) -> dict[str, Any]:
        """task_limits.json for the agent's checker."""
        io = self.net_io()
        (shape,) = io.inputs.values()
        return {**limits.as_json(), "input_shape": list(shape), "output": self.program_output,
                "classes": self.program_classes()}


# ---- converting program answers to the network output each scorer reads --------------------------------------------


def as_labels(out: np.ndarray, rows: int, classes: int) -> np.ndarray:
    if out.shape != (rows,) or not np.all(np.isfinite(out)) or np.any(out != np.round(out)):
        raise ProgramError(f"expected {rows} integer labels, got an array of shape {out.shape}")
    if out.min(initial=0) < 0 or out.max(initial=0) >= classes:
        raise ProgramError(f"labels must be 0 to {classes - 1}")
    return out.astype(np.int64)


def class_logits(out: np.ndarray, rows: int, classes: int) -> np.ndarray:
    """(N,) labels or (N, K) probabilities -> log-probabilities, so softmax / argmax downstream give back p."""
    if out.ndim == 1:
        probs = np.eye(classes)[as_labels(out, rows, classes)]
    else:
        if out.shape != (rows, classes) or not np.all(np.isfinite(out)) or np.any(out < 0):
            raise ProgramError(f"expected {rows} labels or a ({rows}, {classes}) array of probabilities, got {out.shape}")
        total = out.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            raise ProgramError("a row of probabilities sums to 0")
        probs = out / total
    return np.log(np.maximum(probs, 1e-30))


def binary_logits(out: np.ndarray, rows: int, classes: int) -> np.ndarray:
    """(N,) or (N, 1) probabilities in [0, 1] -> (N, 1) logits; the scorers' sigmoids give back p."""
    if out.shape not in ((rows,), (rows, 1)) or not np.all(np.isfinite(out)) or np.any((out < 0) | (out > 1)):
        raise ProgramError(f"expected {rows} probabilities in [0, 1], got an array of shape {out.shape}")
    p = np.clip(out.reshape(rows, 1), PROB_EPS, 1 - PROB_EPS)
    return np.log(p) - np.log1p(-p)


def action_logits(out: np.ndarray, rows: int, classes: int) -> np.ndarray:
    """(N,) actions -> logits that pick exactly that action under argmax and under softmax sampling."""
    logits = np.full((rows, classes), ACTION_OFF_LOGIT)
    logits[np.arange(rows), as_labels(out, rows, classes)] = 0.0
    return logits


OUTPUT_CONVERTERS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "class": class_logits, "binary": binary_logits, "action": action_logits,
}


# ---- scoring --------------------------------------------------------------------------------------------------------


def static_counts(*, source: str, constants: bytes | None, limits: ProgramLimits) -> tuple[dict[str, float], str | None]:
    """(raw counts, None) or (counts so far, why the answer is invalid). Never raises on a hostile answer."""
    report = check_source(source)
    counts = {"program_lines": float(report.lines), "program_literals": float(report.literals),
              "program_fit_calls": float(report.fit_calls), "constants_budget": float(limits.constants_budget)}
    try:
        elements = sum(int(a.size) for a in read_constants(constants, max_bytes=limits.max_constants_bytes).values())
    except ProgramError as error:
        return counts, str(error)
    counts["constants_npz_elements"] = float(elements)
    counts["program_constants"] = float(report.literals + elements)
    within = report.literals + elements <= limits.constants_budget
    counts["within_constants_budget"] = counts["within_budget"] = float(within)
    if report.violations:
        return counts, "; ".join(report.violations[:10])
    if report.lines > limits.max_lines:
        return counts, f"{report.lines} code lines > {limits.max_lines}"
    return counts, None


def score_program(*, config: dict, target: ProgramTarget, solution_path: Path,
                  constants_path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """(reward.json values, report). Every malformed, hostile, slow or crashing answer ends as valid_output 0."""
    limits = ProgramLimits.from_json(config["program"]["limits"])
    if not solution_path.is_file():
        return dict(INVALID), {"invalid": f"missing {solution_path}"}
    if solution_path.stat().st_size > limits.max_source_bytes:
        return dict(INVALID), {"invalid": f"solution.py larger than {limits.max_source_bytes:,} bytes"}
    try:
        source = solution_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        return dict(INVALID), {"invalid": f"solution.py is not UTF-8: {error}"}
    constants = None
    if constants_path.is_file():
        if constants_path.stat().st_size > limits.max_constants_bytes:
            return dict(INVALID), {"invalid": f"constants.npz larger than {limits.max_constants_bytes:,} bytes"}
        constants = constants_path.read_bytes()
    counts, reason = static_counts(source=source, constants=constants, limits=limits)
    if reason is not None:
        return {**INVALID, **counts, "within_budget": 0.0, "valid_output": 0.0}, {"invalid": reason}
    classes = target.program_classes()
    try:
        with ProgramProcess(source=source, entry=limits.entry, constants=constants, limits=limits.run_limits()) as program:

            def call(x: np.ndarray) -> np.ndarray:
                return program.call(x, max_reply_bytes=len(x) * classes * 8 + 4096)

            metrics = target.program_scores_from(call, split="hidden", seed=config["hidden_seed"],
                                                 batch_rows=limits.batch_rows)
    except ProgramError as error:
        print(f"invalid answer at run time: {error}")
        return {**INVALID, **counts, "within_budget": 0.0}, {"invalid": str(error)}
    floor, reference = config.get("reward_floor", 0.0), config["reward_reference"]
    uncapped = (metrics.primary_value - floor) / (reference - floor)
    gap = float(np.clip(uncapped, 0.0, 1.0))
    reward = {
        **{k: float(v) for k, v in metrics.values.items()},
        **counts,
        "program_seconds": float(program.seconds),
        "program_calls": float(program.calls),
        "program_call_limit_s": float(limits.call_seconds),
        "program_cpu_limit_s": float(limits.cpu_seconds),
        "sandbox_chroot": float(program.applied.get("chroot", False)),
        "primary": float(metrics.primary_value),
        "gap_uncapped": float(uncapped),
        "reward_unbudgeted": gap,
        "reward": gap * counts["within_budget"],
        "valid_output": 1.0,
    }
    return reward, {"applied_limits": program.applied, "reason": "ok"}


def reference_timing(target: ProgramTarget, forward: Forward, *, seed: int) -> tuple[float, int]:
    """(seconds, rows) the reference network spends inside forward on the hidden split: the time limits' basis."""
    import time

    spent, rows = 0.0, 0

    def timed(inputs: dict[str, np.ndarray]) -> np.ndarray:
        nonlocal spent, rows
        started = time.perf_counter()
        out = forward(inputs)
        spent += time.perf_counter() - started
        rows += len(next(iter(inputs.values())))
        return out

    target.scores_from(forward=timed, split="hidden", seed=seed)
    return spent, rows


# ---- prompt ---------------------------------------------------------------------------------------------------------

SECTION = re.compile(r"^(THE [A-Z]+|EXAMPLES|EXAMPLE [A-Z]+|SCORING|CLASSES|HOW YOUR NETWORK IS SCORED)\b")
DROPPED = ("THE TASK", "THE NETWORK", "THE OUTPUT", "SCORING", "HOW YOUR NETWORK IS SCORED")
# The fixed tasks forbid fitting to their examples; the program arm allows it (stated in the rules), so that sentence goes.
NO_FIT = re.compile(
    r"(These \d+ labelled images are illustrative only\.|They are illustrations, not a training set:)"
    r".*?(and\s+the\s+architecture\.|understanding\s+of\s+the\s+problem\s+and\s+the\s+network\.)",
    re.DOTALL,
)
MAY_FIT = (
    "{they} a small labelled sample of the training data and the\nonly labelled data you get; your program may use them "
    "in any way (see the rules below)."
)


def may_fit(match: re.Match) -> str:
    opener = match.group(1)
    they = opener.split(" are ")[0] + " are" if opener.startswith("These") else "They are"
    return MAY_FIT.format(they=they)
WORDING = (("The observation the network gets", "The observation your program gets"),
           ("The network input is", "The input is"), ("the network sees", "the program sees"))


def context_sections(description: str) -> str:
    """The fixed description minus its task, network, output and scoring sections, examples reworded for a program."""
    sections: list[list[str]] = [[]]
    for line in description.splitlines():
        if SECTION.match(line) and (not sections[-1] or not sections[-1][-1].strip()):
            sections.append([])
        sections[-1].append(line)
    kept = ["\n".join(s).strip() for s in sections if s and not any(s[0].startswith(d) for d in DROPPED)]
    text = "\n\n".join(k for k in kept if k)
    text = NO_FIT.sub(may_fit, text)
    for fixed, program in WORDING:
        text = text.replace(fixed, program)
    return text


def instruction(target: ProgramTarget, limits: ProgramLimits) -> str:
    entry = limits.entry
    signature = "predict(x)" if entry == "predict" else "act(obs)"
    calls = (
        f"predict is called on batches of at most {limits.batch_rows:,} held-out examples (a batch can be smaller)."
        if entry == "predict" else "act is called once per step of every scored episode, with that step's observation."
    )
    return f"""\
THE TASK
{target.program_task_text().strip()}

{context_sections(target.description())}

THE PROGRAM
Write a Python file, solution.py, that defines {signature}.
{target.program_io_text().strip()}
solution.py runs in a fresh, restricted Python process that is given inputs only. It is loaded once, then {calls}

RULES (an answer that breaks one of them is invalid and scores 0)
  - At most {limits.max_lines} lines of code. Blank lines and comment-only lines are free; every other line counts,
    including each line of a multi-line string. solution.py is at most {limits.max_source_bytes:,} bytes.
  - Imports: numpy and math only (numpy.linalg, numpy.fft and numpy.random are fine). No other module, no file or
    network access, no open / exec / eval / compile / getattr / setattr / globals / vars / type, no name or attribute
    that starts with an underscore, no class or match statements, and none of numpy's file functions (load, save,
    loadtxt, fromfile, tofile, memmap, ...).
  - Time: loading solution.py at most {limits.load_seconds:g} s; each call to {entry} at most {limits.call_seconds:g} s; all calls
    together at most {limits.cpu_seconds:g} s of CPU (at least {TIME_FACTOR}x what the reference network takes). Memory: {limits.memory_mb:,} MB.
  - {entry} must return exactly what is described above: a wrong shape, a value out of range, NaN or an exception
    makes the whole answer invalid.

CONSTANT BUDGET
Numbers written into the program count against a budget of {limits.constants_budget:,} constants, the number of
weights in this task's reference network:
  - every numeric literal in solution.py counts 1 (-3 is one literal; an integer wider than 32 bits counts 1 per 32
    bits; True, False and None are free),
  - every character of a string or bytes literal counts 1 (docstrings are free),
  - every element of every array in the optional constants.npz counts 1.
constants.npz (optional, written with np.savez) holds numeric arrays; solution.py sees them as the dict CONSTANTS
(array name -> numpy array), and CONSTANTS is an empty dict without the file. Over the budget the score is 0.

SCORING
{target.program_scoring_text().strip()}
The 0 point and the reference are the same as in the network-writing versions of this task, so scores compare
directly. Over the constant budget the score is 0.

HOW TO WRITE THE ANSWER
You have a sandbox with Python 3 and numpy. No internet, and no data beyond what this message gives you.
{target.program_rules_text().strip()}
The held-out data is never in the sandbox; the scorer runs your program in a separate process that only sees inputs.
Check an answer with the scorer's own checks (lines, constants, rules) before you submit:
  python /app/check_program.py /app/out/solution.py [/app/out/constants.npz] [--inputs inputs.npz]
With --inputs (an npz holding one array `x`, shape (N, ...) as described above) it also runs {entry} in the same
sandbox and prints what it returns. Write the program to `/app/out/solution.py` and the optional constants to
`/app/out/constants.npz`. Only these two files are read; nothing you print is graded.
"""


def baseline_files(target: ProgramTarget) -> tuple[str, bytes | None]:
    import io

    source, constants = target.program_baseline()
    if not constants:
        return source, None
    buffer = io.BytesIO()
    np.savez(buffer, **constants)
    return source, buffer.getvalue()

