"""Check and run an untrusted solution.py (track write_program) in a process that never sees the hidden labels.

Standalone on purpose (stdlib + numpy, no wsp imports): the verifier runs this file as a fresh interpreter
(`python -I -B program_sandbox.py --serve`), and the agent sandbox gets a copy as /app/check_program.py so the agent can
run the scorer's exact checks before submitting.

Defence in depth; each layer alone stops the usual escapes, together they back each other up:
  1. static check (check_source): imports of numpy / math only; no name or attribute starting with "_" (so no
     __class__ / __mro__ / __globals__ / __builtins__ walks); no open / exec / eval / getattr / globals / ...; none of
     numpy's file functions; no frame or traceback attributes (gi_frame, f_back, ...); no class or match statements.
  2. restricted runtime: solution.py runs with a small builtins dict, an import hook that only returns numpy and math,
     and every module wrapped in SafeModule, so an attribute chain like np.ma.core.builtins cannot reach a non-numpy
     module.
  3. a separate process: the parent sends inputs and reads outputs over a pipe; labels never leave the parent. Before
     solution.py runs, the worker sets CPU, address-space, file-size and process-count rlimits and, when it is root (the
     verifier container), chroots into an empty directory and drops to uid / gid 65534, so even a full escape finds no
     files to read.
  4. the parent (ProgramProcess) enforces a wall-clock limit on loading and on every call, polls the worker's resident
     memory while it waits (and the worker checks its own peak after every call, for where RLIMIT_AS is not
     enforced, e.g. macOS), caps the size of every reply, and turns every failure into ProgramError, never a crash.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import select
import struct
import subprocess
import sys
import tempfile
import time
import tokenize
import types
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ALLOWED_IMPORTS = frozenset({"numpy", "numpy.linalg", "numpy.fft", "numpy.random", "math"})
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "hasattr", "globals", "locals", "vars", "dir",
    "input", "breakpoint", "help", "memoryview", "type", "object", "super", "classmethod", "staticmethod", "property",
    "exit", "quit", "copyright", "credits", "license", "format", "id",
})
# Attributes refused on any object: numpy file I/O, modules that lead out of numpy, string formatting (which can walk
# attributes inside a format string), and the frame / traceback / code attributes that lead back to the worker's globals.
BANNED_ATTRS = frozenset({
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "open_memmap", "fromregex", "load_library", "DataSource", "dump", "dumps", "lib", "ctypeslib", "ctypes",
    "testing", "f2py", "distutils", "os", "sys", "builtins", "show_config", "show_runtime", "get_include", "info",
    "format", "format_map", "mro", "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code", "ag_await", "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "f_trace",
    "tb_frame", "tb_next", "co_code", "co_consts",
})
BANNED_NODES = (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Match)
FIT_CALLS = frozenset({"lstsq", "pinv", "inv", "solve", "polyfit", "svd", "eigh"})
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "hash", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AssertionError", "Exception", "FloatingPointError", "IndexError", "KeyError", "LookupError",
    "NotImplementedError", "OverflowError", "RuntimeError", "StopIteration", "TypeError", "ValueError",
    "ZeroDivisionError",
)
ENTRIES = ("predict", "act")
NOBODY = 65534
WATCH_INTERVAL_S = 0.25
ERROR_TEXT_BYTES = 4000
FRAME = struct.Struct(">cQ")  # status byte (b"O" ok / b"E" error, or b"M" message to the worker), payload length


class ProgramError(Exception):
    """The program is invalid or failed at run time (any reason); the scorer turns it into valid_output 0."""


# ---- static analysis (shared by the verifier and the agent's checker) ----------------------------------------------


@dataclass
class SourceReport:
    lines: int = 0
    literals: int = 0
    fit_calls: int = 0
    violations: list[str] = field(default_factory=list)


def count_code_lines(source: str) -> int:
    """Physical lines holding any token but a comment (blank and comment-only lines are free; a string spanning
    several lines counts each of them)."""
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
            tokenize.ENDMARKER}
    lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in skip:
            lines.update(range(token.start[0], token.end[0] + 1))
    # Statements joined with ';' (or `if x: a; b`) would let one physical line carry many lines of logic, so every
    # statement after the first on a line costs one more line.
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return len(lines)
    starts: dict[int, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            starts[node.lineno] = starts.get(node.lineno, 0) + 1
    return len(lines) + sum(count - 1 for count in starts.values())


def docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.Lambda) and isinstance(getattr(node, "body", None), list):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    ids.add(id(body[0].value))
    return ids


def literal_cost(value: object) -> int:
    """Constants a literal spends: numbers 1 (ints 1 per 32 bits), complex 2, str / bytes 1 per character."""
    if value is None or value is Ellipsis or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(1, math.ceil(abs(value).bit_length() / 32))
    if isinstance(value, float):
        return 1
    if isinstance(value, complex):
        return 2
    if isinstance(value, str | bytes):
        return len(value)
    return 1


def check_source(source: str) -> SourceReport:
    report = SourceReport()
    try:
        report.lines = count_code_lines(source)
        tree = ast.parse(source, filename="solution.py")
    except (SyntaxError, ValueError, RecursionError, MemoryError, tokenize.TokenError) as error:
        report.violations.append(f"cannot parse solution.py: {type(error).__name__}: {error}")
        return report
    docstrings = docstring_nodes(tree)
    bad = report.violations
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, BANNED_NODES):
            bad.append(f"line {line}: {type(node).__name__} statements are not allowed")
        elif isinstance(node, ast.Import):
            bad.extend(f"line {line}: import {a.name} (only numpy and math)" for a in node.names if a.name not in ALLOWED_IMPORTS)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                bad.append(f"line {line}: from {node.module} import (only numpy and math)")
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_") or alias.name in BANNED_ATTRS:
                    bad.append(f"line {line}: cannot import {alias.name!r}")
        elif isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BANNED_NAMES):
            bad.append(f"line {line}: name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BANNED_ATTRS):
            bad.append(f"line {line}: attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.FunctionDef | ast.Lambda | ast.arg | ast.alias):
            name = getattr(node, "name", None) or getattr(node, "arg", None) or getattr(node, "asname", None)
            if name and (name.startswith("__") or name in BANNED_NAMES):
                bad.append(f"line {line}: name {name!r} is not allowed")
        elif isinstance(node, ast.Constant) and id(node) not in docstrings:
            report.literals += literal_cost(node.value)
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            report.fit_calls += name in FIT_CALLS
    return report


def read_constants(data: bytes | None, *, max_bytes: int) -> dict[str, np.ndarray]:
    """constants.npz bytes -> arrays. Refuses pickles, non-numeric arrays and members whose uncompressed size (as the
    zip directory states it, checked before anything is decompressed) would go over max_bytes."""
    if data is None:
        return {}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if sum(info.file_size for info in infos) > max_bytes:
                raise ProgramError(f"constants.npz unpacks to more than {max_bytes:,} bytes")
            if any(not info.filename.endswith(".npy") for info in infos):
                raise ProgramError("constants.npz holds a member that is not a .npy array")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as error:
        raise ProgramError(f"cannot read constants.npz: {type(error).__name__}: {error}") from error
    for key, value in arrays.items():
        if not (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_):
            raise ProgramError(f"constants.npz array {key!r} is not numeric ({value.dtype})")
    return arrays


# ---- the worker (child process) -------------------------------------------------------------------------------------


class SafeModule:
    """Read-only view of numpy / math: public attributes only, and sub-modules only when they are allowed imports."""

    __slots__ = ("_SafeModule__module",)

    def __init__(self, module: types.ModuleType):
        object.__setattr__(self, "_SafeModule__module", module)

    def __getattr__(self, name: str):
        if name.startswith("_") or name in BANNED_ATTRS:
            raise AttributeError(f"{name!r} is not available in the program sandbox")
        value = getattr(object.__getattribute__(self, "_SafeModule__module"), name)
        if isinstance(value, types.ModuleType):
            if value.__name__ not in ALLOWED_IMPORTS:
                raise AttributeError(f"module {value.__name__} is not available in the program sandbox")
            return SafeModule(value)
        return value

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("modules are read-only in the program sandbox")

    def __repr__(self) -> str:
        return f"<module {object.__getattribute__(self, '_SafeModule__module').__name__} (sandboxed)>"


def restricted_globals(constants: dict[str, np.ndarray]) -> dict:
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # numpy's C code imports its own private modules lazily (numpy._core._methods on the first arr.max(axis=...))
        # through the CALLING frame's builtins, i.e. this hook. Only numpy can ask for those: the static check refuses
        # every name with an underscore, so a program cannot spell one. They get the real module, unwrapped.
        if level == 0 and name.startswith("numpy.") and any(part.startswith("_") for part in name.split(".")):
            return real_import(name, globals, locals, fromlist, level)
        if level != 0 or name not in ALLOWED_IMPORTS:
            raise ImportError(f"only numpy and math may be imported, not {name!r}")
        return SafeModule(real_import(name, globals, locals, fromlist, level))

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    safe["__import__"] = guarded_import
    return {"__builtins__": safe, "__name__": "solution", "CONSTANTS": constants}


def preload_numpy() -> None:
    """Import every numpy module numpy itself may import lazily later (it does, e.g. numpy._core._methods)."""
    import pkgutil

    skip = ("tests", "testing", "f2py", "distutils", "_pyinstaller", "conftest", "_build_utils")
    for module in pkgutil.walk_packages(np.__path__, "numpy."):
        if not any(part in skip for part in module.name.split(".")):
            try:
                __import__(module.name)
            except ImportError:  # optional pieces (e.g. a missing compiler toolchain) are simply not available
                continue


def apply_limits(limits: dict) -> dict[str, bool]:
    """rlimits, then (as root only) chroot into the empty run directory and drop to nobody. Returns what took effect."""
    import resource

    def limit(kind: int, soft: int, hard: int | None = None) -> bool:
        try:
            resource.setrlimit(kind, (soft, soft if hard is None else hard))
        except (ValueError, OSError):
            return False
        return True

    cpu = int(math.ceil(limits["cpu_seconds"]))
    applied = {
        "cpu": limit(resource.RLIMIT_CPU, cpu, cpu + 5),
        "memory": limit(resource.RLIMIT_AS, int(limits["memory_mb"]) * 2**20),
        "file_size": limit(resource.RLIMIT_FSIZE, 0),
        "chroot": False,
    }
    if os.geteuid() == 0:
        preload_numpy()  # the jail has no files, so every numpy module must be in sys.modules before entering it
        try:
            os.chroot(limits["jail_dir"])
            os.chdir("/")
            os.setgroups([])
            os.setgid(NOBODY)
            os.setuid(NOBODY)
            applied["chroot"] = True
        except OSError:  # a container without CAP_SYS_CHROOT / CAP_SETUID: the other layers still hold
            applied["chroot"] = False
    applied["processes"] = limit(resource.RLIMIT_NPROC, 0)
    return applied


def read_frame(stream) -> tuple[bytes, bytes] | None:
    header = stream.read(FRAME.size)
    if len(header) < FRAME.size:
        return None
    status, length = FRAME.unpack(header)
    payload = bytearray()
    while len(payload) < length:
        chunk = stream.read(min(length - len(payload), 1 << 20))
        if not chunk:
            return None
        payload += chunk
    return status, bytes(payload)


def write_frame(stream, status: bytes, payload: bytes) -> None:
    stream.write(FRAME.pack(status, len(payload)) + payload)
    stream.flush()


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def run_entry(function, entry: str, x: np.ndarray) -> np.ndarray:
    if entry == "act":
        return np.array([float(np.asarray(function(row)).reshape(())) for row in x], dtype=np.float64)
    out = np.asarray(function(x))
    if not (np.issubdtype(out.dtype, np.number) or out.dtype == np.bool_):
        raise TypeError(f"predict returned a {out.dtype} array, not numbers")
    return out.astype(np.float64)


def peak_memory_mb() -> float:
    """This process's peak resident memory (ru_maxrss is bytes on macOS, KiB on Linux)."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def serve() -> None:
    """Worker loop: init, constants, (limits), load solution.py, then one reply per input batch until EOF."""
    import traceback

    import numpy.fft
    import numpy.linalg
    import numpy.random  # noqa: F401

    proto_in = os.fdopen(os.dup(0), "rb", buffering=0)
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)  # the program's print() and stray writes go nowhere, not into the protocol
    os.dup2(devnull, 1)
    sys.stdout = sys.stdin = open(os.devnull, "r+")
    init_frame, constants_frame = read_frame(proto_in), read_frame(proto_in)
    if init_frame is None or constants_frame is None:
        return
    init = json.loads(init_frame[1])
    entry = init["entry"]

    def fail(where: str, error: BaseException) -> None:
        text = "".join(traceback.format_exception_only(type(error), error)).strip()
        write_frame(proto_out, b"E", f"{where}: {text}"[:ERROR_TEXT_BYTES].encode())

    # Untrusted code runs below: any exception it raises is the answer's failure, reported back, never a crash.
    try:
        constants = read_constants(constants_frame[1] or None, max_bytes=init["max_constants_bytes"])
        report = check_source(init["source"])
        if report.violations:
            raise ProgramError("; ".join(report.violations[:5]))
        applied = apply_limits(init)
        namespace = restricted_globals(constants)
        exec(compile(init["source"], "solution.py", "exec"), namespace)
        function = namespace.get(entry)
        if not callable(function):
            raise ProgramError(f"solution.py does not define {entry}()")
        if peak_memory_mb() > init["memory_mb"]:  # where RLIMIT_AS is not enforced (macOS): checked after the fact
            raise ProgramError(f"memory limit of {init['memory_mb']} MB exceeded")
    except Exception as error:
        fail("loading solution.py", error)
        return
    write_frame(proto_out, b"O", json.dumps(applied).encode())
    while (frame := read_frame(proto_in)) is not None:
        try:
            x = np.load(io.BytesIO(frame[1]), allow_pickle=False)
            payload = npy_bytes(run_entry(function, entry, x))
            if peak_memory_mb() > init["memory_mb"]:
                raise ProgramError(f"memory limit of {init['memory_mb']} MB exceeded")
        except Exception as error:
            fail(f"calling {entry}", error)
            return
        write_frame(proto_out, b"O", payload)


# ---- the parent side ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLimits:
    call_seconds: float
    load_seconds: float
    cpu_seconds: float
    memory_mb: int
    max_constants_bytes: int


def resident_bytes(pid: int) -> int | None:
    statm = Path(f"/proc/{pid}/statm")
    if statm.is_file():
        try:
            return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return int(result.stdout.strip()) * 1024 if result.stdout.strip().isdigit() else None


class ProgramProcess:
    """One worker process running one solution.py. Every failure raises ProgramError; close() always kills it."""

    def __init__(self, *, source: str, entry: str, constants: bytes | None, limits: RunLimits):
        if entry not in ENTRIES:
            raise ValueError(f"entry {entry!r} not in {ENTRIES}")
        self.source, self.entry, self.constants, self.limits = source, entry, constants, limits
        self.seconds = 0.0
        self.calls = 0
        self.applied: dict[str, bool] = {}
        self.process: subprocess.Popen | None = None
        self.jail: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> ProgramProcess:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        self.jail = tempfile.TemporaryDirectory(prefix="wsp-program-")
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=self.jail.name, env=env, close_fds=True, start_new_session=True,
        )
        init = {
            "source": self.source, "entry": self.entry, "jail_dir": self.jail.name,
            "cpu_seconds": self.limits.cpu_seconds, "memory_mb": self.limits.memory_mb,
            "max_constants_bytes": self.limits.max_constants_bytes,
        }
        deadline = time.monotonic() + self.limits.load_seconds
        self._send(b"M", json.dumps(init).encode(), deadline)
        self._send(b"M", self.constants or b"", deadline)
        self.applied = json.loads(self._receive(deadline, max_bytes=ERROR_TEXT_BYTES * 2, what="loading solution.py"))

    def call(self, x: np.ndarray, *, max_reply_bytes: int) -> np.ndarray:
        started = time.monotonic()
        deadline = started + self.limits.call_seconds
        self._send(b"M", npy_bytes(np.ascontiguousarray(x)), deadline)
        payload = self._receive(deadline, max_bytes=max_reply_bytes, what=f"calling {self.entry}")
        self.seconds += time.monotonic() - started
        self.calls += 1
        try:
            return np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise ProgramError(f"unreadable reply from {self.entry}: {error}") from error

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
        if self.process is not None:
            self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        if self.jail is not None:
            self.jail.cleanup()

    def _died(self, what: str) -> ProgramError:
        assert self.process is not None
        self.process.kill()
        code = self.process.wait()
        reason = {-24: "CPU time limit exceeded", -9: "killed", -25: "file size limit exceeded"}.get(code, f"exit code {code}")
        return ProgramError(f"{what}: the program process stopped ({reason})")

    def _check_memory(self, what: str) -> None:
        assert self.process is not None
        rss = resident_bytes(self.process.pid)
        if rss is not None and rss > self.limits.memory_mb * 2**20:
            self.process.kill()
            raise ProgramError(f"{what}: memory limit of {self.limits.memory_mb} MB exceeded")

    def _send(self, status: bytes, payload: bytes, deadline: float) -> None:
        assert self.process is not None and self.process.stdin is not None
        data = memoryview(FRAME.pack(status, len(payload)) + payload)
        fd = self.process.stdin.fileno()
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.process.kill()
                raise ProgramError("time limit exceeded while sending inputs")
            _, ready, _ = select.select([], [fd], [], min(remaining, WATCH_INTERVAL_S))
            if not ready:
                self._check_memory("sending inputs")
                continue
            try:
                data = data[os.write(fd, data[: 1 << 16]) :]
            except (BrokenPipeError, OSError) as error:
                raise self._died("sending inputs") from error

    def _read_exact(self, n: int, deadline: float, what: str) -> bytes:
        assert self.process is not None and self.process.stdout is not None
        fd = self.process.stdout.fileno()
        out = bytearray()
        while len(out) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.process.kill()
                raise ProgramError(f"{what}: time limit exceeded")
            ready, _, _ = select.select([fd], [], [], min(remaining, WATCH_INTERVAL_S))
            if not ready:
                self._check_memory(what)
                continue
            chunk = os.read(fd, min(n - len(out), 1 << 20))
            if not chunk:
                raise self._died(what)
            out += chunk
        return bytes(out)

    def _receive(self, deadline: float, *, max_bytes: int, what: str) -> bytes:
        status, length = FRAME.unpack(self._read_exact(FRAME.size, deadline, what))
        if length > max_bytes:
            self.process.kill()  # type: ignore[union-attr]
            raise ProgramError(f"{what}: reply of {length:,} bytes is larger than the {max_bytes:,} allowed")
        payload = self._read_exact(length, deadline, what)
        if status == b"E":
            raise ProgramError(payload.decode(errors="replace"))
        if status != b"O":
            raise ProgramError(f"{what}: malformed reply")
        return payload


# ---- the agent's checker (/app/check_program.py) -------------------------------------------------------------------


def check_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check solution.py against this task's limits (task_limits.json).")
    parser.add_argument("solution")
    parser.add_argument("constants", nargs="?")
    parser.add_argument("--inputs", help="npz with one array x of shape (N, ...): run the program on it")
    parser.add_argument("--limits", default=str(Path(__file__).with_name("task_limits.json")))
    args = parser.parse_args(argv)
    limits = json.loads(Path(args.limits).read_text())
    source = Path(args.solution).read_text()
    report = check_source(source)
    constants_bytes = Path(args.constants).read_bytes() if args.constants else None
    try:
        elements = sum(int(a.size) for a in read_constants(constants_bytes, max_bytes=limits["max_constants_bytes"]).values())
    except ProgramError as error:
        report.violations.append(str(error))
        elements = 0
    total = report.literals + elements
    print(f"code lines: {report.lines} (limit {limits['max_lines']})")
    print(f"constants: {report.literals} in literals + {elements} in constants.npz = {total} "
          f"(budget {limits['constants_budget']}: {'within' if total <= limits['constants_budget'] else 'OVER, score 0'})")
    if report.lines > limits["max_lines"]:
        report.violations.append(f"{report.lines} code lines > {limits['max_lines']}")
    if len(source.encode()) > limits["max_source_bytes"]:
        report.violations.append(f"solution.py is larger than {limits['max_source_bytes']:,} bytes")
    for violation in report.violations:
        print(f"INVALID: {violation}")
    if report.violations or not args.inputs:
        return 1 if report.violations else 0
    with np.load(args.inputs, allow_pickle=False) as archive:
        x = archive["x"]
    run = RunLimits(call_seconds=limits["call_seconds"], load_seconds=limits["load_seconds"],
                    cpu_seconds=limits["cpu_seconds"], memory_mb=limits["memory_mb"],
                    max_constants_bytes=limits["max_constants_bytes"])
    try:
        with ProgramProcess(source=source, entry=limits["entry"], constants=constants_bytes, limits=run) as program:
            out = program.call(x, max_reply_bytes=max(1, x.size) * 8 * 64 + 4096)
    except ProgramError as error:
        print(f"INVALID at run time: {error}")
        return 1
    np.set_printoptions(threshold=50, linewidth=120)
    print(f"{limits['entry']} returned shape {out.shape}, {program.seconds:.3f} s:\n{out}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        sys.exit(check_main(sys.argv[1:]))
