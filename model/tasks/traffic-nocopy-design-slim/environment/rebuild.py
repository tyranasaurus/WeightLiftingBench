"""Data-free rebuild check: python3 rebuild.py OUT_DIR

Copies OUT_DIR/build.py alone into an empty folder and runs `python3 -I build.py NEW_DIR` there as the unprivileged user
nobody, with no network and CPU / memory limits. In the scoring container nobody cannot read the training data, the test
set or the submitted folder, so the weights build.py writes can only come from its own source. They must match the
submitted OUT_DIR/answer files (weights.npz, constants.npz, network.json, solution.py, whichever it contains) to within REL_TOL, or the answer scores 0.
"""
from __future__ import annotations

import json, os, resource, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

MAX_SOURCE_BYTES = 200_000
BUILD_CPU_SECONDS = 1800
BUILD_WALL_SECONDS = 2400
BUILD_MEMORY_MB = 6144
ANSWER_FILES = ("weights.npz", "constants.npz", "network.json", "solution.py")
REL_TOL = 1e-5  # |rebuilt - submitted| <= REL_TOL * (1 + |submitted|): room for last-bit float differences between machines
NOBODY = 65534


def lock_down(paths) -> None:
    """Make these folders unreadable to nobody (only possible as root; the scoring container runs as root)."""
    if os.geteuid() != 0:
        return
    for p in paths:
        if Path(p).exists():
            os.chmod(p, 0o700)


def limits() -> None:
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (BUILD_CPU_SECONDS, BUILD_CPU_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (BUILD_MEMORY_MB << 20, BUILD_MEMORY_MB << 20))
    except (ValueError, OSError):
        pass
    if os.geteuid() == 0:
        os.setgroups([]); os.setgid(NOBODY); os.setuid(NOBODY)


def load_weights(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def rebuild(out_dir: Path, *, need_network_json: bool, lock: tuple = ()) -> dict:
    """Returns {"ok": bool, "error": str, ...details}."""
    build = out_dir / "build.py"
    if not build.is_file():
        return {"ok": False, "error": f"missing {build}"}
    size = build.stat().st_size
    if size > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"build.py is {size:,} bytes, over the {MAX_SOURCE_BYTES:,} byte limit", "source_bytes": size}
    answer = [f for f in ANSWER_FILES if (out_dir / f).is_file()]
    if need_network_json and "network.json" not in answer:
        return {"ok": False, "error": "missing network.json"}
    if not answer:
        return {"ok": False, "error": "no answer files (weights.npz, constants.npz, network.json or solution.py) to rebuild"}
    submitted = {f: (out_dir / f).read_bytes() for f in answer}
    lock_down(lock)
    work = Path(tempfile.mkdtemp(prefix="rebuild-"))
    src, new = work / "src", work / "new"
    src.mkdir(); new.mkdir()
    shutil.copyfile(build, src / "build.py")
    if os.geteuid() == 0:
        for p in (work, src, new, src / "build.py"):
            os.chown(p, NOBODY, NOBODY)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        proc = subprocess.run([sys.executable, "-I", "build.py", str(new)], cwd=src, env=env, preexec_fn=limits,
                              capture_output=True, text=True, timeout=BUILD_WALL_SECONDS)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"build.py took over {BUILD_WALL_SECONDS} s", "source_bytes": size}
    tail = (proc.stdout[-1500:] + proc.stderr[-1500:]).strip()
    if proc.returncode != 0:
        return {"ok": False, "error": f"build.py exited with code {proc.returncode} when run without the data: {tail}", "source_bytes": size}
    worst, worst_key = 0.0, ""
    for f, data in submitted.items():
        if not (new / f).is_file():
            return {"ok": False, "error": f"build.py ran but wrote no {f} into the folder given as its first argument", "source_bytes": size}
        if f.endswith(".npz"):
            import io
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                s_arr = {k: z[k].astype(np.float32) for k in z.files}
            r_arr = load_weights(new / f)
            if set(r_arr) != set(s_arr):
                return {"ok": False, "error": f"{f}: rebuilt keys differ: missing {sorted(set(s_arr) - set(r_arr))[:10]}, extra {sorted(set(r_arr) - set(s_arr))[:10]}", "source_bytes": size}
            for k, s in s_arr.items():
                r = r_arr[k]
                if r.shape != s.shape:
                    return {"ok": False, "error": f"{f} {k}: rebuilt shape {r.shape} != submitted {s.shape}", "source_bytes": size}
                with np.errstate(all="ignore"):
                    d = float(np.max(np.abs(r.astype(np.float64) - s) / (1 + np.abs(s)), initial=0.0))
                if not np.isfinite(d):
                    d = float("inf")
                if d > worst:
                    worst, worst_key = d, f"{f} {k}"
        elif f.endswith(".json"):
            if json.loads((new / f).read_text()) != json.loads(data):
                return {"ok": False, "error": f"rebuilt {f} differs from the submitted one", "source_bytes": size}
        elif (new / f).read_bytes() != data:
            return {"ok": False, "error": f"rebuilt {f} differs from the submitted one (it must match byte for byte)", "source_bytes": size}
    detail = {"source_bytes": size, "max_relative_difference": worst, "worst_tensor": worst_key, "files": answer}
    if worst > REL_TOL:
        return {"ok": False, "error": f"rebuilt numbers differ from the submitted ones (worst {worst_key}: {worst:.3g} > {REL_TOL:g})", **detail}
    shutil.rmtree(work, ignore_errors=True)
    return {"ok": True, "error": "", **detail}


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
    need = (out / "network.json").exists() or os.environ.get("NEED_NETWORK_JSON") == "1"
    res = rebuild(out, need_network_json=need, lock=("/app", "/root"))
    print(json.dumps(res, indent=2))
    print("REBUILD OK: build.py reproduces your weights without the data." if res["ok"] else "REBUILD FAILED: this answer would score 0.")
    sys.exit(0 if res["ok"] else 1)


def regenerate(out_dir: Path) -> str:
    """Slim tasks carry no weights.npz to the scorer: run build.py (as nobody, no data, same limits) and put the files it
    writes into out_dir before scoring. Returns "" on success or an error message."""
    build = out_dir / "build.py"
    if not build.is_file():
        return f"missing {build}"
    work = Path(tempfile.mkdtemp(prefix="regen-")); src, new = work / "src", work / "new"
    src.mkdir(); new.mkdir(); shutil.copyfile(build, src / "build.py")
    if os.geteuid() == 0:
        for p in (work, src, new, src / "build.py"):
            os.chown(p, NOBODY, NOBODY)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        proc = subprocess.run([sys.executable, "-I", "build.py", str(new)], cwd=src, env=env, preexec_fn=limits,
                              capture_output=True, text=True, timeout=BUILD_WALL_SECONDS)
    except subprocess.TimeoutExpired:
        return f"build.py took over {BUILD_WALL_SECONDS} s"
    if proc.returncode != 0:
        return f"build.py exited with code {proc.returncode} when run without the data: {(proc.stdout[-800:] + proc.stderr[-800:]).strip()}"
    for f in ("weights.npz", "constants.npz", "network.json", "solution.py"):
        if (new / f).is_file() and not (out_dir / f).is_file():
            shutil.copyfile(new / f, out_dir / f)
    shutil.rmtree(work, ignore_errors=True)
    return ""
