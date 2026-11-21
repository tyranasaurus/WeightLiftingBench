"""Score submissions again from this repository, without Docker or Valkyrie.

    uv run --no-project --with numpy python model/code/score_submission.py DATA_DIR [RUN ...]

DATA_DIR holds the test sets of the data release (<task>-test.npz). With no RUN it scores all of results/submissions.
For each run it does what the scorer does: run the submitted build.py alone in an empty folder (python -I, no data),
then run the rebuilt answer on every test image. Design runs go through the task's own interpreter
(tasks/<task>-nocopy-design-slim/tests/blocks.py and check_network.py); program runs import the rebuilt solution.py
and call predict. It prints the accuracy next to the one in results/runs.csv.

What it leaves out: the official scorer runs build.py as an unprivileged user with CPU and memory limits, runs programs
in a restricted process, and checks programs give the same answers on shuffled batches. build.py is code a model wrote:
run this in a sandbox or container if you do not trust it.
"""
import csv, importlib.util, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

M = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1])
RUNS = {r["key"]: r for r in csv.DictReader(open(M / "results/runs.csv"))}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def score(key):
    row = RUNS[key]; task, arm = row["task"], row["arm"]
    sub = M / "results/submissions" / key
    work = Path(tempfile.mkdtemp()); out = work / "out"; out.mkdir()
    (work / "build.py").write_text((sub / "build.py").read_text())
    b = subprocess.run([sys.executable, "-I", "build.py", str(out)], cwd=work, capture_output=True, text=True, timeout=2400)
    if b.returncode: return None, f"build failed: {b.stderr[-200:]}"
    hidden = np.load(DATA / f"{task}-test.npz")
    x = hidden["images"].astype(np.float32) / 255; x = x[:, None] if x.ndim == 3 else x; y = hidden["labels"]
    if arm == "design":
        tests = M / "tasks" / f"{task}-nocopy-design-slim/tests"
        if not (out / "network.json").exists(): (out / "network.json").write_text((sub / "network.json").read_text())
        elif json.loads((out / "network.json").read_text()) != json.loads((sub / "network.json").read_text()): return None, "rebuilt network.json differs from the submitted one"
        check = load(tests / "check_network.py", f"check_{task}").check; run = load(tests / "blocks.py", "blocks").run
        net, weights, problem = check(str(out))[:3]
        if problem: return None, f"invalid network: {problem}"
        pred = []
        with np.errstate(all="ignore"):
            for i in range(0, len(x), 250):
                s = np.asarray(run(net, weights, x[i:i + 250]), dtype=np.float64)
                pred.append(np.nan_to_num(s, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf).argmax(1))
    else:
        if (out / "solution.py").read_text() != (sub / "solution.py").read_text(): return None, "rebuilt solution.py differs from the submitted one"
        os.chdir(out); sys.path.insert(0, str(out))
        consts = dict(np.load(out / "constants.npz", allow_pickle=False)) if (out / "constants.npz").exists() else {}
        spec = importlib.util.spec_from_file_location("solution_" + key.replace("-", "_"), out / "solution.py"); mod = importlib.util.module_from_spec(spec)
        mod.CONSTANTS = consts; spec.loader.exec_module(mod)
        pred = []
        for i in range(0, len(x), 2000):
            o = np.asarray(mod.predict(x[i:i + 2000])); pred.append(o.argmax(1) if o.ndim == 2 else o.astype(int))
        sys.path.remove(str(out))
    return float((np.concatenate(pred) == y).mean()), ""


if __name__ == "__main__":
    keys = sys.argv[2:] or sorted(p.name for p in (M / "results/submissions").iterdir())
    for key in keys:
        t = time.time()
        try: acc, note = score(key)
        except Exception as e: acc, note = None, f"error {e!r}"[:200]
        want = float(RUNS[key]["accuracy"])
        got = "   -  " if acc is None else f"{acc:.4f}"
        flag = "" if acc is not None and abs(acc - want) < 5e-4 else "  <-- differs"
        print(f"{key:32} rescored {got}  recorded {want:.4f}{flag}  {note}  ({time.time() - t:.0f}s)", flush=True)
