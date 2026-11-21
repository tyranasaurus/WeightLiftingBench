"""Test-set predictions of every GPT-6.2 Astra deterministic program: rebuild from build.py with no data, import
solution.py, call predict on the test images (as the verifier does). Writes results/analysis/program_preds/<run>.json.
Usage: python model/code/program_preds.py DATA_DIR   (DATA_DIR holds <task>-test.npz from the data release)"""
import json, subprocess, sys, tempfile, importlib.util, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
M = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1])
S = M / "results/submissions"
OUT = M / "results/analysis/program_preds"; OUT.mkdir(parents=True, exist_ok=True)


def one(run):
  try:
    return one_(run)
  except Exception as e:
    return run, f'error {e!r}'[:300]


def one_(run):
    task = run.split("-")[1]; t0 = time.time()
    d = Path(tempfile.mkdtemp()); (d / "build.py").write_text((S / run / "build.py").read_text())
    out = d / "out"; out.mkdir()
    b = subprocess.run([sys.executable, "build.py", str(out)], cwd=d, capture_output=True, text=True)
    if b.returncode: return run, f"build failed: {b.stderr[-300:]}"
    hidden = np.load(DATA / f"{task}-test.npz")
    x = hidden["images"].astype(np.float32) / 255; x = x[:, None] if x.ndim == 3 else x
    k = int(hidden["labels"].max()) + 1
    os.chdir(out); sys.path.insert(0, str(out))
    cpath = out / "constants.npz"
    consts = dict(np.load(cpath, allow_pickle=False)) if cpath.exists() else {}
    spec = importlib.util.spec_from_file_location("solution_" + run.replace("-", "_"), out / "solution.py"); mod = importlib.util.module_from_spec(spec)
    mod.CONSTANTS = consts; spec.loader.exec_module(mod)
    parts = []
    for i in range(0, len(x), 2000):
        o = np.asarray(mod.predict(x[i:i + 2000])); parts.append(o.argmax(1) if o.ndim == 2 else o.astype(int))
    p = np.concatenate(parts)
    acc = float((p == hidden["labels"]).mean())
    (OUT / f"{run}.json").write_text(json.dumps({"run": run, "accuracy": acc, "pred": p.tolist(), "seconds": time.time() - t0}))
    return run, f"{acc:.4f} in {time.time() - t0:.0f}s"


if __name__ == "__main__":
    runs = sorted(p.name for p in S.iterdir() if p.name.startswith("astra-") and "-program-" in p.name)
    with ProcessPoolExecutor(6) as ex:
        for r, msg in ex.map(one, runs): print(r, msg, flush=True)
