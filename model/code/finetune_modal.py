"""Fine-tune no-copy design networks on Modal GPUs, one container per network, all in parallel.
Each job gets the run's build.py and network.json, rebuilds weights.npz by running build.py with no data (as the
scorer does), then runs scripts/finetune_nocopy.py with the given options against the task data on the nocopy-ft
volume. Results are merged into finetune/<run>.json, logs into finetune/<run>.modal.log.
Usage: uv run --no-project --with modal modal run scripts/finetune_modal.py --runs "a b c" --opts "--mode curve --epochs 3 --rel 0.001 --mask"
       (--runs-file FILE instead of --runs)"""
import json
from pathlib import Path
import modal

W = Path(os.environ.get("NOCOPY_WORKDIR", "."))  # working folder for run state, logs and collected outputs
app = modal.App("nocopy-finetune")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6", "torch==2.7.1")
         .add_local_file(str(W / "scripts/finetune_nocopy.py"), "/root/finetune_nocopy.py"))
vol = modal.Volume.from_name("nocopy-ft")


@app.function(image=image, gpu="L40S", volumes={"/data": vol}, timeout=3 * 3600, cpu=4, memory=65536)
def finetune(run: str, build_py: str, network_json: str, opts: list) -> dict:
    import os, subprocess, tempfile, time
    task = run.split("-")[1]
    work = Path(tempfile.mkdtemp()); src = work / "src"; out = work / "out"; src.mkdir(); out.mkdir()
    (src / "build.py").write_text(build_py)
    t0 = time.time()
    b = subprocess.run(["python", "-I", "build.py", str(out)], cwd=src, capture_output=True, text=True, timeout=3600,
                       env={"PATH": os.environ["PATH"], "HOME": str(work), "OMP_NUM_THREADS": "4"})
    if b.returncode != 0 or not (out / "weights.npz").exists():
        return {"run": run, "ok": False, "log": f"rebuild failed rc {b.returncode}: {(b.stdout[-1500:] + b.stderr[-1500:])}"}
    (out / "network.json").write_text(network_json)
    rebuild_s = time.time() - t0
    res = work / "result.json"
    f = subprocess.run(["python", "/root/finetune_nocopy.py", str(out), f"/data/ftdata/{task}-nocopy-design", str(res), *opts],
                       capture_output=True, text=True)
    return {"run": run, "ok": f.returncode == 0 and res.exists(), "rebuild_seconds": rebuild_s, "minutes": (time.time() - t0) / 60,
            "result": json.loads(res.read_text()) if res.exists() else None, "log": (f.stdout[-20000:] + f.stderr[-5000:])}


def source(run, name, outs="outs2"):
    return next(q for q in (W / outs / run).glob(f"*/artifacts/app/out/**/{name}") if q.is_file()).read_text()


@app.local_entrypoint()
def main(runs: str = "", runs_file: str = "", opts: str = "--mode curve --epochs 3 --rel 0.001 --mask", outs: str = "outs2", suffix: str = ""):
    names = (Path(runs_file).read_text() if runs_file else runs).split()
    args = [(r, source(r, "build.py", outs), source(r, "network.json", outs), opts.split()) for r in names]
    for out in finetune.starmap(args, return_exceptions=True):
        if isinstance(out, Exception):
            print("job raised:", repr(out)[:300]); continue
        r = out["run"]; (W / "finetune" / f"{r}{suffix}.modal.log").write_text(out["log"])
        if not out["ok"]:
            print(f"{r}: FAILED {out['log'][-300:]}"); continue
        p = W / "finetune" / f"{r}{suffix}.json"; cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update({k: v for k, v in out["result"].items() if k != "minutes"}); p.write_text(json.dumps(cur, indent=1))
        last = [v for k, v in out["result"].items() if k.startswith("curve")]
        pts = last[0]["points"] if last else []
        print(f"{r}: done in {out['minutes']:.1f} min, start {pts[0]['accuracy']:.3f} end {pts[-1]['accuracy']:.3f}" if pts else f"{r}: done")
