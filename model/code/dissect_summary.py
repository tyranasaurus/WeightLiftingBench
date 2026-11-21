"""Summarise the fine-tune dissection (finetune/*.dissect.json) per model and task."""
import json, statistics as st
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path(__file__).resolve().parent.parent / "results"
TASKS = ["digits", "clothing", "traffic", "galaxies", "letters", "cifar", "person"]
rows = defaultdict(list)
for f in sorted((W / "finetune").glob("*.dissect.json")):
    run = f.name.split(".")[0]; d = json.loads(f.read_text()).get("dissect")
    if not d: continue
    m, t = run.split("-")[:2]
    y = np.array(d["labels"]); p0 = np.array(d["designed"]["pred"]); p1 = np.array(d["all"]["pred"])
    flips = p0 != p1; fixed = flips & (p1 == y); broke = flips & (p0 == y)
    ch = d["all"]["change"]
    ro = [v["weight"] for v in ch.values() if v["readout"]]; feat = [v["weight"] for v in ch.values() if not v["readout"]]
    rows[(m, t)].append({"run": run, "designed": d["designed"]["accuracy"], "all": d["all"]["accuracy"], "readout": d["readout"]["accuracy"], "bias": d["bias"]["accuracy"],
                         "flip": flips.mean(), "fixed": fixed.mean(), "broke": broke.mean(),
                         "chg_readout": max(ro) if ro else None, "chg_feat": float(np.median(feat)) if feat else None,
                         "noise": {k: sum(v) / len(v) for k, v in d["noise_all_entries"].items()},
                         "train_all": d["all"]["trainable"], "train_ro": d["readout"]["trainable"]})
out = {}
for m in ("astra", "fable"):
    print(f"\n{m}: task designed | all readout bias | flip fixed broke | chg_ro chg_feat | noise 1e-4 1e-3 1e-2 1e-1")
    for t in TASKS:
        R = rows.get((m, t))
        if not R: continue
        mean = lambda k: sum(r[k] for r in R) / len(R)
        nz = {k: sum(r["noise"][k] for r in R) / len(R) for k in R[0]["noise"]}
        cro = [r["chg_readout"] for r in R if r["chg_readout"] is not None]; cf = [r["chg_feat"] for r in R if r["chg_feat"] is not None]
        o = {"n": len(R), **{k: mean(k) for k in ("designed", "all", "readout", "bias", "flip", "fixed", "broke")}, "noise": nz,
             "chg_readout": sum(cro) / len(cro) if cro else None, "chg_feat": sum(cf) / len(cf) if cf else None}
        out[f"{m}-{t}"] = o
        print(f"{t:9s} n{len(R)} {100*o['designed']:5.1f} | {100*o['all']:5.1f} {100*o['readout']:5.1f} {100*o['bias']:5.1f} | {100*o['flip']:4.1f} {100*o['fixed']:4.1f} {100*o['broke']:4.1f} | "
              f"{o['chg_readout'] or 0:.3f} {o['chg_feat'] or 0:.4f} | " + " ".join(f"{100*nz[k]:5.1f}" for k in sorted(nz, key=float)))
(W / "analysis").mkdir(exist_ok=True); (W / "analysis/dissect_summary.json").write_text(json.dumps(out, indent=1))
