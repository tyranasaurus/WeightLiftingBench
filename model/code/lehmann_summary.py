"""One network per model and task: our recipe vs Lehmann's fine-tuning settings (supervised form)."""
import json, glob
from pathlib import Path
W = Path(__file__).resolve().parent.parent / "results"
TASKS = ["digits", "clothing", "traffic", "galaxies", "letters", "cifar", "person"]
pick = {}
for f in sorted(glob.glob(str(W / "finetune/*.dissect.json"))):
    d = json.load(open(f)); r = Path(f).name.split(".")[0]; m, t = r.split("-")[:2]
    if "lehmann" in d and "dissect" in d and (m, t) not in pick: pick[(m, t)] = (r, d)
out = {}
print("model task run | designed | ours | standard (1/32 ep, 3 ep) | structure | sgd | calibrated")
for m in ("astra", "fable"):
    for t in TASKS:
        if (m, t) not in pick: continue
        r, d = pick[(m, t)]; L = d["lehmann"]
        f = lambda v: f"{100 * v:5.1f}"
        row = {"run": r, "designed": d["dissect"]["designed"]["accuracy"], "ours": d["dissect"]["all"]["accuracy"],
               **{k: {"first": v["points"][1]["accuracy"] if len(v["points"]) > 1 else None, "end": v["points"][-1]["accuracy"], "min": min(p["accuracy"] for p in v["points"]), "diverged": v["diverged"]} for k, v in L.items()}}
        out[f"{m}-{t}"] = row
        print(f"{m:5s} {t:9s} {r.split('-')[-1]} | {f(row['designed'])} | {f(row['ours'])} | " + " | ".join(f"{f(row[k]['first'] or 0)} {f(row[k]['end'])}{' DIV' if row[k]['diverged'] else ''}" for k in ("standard", "structure", "sgd", "calibrated")))
(W / "analysis/lehmann_summary.json").write_text(json.dumps(out, indent=1))
