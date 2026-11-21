"""After Lehmann-style fine-tuning: is the network still the written program? Per model and task (mean over networks):
accuracy written / ours / Lehmann standard / structure-preserving; agreement with Astra's programs before and after;
share of the tuned weight norm in previously-zero entries; accuracy with those new entries removed; layer similarity (CKA)."""
import json, glob, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path(__file__).resolve().parent.parent / "results"
TASKS = ["digits", "clothing", "traffic", "galaxies", "letters", "cifar", "person"]
prog = defaultdict(list)
for f in glob.glob(str(W / "analysis/program_preds/astra-*.json")):
    d = json.load(open(f)); prog[d["run"].split("-")[1]].append(np.array(d["pred"]))
R = defaultdict(list)
for f in sorted(glob.glob(str(W / "finetune/*.dissect.json"))):
    d = json.load(open(f)); r = Path(f).name.split(".")[0]; m, t = r.split("-")[:2]
    L = d.get("lehmann", {}); D = d.get("dissect")
    if not D or "pred" not in L.get("standard", {}): continue
    y = np.array(D["labels"]); p0 = np.array(D["designed"]["pred"]); po = np.array(D["all"]["pred"])
    row = {"run": r, "designed": D["designed"]["accuracy"], "ours": D["all"]["accuracy"]}
    for k in ("standard", "structure"):
        v = L.get(k, {})
        if "pred" not in v: continue
        pl = np.array(v["pred"]); row[k] = float((pl == y).mean())
        row[k + "_support_only"] = v["support_only_accuracy"]
        lay = v["layers"]; sizes = {n: x["size"] for n, x in lay.items()}
        row[k + "_new_mass"] = float(np.median([x["new_mass"] for x in lay.values()]))
        row[k + "_written_change"] = float(np.median([x["written_change"] for x in lay.values()]))
        ck = v["cka"]; names = list(ck)
        row[k + "_cka_first"] = ck[names[0]]; row[k + "_cka_last"] = ck[names[-1]]; row[k + "_cka_min"] = min(ck.values()); row[k + "_cka_median"] = float(np.median(list(ck.values())))
        row[k + "_agree_written"] = float((pl == p0).mean())
        if m == "astra" and prog[t]:
            row[k + "_agree_prog"] = float(np.mean([(pl == q).mean() for q in prog[t]]))
    if m == "astra" and prog[t]:
        row["written_agree_prog"] = float(np.mean([(p0 == q).mean() for q in prog[t]])); row["ours_agree_prog"] = float(np.mean([(po == q).mean() for q in prog[t]]))
        row["prog_acc"] = float(np.mean([(q == y).mean() for q in prog[t]]))
    R[(m, t)].append(row)
out = {}
for (m, t), rows in sorted(R.items()):
    keys = set.intersection(*[set(r) for r in rows]) - {"run"}
    out[f"{m}-{t}"] = {"n": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}
(W / "analysis/drift_summary.json").write_text(json.dumps(out, indent=1))
f1 = lambda v: f"{100 * v:5.1f}" if v is not None else "  -  "
print("model-task n | written ours std struct | std: support-only newmass cka(first,median,last) agree-written | agree prog: written ours std")
for k, o in out.items():
    print(f"{k:16s} {o['n']} | {f1(o['designed'])} {f1(o['ours'])} {f1(o.get('standard'))} {f1(o.get('structure'))} | {f1(o.get('standard_support_only'))} {f1(o.get('standard_new_mass'))} "
          f"{o.get('standard_cka_first', 0):.2f} {o.get('standard_cka_median', 0):.2f} {o.get('standard_cka_last', 0):.2f} {f1(o.get('standard_agree_written'))} | "
          f"{f1(o.get('written_agree_prog'))} {f1(o.get('ours_agree_prog'))} {f1(o.get('standard_agree_prog'))}")
