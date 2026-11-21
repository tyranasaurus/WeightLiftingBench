"""Image-by-image comparison of GPT-6.2 Astra's programs and networks on the test sets.
Inputs: analysis/program_preds/<run>.json (programs), finetune/<run>.dissect.json (network predictions, designed and fine-tuned)."""
import json, itertools
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent.parent / "results"
TASKS = ["digits", "clothing", "traffic", "galaxies", "letters", "cifar", "person"]
out = {}
for t in TASKS:
    P = {}; N = {}; T = {}; y = None
    for f in sorted((W / "analysis/program_preds").glob(f"astra-{t}-program-*.json")):
        d = json.loads(f.read_text())
        if d["accuracy"] > 0: P[d["run"]] = np.array(d["pred"])
    for f in sorted((W / "finetune").glob(f"astra-{t}-design-*.dissect.json")):
        d = json.loads(f.read_text())["dissect"]; r = f.name.split(".")[0]
        N[r] = np.array(d["designed"]["pred"]); T[r] = np.array(d["all"]["pred"]); y = np.array(d["labels"])
    if not P or not N: continue
    k = int(y.max()) + 1
    agree = lambda a, b: float((a == b).mean())
    def overlap(a, b):  # of the images b gets wrong, the share a also gets wrong
        return float(((a != y) & (b != y)).sum() / max(1, (b != y).sum()))
    pn = [agree(p, n) for p in P.values() for n in N.values()]
    pt = [agree(p, n) for p in P.values() for n in T.values()]
    pp = [agree(a, b) for a, b in itertools.combinations(P.values(), 2)]
    nn = [agree(a, b) for a, b in itertools.combinations(N.values(), 2)]
    # agreement expected if the two were independent with these accuracies and class mixes
    def indep(a, b):
        pa = np.bincount(a, minlength=k) / len(a); pb = np.bincount(b, minlength=k) / len(b)
        # condition on the true class: independent given the label
        s = 0.0
        for c in range(k):
            m = y == c
            if m.any(): s += m.mean() * float(np.dot(np.bincount(a[m], minlength=k) / m.sum(), np.bincount(b[m], minlength=k) / m.sum()))
        return s
    pn_ind = [indep(p, n) for p in P.values() for n in N.values()]
    net_wrong_prog_wrong = [overlap(p, n) for p in P.values() for n in N.values()]  # P(prog wrong | net wrong)
    prog_err = [float((p != y).mean()) for p in P.values()]
    either = [float(((p == y) | (n == y)).mean()) for p in P.values() for n in N.values()]
    rec = lambda a: np.array([((a == c) & (y == c)).sum() / max(1, (y == c).sum()) for c in range(k)])
    rc = [float(np.corrcoef(rec(p), rec(n))[0, 1]) for p in P.values() for n in N.values()]
    o = {"programs": len(P), "networks": len(N), "prog_acc": float(np.mean([(p == y).mean() for p in P.values()])), "net_acc": float(np.mean([(n == y).mean() for n in N.values()])),
         "ft_acc": float(np.mean([(n == y).mean() for n in T.values()])),
         "agree_prog_net": float(np.mean(pn)), "agree_if_independent": float(np.mean(pn_ind)), "agree_prog_ft": float(np.mean(pt)),
         "agree_prog_prog": float(np.mean(pp)) if pp else None, "agree_net_net": float(np.mean(nn)) if nn else None,
         "net_errors_also_prog": float(np.mean(net_wrong_prog_wrong)), "prog_error_rate": float(np.mean(prog_err)),
         "either_right": float(np.mean(either)), "class_recall_corr": float(np.mean(rc))}
    out[t] = o
    print(f"{t:9s} P{o['programs']} N{o['networks']} acc prog {100*o['prog_acc']:.1f} net {100*o['net_acc']:.1f} ft {100*o['ft_acc']:.1f} | agree p-n {100*o['agree_prog_net']:.1f} (indep {100*o['agree_if_independent']:.1f}) p-ft {100*o['agree_prog_ft']:.1f} p-p {100*(o['agree_prog_prog'] or 0):.1f} n-n {100*(o['agree_net_net'] or 0):.1f} | net errs also prog {100*o['net_errors_also_prog']:.1f} (prog err {100*o['prog_error_rate']:.1f}) | either {100*o['either_right']:.1f} | recall corr {o['class_recall_corr']:.2f}")
(W / "analysis/agreement.json").write_text(json.dumps(out, indent=1))
