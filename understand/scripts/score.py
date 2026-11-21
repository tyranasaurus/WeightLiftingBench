"""Skill score per model and family from results/task_results.csv. Writes results/scores.json.

skill = (accuracy - chance) / (1 - chance); chance is 0.10 for Atari (10 games) and 0.25 for the others (4 options).
"all" covers the 62 scored tasks (Atari, BERT, style) with each task's own chance. LoRA is reported but not scored.
Usage: python scripts/score.py
"""
import csv, json
from collections import defaultdict
from pathlib import Path

U = Path(__file__).resolve().parent.parent
SCORED = {"atari", "bert", "style"}
CHANCE = {"atari": 0.10, "bert": 0.25, "lora": 0.25, "style": 0.25}
agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0])); date = {}; order = []
for r in csv.DictReader(open(U / "results/task_results.csv")):
    if r["model"] not in date: order.append(r["model"])
    date[r["model"]] = r["release_date"]
    for k in (r["family"], "all") if r["family"] in SCORED else (r["family"],):
        a = agg[r["model"]][k]; a[0] += float(r["reward"]); a[1] += 1; a[2] += CHANCE[r["family"]]
skill = lambda c, n, ch: (c / n - ch / n) / (1 - ch / n)
out = {}
for m in order:
    out[m] = dict(date=date[m], **{k: dict(correct=a[0], n=a[1], skill=round(skill(*a), 3)) for k, a in agg[m].items()})
    a = agg[m]["all"]
    print(f"{m:15} {date[m]}  ALL {a[0]:.0f}/{a[1]} {skill(*a):+.2f}   " + "  ".join(f"{k} {v[0]:.0f}/{v[1]} {skill(*v):+.2f}" for k, v in sorted(agg[m].items()) if k != "all"))
json.dump(out, open(U / "results/scores.json", "w"), indent=1)
