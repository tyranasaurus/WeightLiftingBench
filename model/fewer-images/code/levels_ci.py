"""Mean skill score over the 7 tasks at each number of training images, with 95% intervals: the runs within each task are
resampled (tasks stay fixed, so the interval is run to run noise, not task choice). 10,000 resamples, seed 0.
Inputs: fewer-images/runs.csv (1, 10, 100 per class) and results/task_results.csv (full training set, a rule break counts
as 0% accuracy). Runs with valid_output False are left out. Usage: python3 fewer-images/code/levels_ci.py"""
import csv, random
from collections import defaultdict
from pathlib import Path
F = Path(__file__).resolve().parent.parent
CLASSES = {"digits": 10, "clothing": 10, "traffic": 43, "galaxies": 10, "letters": 24, "cifar": 10, "person": 2}
skill = lambda t, a: (a - 1 / CLASSES[t]) / (1 - 1 / CLASSES[t])
cell = defaultdict(list)
for r in csv.DictReader(open(F / "runs.csv")):
    if r["valid_output"] == "True": cell[(r["level"], r["model"], r["arm"], r["task"])].append(float(r["accuracy"]))
for r in csv.DictReader(open(F.parent / "results/task_results.csv")):
    if r["valid_output"] == "True": cell[("full", r["model"], r["arm"], r["task"])].append(float(r["scored_accuracy"]))
rng = random.Random(0)
for name, m, arm in (("GPT-6.2 Astra programs", "openai/gpt-6.2-astra", "program"), ("GPT-6.2 Astra networks", "openai/gpt-6.2-astra", "design"),
                     ("Fable 5.2 networks", "claude-fable-5-2", "design")):
    row = []
    for n in ("1", "10", "100", "full"):
        rs = {t: cell[(n, m, arm, t)] for t in CLASSES}
        est = sum(skill(t, sum(r) / len(r)) for t, r in rs.items()) / 7
        boot = sorted(sum(skill(t, sum(rng.choices(r, k=len(r))) / len(r)) for t, r in rs.items()) / 7 for _ in range(10000))
        row.append((round(est, 3), round(boot[250], 3), round(boot[9749], 3), min(len(r) for r in rs.values())))
    print(name, row)
