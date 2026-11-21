"""Collect the reading runs from Valkyrie into results/task_results.csv: one row per (model, task), the result that counts.

Each model's tasks come from several runs (a first pass, then fills and reruns). Later run labels replace earlier ones
task by task; a rerun's result replaces the first attempt's, right or wrong. Audited rule breaks (a forward pass through
part of the network) are scored 0 from results/rule-break-overrides.json; results/regrade-overrides.json holds answers
regraded after a parser fix. Needs the Valkyrie CLI and access to the runs.

Usage: python scripts/collect_results.py --cache /tmp/reading-cache   (then python scripts/score.py)
"""
import argparse, csv, json, subprocess
from pathlib import Path

U = Path(__file__).resolve().parent.parent
RUNS = {l.split()[0]: l.split()[1] for l in open(U / "results/run_ids.txt") if len(l.split()) == 2}
FAM = {r["task"]: r["family"] for r in csv.DictReader(open(U / "tasks.csv"))}

# model -> (release date, [(run label, families taken from it; None = all)]), in the order they apply
MODELS = {
    "GPT-6.2 Sol":   ("2026-11-17", [("wsu-v7-sol-reading", {"atari"}), ("wsu-v8-sol-rest", {"atari", "lora", "style"}),
                                     ("wsu-v8-sol-fill", {"atari"}), ("wsu-v9-sol-bert", {"bert"})]),
    "Opus 5.5":      ("2026-09-22", [("wsu-v8-opus-smoke", {"atari", "lora", "style"}), ("wsu-v8-opus-slice", {"atari", "lora", "style"}),
                                     ("wsu-v8-opus-rest", {"atari", "lora", "style"}), ("wsu-v9-opus-bert", {"bert"}),
                                     ("wsu-v10-refmt-or-claude-opus-5.5", None), ("wsu-v10-bA-or-claude-opus-5.5", None), ("wsu-v10-runaway7-or-claude-opus-5.5", None)]),
    "GPT-5.5":       ("2026-04-24", [("wsu-v10-full-gpt-5.5", None), ("wsu-v10-fill-gpt-5.5", None), ("wsu-v10-bB-gpt-5.5", None), ("wsu-v10-filter3-gpt-5.5", None), ("wsu-v10-lorafill-or-gpt-5.5", None)]),
    "Opus 4.8":      ("2026-05-27", [("wsu-v10-full-claude-opus-4.8", None), ("wsu-v10-fill-or-claude-opus-4.8", None),
                                     ("wsu-v10-fill2-or-claude-opus-4.8", None), ("wsu-v10-retry-claude-opus-4.8", None), ("wsu-v10-bA-or-claude-opus-4.8", None), ("wsu-v10-rerun-or-claude-opus-4.8", None)]),
    "Fable 5.2":     ("2026-11-09", [("wsu-v10-full-claude-fable-5.2", None), ("wsu-v10-fill-or-claude-fable-5.2", None),
                                     ("wsu-v10-fill2-or-claude-fable-5.2", None), ("wsu-v10-retry-claude-fable-5.2", None), ("wsu-v10-bA-or-claude-fable-5.2", None)]),
    "GPT-6.2 Astra": ("2026-11-17", [("wsu-v10-smoke-gpt-6.2-astra", None), ("wsu-v10-slice20-gpt-6.2-astra", None),
                                     ("wsu-v10-fill-gpt-6.2-astra", None), ("wsu-v10-fill3-gpt-6.2-astra", None), ("wsu-v10-rest50-gpt-6.2-astra", None), ("wsu-v10-bA-gpt-6.2-astra", None), ("wsu-v10-bB-gpt-6.2-astra", None), ("wsu-v10-bC-gpt-6.2-astra", None), ("wsu-v10-fix2-gpt-6.2-astra", None), ("wsu-v10-lorafill-or-gpt-6.2-astra", None)]),
    "GPT-6.1 Sol":   ("2026-09-29", [("wsu-v10-full-gpt-6.1-sol", None), ("wsu-v10-b1-gpt-6.1-sol", None), ("wsu-v10-bA-gpt-6.1-sol", None), ("wsu-v10-bB-gpt-6.1-sol", None), ("wsu-v10-bC-gpt-6.1-sol", None), ("wsu-v10-fix2-gpt-6.1-sol", None)]),
    "GPT-6 Astra":   ("2026-09-04", [("wsu-v10-slice20-gpt-6-astra", None), ("wsu-v10-rest50-gpt-6-astra", None), ("wsu-v10-bA-gpt-6-astra", None), ("wsu-v10-bB-gpt-6-astra", None), ("wsu-v10-bC-gpt-6-astra", None)]),
    "Fable 5.1":     ("2026-09-01", [("wsu-v10-slice20-or-claude-fable-5.1", None), ("wsu-v10-retry-claude-fable-5.1", None), ("wsu-v10-fill3-claude-fable-5.1", None), ("wsu-v10-rest50-claude-fable-5.1", None), ("wsu-v10-b1-or-claude-fable-5.1", None), ("wsu-v10-b2-or-claude-fable-5.1", None), ("wsu-v10-bA-or-claude-fable-5.1", None), ("wsu-v10-full-b1-or-claude-fable-5.1", None), ("wsu-v10-full-b2-or-claude-fable-5.1", None), ("wsu-v10-full-b3-or-claude-fable-5.1", None), ("wsu-v10-lorafill-or-claude-fable-5.1", None)]),
    "Kimi K3.1 Pro": ("2026-10-23", [("wsu-v10-full-kimi-k3.1-pro", None), ("wsu-v10-retry-kimi-k3.1-pro", None), ("wsu-v10-check-kimi-k3.1-pro", None), ("wsu-v10-fix-kimi-k3.1-pro", None), ("wsu-v10-rerun-kimi-k3.1-pro", None)]),
    "GLM-5.3":       ("2026-08-18", [("wsu-v10-full-glm-5.3", None), ("wsu-v10-retry-glm-5.3", None), ("wsu-v10-rerun-glm-5.3", None)]),
}

OVERRIDES = {}
for name in ("regrade-overrides.json", "rule-break-overrides.json"):  # rule breaks apply last
    for rid, tasks in json.loads((U / "results" / name).read_text()).items():
        if not rid.startswith("_"):
            for t, r in tasks.items(): OVERRIDES[(rid, t)] = (r, "rule break, scored 0" if name.startswith("rule") else "regraded")


def rewards(*, run_id, cache):
    f = cache / f"results-{run_id}.json"
    if not f.exists():
        subprocess.run(["valkyrie", "run", "results", run_id], cwd=cache, capture_output=True, input="y\n", text=True)
    res = json.loads(f.read_text()).get("evaluation_results") or {}
    return {t: float(e["rewards"]["reward"]) for t, e in res.items() if e and (e.get("rewards") or {}).get("reward") is not None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); cache = Path(ap.parse_args().cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, (date, parts) in MODELS.items():
        got = {}
        for label, fams in parts:
            if label not in RUNS: continue
            for t, r in rewards(run_id=RUNS[label], cache=cache).items():
                if t in FAM and (fams is None or FAM[t] in fams):
                    r, note = OVERRIDES.get((RUNS[label], t), (r, ""))
                    got[t] = (r, label, RUNS[label], note)
        rows += [[model, date, t, FAM[t], label, rid, r, note] for t, (r, label, rid, note) in sorted(got.items())]
    with open(U / "results/task_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "release_date", "task", "family", "run_label", "run_id", "reward", "note"]); w.writerows(rows)
    print("task_results.csv:", len(rows), "rows")


if __name__ == "__main__":
    main()
