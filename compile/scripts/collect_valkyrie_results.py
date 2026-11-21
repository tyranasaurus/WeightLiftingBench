"""Collect the Compile v1 replies behind the blog from Valkyrie into results/runs.csv and results/task_results.csv.

Provenance only: this needs access to the Valkyrie org the runs were made in, so outside users read the CSVs instead.

A (run, task) counts when results/blog_v1_runs.csv lists it as a Valkyrie reply. Pilots, effort sweeps, superseded
reruns and infrastructure errors are not listed there, so they are left out. Each
(run, task) row records the verifier reward and three flags read from the run's agent log:
  out_of_budget  the reply hit the 128k output cap (while reasoning, or on all 4 continuation requests), or scored 0
                 after a request was cut at the cap: the program never finished
  provider_fail  the reply scored 0 because the provider blocked it or failed (SAFETY, finish error, finish blocked)
  cut            some request of the reply was cut at the output cap
and reward_fixed_count: the reward with the sparse nonzero count fixed (see Known issues in the README). The v1
verifier counted a sparse submission's index arrays too; a reply it zeroed for the cap gets its uncapped reward back
when its real count (mean_nonzero_entries: nonzero values plus biases) is within the cap. Otherwise equal to reward.

Usage: python scripts/collect_valkyrie_results.py --cache /tmp/wsc-cache [--workers 8]
"""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V1_TASKS = [f"{name}-write-nnz500000" for name in (
    "add-10digit", "sort10", "gcd10", "subleq-step", "bowling", "multiply32", "arith-expr", "chess-mate1",
    "regex-match", "decimal-period", "sudoku-any", "inverse-bwt", "empty-circumcircle", "reed-solomon-decode")]
V1_DATASETS = {"compile-easy", "arith-expr", "program-l0-capped", "compile-candidates"}
AGENTS = {"api_wsp"}
OUT_OF_BUDGET = ("hit the max_tokens output cap on all 4 requests", "hit the max_tokens output cap while reasoning")
CUT = "finish length"
PROVIDER_FAIL = ("SAFETY", "finish error", "finish blocked")


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def fetch(*, run: dict, cache: Path) -> dict:
    """Per task status, reward and log flags for one run, cached until the run changes."""
    key = f"{run['id']}-{run.get('finished_at') or 'live'}-{json.dumps(run.get('task_state_counts'), sort_keys=True)}"
    path = cache / f"{run['id']}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["key"] == key:
            return cached
    res_path = cache / f"results-{run['id']}.json"
    sh("valkyrie", "run", "results", run["id"], "--path", str(res_path), "-y")
    results = (json.loads(res_path.read_text()).get("evaluation_results") or {}) if res_path.exists() else {}
    tasks = json.loads(sh("valkyrie", "run", "tasks", run["id"], "--format", "json") or "{}").get("tasks", [])
    budget_log = "".join(sh("valkyrie", "run", "logs", run["id"], "--query", q) for q in OUT_OF_BUDGET)
    cut_log = sh("valkyrie", "run", "logs", run["id"], "--query", CUT)
    provider_log = "".join(sh("valkyrie", "run", "logs", run["id"], "--query", q) for q in PROVIDER_FAIL)
    out = {"key": key, "tasks": {}}
    for task in tasks:
        tid = task["task_id"]
        rewards = (results.get(tid) or {}).get("rewards") or {}
        reward = rewards.get("reward")
        zero = not reward
        fixed = reward
        real_count = rewards.get("mean_nonzero_entries", float("inf"))
        if rewards.get("mean_within_budget", 1.0) < 1.0 and real_count <= rewards["mean_max_nonzero"]:
            fixed = next(v for k, v in rewards.items() if k.endswith("_ignoring_budget"))
        out["tasks"][tid] = {"status": task["status"], "reward": reward, "reward_fixed_count": fixed,
                             "out_of_budget": f"[{tid}]" in budget_log or (zero and f"[{tid}]" in cut_log),
                             "provider_fail": zero and f"[{tid}]" in provider_log, "cut": f"[{tid}]" in cut_log}
    path.write_text(json.dumps(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    runs = [r for r in json.loads(sh("valkyrie", "run", "list", "--all", "--format", "json"))["benchmarks"]
            if r["agent_name"] in AGENTS and (r.get("dataset") or "") in V1_DATASETS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fetched = list(pool.map(lambda run: fetch(run=run, cache=args.cache), runs))
    with open(REPO / "results" / "blog_v1_runs.csv") as fh:
        blog = {(row["run"], row["task"]) for row in csv.DictReader(fh) if row["source"] == "valkyrie"}
    kept = [(run, {tid: t for tid, t in f["tasks"].items()
                   if tid in V1_TASKS and (run["id"], tid.removesuffix("-write-nnz500000")) in blog})
            for run, f in zip(runs, fetched)]
    kept = sorted(((run, tasks) for run, tasks in kept if tasks), key=lambda rt: rt[0]["started_at"])
    out = REPO / "results"
    with open(out / "runs.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "label", "model", "agent", "dataset", "status", "started_at", "finished_at", "tasks"])
        for run, tasks in kept:
            writer.writerow([run["id"], run["label"], run["model"], run["agent_name"], run["dataset"], run["status"],
                             run["started_at"], run.get("finished_at") or "", len(tasks)])
    with open(out / "task_results.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "label", "model", "agent", "task", "status", "reward",
                         "reward_fixed_count", "out_of_budget",
                         "provider_fail", "cut"])
        for run, tasks in kept:
            for tid, t in sorted(tasks.items()):
                writer.writerow([run["id"], run["label"], run["model"], run["agent_name"],
                                 tid.removesuffix("-write-nnz500000"), t["status"], t["reward"],
                                 t["reward_fixed_count"], t["out_of_budget"],
                                 t["provider_fail"], t["cut"]])
    print(f"{len(kept)} runs, {sum(len(t) for _, t in kept)} task results")


if __name__ == "__main__":
    main()
