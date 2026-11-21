"""Each model's easy and hard tier score as in the blog post, from results/blog_v1_runs.csv.

A tier score is the mean over the tier's tasks of the mean reward_fixed_count of that task's replies (at most 3).
Out-of-budget replies are already 0. Usage: python scripts/blog_v1_scores.py
"""

import collections
import csv
from pathlib import Path

ROWS = Path(__file__).resolve().parents[1] / "results" / "blog_v1_runs.csv"


def main() -> None:
    cells = collections.defaultdict(list)  # (model, tier, task) -> rewards
    for row in csv.DictReader(open(ROWS)):
        cells[(row["model"], row["tier"], row["task"])].append(float(row["reward_fixed_count"]))
    tiers = collections.defaultdict(list)  # (model, tier) -> per-task means
    for (model, tier, _), rewards in cells.items():
        tiers[(model, tier)].append(sum(rewards) / len(rewards))
    print(f"{'model':42} {'easy':>13} {'hard':>13}")
    for model in sorted({m for m, _ in tiers}):
        cols = []
        for tier in ("easy", "hard"):
            means = tiers.get((model, tier))
            cols.append(f"{sum(means) / len(means):.3f} ({len(means)}/7)" if means else "-")
        print(f"{model:42} {cols[0]:>13} {cols[1]:>13}")


if __name__ == "__main__":
    main()
