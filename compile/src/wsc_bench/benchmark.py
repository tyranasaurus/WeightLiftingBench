"""The 14 Compile v1 tasks: each is a task module plus fixed parameters, a nonzero cap and a seed count."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsc_bench.tasks.base import Target
from wsc_bench.tasks.registry import get_target

PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Task:
    name: str  # e.g. "gcd10"
    tier: str  # "easy" or "hard"
    target: str  # task module id, e.g. "gcd-10bit"
    params: dict[str, Any]
    max_nonzero: int
    eval_seeds: int  # the reward is the mean primary metric over this many seeds

    def build(self) -> Target:
        return get_target(target_id=self.target, params={**self.params, "max_nonzero": self.max_nonzero})

    def prompt(self) -> str:
        """The exact prompt every model received (the served instruction.md)."""
        return (PACKAGE / "prompts" / f"{self.name}.md").read_text()


TASKS: dict[str, Task] = {
    row["task"]: Task(name=row["task"], tier=row["tier"], target=row["target"], params=row["params"],
                      max_nonzero=row["max_nonzero"], eval_seeds=row["eval_seeds"])
    for row in json.loads((PACKAGE / "benchmark.json").read_text())
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; tasks: {', '.join(TASKS)}")
    return TASKS[name]
