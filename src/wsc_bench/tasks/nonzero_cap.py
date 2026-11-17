"""The nonzero cap: `max_nonzero` in a task's params.

get_target wraps the task class so its prompt states the cap and its primary metric scores 0 when the weights hold more
nonzero entries than the cap. The uncapped value stays in `<primary>_ignoring_budget`, next to nonzero_weights,
max_nonzero and within_budget. Tasks that take `max_nonzero` themselves (inverse-bwt) keep their own handling.
"""

from functools import cache

import numpy as np

from wsc_bench.tasks.base import Metrics, Split, Target, Weights

SCORING_HEADING = "\n## Scoring"
SIZE_LIMIT_HEADING = "\n## Size limit"


SPARSE_KEYS = frozenset({"layer", "row", "col", "val"})


def nonzero_count(weights: Weights) -> int:
    """Nonzero entries in every tensor. A sparse (layer, row, col, val) submission counts nonzero val plus biases
    (keys starting with b), since its index arrays are addresses, not weights."""
    if SPARSE_KEYS <= weights.keys():
        return int(sum(np.count_nonzero(tensor) for name, tensor in weights.items() if name == "val" or name.startswith("b")))
    return int(sum(np.count_nonzero(tensor) for tensor in weights.values()))


def cap_text(max_nonzero: int) -> str:
    return (f"## Size limit\nAll weights and biases together may contain at most {max_nonzero:,} nonzero entries. "
            "Above that the score is 0.")


def without_size_limit(description: str) -> str:
    """The description minus a target's own "## Size limit" section, so a capped prompt states one limit, not two."""
    head, heading, rest = description.partition(SIZE_LIMIT_HEADING)
    if not heading:
        return description
    _, next_heading, tail = rest.partition("\n## ")
    return f"{head.rstrip()}\n{next_heading}{tail}" if next_heading else head.rstrip()


def with_cap_section(*, description: str, max_nonzero: int) -> str:
    """The description with the size limit section placed before its Scoring section (appended if it has none). A
    target's own size limit section is dropped: the cap replaces it."""
    head, heading, tail = without_size_limit(description).partition(SCORING_HEADING)
    if not heading:
        return f"{description.rstrip()}\n\n{cap_text(max_nonzero)}"
    return f"{head.rstrip()}\n\n{cap_text(max_nonzero)}\n{heading}{tail}"


def capped_metrics(*, metrics: Metrics, used: int, max_nonzero: int) -> Metrics:
    within_budget = used <= max_nonzero
    values = {**metrics.values,
              metrics.primary: metrics.primary_value if within_budget else 0.0,
              f"{metrics.primary}_ignoring_budget": metrics.primary_value,
              "nonzero_weights": float(used), "max_nonzero": float(max_nonzero),
              "within_budget": float(within_budget)}
    return Metrics(values=values, primary=metrics.primary)


@cache
def nonzero_capped_class(target_class: type[Target]) -> type[Target]:
    """Subclass of target_class taking a keyword `max_nonzero` on top of the class's own parameters."""

    def __init__(self, *, max_nonzero: int, **params):
        target_class.__init__(self, **params)
        self.max_nonzero = int(max_nonzero)

    def description(self) -> str:
        return with_cap_section(description=target_class.description(self), max_nonzero=self.max_nonzero)

    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        metrics = target_class.evaluate(self, weights=weights, split=split, seed=seed)
        return capped_metrics(metrics=metrics, used=nonzero_count(weights), max_nonzero=self.max_nonzero)

    return type(f"{target_class.__name__}NonzeroCapped", (target_class,),
                {"__init__": __init__, "description": description, "evaluate": evaluate,
                 "__module__": __name__, "__doc__": f"{target_class.__name__} under a nonzero weight cap."})
