"""Target: a model with weights plus a quality metric."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

Weights = dict[str, np.ndarray]
Split = Literal["dev", "hidden"]


@dataclass(frozen=True)
class Metrics:
    values: dict[str, float]
    primary: str

    @property
    def primary_value(self) -> float:
        return self.values[self.primary]


class Target(ABC):
    id: ClassVar[str]
    domain: ClassVar[str]

    @abstractmethod
    def description(self) -> str:
        """Spec text shown to the editor. Must not reveal scoring internals."""

    @abstractmethod
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Name and shape of every tensor in the model."""

    @abstractmethod
    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        """Score weights. The hidden split is only ever used by the validator."""

    @property
    def editable_tensors(self) -> list[str]:
        return list(self.tensor_shapes())

    def load_weights(self) -> Weights:
        raise NotImplementedError(f"target {self.id} has no trained weights")

    def init_weights(self) -> Weights:
        return {name: np.zeros(shape, dtype=np.float32) for name, shape in self.tensor_shapes().items()}

    def diagnosis_question(self) -> str:
        raise NotImplementedError(f"target {self.id} does not support the diagnose track")

    def diagnosis_answer(self) -> str:
        raise NotImplementedError(f"target {self.id} does not support the diagnose track")

    def accepted_answers(self) -> list[str]:
        """Names the task grader accepts (wsp.harbor.answer_grader). Override when several spellings are right."""
        return [self.diagnosis_answer()]

    def diagnosis_matches(self, given: str) -> bool:
        """Whether an `ANSWER:` value is correct. Default: equal to diagnosis_answer up to case and whitespace."""
        return " ".join(given.lower().split()) == " ".join(self.diagnosis_answer().lower().split())

    def edit_goal(self) -> str:
        raise NotImplementedError(f"target {self.id} does not support the edit track")

    # Harbor export hooks (wsp/harbor/export.py); the defaults keep the generic write task.
    answer_file: ClassVar[str] = "weights.npz"  # "edits.txt": the agent writes the edits language as text
    reward_floor: ClassVar[float] = 0.0  # reward = clip((primary - floor) / (reference - floor), 0, 1)

    def harbor_instruction(self) -> str | None:
        """Full instruction.md text, or None to build it from description() and the track contract."""
        return None

    def harbor_agent_files(self) -> dict[str, bytes]:
        """Extra files for the agent sandbox, relative to /app (copied into environment/)."""
        return {}

    def harbor_verifier_files(self, *, hidden_seed: int) -> dict[str, bytes]:
        """Extra files for the verifier, relative to /tests (copied into tests/); WSP_DATA_DIR points there."""
        return {}

    def harbor_verifier_extras(self) -> tuple[str, ...]:
        """Optional dependency extras (pyproject [project.optional-dependencies]) the verifier image installs, not the agent's."""
        return ()

    def harbor_oracle_weights(self) -> Weights | None:
        """Known good weights for solution/solve.sh, or None to use wsp.harbor.oracles."""
        return None
