"""A task: a fixed network envelope (tensor shapes), a spec, and a scorer on freshly generated inputs."""

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
        """The task and network spec shown to the model."""

    @abstractmethod
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Name and shape of every tensor the reply sets."""

    @abstractmethod
    def evaluate(self, *, weights: Weights, split: Split, seed: int) -> Metrics:
        """Score weights on cases generated from (split, seed)."""

    def init_weights(self) -> Weights:
        return {name: np.zeros(shape, dtype=np.float32) for name, shape in self.tensor_shapes().items()}
