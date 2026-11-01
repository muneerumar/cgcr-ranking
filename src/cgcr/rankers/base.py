"""common ranker/backend results with explicit provenance and usage."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankResult:
    permutation: list[int]
    raw_output: str
    parse_status: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    attempts: list[dict[str, Any]] = field(default_factory=list)
    backend_stats: dict[str, Any] = field(default_factory=dict)


class Ranker(ABC):
    @abstractmethod
    def rank(self, query: str, docs: list[str], seed: int) -> RankResult:
        """Return all zero-based document indices, best first."""


def validate_permutation(permutation, n):
    if len(permutation) != n or sorted(permutation) != list(range(n)):
        raise ValueError('Ranker must return each document index exactly once')
