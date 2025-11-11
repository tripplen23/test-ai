from __future__ import annotations

import abc
from typing import Iterable, Sequence

from ..retrieval import Query, RetrievedChunk


class Metric(abc.ABC):
    name: str

    @abc.abstractmethod
    def compute(self, *, query: Query, retrieved: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
        """Return the metric value for a single query."""


class MetricSuite:
    def __init__(self, metrics: Sequence[Metric]):
        self._metrics = metrics

    def evaluate(
        self,
        *,
        query: Query,
        retrieved: Sequence[RetrievedChunk],
        relevant: Iterable[str],
    ) -> dict[str, float]:
        return {metric.name: metric.compute(query=query, retrieved=retrieved, relevant=relevant) for metric in self._metrics}
