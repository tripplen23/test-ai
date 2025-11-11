from __future__ import annotations

import abc
from typing import Iterable

from ..retrieval import Query


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def iter_queries(self) -> Iterable[Query]:
        """Yield evaluation queries."""

    @abc.abstractmethod
    def evaluate(self) -> None:
        """Run the evaluation suite."""
