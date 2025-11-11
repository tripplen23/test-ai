from __future__ import annotations

import abc
from typing import Sequence

from .types import Message, PlanStep


class BaseAgentController(abc.ABC):
    """High-level orchestration contract."""

    @abc.abstractmethod
    def plan(self, history: Sequence[Message]) -> Sequence[PlanStep]:
        """Produce a plan (tool calls, retrieval steps, etc.) given the dialogue history."""

    @abc.abstractmethod
    def run(self, history: Sequence[Message]) -> Message:
        """Execute the plan and return the assistant's next message."""

    def serve(self) -> None:
        """Optional long-running server/CLI entrypoint."""
        raise NotImplementedError("Implement `serve` for long-running agents.")
