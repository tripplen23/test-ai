from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    schema: Mapping[str, Any]
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(slots=True)
class PlanStep:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: Sequence[str] = field(default_factory=tuple)
