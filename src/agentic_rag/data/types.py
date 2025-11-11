from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(slots=True)
class RawRecord:
    identifier: str
    title: str
    body: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    record_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
