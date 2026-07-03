from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class FTSRecord:
    namespace: str
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""


@dataclass(frozen=True)
class FTSHit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""
    namespace: str = ""


class FTSIndex(Protocol):
    backend_name: str

    def upsert(self, record: FTSRecord) -> None: ...

    def delete(self, *, namespace: str, id: str) -> None: ...

    def search(self, *, namespace: str, query: str, top_k: int, filters: dict | None = None) -> list[FTSHit]: ...

    def health_check(self) -> dict: ...
