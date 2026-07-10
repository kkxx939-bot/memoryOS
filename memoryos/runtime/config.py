from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    root: str
    mode: str = "local"
    memory_extractor: Any | None = None
    embedding: Any | None = None
    vector_store: Any | None = None
    reranker: Any | None = None
    retrieval: dict[str, Any] | None = None
    worker: dict[str, Any] | None = None
    http: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"local", "server", "remote_client"}:
            raise ValueError(f"unsupported runtime mode: {self.mode}")

    @property
    def root_path(self) -> Path:
        return Path(self.root)
