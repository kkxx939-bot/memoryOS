"""运行时里的配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    root: str
    mode: str = "local"
    memory_extractor: Any | None = None
    memory_egress_policy: Any | None = None
    memory_aliases: dict[str, dict[str, str]] | None = None
    embedding: Any | None = None
    vector_store: Any | None = None
    reranker: Any | None = None
    retrieval: dict[str, Any] | None = None
    worker: dict[str, Any] | None = None
    http: dict[str, Any] | None = None
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        if self.mode not in {"local", "server", "remote_client"}:
            raise ValueError(f"unsupported runtime mode: {self.mode}")
        if (
            not isinstance(self.tenant_id, str)
            or not self.tenant_id.strip()
            or self.tenant_id in {".", ".."}
            or "/" in self.tenant_id
            or "\\" in self.tenant_id
        ):
            raise ValueError("tenant_id must be one safe non-empty path segment")

    @property
    def root_path(self) -> Path:
        return Path(self.root)
