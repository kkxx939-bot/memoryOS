"""运行时里的配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    root: str
    mode: str = "local"
    memory_extractor: Any | None = None
    memory_egress_policy: Any | None = None
    memory_document_max_bytes: int = 2 * 1024 * 1024
    memory_front_matter_max_bytes: int = 32 * 1024
    memory_front_matter_max_depth: int = 12
    memory_scan_stability_seconds: float = 1.0
    memory_scan_max_files: int = 10_000
    memory_mass_delete_threshold: int = 50
    embedding: Any | None = None
    vector_store: Any | None = None
    reranker: Any | None = None
    retrieval: dict[str, Any] | None = None
    retention: dict[str, Any] | None = None
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
        if self.retention is not None and not isinstance(self.retention, Mapping):
            raise ValueError("retention must be a mapping")
        for field_name in (
            "memory_document_max_bytes",
            "memory_front_matter_max_bytes",
            "memory_front_matter_max_depth",
            "memory_scan_max_files",
            "memory_mass_delete_threshold",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.memory_front_matter_max_bytes >= self.memory_document_max_bytes:
            raise ValueError("memory_front_matter_max_bytes must be smaller than memory_document_max_bytes")
        if self.memory_scan_stability_seconds < 0:
            raise ValueError("memory_scan_stability_seconds cannot be negative")

    @property
    def root_path(self) -> Path:
        raw = str(self.root)
        if not raw or any(marker in raw for marker in ("$", "${", "*", "?", "[", "]")):
            raise ValueError("root must be one explicit path without variables or glob syntax")
        return Path(raw).expanduser().resolve(strict=False)
