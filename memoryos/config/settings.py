from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    memory_root: Path
    default_user_id: str = "gulf"
    embedding_provider: str = "auto"
    rerank_provider: str = "auto"


def load_settings() -> Settings:
    return Settings(
        memory_root=Path(os.environ.get("MEMORYOS_ROOT", "./memory-root")),
        default_user_id=os.environ.get("MEMORYOS_USER", "gulf"),
        embedding_provider=os.environ.get("MEMORYOS_EMBEDDING_PROVIDER", "auto"),
        rerank_provider=os.environ.get("MEMORYOS_RERANK_PROVIDER", "auto"),
    )
