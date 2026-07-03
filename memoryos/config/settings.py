from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    memory_root: Path
    default_user_id: str = "gulf"
    fact_store_backend: str = "sqlite"
    vector_index_backend: str = "sqlite"
    outbox_backend: str = "jsonl"
    chat_provider: str = "auto"
    embedding_provider: str = "auto"
    rerank_provider: str = "auto"
    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_batch_size: int = 32
    normalize_embeddings: bool = False
    max_rerank_candidates: int = 50
    max_llm_input_tokens: int = 8000
    daily_user_budget: float = 0.0
    provider_timeout_seconds: float = 60.0
    provider_retries: int = 2
    provider_backoff_seconds: float = 0.5
    worker_internal_token: str = ""


def load_settings() -> Settings:
    return Settings(
        memory_root=Path(os.environ.get("MEMORYOS_ROOT", "./memory-root")),
        default_user_id=os.environ.get("MEMORYOS_USER", "gulf"),
        fact_store_backend=os.environ.get("MEMORYOS_FACT_STORE_BACKEND", "sqlite"),
        vector_index_backend=os.environ.get("MEMORYOS_VECTOR_INDEX_BACKEND", "sqlite"),
        outbox_backend=os.environ.get("MEMORYOS_OUTBOX_BACKEND", "jsonl"),
        chat_provider=os.environ.get("MEMORYOS_CHAT_PROVIDER", os.environ.get("MEMORYOS_LLM_PROVIDER", "auto")),
        embedding_provider=os.environ.get("MEMORYOS_EMBEDDING_PROVIDER", "auto"),
        rerank_provider=os.environ.get("MEMORYOS_RERANK_PROVIDER", "auto"),
        embedding_model=os.environ.get("MEMORYOS_EMBEDDING_MODEL", ""),
        embedding_dimension=int(os.environ.get("MEMORYOS_EMBEDDING_DIMENSION", "0")),
        embedding_batch_size=int(os.environ.get("MEMORYOS_EMBEDDING_BATCH_SIZE", "32")),
        normalize_embeddings=os.environ.get("MEMORYOS_NORMALIZE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"},
        max_rerank_candidates=int(os.environ.get("MEMORYOS_MAX_RERANK_CANDIDATES", "50")),
        max_llm_input_tokens=int(os.environ.get("MEMORYOS_MAX_LLM_INPUT_TOKENS", "8000")),
        daily_user_budget=float(os.environ.get("MEMORYOS_DAILY_USER_BUDGET", "0")),
        provider_timeout_seconds=float(os.environ.get("MEMORYOS_PROVIDER_TIMEOUT_SECONDS", "60")),
        provider_retries=int(os.environ.get("MEMORYOS_PROVIDER_RETRIES", "2")),
        provider_backoff_seconds=float(os.environ.get("MEMORYOS_PROVIDER_BACKOFF_SECONDS", "0.5")),
        worker_internal_token=os.environ.get("MEMORYOS_WORKER_INTERNAL_TOKEN", ""),
    )
