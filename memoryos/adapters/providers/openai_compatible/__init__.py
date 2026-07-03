from __future__ import annotations

import os

from memoryos.adapters.providers.openai_compatible.chat import OpenAICompatibleChatProvider
from memoryos.adapters.providers.openai_compatible.embedding import OpenAICompatibleEmbeddingProvider
from memoryos.adapters.providers.openai_compatible.rerank import OpenAICompatibleRerankProvider
from memoryos.ports.providers.provider_errors import ProviderError as APIProviderError
from memoryos.ports.providers.rerank_provider import RerankProvider


def build_chat_provider_from_env() -> OpenAICompatibleChatProvider | None:
    model = os.getenv("MEMORYOS_LLM_MODEL")
    if not model:
        return None
    return OpenAICompatibleChatProvider(
        model=model,
        base_url=os.getenv("MEMORYOS_LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("MEMORYOS_LLM_API_KEY"),
        timeout=float(os.getenv("MEMORYOS_LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("MEMORYOS_LLM_TEMPERATURE", "0")),
        retries=int(os.getenv("MEMORYOS_PROVIDER_RETRIES", "2")),
        backoff_seconds=float(os.getenv("MEMORYOS_PROVIDER_BACKOFF_SECONDS", "0.5")),
    )


def build_embedding_provider_from_env() -> OpenAICompatibleEmbeddingProvider | None:
    model = os.getenv("MEMORYOS_EMBEDDING_MODEL")
    if not model:
        return None
    return OpenAICompatibleEmbeddingProvider(
        model=model,
        base_url=os.getenv("MEMORYOS_EMBEDDING_BASE_URL", os.getenv("MEMORYOS_LLM_BASE_URL", "https://api.openai.com/v1")),
        api_key=os.getenv("MEMORYOS_EMBEDDING_API_KEY", os.getenv("MEMORYOS_LLM_API_KEY")),
        timeout=float(os.getenv("MEMORYOS_EMBEDDING_TIMEOUT", "60")),
        retries=int(os.getenv("MEMORYOS_PROVIDER_RETRIES", "2")),
        backoff_seconds=float(os.getenv("MEMORYOS_PROVIDER_BACKOFF_SECONDS", "0.5")),
        normalize_embeddings=os.getenv("MEMORYOS_NORMALIZE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"},
    )


def build_rerank_provider_from_env() -> RerankProvider | None:
    model = os.getenv("MEMORYOS_RERANK_MODEL")
    if not model:
        return None
    return OpenAICompatibleRerankProvider(
        model=model,
        base_url=os.getenv("MEMORYOS_RERANK_BASE_URL", os.getenv("MEMORYOS_LLM_BASE_URL", "https://api.openai.com/v1")),
        endpoint=os.getenv("MEMORYOS_RERANK_ENDPOINT", "/rerank"),
        api_key=os.getenv("MEMORYOS_RERANK_API_KEY", os.getenv("MEMORYOS_LLM_API_KEY")),
        timeout=float(os.getenv("MEMORYOS_RERANK_TIMEOUT", "60")),
        retries=int(os.getenv("MEMORYOS_PROVIDER_RETRIES", "2")),
        backoff_seconds=float(os.getenv("MEMORYOS_PROVIDER_BACKOFF_SECONDS", "0.5")),
    )


__all__ = [
    "APIProviderError",
    "OpenAICompatibleChatProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleRerankProvider",
    "build_chat_provider_from_env",
    "build_embedding_provider_from_env",
    "build_rerank_provider_from_env",
]
