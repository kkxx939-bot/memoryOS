from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memoryos.adapters.providers.openai_compatible import (
    build_chat_provider_from_env,
    build_embedding_provider_from_env,
    build_rerank_provider_from_env,
)
from memoryos.config.settings import Settings
from memoryos.ports.providers.chat_provider import ChatProvider
from memoryos.ports.providers.embedding_provider import EmbeddingProvider, HashingEmbeddingProvider
from memoryos.ports.providers.rerank_provider import RerankProvider


@dataclass
class ProviderRegistry:
    settings: Settings

    def get_chat_provider(self, name: str | None = None) -> ChatProvider | None:
        selected = name or self.settings.chat_provider
        if selected in {"auto", "openai_compatible"}:
            provider = build_chat_provider_from_env()
            if provider is not None:
                provider.timeout = self.settings.provider_timeout_seconds
                provider.retries = self.settings.provider_retries
                provider.backoff_seconds = self.settings.provider_backoff_seconds
            return provider
        if selected in {"none", "disabled", ""}:
            return None
        raise ValueError(f"Unknown chat provider: {selected}")

    def get_embedding_provider(self, name: str | None = None) -> EmbeddingProvider:
        selected = name or self.settings.embedding_provider
        if selected in {"auto", "openai_compatible"}:
            provider = build_embedding_provider_from_env()
            if provider is not None:
                provider.timeout = self.settings.provider_timeout_seconds
                provider.retries = self.settings.provider_retries
                provider.backoff_seconds = self.settings.provider_backoff_seconds
                provider.normalize_embeddings = self.settings.normalize_embeddings
                return provider
        if selected in {"auto", "local", "hashing"}:
            dimensions = self.settings.embedding_dimension or 128
            return HashingEmbeddingProvider(dimensions=dimensions)
        raise ValueError(f"Unknown embedding provider: {selected}")

    def get_rerank_provider(self, name: str | None = None) -> RerankProvider | None:
        selected = name or self.settings.rerank_provider
        if selected in {"auto", "openai_compatible"}:
            provider = build_rerank_provider_from_env()
            if provider is not None:
                concrete_provider: Any = provider
                concrete_provider.timeout = self.settings.provider_timeout_seconds
                concrete_provider.retries = self.settings.provider_retries
                concrete_provider.backoff_seconds = self.settings.provider_backoff_seconds
            return provider
        if selected in {"none", "disabled", ""}:
            return None
        raise ValueError(f"Unknown rerank provider: {selected}")
