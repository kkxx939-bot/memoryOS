from __future__ import annotations

import time
from dataclasses import dataclass

from memoryos.adapters.providers.openai_compatible.http_client import post_json
from memoryos.ports.providers.embedding_provider import EmbeddingResult, content_hash, normalize
from memoryos.ports.providers.provider_errors import ProviderBadResponse


@dataclass
class OpenAICompatibleEmbeddingProvider:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout: float = 60.0
    retries: int = 2
    backoff_seconds: float = 0.5
    normalize_embeddings: bool = False
    provider_name: str = "openai_compatible"
    dimension: int = 0

    def embed(self, text: str) -> list[float]:
        return self.embed_text(text).vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [result.vector for result in self.embed_texts(texts)]

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        started = time.monotonic()
        response = post_json(
            base_url=self.base_url,
            endpoint="/embeddings",
            payload={"model": self.model, "input": texts[0] if len(texts) == 1 else texts},
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
            provider=self.provider_name,
        )
        data = response.get("data") or []
        if not data:
            raise ProviderBadResponse("Embedding response has no data", provider=self.provider_name)
        vectors = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise ProviderBadResponse("Embedding response has no embedding vector", provider=self.provider_name)
            vector = [float(value) for value in embedding]
            vectors.append(normalize(vector) if self.normalize_embeddings else vector)
        if len(vectors) != len(texts):
            raise ProviderBadResponse("Embedding response count does not match input count", provider=self.provider_name)
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
        results = []
        for text, vector in zip(texts, vectors, strict=True):
            results.append(
                EmbeddingResult(
                    vector=vector,
                    model=self.model,
                    provider=self.provider_name,
                    dimension=len(vector),
                    content_hash=content_hash(text),
                    latency_ms=latency_ms,
                    normalized=self.normalize_embeddings,
                    usage=usage,
                )
            )
        if results:
            self.dimension = results[0].dimension
        return results

    def health_check(self) -> dict:
        return {"status": "configured", "provider": self.provider_name, "model": self.model, "dimension": self.dimension}
