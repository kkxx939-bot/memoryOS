from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from memoryos.infrastructure.providers.embedding_provider import normalize
from memoryos.infrastructure.providers.rerank_provider import RerankProvider


class APIProviderError(RuntimeError):
    pass


def _post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise APIProviderError(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise APIProviderError(f"API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise APIProviderError(f"API response is not valid JSON: {exc}") from exc


@dataclass
class OpenAICompatibleChatProvider:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout: float = 60.0
    temperature: float = 0.0

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
        }
        response = _post_json(
            base_url=self.base_url,
            endpoint="/chat/completions",
            payload=payload,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        choices = response.get("choices") or []
        if not choices:
            raise APIProviderError("Chat completion response has no choices")
        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content") or first.get("text")
        if not isinstance(content, str) or not content.strip():
            raise APIProviderError("Chat completion response has no text content")
        return content


@dataclass
class OpenAICompatibleEmbeddingProvider:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout: float = 60.0

    def embed(self, text: str) -> list[float]:
        response = _post_json(
            base_url=self.base_url,
            endpoint="/embeddings",
            payload={
                "model": self.model,
                "input": text,
            },
            api_key=self.api_key,
            timeout=self.timeout,
        )
        data = response.get("data") or []
        if not data:
            raise APIProviderError("Embedding response has no data")
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise APIProviderError("Embedding response has no embedding vector")
        return normalize([float(value) for value in embedding])


@dataclass
class OpenAICompatibleRerankProvider:
    model: str
    base_url: str
    endpoint: str = "/rerank"
    api_key: str | None = None
    timeout: float = 60.0

    def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        if not documents:
            return []
        response = _post_json(
            base_url=self.base_url,
            endpoint=self.endpoint,
            payload={
                "model": self.model,
                "query": query,
                "documents": documents,
            },
            api_key=self.api_key,
            timeout=self.timeout,
        )
        if isinstance(response.get("scores"), list):
            scores = response["scores"]
            return [float(score) for score in scores] if len(scores) == len(documents) else None
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            return None
        scores = [0.0] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                return None
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                return None
            scores[index] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores


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
    )


def build_embedding_provider_from_env() -> OpenAICompatibleEmbeddingProvider | None:
    model = os.getenv("MEMORYOS_EMBEDDING_MODEL")
    if not model:
        return None
    return OpenAICompatibleEmbeddingProvider(
        model=model,
        base_url=os.getenv(
            "MEMORYOS_EMBEDDING_BASE_URL",
            os.getenv("MEMORYOS_LLM_BASE_URL", "https://api.openai.com/v1"),
        ),
        api_key=os.getenv("MEMORYOS_EMBEDDING_API_KEY", os.getenv("MEMORYOS_LLM_API_KEY")),
        timeout=float(os.getenv("MEMORYOS_EMBEDDING_TIMEOUT", "60")),
    )


def build_rerank_provider_from_env() -> RerankProvider | None:
    model = os.getenv("MEMORYOS_RERANK_MODEL")
    if not model:
        return None
    return OpenAICompatibleRerankProvider(
        model=model,
        base_url=os.getenv(
            "MEMORYOS_RERANK_BASE_URL",
            os.getenv("MEMORYOS_LLM_BASE_URL", "https://api.openai.com/v1"),
        ),
        endpoint=os.getenv("MEMORYOS_RERANK_ENDPOINT", "/rerank"),
        api_key=os.getenv("MEMORYOS_RERANK_API_KEY", os.getenv("MEMORYOS_LLM_API_KEY")),
        timeout=float(os.getenv("MEMORYOS_RERANK_TIMEOUT", "60")),
    )
