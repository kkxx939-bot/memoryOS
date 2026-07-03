from __future__ import annotations

import time
from dataclasses import dataclass

from memoryos.adapters.providers.openai_compatible.http_client import post_json
from memoryos.ports.providers.provider_errors import ProviderBadResponse
from memoryos.ports.providers.rerank_provider import RerankDocument, RerankHit


@dataclass
class OpenAICompatibleRerankProvider:
    model: str
    base_url: str
    endpoint: str = "/rerank"
    api_key: str | None = None
    timeout: float = 60.0
    retries: int = 2
    backoff_seconds: float = 0.5
    provider_name: str = "openai_compatible"

    def rerank(self, query: str, documents: list[str] | list[RerankDocument]) -> list[float] | list[RerankHit] | None:
        if not documents:
            return []
        structured = all(isinstance(document, RerankDocument) for document in documents)
        texts = [document.text if isinstance(document, RerankDocument) else str(document) for document in documents]
        ids = [document.id if isinstance(document, RerankDocument) else str(index) for index, document in enumerate(documents)]
        started = time.monotonic()
        response = post_json(
            base_url=self.base_url,
            endpoint=self.endpoint,
            payload={"model": self.model, "query": query, "documents": texts},
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
            provider=self.provider_name,
        )
        scores = self._scores_from_response(response, len(texts))
        if scores is None:
            return None
        if not structured:
            return scores
        latency_ms = int((time.monotonic() - started) * 1000)
        return [
            RerankHit(
                id=document_id,
                score=score,
                model=self.model,
                provider=self.provider_name,
                metadata={"latency_ms": latency_ms},
            )
            for document_id, score in zip(ids, scores, strict=True)
        ]

    def health_check(self) -> dict:
        return {"status": "configured", "provider": self.provider_name, "model": self.model}

    def _scores_from_response(self, response: dict, expected: int) -> list[float] | None:
        if isinstance(response.get("scores"), list):
            scores = response["scores"]
            return [float(score) for score in scores] if len(scores) == expected else None
        results = response.get("results")
        if not isinstance(results, list) or len(results) != expected:
            return None
        scores = [0.0] * expected
        for item in results:
            if not isinstance(item, dict):
                raise ProviderBadResponse("Rerank result item is not an object", provider=self.provider_name)
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= expected:
                return None
            raw_score = item.get("relevance_score", item.get("score", 0.0))
            scores[index] = float(raw_score if raw_score is not None else 0.0)
        return scores
