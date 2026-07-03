from __future__ import annotations

import unittest

from memoryos.config.provider_registry import ProviderRegistry
from memoryos.config.settings import Settings
from memoryos.ports.providers.chat_provider import ChatMessage, ChatRequest, ModelResponse
from memoryos.ports.providers.embedding_provider import HashingEmbeddingProvider
from memoryos.ports.providers.rerank_provider import RerankDocument, RerankHit, rerank_with_fallback


class FakeRerankProvider:
    provider_name = "fake"
    model = "fake-rerank"

    def rerank(self, query: str, documents: list[str] | list[RerankDocument]) -> list[RerankHit]:
        return [RerankHit(id=str(index), score=0.9 - index * 0.1, model=self.model, provider=self.provider_name) for index, _ in enumerate(documents)]

    def health_check(self) -> dict:
        return {"status": "ok"}


class ProviderContractsTest(unittest.TestCase):
    def test_embedding_result_contains_metadata_and_hash(self) -> None:
        result = HashingEmbeddingProvider(dimensions=16).embed_text("hello memory")

        self.assertEqual(result.provider, "local_hashing")
        self.assertEqual(result.dimension, 16)
        self.assertTrue(result.content_hash)
        self.assertTrue(result.normalized)

    def test_rerank_fallback_accepts_structured_hits(self) -> None:
        scores = rerank_with_fallback(
            FakeRerankProvider(),
            "query",
            ["a", "b"],
            [0.1, 0.1],
        )

        self.assertEqual(scores, [0.9, 0.8])

    def test_provider_registry_returns_local_embedding_without_env(self) -> None:
        registry = ProviderRegistry(Settings(memory_root=__import__("pathlib").Path("."), embedding_provider="local"))

        self.assertEqual(registry.get_embedding_provider().health_check()["provider"], "local_hashing")

    def test_model_response_contract(self) -> None:
        request = ChatRequest(messages=[ChatMessage(role="user", content="hi")], prompt_version="v1")
        response = ModelResponse(text="ok", model="m", provider="p", prompt_version=request.prompt_version)

        self.assertEqual(response.to_dict()["prompt_version"], "v1")


if __name__ == "__main__":
    unittest.main()
