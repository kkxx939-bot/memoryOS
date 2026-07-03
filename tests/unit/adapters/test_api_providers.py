from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import patch

from memoryos.adapters.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleRerankProvider,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class APIProviderTest(unittest.TestCase):
    def test_chat_provider_calls_openai_compatible_endpoint(self) -> None:
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            seen["body"] = json.loads(request.data.decode("utf-8"))
            seen["authorization"] = request.headers.get("Authorization")
            return FakeHTTPResponse({"choices": [{"message": {"content": "{\"operations\": []}"}}]})

        with patch("memoryos.adapters.providers.openai_compatible.urllib.request.urlopen", fake_urlopen):
            provider = OpenAICompatibleChatProvider(
                model="memory-llm",
                base_url="https://api.example.test/v1",
                api_key="test-key",
                timeout=12,
            )
            result = provider.complete("extract memory")

        self.assertEqual(result, "{\"operations\": []}")
        self.assertEqual(seen["url"], "https://api.example.test/v1/chat/completions")
        self.assertEqual(seen["timeout"], 12)
        self.assertEqual(seen["authorization"], "Bearer test-key")
        self.assertEqual(seen["body"]["model"], "memory-llm")
        self.assertEqual(seen["body"]["messages"][0]["content"], "extract memory")

    def test_embedding_provider_parses_and_normalizes_embedding(self) -> None:
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse({"data": [{"embedding": [3, 4]}]})

        with patch("memoryos.adapters.providers.openai_compatible.urllib.request.urlopen", fake_urlopen):
            provider = OpenAICompatibleEmbeddingProvider(
                model="memory-embedding",
                base_url="http://localhost:8000/v1",
            )
            result = provider.embed("hot room")

        self.assertEqual(seen["url"], "http://localhost:8000/v1/embeddings")
        self.assertEqual(seen["body"]["model"], "memory-embedding")
        self.assertEqual(seen["body"]["input"], "hot room")
        self.assertAlmostEqual(result[0], 0.6)
        self.assertAlmostEqual(result[1], 0.8)

    def test_embedding_provider_supports_batch_embedding(self) -> None:
        seen = {}

        def fake_urlopen(request, timeout):
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse({"data": [{"embedding": [3, 4]}, {"embedding": [0, 5]}]})

        with patch("memoryos.adapters.providers.openai_compatible.urllib.request.urlopen", fake_urlopen):
            provider = OpenAICompatibleEmbeddingProvider(
                model="memory-embedding",
                base_url="http://localhost:8000/v1",
            )
            result = provider.embed_batch(["hot room", "tea"])

        self.assertEqual(seen["body"]["input"], ["hot room", "tea"])
        self.assertEqual(len(result), 2)

    def test_chat_provider_retries_transient_url_errors(self) -> None:
        calls = {"count": 0}

        def fake_urlopen(request, timeout):
            calls["count"] += 1
            if calls["count"] == 1:
                raise urllib.error.URLError("temporary network failure")
            return FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

        with (
            patch("memoryos.adapters.providers.openai_compatible.urllib.request.urlopen", fake_urlopen),
            patch("memoryos.adapters.providers.openai_compatible.time.sleep", lambda _: None),
        ):
            provider = OpenAICompatibleChatProvider(
                model="memory-llm",
                base_url="https://api.example.test/v1",
                retries=1,
                backoff_seconds=0,
            )
            result = provider.complete("extract memory")

        self.assertEqual(result, "ok")
        self.assertEqual(calls["count"], 2)

    def test_rerank_provider_parses_indexed_scores(self) -> None:
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.12},
                    ]
                }
            )

        with patch("memoryos.adapters.providers.openai_compatible.urllib.request.urlopen", fake_urlopen):
            provider = OpenAICompatibleRerankProvider(
                model="memory-rerank",
                base_url="https://api.example.test/v1",
                endpoint="/rerank",
                api_key="test-key",
            )
            result = provider.rerank("hot room", ["tea memory", "cooling memory"])

        self.assertEqual(seen["url"], "https://api.example.test/v1/rerank")
        self.assertEqual(seen["body"]["model"], "memory-rerank")
        self.assertEqual(seen["body"]["query"], "hot room")
        self.assertEqual(result, [0.12, 0.91])


if __name__ == "__main__":
    unittest.main()
