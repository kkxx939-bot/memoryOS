from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.domain.memory.memory_item import MemoryItem
from memoryos.interfaces.api.app import handle
from memoryos.interfaces.api.request_context import APIRequestContext
from memoryos.ports.providers.embedding_provider import EmbeddingResult, content_hash
from memoryos.services.memory.extractor import JsonLLMMemoryExtractor, MemoryOperation
from memoryos.services.memory.markdown import render_memory_markdown
from memoryos.services.memory.update_service import MemoryUpdateContext, MemoryUpdateService
from memoryos.usecases.episode.process_observation import EpisodeProcessor


class StaticChatProvider:
    provider_name = "static"
    model = "static-chat"

    def __init__(self, response: object) -> None:
        self.response = response

    def complete(self, request) -> str:  # noqa: ANN001
        return json.dumps(self.response, ensure_ascii=False)

    def health_check(self) -> dict[str, object]:
        return {"ok": True}


class BadDimensionEmbeddingProvider:
    provider_name = "bad"
    model = "bad-embedding"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=self.embed(text),
            provider=self.provider_name,
            model=self.model,
            dimension=2,
            content_hash=content_hash(text),
        )

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]

    def health_check(self) -> dict[str, object]:
        return {"ok": True}


class DraftExtractor:
    def extract(self, messages: list[dict[str, str]]) -> list[MemoryOperation]:
        return [
            MemoryOperation(
                action="add",
                memory_type="event",
                title="rawhot draft",
                text="rawhot123 should not be used as historical memory in this episode.",
                tags=["event"],
            )
        ]


class ProductionSealTest(unittest.TestCase):
    def test_update_supersedes_old_preference_and_delete_resolves_without_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            service = MemoryUpdateService(store)
            context = MemoryUpdateContext(user_id="gulf", source="test", diff_id="seal", explicit_user_intent=True)
            service.apply(
                [
                    MemoryOperation(
                        action="add",
                        memory_type="preference",
                        title="饮品偏好",
                        text="用户喜欢咖啡。",
                        tags=["drink"],
                    )
                ],
                context,
            )
            diff = service.apply(
                [
                    MemoryOperation(
                        action="update",
                        memory_type="preference",
                        title="饮品偏好更新",
                        text="用户现在不喜欢咖啡了，改喝茶。",
                        tags=["drink"],
                    )
                ],
                context,
            )
            self.assertEqual(diff["summary"]["total_pending"], 0)
            self.assertEqual(diff["summary"]["total_updates"], 1)
            active = store.hybrid_search("喝茶", user_id="gulf", memory_type="preference")
            self.assertEqual(len(active), 1)
            self.assertIn("茶", active[0]["content"])
            self.assertEqual(active[0]["supersedes"][0], diff["operations"]["updates"][0]["uri"])
            obsolete = store.resolve_memory(diff["operations"]["updates"][0]["uri"], "gulf")
            self.assertEqual(obsolete["status"], "obsolete")

            delete_diff = service.apply(
                [
                    MemoryOperation(
                        action="delete",
                        memory_type="preference",
                        title="忘记茶偏好",
                        text="忘记我喜欢茶这件事。",
                        tags=["drink"],
                    )
                ],
                context,
            )
            self.assertEqual(delete_diff["summary"]["total_deletes"], 1)
            self.assertEqual(store.hybrid_search("茶", user_id="gulf", memory_type="preference"), [])

    def test_llm_extractor_keeps_valid_ops_and_reports_bad_or_pending_ops(self) -> None:
        extractor = JsonLLMMemoryExtractor(
            StaticChatProvider(
                [
                    {"action": "add", "memory_type": "preference", "title": "茶", "text": "用户喜欢茶。", "tags": [], "confidence": 0.8},
                    {"action": "add", "memory_type": "preference", "title": "坏", "text": "bad", "tags": [], "confidence": "high"},
                    {"action": "add", "memory_type": "event", "title": "低置信", "text": "也许用户今天很忙。", "tags": [], "confidence": 0.2},
                ]
            )
        )
        operations = extractor.extract([{"role": "user", "text": "demo"}])
        self.assertEqual(len(operations), 1)
        self.assertEqual(len(extractor.last_result.rejected), 1)
        self.assertEqual(len(extractor.last_result.pending), 1)

    def test_before_prediction_does_not_commit_inferred_memory_into_current_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            result = EpisodeProcessor(store, extractor=DraftExtractor()).process(
                user_id="gulf",
                episode_id="ep1",
                scene="rawhot123 current observation",
                memory_commit_timing="before_prediction",
            )
            self.assertTrue(result["isolated_current_episode_inferred_memory"])
            self.assertEqual(result["memory_diff"]["summary"]["total_adds"], 0)
            self.assertEqual(len(result["pending_memory_operations"]), 1)
            self.assertEqual(store.search("rawhot123", user_id="gulf"), [])

    def test_chinese_query_retrieves_chinese_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            store.add_memory(
                MemoryItem(
                    user_id="gulf",
                    memory_type="preference",
                    title="饮品偏好",
                    text="用户喜欢咖啡，下午通常会喝咖啡。",
                    tags=["饮品"],
                )
            )
            rows = store.search("喜欢咖啡", user_id="gulf")
            self.assertEqual(len(rows), 1)

    def test_sensitive_memory_goes_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            diff = MemoryUpdateService(store).apply(
                [
                    MemoryOperation(
                        action="add",
                        memory_type="preference",
                        title="手机号",
                        text="用户手机号是 13812345678。",
                        tags=[],
                    )
                ],
                MemoryUpdateContext(user_id="gulf", source="test", diff_id="sensitive"),
            )
            self.assertEqual(diff["summary"]["total_pending"], 1)
            self.assertEqual(diff["summary"]["total_adds"], 0)

    def test_api_user_id_comes_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            response = handle(
                "GET /memory/search",
                store,
                {"user_id": "attacker", "query": "x"},
                context=APIRequestContext(user_id="gulf"),
            )
            self.assertEqual(response["results"], [])

    def test_embedding_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp), embedding_provider=BadDimensionEmbeddingProvider())
            store.init("gulf")
            with self.assertRaises(ValueError):
                store.add_memory(
                    MemoryItem(
                        user_id="gulf",
                        memory_type="event",
                        title="bad vector",
                        text="This should fail because embedding dimension is inconsistent.",
                    )
                )

    def test_redo_recovery_rebuilds_missing_index_from_markdown_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(user_id="gulf", memory_type="event", title="recovery fact", text="recoverable content")
            path = Path(tmp) / (item.path or "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_memory_markdown(item.metadata(), item.text), encoding="utf-8")
            store._append_operation_log("gulf", "redo-test", "pending", "add_memory", {"path": item.path})
            report = store.recover_user_operations("gulf")
            self.assertTrue(report["recovered"])
            self.assertEqual(len(store.search("recoverable", user_id="gulf")), 1)


if __name__ == "__main__":
    unittest.main()
