from __future__ import annotations

import json
import tempfile
import unittest

from memoryos.action_policy.model import ActionPolicy
from memoryos.api.http import handle
from memoryos.api.mcp import MemoryOSMCPServer
from memoryos.api.sdk import MemoryOSClient
from memoryos.behavior.extraction import BehaviorExtractor
from memoryos.behavior.model import Observation
from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.store import FileSystemSourceStore, InMemoryIndexStore, InMemoryQueueStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.extraction import LLMMemoryExtractor, RuleMemoryExtractor
from memoryos.operations.commit import OperationCommitter, RedoLog
from memoryos.operations.model import ContextOperation, OperationAction
from memoryos.prediction.model import PredictionRequest
from memoryos.workers.embedding_worker import EmbeddingWorker
from memoryos.workers.recovery_worker import RecoveryWorker
from memoryos.workers.semantic_worker import SemanticWorker
from memoryos.workers.session_commit_worker import SessionCommitWorker


class StaticProvider:
    provider_name = "static"
    model = "static-test"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, request):  # noqa: ANN001
        return self.text

    def health_check(self) -> dict:
        return {"ok": True}


class FinalPipelineComponentsTest(unittest.TestCase):
    def test_rule_and_llm_extractors_emit_context_operations_with_pending_and_rejected(self) -> None:
        archive = SessionArchive(
            user_id="gulf",
            session_id="s1",
            archive_uri="memoryos://user/gulf/sessions/history/s1",
            messages=[{"content": "记住：以后别自动开空调"}],
        )
        rule_result = RuleMemoryExtractor().extract(archive)
        self.assertEqual(rule_result.accepted[0].context_type.value, "memory")

        payload = {
            "operations": [
                {"action": "update", "title": "needs target", "text": "x", "confidence": 0.8},
                {"action": "add", "title": "bad confidence", "text": "x", "confidence": "bad"},
                {"action": "add", "title": "secret", "text": "password=123", "confidence": 0.9},
            ]
        }
        result = LLMMemoryExtractor(StaticProvider(json.dumps(payload))).parse_response(
            json.dumps(payload),
            user_id="gulf",
            session_id="s1",
        )
        self.assertEqual(len(result.pending), 2)
        self.assertEqual(len(result.rejected), 1)
        bad = LLMMemoryExtractor(StaticProvider("bad")).parse_response("bad", user_id="gulf")
        self.assertEqual(len(bad.rejected), 1)

    def test_behavior_extractor_api_and_mcp_use_prediction_engine_without_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observation = Observation(user_id="gulf", raw_text="hot room", location="home", environment={"temperature": 30})
            case = BehaviorExtractor().extract_case(observation, selected_action="turn_on_ac")
            self.assertEqual(case.scene_key, observation.scene_key)
            policy = ActionPolicy(
                user_id="gulf",
                scene_key=observation.scene_key,
                action="turn_on_ac",
                memory_anchor_uri="memoryos://user/gulf/memories/anchors/hot",
                auto_execute_allowed=True,
                q_value=0.95,
                confidence=0.95,
            )
            client = MemoryOSClient(tmp, InMemoryIndexStore())
            request = PredictionRequest(
                user_id="gulf",
                episode_id="ep",
                observation=observation,
                available_actions=["turn_on_ac"],
                request_id="r1",
            )
            result = client.predict(request, [policy])
            self.assertEqual(result.memory_operations, [])
            http_result = handle("POST /predict", client, {"request": request.__dict__, "policies": [policy.to_dict()]})
            self.assertEqual(http_result["memory_operations"], [])
            mcp_result = MemoryOSMCPServer(client).call_tool("memoryos_predict", {"request": request.__dict__, "policies": [policy.to_dict()]})
            self.assertEqual(mcp_result["episode_id"], "ep")

    def test_workers_process_semantic_embedding_session_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            queue = InMemoryQueueStore()
            obj_uri = "memoryos://user/gulf/memories/anchors/hot"
            from memoryos.contextdb.model import ContextObject, ContextType
            from memoryos.contextdb.store import QueueJob

            obj = ContextObject(uri=obj_uri, context_type=ContextType.MEMORY, title="Hot", owner_user_id="gulf")
            source.write_object(obj, content="hot weather")
            queue.enqueue(QueueJob(job_id="semantic1", queue_name="semantic", action="refresh", target_uri=obj_uri))
            self.assertEqual(SemanticWorker(source, queue).process_pending()["processed"], ["semantic1"])
            queue.enqueue(QueueJob(job_id="embedding1", queue_name="embedding", action="embed", target_uri=obj_uri))
            vector = InMemoryVectorStore()
            self.assertEqual(EmbeddingWorker(source, queue, vector).process_pending()["processed"], ["embedding1"])
            archive = SessionArchive(user_id="gulf", session_id="s1", archive_uri="memoryos://user/gulf/sessions/history/s1")
            service = SessionCommitService(SessionArchiveStore(tmp), queue)
            self.assertTrue(SessionCommitWorker(service).process_archive(archive)["done"])

            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, tmp)
            op = ContextOperation(
                user_id="gulf",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj_uri,
                payload={"context_object": obj.to_dict(), "content": "hot weather"},
            )
            RedoLog(tmp).begin(op)
            recovered = RecoveryWorker(
                __import__("memoryos.contextdb.transaction", fromlist=["RecoveryService"]).RecoveryService(RedoLog(tmp), committer)
            ).process_pending("gulf")
            self.assertEqual(recovered["recovered_count"], 1)


if __name__ == "__main__":
    unittest.main()
