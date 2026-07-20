from __future__ import annotations

import tempfile
import unittest

from behavior.core import BehaviorExtractor, Observation
from infrastructure.context.maintenance.embedding_worker import EmbeddingWorker
from infrastructure.context.maintenance.semantic_worker import SemanticWorker
from infrastructure.store.operation.redo import RedoLog
from memory.commit.session_commit import SessionCommitService
from memory.worker.session_commit import SessionCommitWorker
from openApi.http import handle
from openApi.mcp import MemoryOSMCPServer
from openApi.mcp.config import MCPServerConfig
from openApi.sdk import MemoryOSClient
from policy.action_policy.decision import PredictionRequest
from policy.action_policy.model import ActionPolicy
from pre.connect import ConnectMetadata
from pre.session import SessionArchive
from runtime.recovery.transaction_worker import RecoveryWorker
from tests.support.embedding import DeterministicEmbeddingProvider
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore, InMemoryQueueStore, InMemoryVectorStore
from tests.support.session_archive import build_session_archive_store
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.commit.recovery import RecoveryService
from transaction.model import ContextOperation, OperationAction


class FinalPipelineComponentsTest(unittest.TestCase):
    def test_behavior_extractor_api_and_mcp_use_prediction_engine_without_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observation = Observation(
                user_id="gulf", raw_text="hot room", location="home", environment={"temperature": 30}
            )
            case = BehaviorExtractor().extract_case(observation, selected_action="turn_on_ac")
            self.assertEqual(case.scene_key, observation.scene_key)
            policy = ActionPolicy(
                user_id="gulf",
                scene_key=observation.scene_key,
                action="turn_on_ac",
                support_anchor_uri="memoryos://user/gulf/support/behavior/hot",
                auto_execute_allowed=True,
                q_value=0.95,
                confidence=0.95,
            )
            client = MemoryOSClient(tmp)
            request = PredictionRequest(
                user_id="gulf",
                episode_id="ep",
                observation=observation,
                available_actions=["turn_on_ac"],
                request_id="r1",
                connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
            )
            result = client.predict(request, [policy])
            self.assertNotIn("memory_operations", result.to_dict())
            http_result = handle("POST /predict", client, {"request": request.__dict__, "policies": [policy.to_dict()]})
            self.assertNotIn("memory_operations", http_result)
            mcp_result = MemoryOSMCPServer(client).call_tool(
                "memoryos_predict", {"request": request.__dict__, "policies": [policy.to_dict()]}
            )
            self.assertEqual(mcp_result["error"]["code"], "PERMISSION_DENIED")
            enabled_mcp = MemoryOSMCPServer(
                client,
                MCPServerConfig(
                    root=tmp,
                    user_id="gulf",
                    adapter_id="codex",
                    agent_name="codex",
                    enable_action_tools=True,
                ),
            )
            action_request = {
                **request.__dict__,
                "connect_metadata": ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
            }
            allowed_mcp_result = enabled_mcp.call_tool(
                "memoryos_predict", {"request": action_request, "policies": [policy.to_dict()]}
            )
            self.assertEqual(allowed_mcp_result["prediction"]["episode_id"], "ep")

    def test_workers_process_semantic_embedding_session_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            queue = InMemoryQueueStore()
            obj_uri = "memoryos://user/gulf/resources/hot-weather"
            from infrastructure.store.contracts import QueueJob
            from infrastructure.store.model.context import ContextObject, ContextType

            obj = ContextObject(uri=obj_uri, context_type=ContextType.RESOURCE, title="Hot", owner_user_id="gulf")
            source.write_object(obj, content="hot weather")
            queue.enqueue(QueueJob(job_id="semantic1", queue_name="semantic", action="refresh", target_uri=obj_uri))
            self.assertEqual(SemanticWorker(source, queue).process_pending()["processed"], ["semantic1"])
            queue.enqueue(QueueJob(job_id="embedding1", queue_name="embedding", action="embed", target_uri=obj_uri))
            vector = InMemoryVectorStore()
            self.assertEqual(
                EmbeddingWorker(
                    source,
                    queue,
                    vector,
                    DeterministicEmbeddingProvider(),
                ).process_pending()["processed"],
                ["embedding1"],
            )
            archive = SessionArchive(
                user_id="gulf", session_id="s1", archive_uri="memoryos://user/gulf/sessions/history/s1"
            )
            service = SessionCommitService(build_session_archive_store(tmp), queue)
            self.assertTrue(SessionCommitWorker(service).process_archive(archive)["done"])

            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, tmp)
            op = ContextOperation(
                user_id="gulf",
                context_type=ContextType.RESOURCE,
                action=OperationAction.ADD,
                target_uri=obj_uri,
                payload={"context_object": obj.to_dict(), "content": "hot weather"},
            )
            RedoLog(tmp).begin(op)
            recovered = RecoveryWorker(RecoveryService(RedoLog(tmp), committer)).process_pending("gulf")
            self.assertEqual(recovered["recovered_count"], 1)


if __name__ == "__main__":
    unittest.main()
