from __future__ import annotations

from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.planners import ActionPolicyCommitPlanner, BehaviorCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store import FileSystemSourceStore, IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.source_store import LockStore, QueueStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_lock_store import SQLiteLockStore
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.prediction.pipeline.executor import ActionExecutor
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.skill.tool_registry import ToolRegistry


class MemoryOSClient:
    def __init__(
        self,
        root: str,
        index_store: IndexStore | None = None,
        source_store: SourceStore | None = None,
        relation_store: RelationStore | None = None,
        queue_store: QueueStore | None = None,
        lock_store: LockStore | None = None,
        tool_registry: ToolRegistry | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        hybrid_search: HybridSearch | None = None,
    ) -> None:
        self.root = root
        root_path = Path(root)
        self.source_store = source_store or FileSystemSourceStore(root_path)
        self.index_store = index_store or SQLiteIndexStore(root_path / "indexes" / "context.sqlite3")
        self.relation_store = relation_store or SQLiteRelationStore(root_path / "indexes" / "relations.sqlite3")
        self.queue_store = queue_store or SQLiteQueueStore(root_path / "queues" / "jobs.sqlite3")
        self.lock_store = lock_store or SQLiteLockStore(root_path / "system" / "locks.sqlite3")
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.hybrid_search = hybrid_search or (
            HybridSearch(self.index_store, vector_store=vector_store, embedding_provider=embedding_provider, source_store=self.source_store)
            if vector_store is not None and embedding_provider is not None
            else None
        )
        self.committer = OperationCommitter(
            self.source_store,
            self.index_store,
            root,
            lock_store=self.lock_store,
            relation_store=self.relation_store,
        )
        self.session_archive_store = SessionArchiveStore(root_path)
        self.session_commit_service = SessionCommitService(
            self.session_archive_store,
            self.queue_store,
            committer=self.committer,
            behavior_planner=BehaviorCommitPlanner(index_store=self.index_store, source_store=self.source_store),
            action_policy_planner=ActionPolicyCommitPlanner(index_store=self.index_store, source_store=self.source_store),
        )
        self.context_db = ContextDB(
            self.source_store,
            self.index_store,
            self.relation_store,
            queue_store=self.queue_store,
            session_commit_service=self.session_commit_service,
            committer=self.committer,
        )
        self.engine = PredictionEngine(
            self.index_store,
            PredictionLedger(root),
            source_store=self.source_store,
            relation_store=self.relation_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            hybrid_search=self.hybrid_search,
        )
        self.executor = ActionExecutor(tool_registry)

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        return self.engine.process(request, policies=policies)

    def process_observation(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy] | None = None,
        *,
        archive_session: bool = True,
        async_commit: bool = True,
    ) -> PredictionResult:
        result = self.engine.process(request, policies=policies)
        action_result = self.executor.execute(result.decision, result.action_context)
        if not archive_session:
            return result
        policy_uri = result.candidates[0].policy_uri if result.candidates else ""
        feedback = []
        if action_result.status in {"success", "failed", "blocked"} and policy_uri:
            feedback.append(
                action_result.to_feedback(
                    user_id=request.user_id,
                    episode_id=request.episode_id,
                    policy_uri=policy_uri,
                    scene_key=result.observation.scene_key,
                )
            )
        observation_payload = {
            **result.observation.__dict__,
            "episode_id": request.episode_id,
            "request_id": request.request_id or result.request_id,
            "scene_key": result.observation.scene_key,
        }
        archive = SessionArchive(
            user_id=request.user_id,
            session_id=request.episode_id,
            archive_uri=request.session_uri or f"memoryos://user/{request.user_id}/sessions/history/{request.episode_id}",
            observations=[observation_payload],
            predictions=[result.to_dict()],
            action_results=[
                {
                    "request_id": result.request_id,
                    "episode_id": result.episode_id,
                    "decision": result.decision.to_dict(),
                    "selected_action": result.decision.action,
                    "action_result": action_result.to_dict(),
                }
            ],
            feedback=feedback,
            used_contexts=[{"uri": uri} for uri in result.action_context.source_uris],
            used_skills=[
                {"uri": uri}
                for uri in result.action_context.source_uris
                if uri.startswith("memoryos://skills/")
            ],
        )
        self.context_db.commit_session(archive, async_commit=async_commit)
        return result
