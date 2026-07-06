from __future__ import annotations

from typing import Any

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.connect import ConnectMetadata, PipelineMode
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store import IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.source_store import LockStore, QueueStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.runtime import RuntimeConfig, build_runtime_container
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
        container = build_runtime_container(
            RuntimeConfig(root=root),
            index_store=index_store,
            source_store=source_store,
            relation_store=relation_store,
            queue_store=queue_store,
            lock_store=lock_store,
            tool_registry=tool_registry,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            hybrid_search=hybrid_search,
        )
        self.source_store = container.source_store
        self.index_store = container.index_store
        self.relation_store = container.relation_store
        self.queue_store = container.queue_store
        self.lock_store = container.lock_store
        self.vector_store = container.vector_store
        self.embedding_provider = container.embedding_provider
        self.hybrid_search = container.hybrid_search
        self.committer = container.committer
        self.session_archive_store = container.session_archive_store
        self.session_commit_service = container.session_commit_service
        self.context_db = container.context_db
        self.engine = container.engine
        self.executor = container.executor

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        if request.connect_metadata:
            metadata = self._parse_connect_metadata(request.connect_metadata)
            if (
                metadata.run_mode != PipelineMode.ACTION_CAPABLE
                or not metadata.capabilities.can_predict_behavior
            ):
                raise ValueError(
                    "predict() requires action_capable connect metadata with can_predict_behavior=True; "
                    "use assemble_context() for context_reduction agents."
                )
        return self.engine.process(request, policies=policies)

    def process_observation(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy] | None = None,
        *,
        archive_session: bool = True,
        async_commit: bool = True,
    ) -> PredictionResult:
        connect_metadata: dict[str, Any] = {}
        if request.connect_metadata:
            metadata = self._parse_connect_metadata(request.connect_metadata)
            connect_metadata = metadata.to_dict()
            if (
                metadata.run_mode != PipelineMode.ACTION_CAPABLE
                or not metadata.capabilities.can_predict_behavior
                or not metadata.capabilities.can_execute_action
            ):
                raise PermissionError(
                    "process_observation() requires action_capable connect metadata "
                    "with can_predict_behavior=True and can_execute_action=True."
                )
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
            metadata={"connect": connect_metadata} if connect_metadata else {},
        )
        self.context_db.commit_session(archive, async_commit=async_commit)
        return result

    def search_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        return ContextAssembler(self.context_db).search(
            query,
            user_id=user_id,
            context_type=context_type,
            limit=limit,
            connect_filters=connect_filters,
        )

    def assemble_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        token_budget: int = 2000,
        context_types: list[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = self._parse_connect_metadata(connect_metadata)
        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        parsed_types: list[object] | None = (
            [self._parse_context_type(item) for item in context_types] if context_types else None
        )
        return ContextAssembler(self.context_db).assemble(
            query,
            user_id=user_id,
            token_budget=token_budget,
            context_types=parsed_types,
            limit=limit,
            connect_metadata=metadata.to_dict(),
            connect_filters=connect_filters,
        )

    def commit_agent_session(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        used_contexts: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        connect_metadata: dict[str, Any] | None = None,
        async_commit: bool = True,
    ) -> None:
        metadata = self._parse_connect_metadata(connect_metadata)
        archive = SessionArchive(
            user_id=user_id,
            session_id=session_id,
            archive_uri=f"memoryos://user/{user_id}/sessions/history/{session_id}",
            messages=messages or [],
            used_contexts=used_contexts or [],
            tool_results=tool_results or [],
            metadata={"connect": metadata.to_dict()},
        )
        self.context_db.commit_session(archive, async_commit=async_commit)

    def _parse_connect_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        return ConnectMetadata.from_dict(payload)

    def _connect_filters_from_metadata(self, connect_metadata: dict[str, Any] | None) -> dict[str, str]:
        if not connect_metadata:
            return {}
        metadata = self._parse_connect_metadata(connect_metadata)
        return {
            "connect_type": metadata.connect_type,
            "adapter_id": metadata.adapter_id,
            "run_mode": metadata.run_mode,
            "world_domain": metadata.world_domain,
            "source_kind": metadata.source_kind,
        }

    def _parse_context_type(self, context_type: object) -> ContextType:
        if isinstance(context_type, ContextType):
            return context_type
        return ContextType(str(context_type))
