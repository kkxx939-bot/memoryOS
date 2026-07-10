from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.result import ProcessObservationResult
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.retrieval.service import RetrievalService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store import IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.source_store import LockStore, QueueStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.core.ids import stable_hash
from memoryos.core.time import utc_now
from memoryos.memory.extraction import MemoryExtractorBackend
from memoryos.memory.schema import MemoryType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.providers.rerank import Reranker
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
        reranker: Reranker | None = None,
        memory_extractor: MemoryExtractorBackend | None = None,
        mode: str = "local",
    ) -> None:
        self.root = root
        self.mode = mode
        container = build_runtime_container(
            RuntimeConfig(root=root, mode=mode, memory_extractor=memory_extractor, reranker=reranker),
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
        self.reranker = container.reranker
        self.committer = container.committer
        self.session_archive_store = container.session_archive_store
        self.session_commit_service = container.session_commit_service
        self.context_db = container.context_db
        self.engine = container.engine
        self.executor = container.executor

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        self._require_predict_metadata(request.connect_metadata)
        return self.engine.process(request, policies=policies)

    def process_observation(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy] | None = None,
        *,
        archive_session: bool = True,
        async_commit: bool = True,
    ) -> ProcessObservationResult:
        metadata = self._require_process_observation_metadata(request.connect_metadata)
        connect_metadata = metadata.to_dict()
        result = self.engine.process(request, policies=policies)
        try:
            action_result = self.executor.execute(result.decision, result.action_context)
        except Exception as exc:
            from memoryos.prediction.model.action_result import ActionResult

            action_result = ActionResult(
                action=result.decision.action,
                status="failed",
                executed=False,
                reason="ActionExecutor raised",
                error=exc.__class__.__name__,
            )
        if not archive_session:
            return ProcessObservationResult(
                prediction_result=result,
                action_result=action_result,
                session_commit_result=None,
                archive_uri=None,
            )
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
        used_contexts = self._merge_uri_items(
            [{"uri": uri} for uri in result.action_context.source_uris],
            [{"uri": uri, "refresh_layers": False} for uri in action_result.resource_uris],
        )
        used_skills = self._uri_items(
            [
                *[uri for uri in result.action_context.source_uris if uri.startswith("memoryos://skills/")],
                *action_result.skill_uris,
            ]
        )
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
            used_contexts=used_contexts,
            used_skills=used_skills,
            metadata={"connect": connect_metadata},
        )
        archive_error = None
        try:
            commit_result = self.context_db.commit_session(archive, async_commit=async_commit)
        except Exception as exc:
            commit_result = None
            archive_error = {"code": "ARCHIVE_COMMIT_FAILED", "message": exc.__class__.__name__}
        return ProcessObservationResult(
            prediction_result=result,
            action_result=action_result,
            session_commit_result=commit_result,
            archive_uri=archive.archive_uri,
            archive_error=archive_error,
        )

    def search_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
    ) -> list[dict[str, Any]]:
        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        metadata = self._parse_connect_metadata(connect_metadata)
        assembler = self._context_assembler()
        kwargs = {
            "user_id": user_id,
            "context_type": context_type,
            "limit": limit,
            "connect_filters": connect_filters,
            "search_scope": search_scope,
            "retrieval_views": retrieval_views,
            "project_id": project_id or self._project_id_from_metadata(connect_metadata),
            "adapter_id": metadata.adapter_id,
        }
        service = RetrievalService(assembler, _trace_root(self))
        results, self.last_recall_trace_id = service.search(query, **_supported_kwargs(assembler.search, kwargs))
        return results

    def assemble_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        token_budget: int = 2000,
        context_types: list[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
    ) -> dict[str, Any]:
        metadata = self._parse_connect_metadata(connect_metadata)
        connect_filters = self._connect_filters_from_metadata(connect_metadata)
        parsed_types: list[object] | None = (
            [self._parse_context_type(item) for item in context_types] if context_types else None
        )
        assembler = self._context_assembler()
        kwargs = {
            "user_id": user_id,
            "token_budget": token_budget,
            "context_types": parsed_types,
            "limit": limit,
            "connect_metadata": metadata.to_dict(),
            "connect_filters": connect_filters,
            "search_scope": search_scope,
            "retrieval_views": retrieval_views,
            "project_id": project_id or self._project_id_from_metadata(connect_metadata),
            "adapter_id": metadata.adapter_id,
        }
        service = RetrievalService(assembler, _trace_root(self))
        result = service.assemble(query, **_supported_kwargs(assembler.assemble, kwargs))
        self.last_recall_trace_id = str(result.get("trace_id", ""))
        return result

    def recall_trace(self, trace_id: str) -> dict[str, Any]:
        return RetrievalService(self._context_assembler(), _trace_root(self)).read_trace(trace_id)

    def read(self, uri: str, *, layer: str = "L2") -> dict[str, Any]:
        obj = self.context_db.read_object(uri)
        layer_uri = {"L0": obj.layers.l0_uri, "L1": obj.layers.l1_uri, "L2": obj.layers.l2_uri or obj.uri}.get(layer.upper())
        if not layer_uri:
            raise FileNotFoundError(f"layer unavailable: {layer}")
        return {"object": obj.to_dict(), "layer": layer.upper(), "content": self.source_store.read_content(layer_uri)}

    def remember(
        self,
        *,
        user_id: str,
        content: str,
        title: str = "",
        memory_type: str = "project_decision",
        project_id: str = "",
        connect_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not content.strip():
            raise ValueError("content is required")
        normalized_type = _normalize_explicit_memory_type(memory_type)
        retrieval_views = _explicit_retrieval_views(normalized_type, user_id=user_id, project_id=project_id)
        digest = stable_hash([user_id, project_id, normalized_type, content], length=20)
        uri = f"memoryos://user/{user_id}/memories/explicit/{digest}"
        connect = self._parse_connect_metadata(connect_metadata).to_dict()
        metadata = {
            "memory_type": normalized_type,
            "admission": {"decision": "accept", "confidence": 1.0},
            "retrieval_views": retrieval_views,
            "connect": connect,
            "scope": {"user_id": user_id, "project_id": project_id},
            "provenance": {"source_kind": "explicit", "captured_at": utc_now()},
            "confidence": 1.0,
        }
        obj = ContextObject(uri=uri, context_type=ContextType.MEMORY, title=(title or content[:64]), owner_user_id=user_id, metadata=metadata)
        operation = ContextOperation(
            user_id=user_id,
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=uri,
            payload={"context_object": obj.to_dict(), "content": content},
            evidence=[{"source": "explicit_remember"}],
        )
        diff = self.context_db.commit_operation(operation)
        return {"uri": uri, "status": "COMMITTED", "diff_id": diff.diff_id}

    def forget(self, *, user_id: str, uri: str) -> dict[str, Any]:
        if not uri.startswith(f"memoryos://user/{user_id}/"):
            raise PermissionError("forget requires an exact URI owned by user_id")
        obj = self.context_db.read_object(uri)
        operation = ContextOperation(
            user_id=user_id,
            context_type=obj.context_type,
            action=OperationAction.DELETE,
            target_uri=uri,
            payload={"reason": "explicit_forget"},
            evidence=[{"source": "explicit_forget"}],
        )
        self.context_db.commit_operation(operation)
        return {"uri": uri, "status": "COMMITTED", "lifecycle_state": LifecycleState.DELETED.value}

    def archive_read(self, archive_uri: str) -> dict[str, Any]:
        archive = self.session_archive_store.read_archive(archive_uri)
        return {"archive": archive.manifest(), "messages": archive.messages, "tool_results": archive.tool_results}

    def archive_search(self, query: str, *, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        results = []
        needle = query.lower()
        for manifest_path in Path(self.root).glob("tenants/*/users/*/sessions/history/**/commit_manifest.json"):
            try:
                manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.get("user_id") != user_id:
                continue
            archive = self.session_archive_store.read_archive(str(manifest["archive_uri"]))
            text = "\n".join(str(item.get("content", item.get("text", ""))) for item in archive.messages)
            if needle in text.lower():
                results.append({"archive_uri": archive.archive_uri, "session_id": archive.session_id, "preview": text[:500]})
            if len(results) >= limit:
                break
        return results

    def health(self) -> dict[str, Any]:
        heartbeat = Path(self.root) / "system" / "worker-health.json"
        queue_stats: dict[str, int] = getattr(self.queue_store, "stats", lambda: {})()
        return {
            "source_store": "ready",
            "index_store": "ready",
            "queue_store": "ready",
            "worker": "ready" if heartbeat.exists() else "stopped",
            "memory_extractor": "ready" if self.session_commit_service.memory_planner.extractor.__class__.__name__ != "RuleFallbackExtractor" else "fallback",
            "embedding": "ready" if self.embedding_provider else "disabled",
            "vector_store": "ready" if self.vector_store else "disabled",
            "reranker": "ready" if self.reranker else "disabled",
            "http_server": "ready" if self.mode == "server" else "disabled",
            "queue": queue_stats,
            "degraded_features": [name for name, value in (("embedding", self.embedding_provider), ("vector_store", self.vector_store), ("reranker", self.reranker)) if value is None],
        }

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
        project_id: str = "",
        session_key: str = "",
        scope: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Any:
        metadata = self._parse_connect_metadata(connect_metadata)
        stable_session_id = session_key or session_id
        archive_uri = f"memoryos://user/{user_id}/sessions/history/{stable_session_id}"
        normalized_metadata = metadata.to_dict()
        normalized_scope = {
            "user_id": user_id,
            "project_id": project_id or self._project_id_from_metadata(connect_metadata),
            "session_key": stable_session_id,
            **dict(scope or {}),
        }
        normalized_provenance = {"native_session_id": session_id, **dict(provenance or {})}
        task_id = _stable_session_commit_task_id(
            {
                "user_id": user_id,
                "session_id": session_id,
                "archive_uri": archive_uri,
                "messages": messages or [],
                "used_contexts": used_contexts or [],
                "tool_results": tool_results or [],
                "metadata": {"connect": normalized_metadata, "scope": normalized_scope, "provenance": normalized_provenance},
            }
        )
        archive = SessionArchive(
            user_id=user_id,
            session_id=session_id,
            archive_uri=archive_uri,
            messages=messages or [],
            used_contexts=used_contexts or [],
            tool_results=tool_results or [],
            metadata={
                "connect": normalized_metadata,
                "scope": normalized_scope,
                "provenance": normalized_provenance,
                "project_id": normalized_scope.get("project_id", ""),
            },
            task_id=task_id,
        )
        return self.context_db.commit_session(archive, async_commit=async_commit)

    def _parse_connect_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        return ConnectMetadata.from_dict(payload)

    def _require_predict_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        if not payload:
            raise PermissionError(
                "predict() requires explicit embodied/action_capable connect metadata "
                "with can_predict_behavior=True."
            )
        metadata = self._parse_connect_metadata(payload)
        if (
            metadata.connect_type != ConnectType.EMBODIED
            or metadata.run_mode != PipelineMode.ACTION_CAPABLE
            or not metadata.capabilities.can_predict_behavior
        ):
            raise PermissionError(
                "predict() requires embodied/action_capable connect metadata "
                "with can_predict_behavior=True; use assemble_context() for context_reduction agents."
            )
        return metadata

    def _require_process_observation_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        metadata = self._require_predict_metadata(payload)
        if not metadata.capabilities.can_execute_action:
            raise PermissionError(
                "process_observation() requires embodied/action_capable connect metadata "
                "with can_predict_behavior=True and can_execute_action=True."
            )
        return metadata

    def _connect_filters_from_metadata(self, connect_metadata: dict[str, Any] | None) -> dict[str, str]:
        if not connect_metadata:
            return {}
        allowed = {"connect_type", "adapter_id", "run_mode", "world_domain", "source_kind"}
        metadata = self._parse_connect_metadata(connect_metadata)
        metadata.validate()
        metadata_dict = metadata.to_dict()
        return {
            key: str(metadata_dict[key])
            for key in allowed
            if key in connect_metadata and metadata_dict.get(key) not in {None, ""}
        }

    def _parse_context_type(self, context_type: object) -> ContextType:
        if isinstance(context_type, ContextType):
            return context_type
        return ContextType(str(context_type))

    def _context_assembler(self) -> ContextAssembler:
        reranker = getattr(self, "reranker", None)
        hybrid_search = getattr(self, "hybrid_search", None)
        kwargs = _supported_kwargs(ContextAssembler, {"reranker": reranker, "hybrid_search": hybrid_search})
        return ContextAssembler(self.context_db, **kwargs)

    def _project_id_from_metadata(self, connect_metadata: dict[str, Any] | None) -> str:
        metadata = dict(connect_metadata or {})
        for key in ("project_id", "project"):
            if metadata.get(key):
                return str(metadata[key])
        extra = metadata.get("extra")
        if isinstance(extra, dict):
            for key in ("project_id", "project", "repo"):
                if extra.get(key):
                    return str(extra[key])
        return ""

    def _uri_items(self, uris: list[str]) -> list[dict[str, str]]:
        return [{"uri": uri} for uri in dict.fromkeys(str(uri) for uri in uris if uri)]

    def _merge_uri_items(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                uri = str(item.get("uri", ""))
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                merged.append(dict(item))
        return merged


def _stable_session_commit_task_id(payload: dict[str, Any]) -> str:
    return f"session_commit_{stable_hash(payload, length=32)}"


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize a protocol boundary without hiding errors raised inside the call."""
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _trace_root(client: Any) -> Path:
    return Path(str(getattr(client, "root", "/tmp/memoryos-test"))) / "recall-traces"


def _normalize_explicit_memory_type(memory_type: str) -> str:
    aliases = {"user_profile": MemoryType.PROFILE.value, "user_preference": MemoryType.PREFERENCE.value}
    return aliases.get(memory_type, memory_type)


def _explicit_retrieval_views(memory_type: str, *, user_id: str, project_id: str) -> list[str]:
    user_views = {
        MemoryType.PROFILE.value: f"user:{user_id}:profile",
        MemoryType.PREFERENCE.value: f"user:{user_id}:preferences",
    }
    if memory_type in user_views:
        return [user_views[memory_type]]
    project_suffix = {
        MemoryType.PROJECT_RULE.value: "rules",
        MemoryType.PROJECT_DECISION.value: "decisions",
        MemoryType.AGENT_EXPERIENCE.value: "agent_experience",
        MemoryType.ENTITY.value: "knowledge",
        MemoryType.EVENT.value: "knowledge",
    }.get(memory_type, "knowledge")
    return [f"project:{project_id}:{project_suffix}"] if project_id else [f"user:{user_id}:profile"]


class LocalMemoryOSClient(MemoryOSClient):
    """Explicit name for the in-process ContextDB transport."""
