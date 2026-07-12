"""接口层里的客户端。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.result import ProcessObservationResult
from memoryos.api.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    COMMIT_SESSION,
    PRINCIPAL_ONLY_WORKSPACE,
    READ_CONTEXT,
    TrustedRequestContext,
    sanitize_ingress_messages,
    sanitize_ingress_tool_results,
    sanitize_session_provenance,
    sanitize_session_scope,
    workspace_ids_from_metadata,
)
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.retrieval.service import RetrievalService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store import IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.source_store import LockStore, QueueStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical import (
    IDENTITY_ALGORITHM_V2,
    Atomicity,
    Attribution,
    CanonicalMemoryRepository,
    Durability,
    EpistemicStatus,
    EvidenceRef,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemorySemanticReconciler,
    MemoryTransactionPlanner,
    MemoryTransitionPolicy,
    ModalForce,
    ProposalEvidenceValidator,
    ResolvedMemoryIdentity,
    SemanticAssessment,
    UtteranceMode,
    bind_field_evidence,
    scope_key_from_payload,
)
from memoryos.memory.canonical.visibility import read_committed_canonical
from memoryos.memory.extraction import MemoryExtractorBackend
from memoryos.memory.schema import MemoryType
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.providers.rerank import Reranker
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.skill.tool_registry import ToolRegistry


class MemoryOSClient:
    """对外提供记忆写入、检索和会话提交这些常用入口。"""

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
        memory_aliases: dict[str, dict[str, str]] | None = None,
        mode: str = "local",
        tenant_id: str = "default",
    ) -> None:
        self.root = root
        self.mode = mode
        self.tenant_id = tenant_id
        container = build_runtime_container(
            RuntimeConfig(
                root=root,
                mode=mode,
                tenant_id=tenant_id,
                memory_extractor=memory_extractor,
                memory_aliases=memory_aliases,
                reranker=reranker,
            ),
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
        self.memory_projection_worker = container.memory_projection_worker

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        """处理 predict 这一步。"""

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
        """处理一次观察并返回预测结果，记忆仍由会话提交形成。"""

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
            archive_uri=request.session_uri
            or f"memoryos://user/{request.user_id}/sessions/history/{request.episode_id}",
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
        tenant_id: str = "default",
        applicability_scopes: list[dict[str, Any]] | None = None,
        memory_states: list[str] | None = None,
        memory_types: list[str] | None = None,
        claim_uris: list[str] | None = None,
        slot_uris: list[str] | None = None,
        query_intent: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> list[dict[str, Any]]:
        """按用户、工作区、状态和查询意图检索上下文。"""

        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            caller.assert_applicability_scopes(applicability_scopes)
            user_id = caller.user_id
            tenant_id = caller.tenant_id
            project_id = caller.bind_read_workspace(project_id)

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
            "tenant_id": tenant_id,
            "applicability_scope_keys": _scope_keys(applicability_scopes),
            "memory_states": memory_states,
            "memory_types": memory_types,
            "claim_uris": claim_uris,
            "slot_uris": slot_uris,
            "query_intent": query_intent,
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
        tenant_id: str = "default",
        applicability_scopes: list[dict[str, Any]] | None = None,
        memory_states: list[str] | None = None,
        memory_types: list[str] | None = None,
        claim_uris: list[str] | None = None,
        slot_uris: list[str] | None = None,
        query_intent: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """检索并打包本次请求能看到的上下文。"""

        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            caller.assert_applicability_scopes(applicability_scopes)
            user_id = caller.user_id
            tenant_id = caller.tenant_id
            project_id = caller.bind_read_workspace(project_id)

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
            "tenant_id": tenant_id,
            "applicability_scope_keys": _scope_keys(applicability_scopes),
            "memory_states": memory_states,
            "memory_types": memory_types,
            "claim_uris": claim_uris,
            "slot_uris": slot_uris,
            "query_intent": query_intent,
        }
        service = RetrievalService(assembler, _trace_root(self))
        result = service.assemble(query, **_supported_kwargs(assembler.assemble, kwargs))
        self.last_recall_trace_id = str(result.get("trace_id", ""))
        return result

    def recall_trace(
        self,
        trace_id: str,
        *,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        if caller is not None:
            caller.require(READ_CONTEXT)
        trace = RetrievalService(self._context_assembler(), _trace_root(self)).read_trace(trace_id)
        if caller is not None:
            scope = dict(trace.get("scope", {}) or {})
            if scope.get("user_id") != caller.user_id or scope.get("tenant_id") != caller.tenant_id:
                raise FileNotFoundError(trace_id)
            self._require_exact_workspace({"project_id": scope.get("project_id")}, caller, trace_id)
        return trace

    def read(
        self,
        uri: str,
        *,
        layer: str = "L2",
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        if caller is not None:
            caller.require(READ_CONTEXT)
            parsed = ContextURI.parse(uri)
        obj = self.context_db.read_object(uri)
        committed = None
        if caller is not None and dict(obj.metadata or {}).get("canonical_kind"):
            committed = read_committed_canonical(self.source_store, uri)
            obj = committed.object
        if caller is not None:
            self._require_exact_read_visibility(uri, obj, caller)
        layer_uri = {"L0": obj.layers.l0_uri, "L1": obj.layers.l1_uri, "L2": obj.layers.l2_uri or obj.uri}.get(
            layer.upper()
        )
        if not layer_uri:
            raise FileNotFoundError(f"layer unavailable: {layer}")
        if committed is not None and committed.from_before_image and layer.upper() != "L2":
            raise FileNotFoundError(f"committed layer unavailable: {layer}")
        if caller is not None:
            layer_parsed = ContextURI.parse(layer_uri)
            if layer_parsed.authority != parsed.authority or layer_parsed.user_id != parsed.user_id:
                raise FileNotFoundError(uri)
        content = (
            committed.content_override
            if committed is not None and committed.content_override is not None and layer.upper() == "L2"
            else self.source_store.read_content(layer_uri)
        )
        return {"object": obj.to_dict(), "layer": layer.upper(), "content": content}

    def remember(
        self,
        *,
        user_id: str,
        content: str,
        title: str = "",
        memory_type: str = "project_decision",
        project_id: str = "",
        constraint_polarity: str = "",
        condition: str = "",
        exception: str = "",
        connect_metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """Commit a structured explicit-memory command through the canonical chain."""

        if caller is not None:
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            if caller.actor_kind != "user":
                raise PermissionError("authoritative remember requires a trusted user actor")
        if not content.strip():
            raise ValueError("content is required")
        normalized_type = _normalize_explicit_memory_type(memory_type)
        if caller is not None:
            if normalized_type in {MemoryType.PROFILE.value, MemoryType.PREFERENCE.value} and not project_id:
                project_id = ""
            else:
                project_id = caller.bind_write_workspace(project_id)
        retrieval_views = _explicit_retrieval_views(normalized_type, user_id=user_id, project_id=project_id)
        connect = self._parse_connect_metadata(connect_metadata).to_dict()
        event_id = "explicit_" + stable_hash(
            [
                user_id,
                project_id,
                normalized_type,
                title,
                content,
                constraint_polarity,
                condition,
                exception,
            ],
            length=32,
        )
        identity_fields = _explicit_identity_fields(
            normalized_type,
            title=title,
            user_id=user_id,
            project_id=project_id,
            event_id=event_id,
        )
        value_fields: dict[str, Any] = {"canonical_value": content}
        modal_force = ModalForce.PREFER
        if normalized_type == MemoryType.PROJECT_RULE.value:
            modal_force = _explicit_rule_modal_force(
                constraint_polarity,
                has_condition=bool(condition.strip() or exception.strip()),
            )
            value_fields["constraint_polarity"] = modal_force.value
            value_fields["rule"] = content
            if condition.strip():
                value_fields["condition"] = condition.strip()
            if exception.strip():
                value_fields["exception"] = exception.strip()
        elif normalized_type != MemoryType.PREFERENCE.value:
            modal_force = ModalForce.NONE
        command_payload = {
            "command": "REMEMBER_CANONICAL_VALUE",
            "memory_type": normalized_type,
            "identity_fields": identity_fields,
            "value_fields": value_fields,
        }
        command_text = json.dumps(command_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        archive_uri = f"memoryos://user/{user_id}/sessions/history/{event_id}"
        archive = SessionArchive(
            user_id=user_id,
            session_id=event_id,
            archive_uri=archive_uri,
            messages=[
                {
                    "id": event_id,
                    "role": "user",
                    "actor_id": user_id,
                    "event_type": "EXPLICIT_MEMORY_COMMAND",
                    "content": command_text,
                }
            ],
            metadata={
                "connect": connect,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "structured_memory_command": True,
            },
        )
        archive = self._persist_structured_command_archive(archive)
        connect = dict(archive.metadata.get("connect", {}) or {})
        planner = self.session_commit_service.memory_planner
        episode = planner.episode_adapter.adapt(archive)
        system_fields = tuple(identity_fields)
        suggested_scopes = tuple(
            scope
            for scope in episode.legal_scope_candidates()
            if (
                normalized_type in {MemoryType.PROFILE.value, MemoryType.PREFERENCE.value} and scope.kind == "principal"
            )
            or (
                normalized_type not in {MemoryType.PROFILE.value, MemoryType.PREFERENCE.value}
                and scope.kind == ("workspace" if project_id else "principal")
            )
        )
        event_text = episode.events[0].text()
        evidence_refs = (
            EvidenceRef.from_event(
                episode.events[0],
                source_uri=archive.archive_uri,
                span_start=0,
                span_end=len(event_text),
            ),
        )
        proposal = MemorySemanticProposal(
            proposal_id=f"proposal_{event_id}",
            memory_type=normalized_type,
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(
                "confirmation",
                "confirmed",
                "current",
                "unrelated",
                UtteranceMode.ASSERTION.value,
                Attribution.SOURCE_ACTOR.value,
                Durability.DURABLE.value,
                modal_force.value,
                Atomicity.ATOMIC.value,
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=suggested_scopes,
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=_explicit_field_evidence(identity_fields, value_fields, evidence_refs),
            confidence=1.0,
            extractor_version="explicit_remember_v3",
            prompt_version="explicit_remember_v3",
            semantic_contract_version="v3",
            atomic_evidence_ref=evidence_refs[0],
            metadata={
                "source_role": "user",
                "source_adapter_id": str(connect.get("adapter_id", "")),
                "source_session_id": event_id,
                "system_identity_fields": system_fields,
                "effect_authority": "structured_explicit_command",
            },
        )
        formed = planner.formation.plan(
            proposal,
            archive=archive,
            episode=episode,
            retrieval_views=retrieval_views,
        )
        operations = list(formed.operations)
        if formed.decision.value == "PENDING":
            diff = self.committer.commit(user_id, operations) if operations else None
            pending_uri = formed.pending_uri or next(
                (
                    str(operation.target_uri)
                    for operation in operations
                    if operation.payload.get("canonical_pending_proposal") is True
                ),
                "",
            )
            if not pending_uri:
                raise RuntimeError("pending formation did not identify its durable proposal")
            lifecycle_state = (formed.pending_lifecycle_state or "PENDING").upper()
            pending_outstanding = lifecycle_state in {"PENDING", "CONFIRMED", "RETRYABLE"}
            return {
                "uri": pending_uri,
                "status": lifecycle_state,
                "lifecycle_revision": formed.pending_lifecycle_revision or 1,
                "diff_id": diff.diff_id if diff is not None else "",
                "pending_count": 1 if pending_outstanding else 0,
                "pending_persisted": pending_outstanding,
                "proposal_record_persisted": True,
                "canonical_active_operation_count": 0,
            }
        if formed.decision.value != "ACCEPT_FOR_RECONCILE":
            raise ValueError(f"explicit memory was not admitted: {formed.reason}")
        if not operations:
            existing = self.search_context(
                content[:120],
                user_id=user_id,
                project_id=project_id,
                context_type=ContextType.MEMORY,
                query_intent="CURRENT",
            )
            uri = str(existing[0]["uri"]) if existing else ""
            return {"uri": uri, "status": "COMMITTED", "diff_id": ""}
        diff = self.committer.commit(user_id, operations)
        self.memory_projection_worker.process_pending()
        uri = next(
            str(operation.target_uri)
            for operation in operations
            if dict(operation.payload.get("context_object", {}).get("metadata", {}) or {}).get("canonical_kind")
            == "claim"
        )
        return {"uri": uri, "status": "COMMITTED", "diff_id": diff.diff_id}

    def forget(
        self,
        *,
        user_id: str,
        uri: str,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """撤回或软删除自己拥有的记忆，同时保留审计信息。"""

        if caller is not None:
            caller.require(AUTHORITATIVE_FORGET)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
        parsed = ContextURI.parse(uri)
        obj = self.context_db.read_object(uri)
        metadata = dict(obj.metadata or {})
        if caller is not None:
            self._require_exact_workspace(metadata, caller, uri)
        scope = dict(metadata.get("scope", {}) or {})
        authority = dict(scope.get("authority", {}) or {})
        authority_principals = {str(item) for item in authority.get("principal_ids", []) or []}
        if tenant_id is not None and str(obj.tenant_id or "default") != tenant_id:
            raise PermissionError("forget tenant does not match trusted identity")
        if (
            obj.owner_user_id != user_id
            and metadata.get("asserted_by") != user_id
            and user_id not in authority_principals
        ):
            raise PermissionError("forget requires an exact URI owned by user_id")
        canonical_kind = str(metadata.get("canonical_kind") or "")
        if parsed.authority != "user" or (
            parsed.user_id != user_id and not (canonical_kind == "claim" and obj.owner_user_id == user_id)
        ):
            raise PermissionError("forget URI owner does not match user_id")
        if obj.metadata.get("canonical_kind") == "claim":
            return self._forget_canonical_claim(user_id, obj)
        operation = ContextOperation(
            user_id=user_id,
            context_type=obj.context_type,
            action=OperationAction.DELETE,
            target_uri=uri,
            payload={"reason": "explicit_forget"},
            evidence=[{"source": "explicit_forget"}],
        )
        diff = self.context_db.commit_operation(operation)
        _require_committed_diff(diff, {operation.operation_id})
        return {
            "uri": uri,
            "status": "COMMITTED",
            "lifecycle_state": LifecycleState.DELETED.value,
            "diff_id": diff.diff_id,
        }

    def list_pending(
        self,
        *,
        user_id: str,
        tenant_id: str = "default",
        lifecycle_states: list[str] | None = None,
        project_id: str = "",
        caller: TrustedRequestContext | None = None,
    ) -> list[dict[str, Any]]:
        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            project_id = caller.bind_read_workspace(project_id)
        records = CanonicalMemoryRepository(self.source_store).list_pending(
            tenant_id=tenant_id,
            owner_user_id=user_id,
            lifecycle_states=tuple(lifecycle_states or ()),
        )
        visible = []
        for record in records:
            metadata = {"scope": record.scope.to_dict()}
            if self._workspace_matches(metadata, project_id, caller):
                visible.append({"uri": record.uri, **record.to_payload()})
        return visible

    def review_pending(
        self,
        *,
        user_id: str,
        pending_uri: str,
        decision: str,
        expected_lifecycle_revision: int,
        expected_proposal_fingerprint: str,
        command_id: str,
        tenant_id: str = "default",
        reason: str = "",
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        """Apply a user-owned structured review without accepting arbitrary operations or targets."""

        if expected_lifecycle_revision < 1:
            raise ValueError("expected_lifecycle_revision must be positive")
        if not expected_proposal_fingerprint or not command_id:
            raise ValueError("pending review requires proposal fingerprint and command_id")
        repository = CanonicalMemoryRepository(self.source_store)
        pending = repository.load_pending(
            pending_uri,
            tenant_id=tenant_id,
            owner_user_id=user_id,
        )
        if caller is not None:
            caller.require(AUTHORITATIVE_REMEMBER)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            self._require_exact_workspace({"scope": pending.scope.to_dict()}, caller, pending_uri)
        if (
            pending.lifecycle_revision != expected_lifecycle_revision
            or pending.proposal.fingerprint != expected_proposal_fingerprint
        ):
            raise ValueError("pending review expected revision or proposal fingerprint mismatch")
        normalized_decision = str(decision or "").strip().upper()
        formation = self.session_commit_service.memory_planner.formation
        review_reason = f"structured_review:{command_id}:{reason}".rstrip(":")
        terminal = {
            "REJECT": LifecycleState.REJECTED,
            "EXPIRE": LifecycleState.EXPIRED,
            "RETRY": LifecycleState.RETRYABLE,
            "CONFIRM": LifecycleState.CONFIRMED,
        }
        if normalized_decision in terminal:
            operation = formation.plan_pending_lifecycle_transition(
                pending_uri,
                terminal[normalized_decision],
                tenant_id=tenant_id,
                owner_user_id=user_id,
                commit_group_id=f"pending-review:{command_id}",
                reason=review_reason,
                retry_increment=normalized_decision == "RETRY",
            )
            diff = self.committer.commit(user_id, [operation])
            updated = repository.load_pending(
                pending_uri,
                tenant_id=tenant_id,
                owner_user_id=user_id,
            )
            return {
                "uri": pending_uri,
                "status": updated.lifecycle_state.value,
                "lifecycle_revision": updated.lifecycle_revision,
                "diff_id": diff.diff_id,
            }
        if normalized_decision != "CONFIRM_AND_APPLY":
            raise ValueError("pending review decision must be CONFIRM, CONFIRM_AND_APPLY, REJECT, EXPIRE, or RETRY")
        if pending.lifecycle_state in {LifecycleState.PENDING, LifecycleState.RETRYABLE}:
            confirmation = formation.plan_pending_lifecycle_transition(
                pending_uri,
                LifecycleState.CONFIRMED,
                tenant_id=tenant_id,
                owner_user_id=user_id,
                commit_group_id=f"pending-review:{command_id}",
                reason=review_reason,
            )
            self.committer.commit(user_id, [confirmation])
            pending = repository.load_pending(
                pending_uri,
                tenant_id=tenant_id,
                owner_user_id=user_id,
            )
        elif pending.lifecycle_state != LifecycleState.CONFIRMED:
            raise ValueError("only PENDING, RETRYABLE, or CONFIRMED proposals can be applied")
        evidence = pending.proposal.atomic_evidence_ref or (
            pending.proposal.evidence_refs[0] if pending.proposal.evidence_refs else None
        )
        if evidence is None or not evidence.source_uri:
            raise ValueError("confirmed pending proposal has no durable source archive")
        archive = self.session_archive_store.read_archive(evidence.source_uri, tenant_id=tenant_id)
        episode = self.session_commit_service.memory_planner.episode_adapter.adapt(archive)
        resolved = formation.plan_confirmed_pending_resolution(
            pending_uri,
            pending.proposal,
            archive=archive,
            episode=episode,
            tenant_id=tenant_id,
            owner_user_id=user_id,
            commit_group_id=f"pending-resolution:{command_id}",
            retrieval_views=list(pending.retrieval_views),
            reason=review_reason,
        )
        diff = self.committer.commit(user_id, list(resolved.operations))
        self.memory_projection_worker.process_pending()
        final = repository.load_pending(
            pending_uri,
            tenant_id=tenant_id,
            owner_user_id=user_id,
        )
        claim_uris = [
            str(operation.target_uri)
            for operation in resolved.operations
            if isinstance((payload := operation.payload.get("context_object")), dict)
            and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
        ]
        return {
            "uri": pending_uri,
            "status": final.lifecycle_state.value,
            "lifecycle_revision": final.lifecycle_revision,
            "resolved_claim_uris": claim_uris,
            "diff_id": diff.diff_id,
        }

    def _forget_canonical_claim(self, user_id: str, obj) -> dict[str, Any]:  # noqa: ANN001
        metadata = dict(obj.metadata or {})
        slot_uri = obj.uri.rsplit("/claims/", 1)[0]
        slot_obj = self.source_store.read_object(slot_uri)
        slot_metadata = dict(slot_obj.metadata or {})
        memory_scope = MemoryScope.from_dict(dict(metadata.get("scope", {}) or {}))
        canonical_subject = memory_scope.canonical_subject
        if canonical_subject is None:
            raise ValueError("Identity V2 canonical memory is missing its subject")
        identity = ResolvedMemoryIdentity(
            slot_id=str(metadata["slot_id"]),
            slot_uri=slot_uri,
            claim_id=str(metadata["claim_id"]),
            claim_uri=obj.uri,
            slot_identity=dict(slot_metadata.get("identity_fields", {}) or {}),
            canonical_value=str(metadata["canonical_value"]),
            scope_keys=tuple(str(item) for item in slot_metadata.get("scope_keys", []) or []),
            identity_algorithm_version=str(metadata.get("identity_algorithm_version") or IDENTITY_ALGORITHM_V2),
            canonical_subject=canonical_subject,
        )
        slot, claims = CanonicalMemoryRepository(self.source_store).load(identity)
        if slot is None:
            raise FileNotFoundError(slot_uri)
        claim = next(item for item in claims if item.claim_id == identity.claim_id)
        state = "RETRACTED"
        if claim.current.state == state:
            return {"uri": obj.uri, "status": "COMMITTED", "memory_state": state, "diff_id": ""}
        event_id = f"forget:{stable_hash([user_id, obj.uri, claim.latest_revision.revision], length=24)}"
        command_payload = {
            "command": "RETRACT_CANONICAL_CLAIM",
            "claim_id": claim.claim_id,
            "claim_uri": obj.uri,
            "memory_type": str(metadata["memory_type"]),
        }
        command_text = json.dumps(command_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        archive = SessionArchive(
            user_id=user_id,
            session_id=event_id,
            archive_uri=f"memoryos://user/{user_id}/sessions/history/{event_id}",
            messages=[
                {
                    "id": event_id,
                    "role": "user",
                    "event_type": "EXPLICIT_MEMORY_COMMAND",
                    "content": command_text,
                }
            ],
            metadata={
                "tenant_id": str(obj.tenant_id or "default"),
                "structured_memory_command": True,
            },
        )
        archive = self._persist_structured_command_archive(archive)
        episode = self.session_commit_service.memory_planner.episode_adapter.adapt(archive)
        event_text = episode.events[0].text()
        evidence = EvidenceRef.from_event(
            episode.events[0],
            source_uri=archive.archive_uri,
            span_start=0,
            span_end=len(event_text),
        )
        raw_proposal = MemorySemanticProposal(
            proposal_id=event_id,
            memory_type=str(metadata["memory_type"]),
            identity_fields=slot.identity_fields,
            value_fields=claim.current.value_fields,
            semantic=SemanticAssessment(
                "retraction",
                "confirmed",
                "current",
                "corrects",
                UtteranceMode.ASSERTION.value,
                Attribution.SOURCE_ACTOR.value,
                Durability.DURABLE.value,
                ModalForce.NONE.value,
                Atomicity.ATOMIC.value,
            ),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=memory_scope.applicability.all_of,
            related_memory_ids=(claim.claim_id,),
            evidence_refs=(evidence,),
            field_evidence_refs=_explicit_field_evidence(
                slot.identity_fields,
                claim.current.value_fields,
                (evidence,),
            ),
            confidence=1.0,
            extractor_version="explicit_forget_v3",
            prompt_version="explicit_forget_v3",
            semantic_contract_version="v3",
            atomic_evidence_ref=evidence,
            metadata={
                "source_role": "user",
                "source_session_id": event_id,
                "asserted_by": user_id,
                "system_identity_fields": list(slot.identity_fields),
                "system_value_fields": list(claim.current.value_fields),
                "effect_authority": "structured_explicit_command",
            },
        )
        validated = ProposalEvidenceValidator().validate(raw_proposal, episode)
        if not validated.valid:
            raise ValueError(f"explicit forget evidence validation failed: {','.join(validated.errors)}")
        proposal = MemorySemanticNormalizer().normalize(validated.proposal)
        reconciliation = MemorySemanticReconciler().reconcile(
            proposal,
            identity,
            slot=slot,
            claims=claims,
        )
        transition_policy = MemoryTransitionPolicy()
        transition = transition_policy._apply_structured_retraction(
            proposal,
            identity,
            reconciliation,
            authorization_id=event_id,
            owner_user_id=user_id,
            tenant_id=str(obj.tenant_id or "default"),
        )
        plan = MemoryTransactionPlanner().build(
            proposal,
            memory_scope,
            transition,
            tenant_id=str(obj.tenant_id or "default"),
            owner_user_id=user_id,
            episode_id=event_id,
        )
        operations = plan.to_context_operations(
            user_id=user_id,
            tenant_id=str(obj.tenant_id or "default"),
            episode_id=event_id,
        )
        for operation in operations:
            payload = operation.payload.get("context_object")
            if isinstance(payload, dict) and payload.get("uri") == obj.uri:
                payload["relations"] = [relation.to_dict() for relation in obj.relations]
        diff = self.committer.commit(
            user_id,
            operations,
        )
        _require_committed_diff(diff, {operation.operation_id for operation in operations})
        self.memory_projection_worker.process_pending()
        return {"uri": obj.uri, "status": "COMMITTED", "memory_state": state, "diff_id": diff.diff_id}

    def _persist_structured_command_archive(self, archive: SessionArchive) -> SessionArchive:
        """Create one immutable evidence archive for a stable structured command id."""

        tenant_id = self.session_archive_store.archive_tenant(archive)
        with PathLock(self.lock_store).acquire(f"structured-command:{tenant_id}:{archive.archive_uri}"):
            if not self.session_archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self.session_archive_store.write_sync_archive(archive)
                return archive
            persisted = self.session_archive_store.read_archive(archive.archive_uri, tenant_id=tenant_id)
            stable_metadata = ("tenant_id", "project_id", "structured_memory_command")
            if (
                persisted.user_id != archive.user_id
                or persisted.session_id != archive.session_id
                or persisted.messages != archive.messages
                or any(persisted.metadata.get(key) != archive.metadata.get(key) for key in stable_metadata)
            ):
                raise ValueError("structured memory command archive identity conflict")
            return persisted

    def archive_read(
        self,
        archive_uri: str,
        *,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        if caller is not None:
            caller.require(READ_CONTEXT)
            if ContextURI.parse(archive_uri).user_id != caller.user_id:
                raise FileNotFoundError(archive_uri)
        tenant_id = caller.tenant_id if caller is not None else None
        archive = self.session_archive_store.read_archive(archive_uri, tenant_id=tenant_id)
        if caller is not None:
            if archive.user_id != caller.user_id:
                raise FileNotFoundError(archive_uri)
            self._require_exact_workspace(dict(archive.metadata or {}), caller, archive_uri)
        return {"archive": archive.manifest(), "messages": archive.messages, "tool_results": archive.tool_results}

    def archive_search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 20,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
        project_id: str = "",
    ) -> list[dict[str, Any]]:
        if caller is not None:
            caller.require(READ_CONTEXT)
            caller.assert_identity(user_id=user_id, tenant_id=tenant_id)
            tenant_id = caller.tenant_id
            project_id = caller.bind_read_workspace(project_id)
        results = []
        needle = query.lower()
        for head_path in Path(self.root).glob("tenants/*/users/*/sessions/history/**/commit_head.json"):
            try:
                head = json.loads(head_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            head_tenant = str(head.get("tenant_id") or "default")
            if head.get("user_id") != user_id or (tenant_id is not None and head_tenant != tenant_id):
                continue
            archive = self.session_archive_store.read_archive(str(head["archive_uri"]), tenant_id=head_tenant)
            if not self._workspace_matches(dict(archive.metadata or {}), project_id, caller):
                continue
            text = "\n".join(str(item.get("content", item.get("text", ""))) for item in archive.messages)
            if needle in text.lower():
                results.append(
                    {"archive_uri": archive.archive_uri, "session_id": archive.session_id, "preview": text[:500]}
                )
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
            "memory_extractor": "ready"
            if self.session_commit_service.memory_planner.extractor is not None
            else "disabled",
            "embedding": "ready" if self.embedding_provider else "disabled",
            "vector_store": "ready" if self.vector_store else "disabled",
            "reranker": "ready" if self.reranker else "disabled",
            "http_server": "ready" if self.mode == "server" else "disabled",
            "queue": queue_stats,
            "degraded_features": [
                name
                for name, value in (
                    ("embedding", self.embedding_provider),
                    ("vector_store", self.vector_store),
                    ("reranker", self.reranker),
                )
                if value is None
            ],
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
        caller: TrustedRequestContext | None = None,
    ) -> Any:
        """归档并提交一次 Agent 会话。"""

        if caller is not None:
            caller.require(COMMIT_SESSION)
            caller.assert_identity(user_id=user_id, tenant_id=dict(scope or {}).get("tenant_id"))
        metadata = self._parse_connect_metadata(connect_metadata)
        stable_session_id = session_key or session_id
        archive_uri = f"memoryos://user/{user_id}/sessions/history/{stable_session_id}"
        normalized_metadata = metadata.to_dict()
        normalized_project_id = project_id or self._project_id_from_metadata(connect_metadata)
        if caller is not None:
            normalized_project_id = caller.bind_write_workspace(normalized_project_id)
        if caller is None:
            normalized_scope = {
                "user_id": user_id,
                "project_id": normalized_project_id,
                "session_key": stable_session_id,
                **dict(scope or {}),
            }
            normalized_provenance = {"native_session_id": session_id, **dict(provenance or {})}
            normalized_messages = messages or []
            normalized_tool_results = tool_results or []
        else:
            normalized_scope = sanitize_session_scope(
                scope,
                caller,
                project_id=normalized_project_id,
                session_key=stable_session_id,
            )
            normalized_provenance = sanitize_session_provenance(
                provenance,
                caller,
                native_session_id=session_id,
            )
            normalized_messages = sanitize_ingress_messages(messages, caller)
            normalized_tool_results = sanitize_ingress_tool_results(tool_results, caller)
        task_id = _stable_session_commit_task_id(
            {
                "user_id": user_id,
                "session_id": session_id,
                "archive_uri": archive_uri,
                "messages": normalized_messages,
                "used_contexts": used_contexts or [],
                "tool_results": normalized_tool_results,
                "metadata": {
                    "connect": normalized_metadata,
                    "scope": normalized_scope,
                    "provenance": normalized_provenance,
                },
            }
        )
        archive = SessionArchive(
            user_id=user_id,
            session_id=session_id,
            archive_uri=archive_uri,
            messages=normalized_messages,
            used_contexts=used_contexts or [],
            tool_results=normalized_tool_results,
            metadata={
                "connect": normalized_metadata,
                "scope": normalized_scope,
                "provenance": normalized_provenance,
                "project_id": normalized_scope.get("project_id", ""),
                "tenant_id": normalized_scope.get("tenant_id", "default"),
            },
            task_id=task_id,
        )
        return self.context_db.commit_session(archive, async_commit=async_commit)

    def _require_exact_read_visibility(
        self,
        uri: str,
        obj: Any,
        caller: TrustedRequestContext,
    ) -> None:
        if obj.uri != uri or str(obj.tenant_id or "default") != caller.tenant_id:
            raise FileNotFoundError(uri)
        if obj.owner_user_id != caller.user_id:
            raise FileNotFoundError(uri)
        parsed = ContextURI.parse(uri)
        if parsed.authority == "user":
            canonical_kind = str(dict(obj.metadata or {}).get("canonical_kind") or "")
            path_is_bound = parsed.user_id == caller.user_id or canonical_kind in {"claim", "slot"}
            if not path_is_bound:
                raise FileNotFoundError(uri)
        if obj.lifecycle_state != LifecycleState.ACTIVE:
            raise FileNotFoundError(uri)
        metadata = dict(obj.metadata or {})
        self._require_exact_workspace(metadata, caller, uri)
        admission = dict(metadata.get("admission", {}) or {})
        if admission.get("decision") in {"pending", "restricted", "archive_only", "reject"}:
            raise FileNotFoundError(uri)
        if metadata.get("canonical_kind") == "claim" and metadata.get("state") != "ACTIVE":
            raise FileNotFoundError(uri)
        visibility = dict(dict(metadata.get("scope", {}) or {}).get("visibility", {}) or {})
        if visibility:
            if str(visibility.get("tenant_id") or "default") != caller.tenant_id:
                raise FileNotFoundError(uri)
            principals = {str(item) for item in visibility.get("allowed_principal_ids", []) or []}
            services = {str(item) for item in visibility.get("allowed_service_ids", []) or []}
            private = bool(visibility.get("private", False))
            if principals or services or private:
                principal_allowed = caller.user_id in principals
                service_allowed = caller.actor_kind == "service" and caller.actor_id in services
                if not principal_allowed and not service_allowed:
                    raise FileNotFoundError(uri)

    def _require_exact_workspace(
        self,
        metadata: dict[str, Any],
        caller: TrustedRequestContext,
        target: str,
    ) -> None:
        try:
            workspace_ids = workspace_ids_from_metadata(metadata)
        except (TypeError, ValueError):
            raise FileNotFoundError(target) from None
        if workspace_ids and not workspace_ids.issubset(caller.allowed_workspace_ids):
            raise FileNotFoundError(target)

    def _workspace_matches(
        self,
        metadata: dict[str, Any],
        project_id: str,
        caller: TrustedRequestContext | None,
    ) -> bool:
        try:
            workspace_ids = workspace_ids_from_metadata(metadata)
        except (TypeError, ValueError):
            return False
        if caller is not None and workspace_ids and not workspace_ids.issubset(caller.allowed_workspace_ids):
            return False
        if caller is None and not project_id:
            return True
        if project_id == PRINCIPAL_ONLY_WORKSPACE:
            return not workspace_ids
        return not workspace_ids or workspace_ids == {project_id}

    def _parse_connect_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        return ConnectMetadata.from_dict(payload)

    def _require_predict_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        if not payload:
            raise PermissionError(
                "predict() requires explicit embodied/action_capable connect metadata with can_predict_behavior=True."
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


def _require_committed_diff(diff: ContextDiff, expected_operation_ids: set[str]) -> None:
    committed = {operation.operation_id for operation in diff.operations}
    pending = {operation.operation_id for operation in diff.pending_operations}
    rejected = {operation.operation_id for operation in diff.rejected_operations}
    if (
        not expected_operation_ids
        or expected_operation_ids - committed
        or expected_operation_ids & (pending | rejected)
    ):
        raise RuntimeError("forget operation was not fully committed")


def _explicit_field_evidence(
    identity_fields: Any,
    value_fields: Any,
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, tuple[EvidenceRef, ...]]:
    """Declare evidence for the SDK's own fully materialized remember/forget event."""

    bindings = {
        **{f"identity.{key}": evidence_refs for key in identity_fields},
        **{f"value.{key}": evidence_refs for key in value_fields},
        "semantic.speech_act": evidence_refs,
        "semantic.commitment": evidence_refs,
        "semantic.temporal_scope": evidence_refs,
        "semantic.relation_to_existing": evidence_refs,
        "semantic.utterance_mode": evidence_refs,
        "semantic.attribution": evidence_refs,
        "semantic.durability": evidence_refs,
        "semantic.modal_force": evidence_refs,
        "semantic.atomicity": evidence_refs,
        "transition": evidence_refs,
    }
    return bind_field_evidence(
        identity_fields,
        value_fields,
        evidence_refs,
        bindings=bindings,
        semantic_contract_version="v3",
    )


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """处理  supported kwargs 这一步。"""
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _trace_root(client: Any) -> Path:
    return Path(str(getattr(client, "root", "/tmp/memoryos-test"))) / "recall-traces"


def _scope_keys(scopes: list[dict[str, Any]] | None) -> list[str]:
    keys = []
    for scope in scopes or []:
        if not isinstance(scope, dict) or not scope.get("kind") or not scope.get("id"):
            raise ValueError("applicability_scopes must contain scope objects with kind and id")
        keys.append(scope_key_from_payload(scope))
    return list(dict.fromkeys(keys))


def _normalize_explicit_memory_type(memory_type: str) -> str:
    aliases = {"user_profile": MemoryType.PROFILE.value, "user_preference": MemoryType.PREFERENCE.value}
    return aliases.get(memory_type, memory_type)


def _explicit_rule_modal_force(raw: str, *, has_condition: bool) -> ModalForce:
    normalized = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "REQUIRED": ModalForce.REQUIRE,
        "FORBIDDEN": ModalForce.FORBID,
        "ALLOWED": ModalForce.ALLOW,
        "PREFERRED": ModalForce.PREFER,
        "DISCOURAGED": ModalForce.DISCOURAGE,
    }
    try:
        force = aliases.get(normalized)
        if force is None:
            force = ModalForce(normalized)
    except ValueError as exc:
        allowed = ", ".join(
            item.value
            for item in (
                ModalForce.REQUIRE,
                ModalForce.FORBID,
                ModalForce.ALLOW,
                ModalForce.PREFER,
                ModalForce.DISCOURAGE,
                ModalForce.CONDITIONAL_REQUIRE,
                ModalForce.CONDITIONAL_FORBID,
            )
        )
        raise ValueError(f"project_rule requires constraint_polarity in {{{allowed}}}") from exc
    if has_condition and force == ModalForce.REQUIRE:
        return ModalForce.CONDITIONAL_REQUIRE
    if has_condition and force == ModalForce.FORBID:
        return ModalForce.CONDITIONAL_FORBID
    if has_condition and force not in {ModalForce.CONDITIONAL_REQUIRE, ModalForce.CONDITIONAL_FORBID}:
        raise ValueError("project_rule condition or exception requires REQUIRE or FORBID polarity")
    if not has_condition and force in {ModalForce.CONDITIONAL_REQUIRE, ModalForce.CONDITIONAL_FORBID}:
        raise ValueError("conditional project_rule requires condition or exception")
    return force


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


def _explicit_identity_fields(
    memory_type: str,
    *,
    title: str,
    user_id: str,
    project_id: str,
    event_id: str,
) -> dict[str, str]:
    topic = title.strip() or memory_type.replace("_", " ")
    if memory_type == MemoryType.PROFILE.value:
        return {"attribute_key": topic}
    if memory_type == MemoryType.PREFERENCE.value:
        return {"subject": user_id, "dimension": topic}
    if memory_type == MemoryType.ENTITY.value:
        return {"entity_type": "entity", "canonical_entity_id": topic}
    if memory_type == MemoryType.PROJECT_RULE.value:
        return {"rule_topic": topic}
    if memory_type == MemoryType.PROJECT_DECISION.value:
        return {"decision_topic": topic}
    if memory_type == MemoryType.EVENT.value:
        return {"event_key": event_id}
    return {
        "task_pattern": topic,
        "environment_signature": project_id or "global",
    }


class LocalMemoryOSClient(MemoryOSClient):
    """负责 LocalMemoryOSClient 这部分逻辑。"""
