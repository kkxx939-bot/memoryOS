"""Stable in-process SDK facade for Context, Session and Markdown memory."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.memory_contract import validate_memory_request, validate_memory_response
from memoryos.api.retrieval_contract import parse_retrieval_options
from memoryos.application.context.assembler import ContextAssembler
from memoryos.application.context.orchestrator import UnifiedRetrievalOrchestrator
from memoryos.application.context.query_service import ContextQueryService
from memoryos.application.context.query_support import (
    _compatible_scalar,
    _supported_kwargs,
)
from memoryos.application.context.query_support import (
    _scope_keys as _scope_keys,
)
from memoryos.application.context.reranking import Reranker
from memoryos.application.memory.command_service import MemoryCommandService
from memoryos.application.memory.pending_review_service import MemoryEditReviewService
from memoryos.application.prediction.result import ProcessObservationResult
from memoryos.application.prediction.service import PredictionApplicationService
from memoryos.application.session.service import SessionApplicationService
from memoryos.connect import ConnectMetadata, ConnectType, PipelineMode
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.lock_store import LockStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.store.vector import VectorStore
from memoryos.execution.tool_registry import ToolRegistry
from memoryos.memory.extraction import MemoryEgressPolicy, MemoryExtractorBackend
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.security.trusted_context import (
    PRINCIPAL_ONLY_WORKSPACE,
    TrustedRequestContext,
    workspace_ids_from_metadata,
)


class MemoryOSClient:
    """Public facade for the MemoryOS in-process application."""

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
        memory_egress_policy: MemoryEgressPolicy | None = None,
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
                memory_egress_policy=memory_egress_policy,
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
        self.agent_session_service = container.agent_session_service
        self.session_archive_store = container.session_archive_store
        self.session_commit_service = container.session_commit_service
        self.context_db = container.context_db
        self.engine = container.engine
        self.executor = container.executor
        self.memory_projection_worker = container.memory_projection_worker
        self.memory_command_service = container.memory_command_service
        self.memory_review_service = container.memory_review_service
        self.memory_document_store = container.memory_document_store
        self.memory_document_control_store = container.memory_document_control_store
        self.memory_document_revision_store = container.memory_document_revision_store
        self.memory_document_planner = container.memory_document_planner
        self.memory_document_committer = container.memory_document_committer
        self.memory_document_consolidation_store = container.memory_document_consolidation_store
        self.memory_document_consolidator = container.memory_document_consolidator
        self.memory_document_projector = container.memory_document_projector
        self.memory_document_scanner = container.memory_document_scanner
        self.memory_document_edit_worker = container.memory_document_edit_worker
        self.memory_document_scan_worker = container.memory_document_scan_worker
        self.memory_document_eraser = container.memory_document_eraser
        self.recovery_service = container.recovery_service
        self.recovery_worker = container.recovery_worker
        self.readiness = container.readiness
        self._tenant_clients: dict[str, MemoryOSClient] = {}
        self._tenant_clients_lock = threading.RLock()
        self._tenant_mode = mode
        self._tenant_memory_extractor = memory_extractor
        self._tenant_memory_egress_policy = memory_egress_policy
        self._tenant_reranker = reranker
        self._tenant_embedding_provider = embedding_provider
        self.last_recall_trace_id = ""
        self._context_queries = ContextQueryService(self)
        self._prediction_application = PredictionApplicationService(self)
        self._session_application = SessionApplicationService(self, self._context_queries)

    def _get_context_queries(self) -> ContextQueryService:
        service = getattr(self, "_context_queries", None)
        if service is None:
            service = ContextQueryService(self)
            self._context_queries = service
        return service

    def _get_prediction_application(self) -> PredictionApplicationService:
        service = getattr(self, "_prediction_application", None)
        if service is None:
            service = PredictionApplicationService(self)
            self._prediction_application = service
        return service

    def _get_memory_commands(self) -> MemoryCommandService:
        return self.memory_command_service

    def _get_memory_reviews(self) -> MemoryEditReviewService:
        return self.memory_review_service

    def _get_session_application(self) -> SessionApplicationService:
        service = getattr(self, "_session_application", None)
        if service is None:
            service = SessionApplicationService(self, self._get_context_queries())
            self._session_application = service
        return service

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        """Process an embodied prediction request."""

        return self._get_prediction_application().predict(request, policies=policies)

    def process_observation(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy] | None = None,
        *,
        archive_session: bool = True,
        async_commit: bool = True,
    ) -> ProcessObservationResult:
        """Process an observation, action execution, and optional session archive."""

        return self._get_prediction_application().process_observation(
            request,
            policies=policies,
            archive_session=archive_session,
            async_commit=async_commit,
        )

    def search_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        tenant_id: str | None = None,
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        document_ids: list[str] | None = None,
        document_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> list[dict[str, Any]]:
        structured = parse_retrieval_options(options)
        effective_tenant = self._effective_tenant(
            caller,
            _compatible_scalar(tenant_id, structured.tenant_id if structured else None, "tenant_id"),
        )
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.search_context(
                query,
                options=structured,
                user_id=user_id,
                context_type=context_type,
                limit=limit,
                connect_metadata=connect_metadata,
                search_scope=search_scope,
                retrieval_views=retrieval_views,
                project_id=project_id,
                tenant_id=effective_tenant,
                applicability_scopes=applicability_scopes,
                record_kinds=record_kinds,
                document_ids=document_ids,
                document_kinds=document_kinds,
                query_intent=query_intent,
                caller=caller,
            )
        return self._get_context_queries().search_context(
            query,
            options=structured,
            user_id=user_id,
            context_type=context_type,
            limit=limit,
            connect_metadata=connect_metadata,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            tenant_id=effective_tenant,
            applicability_scopes=applicability_scopes,
            record_kinds=record_kinds,
            document_ids=document_ids,
            document_kinds=document_kinds,
            query_intent=query_intent,
            caller=caller,
        )

    def assemble_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        token_budget: int = 2000,
        context_types: list[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        tenant_id: str | None = None,
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        document_ids: list[str] | None = None,
        document_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        structured = parse_retrieval_options(options)
        effective_tenant = self._effective_tenant(
            caller,
            _compatible_scalar(tenant_id, structured.tenant_id if structured else None, "tenant_id"),
        )
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.assemble_context(
                query,
                options=structured,
                user_id=user_id,
                token_budget=token_budget,
                context_types=context_types,
                limit=limit,
                connect_metadata=connect_metadata,
                search_scope=search_scope,
                retrieval_views=retrieval_views,
                project_id=project_id,
                tenant_id=effective_tenant,
                applicability_scopes=applicability_scopes,
                record_kinds=record_kinds,
                document_ids=document_ids,
                document_kinds=document_kinds,
                query_intent=query_intent,
                caller=caller,
            )
        return self._get_context_queries().assemble_context(
            query,
            options=structured,
            user_id=user_id,
            token_budget=token_budget,
            context_types=context_types,
            limit=limit,
            connect_metadata=connect_metadata,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            tenant_id=effective_tenant,
            applicability_scopes=applicability_scopes,
            record_kinds=record_kinds,
            document_ids=document_ids,
            document_kinds=document_kinds,
            query_intent=query_intent,
            caller=caller,
        )

    def recall_trace(
        self,
        trace_id: str,
        *,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        return self._get_context_queries().recall_trace(trace_id, caller=caller)

    def read(
        self,
        uri: str,
        *,
        layer: str = "L2",
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.read(uri, layer=layer, tenant_id=effective_tenant, caller=caller)
        return self._get_context_queries().read(uri, layer=layer, tenant_id=effective_tenant, caller=caller)

    def remember(
        self,
        content: str,
        occurred_at: str | None = None,
        target_hint: str | None = None,
        expected_document_digest: str | None = None,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "remember",
            {
                "content": content,
                "occurred_at": occurred_at,
                "target_hint": target_hint,
                "expected_document_digest": expected_document_digest,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.remember(
                **request,
                tenant_id=effective_tenant,
                caller=caller,
            )
        self._require_ready()
        return validate_memory_response("remember",
            self._get_memory_commands().remember(**request, caller=caller)
        )

    def adopt_memory_document(
        self,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "adopt",
            {
                "relative_path": relative_path,
                "expected_raw_sha256": expected_raw_sha256,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.adopt_memory_document(
                **request,
                tenant_id=effective_tenant,
                caller=caller,
            )
        self._require_ready()
        return validate_memory_response("adopt",
            self._get_memory_commands().adopt_memory_document(**request, caller=caller)
        )

    def edit_memory_document(
        self,
        document_uri: str,
        edit: str,
        expected_digest: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "edit",
            {"document_uri": document_uri, "edit": edit, "expected_digest": expected_digest},
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.edit_memory_document(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response("edit",
            self._get_memory_commands().edit_memory_document(**request, caller=caller)
        )

    def rename_memory_document(
        self,
        document_uri: str,
        new_relative_path: str,
        expected_digest: str,
        edit: str | None = None,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "rename",
            {
                "document_uri": document_uri,
                "new_relative_path": new_relative_path,
                "expected_digest": expected_digest,
                "edit": edit,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.rename_memory_document(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response(
            "rename",
            self._get_memory_commands().rename_memory_document(**request, caller=caller),
        )

    def merge_memory_documents(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: list[dict[str, str]],
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "merge",
            {
                "target_document_uri": target_document_uri,
                "merged_edit": merged_edit,
                "expected_target_digest": expected_target_digest,
                "source_documents": source_documents,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.merge_memory_documents(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response(
            "merge",
            self._get_memory_commands().merge_memory_documents(**request, caller=caller),
        )

    def propose_memory_consolidation(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: list[dict[str, str]],
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "merge_propose",
            {
                "target_document_uri": target_document_uri,
                "merged_edit": merged_edit,
                "expected_target_digest": expected_target_digest,
                "source_documents": source_documents,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.propose_memory_consolidation(
                **request,
                tenant_id=effective_tenant,
                caller=caller,
            )
        self._require_ready()
        return validate_memory_response(
            "merge_propose",
            self._get_memory_commands().propose_memory_consolidation(
                **request,
                caller=caller,
            ),
        )

    def resume_memory_consolidation(
        self,
        saga_id: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request("merge_resume", {"saga_id": saga_id})
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.resume_memory_consolidation(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response(
            "merge_resume",
            self._get_memory_commands().resume_memory_consolidation(**request, caller=caller),
        )

    def forget(
        self,
        document_uri: str,
        section_anchor: str | None = None,
        mode: str = "SOFT_FORGET",
        expected_digest: str | None = None,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "forget",
            {
                "document_uri": document_uri,
                "section_anchor": section_anchor,
                "mode": mode,
                "expected_digest": expected_digest,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.forget(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response("forget",
            self._get_memory_commands().forget(**request, caller=caller)
        )

    def list_memory_history(
        self,
        document_uri: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request("history", {"document_uri": document_uri})
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.list_memory_history(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response("history",
            self._get_memory_commands().list_memory_history(**request, caller=caller)
        )

    def restore_memory_revision(
        self,
        document_uri: str,
        revision: int,
        expected_digest: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "restore",
            {
                "document_uri": document_uri,
                "revision": revision,
                "expected_digest": expected_digest,
            },
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.restore_memory_revision(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response("restore",
            self._get_memory_commands().restore_memory_revision(**request, caller=caller)
        )

    def review_memory_edit(
        self,
        proposal_id: str,
        decision: str,
        corrected_edit: str | None = None,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request(
            "review",
            {"proposal_id": proposal_id, "decision": decision, "corrected_edit": corrected_edit},
        )
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.review_memory_edit(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response("review",
            self._get_memory_reviews().review_edit(**request, caller=caller)
        )

    def preview_memory_edit(
        self,
        proposal_id: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext,
    ) -> dict[str, Any]:
        request = validate_memory_request("review_preview", {"proposal_id": proposal_id})
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.preview_memory_edit(**request, tenant_id=effective_tenant, caller=caller)
        self._require_ready()
        return validate_memory_response(
            "review_preview",
            self._get_memory_reviews().preview_edit(**request, caller=caller),
        )

    def archive_read(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.archive_read(archive_uri, tenant_id=effective_tenant, caller=caller)
        return self._get_session_application().archive_read(
            archive_uri,
            tenant_id=effective_tenant,
            caller=caller,
        )

    def archive_search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 20,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
        project_id: str = "",
        timezone_name: str = "UTC",
    ) -> list[dict[str, Any]]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.archive_search(
                query,
                user_id=user_id,
                limit=limit,
                tenant_id=effective_tenant,
                caller=caller,
                project_id=project_id,
                timezone_name=timezone_name,
            )
        return self._get_session_application().archive_search(
            query,
            user_id=user_id,
            limit=limit,
            tenant_id=effective_tenant,
            caller=caller,
            project_id=project_id,
            timezone_name=timezone_name,
            search_context=self.search_context,
            archive_read=self.archive_read,
        )

    def health(self) -> dict[str, Any]:
        return self._get_session_application().health()

    def commit_agent_session(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        used_contexts: list[dict[str, Any]] | None = None,
        used_skills: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        connect_metadata: dict[str, Any] | None = None,
        async_commit: bool = True,
        project_id: str = "",
        session_key: str = "",
        scope: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> Any:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.commit_agent_session(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
                used_contexts=used_contexts,
                used_skills=used_skills,
                tool_results=tool_results,
                connect_metadata=connect_metadata,
                async_commit=async_commit,
                project_id=project_id,
                session_key=session_key,
                scope=scope,
                provenance=provenance,
                tenant_id=effective_tenant,
                caller=caller,
            )
        return self._get_session_application().commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            used_contexts=used_contexts,
            used_skills=used_skills,
            tool_results=tool_results,
            connect_metadata=connect_metadata,
            async_commit=async_commit,
            project_id=project_id,
            session_key=session_key,
            scope=scope,
            provenance=provenance,
            tenant_id=effective_tenant,
            caller=caller,
        )

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
        if parsed.authority == "user" and parsed.user_id != caller.user_id:
            raise FileNotFoundError(uri)
        if obj.lifecycle_state != LifecycleState.ACTIVE:
            raise FileNotFoundError(uri)
        metadata = dict(obj.metadata or {})
        self._require_exact_workspace(metadata, caller, uri)
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

    def _effective_tenant(
        self,
        caller: TrustedRequestContext | None,
        explicit_tenant_id: str | None,
    ) -> str:
        if caller is not None:
            if explicit_tenant_id is not None and explicit_tenant_id != caller.tenant_id:
                raise PermissionError("tenant_id does not match trusted caller")
            return caller.tenant_id
        effective = explicit_tenant_id or getattr(self, "tenant_id", "default")
        if not isinstance(effective, str) or not effective.strip():
            raise ValueError("tenant_id is required")
        return effective

    def _client_for_tenant(self, tenant_id: str) -> MemoryOSClient:
        """Return a runtime whose stores and recovery artifacts are bound to tenant_id."""

        if tenant_id == getattr(self, "tenant_id", "default"):
            return self
        with self._tenant_clients_lock:
            existing = self._tenant_clients.get(tenant_id)
            if existing is not None:
                return existing
            client = MemoryOSClient(
                self.root,
                mode=self._tenant_mode,
                tenant_id=tenant_id,
                memory_extractor=self._tenant_memory_extractor,
                memory_egress_policy=self._tenant_memory_egress_policy,
                reranker=self._tenant_reranker,
                embedding_provider=self._tenant_embedding_provider,
            )
            self._tenant_clients[tenant_id] = client
            return client

    def _require_ready(self) -> None:
        readiness = getattr(self, "readiness", None)
        if readiness is not None:
            readiness.require_ready()

    def _process_memory_projections_or_raise(self) -> dict[str, list[str]]:
        """Boundedly assist durable document projection without hiding failure."""

        combined: dict[str, list[str]] = {key: [] for key in ("processed", "stale", "failed")}
        # A larger backlog remains durable and explicitly pending.
        for _ in range(10):
            run = self.memory_projection_worker.process_pending(limit=10)
            for key in combined:
                combined[key].extend(str(item) for item in getattr(run, key))
            if run.failed:
                raise RuntimeError(
                    "memory document committed but its serving projection is unavailable; "
                    f"failed={len(run.failed)}"
                )
            stats = self.queue_store.stats(queue_name="memory_projection")
            if not any(int(stats.get(status, 0) or 0) for status in ("pending", "leased")):
                return combined
            if not run.processed and not run.stale:
                break
        raise RuntimeError(
            "memory document committed but its serving projection remains pending after bounded replay"
        )

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

    def _retrieval_orchestrator(self) -> UnifiedRetrievalOrchestrator:
        return UnifiedRetrievalOrchestrator(
            self.context_db,
            vector_store=getattr(self, "vector_store", None),
            embedding_provider=getattr(self, "embedding_provider", None),
            reranker=getattr(self, "reranker", None),
        )

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



class LocalMemoryOSClient(MemoryOSClient):
    """Local compatibility name for the in-process client."""
