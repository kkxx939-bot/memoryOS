"""Stable in-process SDK facade.

Business orchestration lives in :mod:`memoryos.application`; this module keeps
the public constructor, method signatures, tenant routing, and compatibility
hooks.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from memoryos.action_policy.model.action_policy import ActionPolicy
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
from memoryos.application.memory.pending_review_service import PendingReviewService
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
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.lock_store import LockStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.store.vector import VectorStore
from memoryos.execution.tool_registry import ToolRegistry
from memoryos.memory.canonical import MemorySemanticProposal
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
                memory_egress_policy=memory_egress_policy,
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
        self.agent_session_service = container.agent_session_service
        self.session_archive_store = container.session_archive_store
        self.session_commit_service = container.session_commit_service
        self.context_db = container.context_db
        self.engine = container.engine
        self.executor = container.executor
        self.memory_projection_worker = container.memory_projection_worker
        self.recovery_service = container.recovery_service
        self.recovery_worker = container.recovery_worker
        self.readiness = container.readiness
        self.migration_gate = container.migration_gate
        self.unified_context_migration = container.unified_context_migration
        self._tenant_clients: dict[str, MemoryOSClient] = {}
        self._tenant_clients_lock = threading.RLock()
        self._tenant_mode = mode
        self._tenant_memory_extractor = memory_extractor
        self._tenant_memory_egress_policy = memory_egress_policy
        self._tenant_memory_aliases = memory_aliases
        self._tenant_reranker = reranker
        self._tenant_embedding_provider = embedding_provider
        self.last_recall_trace_id = ""
        self._context_queries = ContextQueryService(self)
        self._prediction_application = PredictionApplicationService(self)
        self._memory_commands = MemoryCommandService(self)
        self._pending_reviews = PendingReviewService(self)
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
        service = getattr(self, "_memory_commands", None)
        if service is None:
            service = MemoryCommandService(self)
            self._memory_commands = service
        return service

    def _get_pending_reviews(self) -> PendingReviewService:
        service = getattr(self, "_pending_reviews", None)
        if service is None:
            service = PendingReviewService(self)
            self._pending_reviews = service
        return service

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
        memory_states: list[str] | None = None,
        memory_types: list[str] | None = None,
        claim_uris: list[str] | None = None,
        slot_uris: list[str] | None = None,
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
                memory_states=memory_states,
                memory_types=memory_types,
                claim_uris=claim_uris,
                slot_uris=slot_uris,
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
            memory_states=memory_states,
            memory_types=memory_types,
            claim_uris=claim_uris,
            slot_uris=slot_uris,
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
        memory_states: list[str] | None = None,
        memory_types: list[str] | None = None,
        claim_uris: list[str] | None = None,
        slot_uris: list[str] | None = None,
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
                memory_states=memory_states,
                memory_types=memory_types,
                claim_uris=claim_uris,
                slot_uris=slot_uris,
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
            memory_states=memory_states,
            memory_types=memory_types,
            claim_uris=claim_uris,
            slot_uris=slot_uris,
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
        *,
        user_id: str,
        content: str,
        title: str = "",
        memory_type: str = "project_decision",
        project_id: str = "",
        constraint_polarity: str = "",
        condition: str = "",
        exception: str = "",
        identity_fields: dict[str, Any] | None = None,
        connect_metadata: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.remember(
                user_id=user_id,
                content=content,
                title=title,
                memory_type=memory_type,
                project_id=project_id,
                constraint_polarity=constraint_polarity,
                condition=condition,
                exception=exception,
                identity_fields=identity_fields,
                connect_metadata=connect_metadata,
                tenant_id=effective_tenant,
                caller=caller,
            )
        return self._get_memory_commands().remember(
            user_id=user_id,
            content=content,
            title=title,
            memory_type=memory_type,
            project_id=project_id,
            constraint_polarity=constraint_polarity,
            condition=condition,
            exception=exception,
            identity_fields=identity_fields,
            connect_metadata=connect_metadata,
            tenant_id=effective_tenant,
            caller=caller,
        )

    def forget(
        self,
        *,
        user_id: str,
        uri: str,
        tenant_id: str | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.forget(user_id=user_id, uri=uri, tenant_id=effective_tenant, caller=caller)
        return self._get_memory_commands().forget(
            user_id=user_id,
            uri=uri,
            tenant_id=effective_tenant,
            caller=caller,
        )

    def list_pending(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        lifecycle_states: list[str] | None = None,
        project_id: str = "",
        caller: TrustedRequestContext | None = None,
    ) -> list[dict[str, Any]]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.list_pending(
                user_id=user_id,
                tenant_id=effective_tenant,
                lifecycle_states=lifecycle_states,
                project_id=project_id,
                caller=caller,
            )
        return self._get_memory_commands().list_pending(
            user_id=user_id,
            tenant_id=effective_tenant,
            lifecycle_states=lifecycle_states,
            project_id=project_id,
            caller=caller,
        )

    def review_pending(
        self,
        *,
        user_id: str,
        pending_uri: str,
        decision: str,
        expected_lifecycle_revision: int,
        expected_proposal_fingerprint: str,
        command_id: str,
        tenant_id: str | None = None,
        reason: str = "",
        corrected_proposal: MemorySemanticProposal | dict[str, Any] | None = None,
        caller: TrustedRequestContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, tenant_id)
        scoped = self._client_for_tenant(effective_tenant)
        if scoped is not self:
            return scoped.review_pending(
                user_id=user_id,
                pending_uri=pending_uri,
                decision=decision,
                expected_lifecycle_revision=expected_lifecycle_revision,
                expected_proposal_fingerprint=expected_proposal_fingerprint,
                command_id=command_id,
                tenant_id=effective_tenant,
                reason=reason,
                corrected_proposal=corrected_proposal,
                caller=caller,
            )
        return self._get_pending_reviews().review_pending(
            user_id=user_id,
            pending_uri=pending_uri,
            decision=decision,
            expected_lifecycle_revision=expected_lifecycle_revision,
            expected_proposal_fingerprint=expected_proposal_fingerprint,
            command_id=command_id,
            tenant_id=effective_tenant,
            reason=reason,
            corrected_proposal=corrected_proposal,
            caller=caller,
            review_locked=self._review_pending_locked,
        )

    def _review_pending_locked(
        self,
        *,
        user_id: str,
        pending_uri: str,
        normalized_decision: str,
        expected_lifecycle_revision: int,
        expected_proposal_fingerprint: str,
        command_id: str,
        tenant_id: str,
        reason: str,
        corrected_proposal: MemorySemanticProposal | None,
        caller: TrustedRequestContext | None,
        review_request_digest: str,
        command_proof_preexisting: bool,
    ) -> dict[str, Any]:
        return self._get_pending_reviews()._review_pending_locked(
            user_id=user_id,
            pending_uri=pending_uri,
            normalized_decision=normalized_decision,
            expected_lifecycle_revision=expected_lifecycle_revision,
            expected_proposal_fingerprint=expected_proposal_fingerprint,
            command_id=command_id,
            tenant_id=tenant_id,
            reason=reason,
            corrected_proposal=corrected_proposal,
            caller=caller,
            review_request_digest=review_request_digest,
            command_proof_preexisting=command_proof_preexisting,
        )

    def _pending_review_recovered_result(
        self,
        pending_uri: str,
        pending: Any,
        claim_uris: tuple[str, ...],
    ) -> dict[str, Any]:
        return self._get_pending_reviews()._pending_review_recovered_result(pending_uri, pending, claim_uris)

    def _forget_canonical_claim(self, user_id: str, obj) -> dict[str, Any]:  # noqa: ANN001
        return self._get_memory_commands()._forget_canonical_claim(user_id, obj)

    def _persist_structured_command_archive(self, archive: SessionArchive) -> SessionArchive:
        return self._get_memory_commands()._persist_structured_command_archive(archive)

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
            )
        return self._get_session_application().archive_search(
            query,
            user_id=user_id,
            limit=limit,
            tenant_id=effective_tenant,
            caller=caller,
            project_id=project_id,
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
                memory_aliases=self._tenant_memory_aliases,
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
        """A committed canonical write must never report a false serving success."""

        combined: dict[str, list[str]] = {
            key: []
            for key in ("processed", "stale", "failed", "dead_letter", "quarantine", "released")
        }
        # Bound synchronous assistance. A larger backlog remains durable and
        # explicitly unavailable instead of making this request wait without
        # limit or falsely claiming that its CurrentSlot row is serving.
        for _ in range(10):
            result = self.memory_projection_worker.process_pending(limit=10)
            for key in combined:
                combined[key].extend(str(item) for item in result.get(key, ()))
            terminal = tuple(result.get("dead_letter", ())) + tuple(result.get("quarantine", ()))
            failed = tuple(result.get("failed", ()))
            if failed or terminal:
                raise RuntimeError(
                    "canonical transaction committed but its serving projection is unavailable; "
                    f"failed={len(failed)}, terminal={len(terminal)}"
                )
            stats = self.queue_store.stats(queue_name="memory_projection")
            if not any(int(stats.get(status, 0) or 0) for status in ("pending", "leased")):
                return combined
            if not result.get("processed"):
                break
        raise RuntimeError(
            "canonical transaction committed but its serving projection remains pending after bounded replay"
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
            projection_store=getattr(self.context_db, "projection_store", None),
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
