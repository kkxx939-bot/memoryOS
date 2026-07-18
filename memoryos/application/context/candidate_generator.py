"""Bounded candidate generation over the single Unified Context Catalog."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind, ServingTier
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.retrieval.fusion import RetrievalCandidate
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from memoryos.contextdb.store.index_store import IndexHit, IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.vector import VectorHit, VectorStore, vector_capabilities, vector_row_id
from memoryos.core.types import scope_keys_from_payloads
from memoryos.security.context_projection import ContextProjectionSanitizer

_PRINCIPAL_ONLY_WORKSPACE = "__memoryos_principal_only__"
_TEMPORAL_TEXT_SATISFACTION_SCORE = 0.5
_DOCUMENT_RECORD_KINDS = frozenset(
    {
        CatalogRecordKind.MEMORY_DOCUMENT.value,
        CatalogRecordKind.MEMORY_BLOCK.value,
    }
)


@dataclass(frozen=True)
class CandidateGenerationResult:
    branches: Mapping[str, tuple[RetrievalCandidate, ...]]
    structured_candidates: int = 0
    exact_candidates: int = 0
    fts_candidates: int = 0
    vector_candidates: int = 0
    relation_candidates: int = 0
    vector_overfetch: int = 0
    source_reads: int = 0
    degraded_modes: tuple[str, ...] = ()


class CandidateGenerator:
    """Generate exact/FTS/vector/relation branches with hard query bounds."""

    MAX_RELATION_SEEDS = 20
    MAX_RELATIONS_PER_SEED = 5
    MAX_RELATION_IDENTITIES_PER_SEED = 3
    MAX_RECORDS_PER_RELATION_TARGET = 2
    MAX_VECTOR_OVERFETCH = 200

    def __init__(
        self,
        index_store: IndexStore,
        *,
        relation_store: RelationStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        sanitizer: ContextProjectionSanitizer | None = None,
    ) -> None:
        self.index_store = index_store
        self.relation_store = relation_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.sanitizer = sanitizer or ContextProjectionSanitizer()

    def generate(self, plan: RetrievalQueryPlan) -> CandidateGenerationResult:
        filters = self._filters(plan)
        exact_records = self._exact_records(plan, filters)
        record_exact = tuple(self._from_record(record, branch="exact", score=1.0) for record in exact_records)

        lexical_hits = self._search_catalog(plan.semantic_query, filters, plan.candidate_limit)
        hit_candidates = tuple(self._from_hit(hit) for hit in lexical_hits)
        search_exact = tuple(item for item in hit_candidates if item.branch_scores.get("exact", 0.0) > 0)
        lexical = tuple(item for item in hit_candidates if item.branch_scores.get("exact", 0.0) <= 0)
        exact_by_key = {item.record_key: item for item in (*record_exact, *search_exact)}
        exact = tuple(exact_by_key.values())
        structured_records = self._structured_records(
            plan,
            filters,
            has_bounded_text_candidates=self._has_sufficient_text_candidates(exact, lexical),
        )
        structured_candidates = tuple(
            self._from_record(record, branch="structured", score=0.0) for record in structured_records
        )

        seeds = (*structured_candidates, *exact, *lexical)
        vector, vector_degraded, vector_source_reads, vector_overfetch = self._vector_candidates(seeds, plan)
        relation = self._relation_candidates(seeds, plan)
        fts_degraded = (
            "fts_unavailable" if plan.semantic_query and getattr(self.index_store, "fts_enabled", True) is False else ""
        )
        modes = tuple(dict.fromkeys(mode for mode in (fts_degraded, vector_degraded) if mode))
        return CandidateGenerationResult(
            branches={
                "structured": structured_candidates[: plan.candidate_limit],
                "exact": exact[: plan.candidate_limit],
                "lexical": lexical[: plan.candidate_limit],
                "vector": vector[: plan.candidate_limit],
                "relation": relation[: plan.candidate_limit],
            },
            structured_candidates=len(structured_records),
            exact_candidates=len(exact),
            fts_candidates=len(lexical),
            vector_candidates=len(vector),
            relation_candidates=len(relation),
            vector_overfetch=vector_overfetch,
            source_reads=vector_source_reads,
            degraded_modes=modes,
        )

    def _structured_records(
        self,
        plan: RetrievalQueryPlan,
        filters: Mapping[str, Any],
        *,
        has_bounded_text_candidates: bool = False,
    ) -> tuple[CatalogRecord, ...]:
        """Return a bounded SQL-filtered stream for deterministic list plans.

        Empty-query CURRENT/HISTORY calls are supported list views,
        not permission to enumerate Source state.  The Catalog
        store applies every tenant/ACL/type/path/time filter before the hard
        candidate LIMIT.

        A semantic query normally disables this branch.  The only exception
        is a temporal plan whose intent and indexed time constraints make the
        candidate set deterministic: event-time OPEN_RECALL or transaction-time
        HISTORY. Exact/FTS run first; a bounded text hit
        already satisfies the query and suppresses the broader temporal list.
        With no text hit, temporal SQL provides the non-lexical fallback while
        still applying time/path/ACL predicates before the bounded LIMIT.
        """

        if plan.target_uris:
            return ()
        if plan.semantic_query:
            if has_bounded_text_candidates or not self._is_bounded_temporal_plan(plan):
                return ()
        lister = getattr(self.index_store, "list_catalog", None)
        if not callable(lister):
            return ()
        raw_values: Any = lister(
            tenant_id=plan.tenant_id or "default",
            filters=dict(filters),
            limit=plan.candidate_limit,
        )
        values = raw_values if isinstance(raw_values, Sequence) else ()
        return tuple(record for record in values if isinstance(record, CatalogRecord))

    @staticmethod
    def _is_bounded_temporal_plan(plan: RetrievalQueryPlan) -> bool:
        """Admit only complete, intent-specific indexed temporal constraints."""

        if plan.query_intent == RetrievalQueryIntent.OPEN_RECALL:
            return bool(plan.event_time_from and plan.event_time_to)
        if plan.query_intent == RetrievalQueryIntent.HISTORY:
            return bool(plan.transaction_time_from and plan.transaction_time_to)
        return False

    @staticmethod
    def _has_sufficient_text_candidates(
        exact: Sequence[RetrievalCandidate],
        lexical: Sequence[RetrievalCandidate],
    ) -> bool:
        """Avoid a broader temporal list only when text retrieval is decisive.

        Calendar digits or generic date terms can create a weak FTS hit even
        when the natural-language question has no semantic overlap with the
        stored content.  Exact identity always satisfies the query; lexical
        candidates suppress the temporal fallback only when term coverage is
        high enough to be a selective bounded result.
        """

        if exact:
            return True
        return any(
            float(candidate.branch_scores.get("lexical", 0.0)) >= _TEMPORAL_TEXT_SATISFACTION_SCORE
            for candidate in lexical
        )

    def _filters(self, plan: RetrievalQueryPlan) -> dict[str, Any]:
        shared_view = any(view.startswith(("project:", "user:")) for view in plan.legacy_retrieval_views)
        filters: dict[str, Any] = {
            "tenant_id": plan.tenant_id or "default",
            "record_kinds": self._record_kinds(plan),
        }
        for key, value in (
            ("session_ids", plan.session_ids),
            ("context_types", tuple(item.value for item in plan.context_types)),
            ("source_kinds", plan.source_kinds),
            ("record_kinds", self._record_kinds(plan)),
            ("document_ids", plan.document_ids),
            ("document_kinds", plan.document_kinds),
            ("target_uris", plan.target_uris),
            ("target_paths", plan.target_paths),
            ("event_time_from", plan.event_time_from),
            ("event_time_to", plan.event_time_to),
            ("transaction_time_from", plan.transaction_time_from),
            ("transaction_time_to", plan.transaction_time_to),
            ("updated_at_from", plan.updated_at_from),
            ("updated_at_to", plan.updated_at_to),
        ):
            if value not in (None, (), ""):
                filters[key] = value
        if plan.owner_user_id:
            filters["principal_owner_id"] = plan.owner_user_id
        if plan.service_id:
            filters["service_access_id"] = plan.service_id
        if plan.adapter_id:
            agent_root = f"agents/{plan.adapter_id}"
            exact_agent_path = bool(plan.target_paths) and all(
                path == agent_root or path.startswith(f"{agent_root}/") for path in plan.target_paths
            )
            filter_name = (
                "adapter_id" if plan.legacy_search_scope == "agent_private" or exact_agent_path else "adapter_access_id"
            )
            filters[filter_name] = plan.adapter_id
        if plan.workspace_ids:
            # A trusted workspace scope can also read owner-scoped records
            # whose workspace is intentionally empty.  The reserved
            # principal-only value is a deny-all for workspace-bound rows.
            filters["workspace_access_ids"] = (
                ("",)
                if plan.workspace_ids == (_PRINCIPAL_ONLY_WORKSPACE,)
                else tuple(dict.fromkeys(("", *plan.workspace_ids)))
            )
        metadata = dict(plan.metadata_filters)
        raw_connect_filters = metadata.get("connect_filters", metadata.get("connect"))
        if isinstance(raw_connect_filters, Mapping):
            allowed_connect_fields = {"connect_type", "adapter_id", "run_mode", "world_domain", "source_kind"}
            connect_filters = {
                str(key): value
                for key, value in raw_connect_filters.items()
                if str(key) in allowed_connect_fields and value not in (None, "")
            }
            if shared_view:
                # Project/user shared views are intentionally cross-adapter.
                # Connect metadata describes the trusted caller and must not
                # become a content filter that hides tenant-shared state.
                connect_filters.clear()
            if connect_filters:
                filters["connect_filters"] = connect_filters
        if metadata.get("principal_absent") or plan.owner_user_id is None:
            filters["owner_user_id"] = ""
            filters["require_unscoped"] = True
        if metadata.get("lifecycle_state"):
            filters["lifecycle_state"] = metadata["lifecycle_state"]
        if "applicability_scope_keys" in metadata:
            scope_keys = tuple(metadata.get("applicability_scope_keys") or ())
            if scope_keys:
                filters["applicability_scope_keys"] = scope_keys
            else:
                filters["require_unscoped"] = True
        if metadata.get("require_unscoped"):
            filters["require_unscoped"] = True
        if metadata.get("retrieval_views"):
            filters["retrieval_views"] = metadata["retrieval_views"]
        if metadata.get("include_candidates"):
            filters["include_candidates"] = True
        requested_record_kinds = set(plan.record_kinds)
        if (
            plan.document_kinds
            and requested_record_kinds.intersection(_DOCUMENT_RECORD_KINDS)
            and requested_record_kinds.difference(_DOCUMENT_RECORD_KINDS)
        ):
            # A mixed Session + Markdown query uses document_kind to narrow
            # only document projections; ordinary Session rows have no such
            # field and must remain eligible.
            filters["document_kinds_apply_to_documents_only"] = True
        if metadata.get("minimum_lexical_relevance") is not None:
            try:
                minimum_lexical_relevance = float(metadata["minimum_lexical_relevance"])
            except (TypeError, ValueError) as exc:
                raise ValueError("minimum_lexical_relevance must be a finite score") from exc
            if not 0.0 <= minimum_lexical_relevance <= 1.0:
                raise ValueError("minimum_lexical_relevance must be between 0 and 1")
            filters["minimum_lexical_relevance"] = minimum_lexical_relevance
        if plan.query_intent in {
            RetrievalQueryIntent.HISTORY,
            RetrievalQueryIntent.OPEN_RECALL,
        }:
            filters["serving_tier"] = tuple(item.value for item in ServingTier)
        return filters

    def _record_kinds(self, plan: RetrievalQueryPlan) -> tuple[str, ...]:
        if plan.record_kinds:
            return plan.record_kinds
        all_kinds = tuple(item.value for item in CatalogRecordKind)
        return all_kinds

    def _exact_records(self, plan: RetrievalQueryPlan, filters: Mapping[str, Any]) -> tuple[CatalogRecord, ...]:
        lister = getattr(self.index_store, "list_catalog", None)
        if not callable(lister):
            return ()
        target_uris = plan.target_uris
        if (
            not target_uris
            and plan.query_intent == RetrievalQueryIntent.EXACT
            and plan.semantic_query.startswith("memoryos://")
        ):
            target_uris = (plan.semantic_query,)
        if not target_uris:
            return ()
        exact_filters = self._target_identity_filters(
            filters,
            target_uris,
            limit=plan.candidate_limit,
        )
        raw_values: Any = lister(
            tenant_id=plan.tenant_id or "default",
            filters=exact_filters,
            limit=plan.candidate_limit,
        )
        values = raw_values if isinstance(raw_values, Sequence) else ()
        return tuple(record for record in values if isinstance(record, CatalogRecord))

    def _search_catalog(self, query: str, filters: Mapping[str, Any], limit: int) -> list[IndexHit]:
        search_catalog = getattr(self.index_store, "search_catalog", None)
        if callable(search_catalog):
            raw_values: Any = search_catalog(
                query,
                tenant_id=str(filters.get("tenant_id") or "default"),
                filters=filters,
                limit=limit,
            )
            values = raw_values if isinstance(raw_values, Sequence) else ()
        else:
            search = getattr(self.index_store, "search", None)
            if not callable(search):
                return []
            parameters = inspect.signature(search).parameters
            supports_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
            if "tenant_id" not in parameters and not supports_kwargs:
                # IndexStore exact/search contracts are tenant-qualified.  A
                # backend that cannot bind the tenant before searching must
                # not be used and then filtered after the fact: doing so can
                # both scan another tenant and apply LIMIT before isolation.
                return []
            kwargs: dict[str, Any] = {
                "tenant_id": str(filters.get("tenant_id") or "default"),
                "limit": limit,
            }
            if "filters" in parameters or supports_kwargs:
                kwargs["filters"] = dict(filters)
            else:
                if "owner_user_id" in parameters and filters.get("principal_owner_id") is not None:
                    kwargs["owner_user_id"] = filters["principal_owner_id"]
                context_types = tuple(filters.get("context_types", ()) or ())
                if "context_type" in parameters and len(context_types) == 1:
                    kwargs["context_type"] = context_types[0]
            raw_values = search(query, **kwargs)
            values = raw_values if isinstance(raw_values, Sequence) else ()
        minimum_lexical = float(filters.get("minimum_lexical_relevance") or 0.0)
        accepted: list[IndexHit] = []
        for hit in values:
            if not isinstance(hit, IndexHit) or not self._hit_matches_filters(hit, filters):
                continue
            scores = dict(hit.metadata.get("retrieval_scores", {}) or {})
            lexical = self._finite_score(scores.get("lexical"))
            identity = self._finite_score(scores.get("identity"))
            if minimum_lexical and lexical < minimum_lexical and identity <= 0.0:
                continue
            accepted.append(hit)
            if len(accepted) >= limit:
                break
        return accepted

    @staticmethod
    def _finite_score(value: Any) -> float:
        try:
            score = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return score if score == score and score not in {float("inf"), float("-inf")} else 0.0

    @staticmethod
    def _hit_matches_filters(hit: IndexHit, filters: Mapping[str, Any]) -> bool:
        metadata = dict(hit.metadata or {})
        connect = dict(metadata.get("connect", {}) or {})
        hit_context_type = str(getattr(hit.context_type, "value", hit.context_type))
        if str(metadata.get("tenant_id") or "default") != str(filters.get("tenant_id") or "default"):
            return False
        principal_owner = filters.get("principal_owner_id")
        hit_owner = str(metadata.get("owner_user_id") or "")
        hit_workspace = str(metadata.get("workspace_id") or metadata.get("project_id") or "")
        if principal_owner is not None:
            shared_workspaces = {
                str(item)
                for item in filters.get("workspace_access_ids", ()) or ()
                if str(item) not in {"", _PRINCIPAL_ONLY_WORKSPACE}
            }
            scope = metadata.get("scope")
            visibility = scope.get("visibility") if isinstance(scope, Mapping) else None
            explicit_principal_grant = bool(
                isinstance(visibility, Mapping)
                and str(visibility.get("tenant_id") or "") == str(metadata.get("tenant_id") or "default")
                and str(principal_owner)
                in {str(item) for item in visibility.get("allowed_principal_ids", ()) or ()}
            )
            explicit_service_grant = bool(
                isinstance(visibility, Mapping)
                and str(visibility.get("tenant_id") or "") == str(metadata.get("tenant_id") or "default")
                and str(filters.get("service_access_id") or "")
                in {str(item) for item in visibility.get("allowed_service_ids", ()) or ()}
            )
            tenant_public_grant = bool(
                isinstance(visibility, Mapping)
                and str(visibility.get("tenant_id") or "") == str(metadata.get("tenant_id") or "default")
                and visibility.get("private") is False
                and not (visibility.get("allowed_principal_ids", ()) or ())
                and not (visibility.get("allowed_service_ids", ()) or ())
            )
            shared_record = bool(
                explicit_principal_grant
                or explicit_service_grant
                or tenant_public_grant
                or (metadata.get("workspace_shared") is True and hit_workspace in shared_workspaces)
            )
            public_context = hit_owner == "" and hit_context_type in {"resource", "skill"}
            if hit_owner != str(principal_owner) and not public_context and not shared_record:
                return False
        workspace_access = filters.get("workspace_access_ids")
        if workspace_access is not None and hit_workspace not in {str(item) for item in workspace_access}:
            return False
        context_types = tuple(str(item) for item in filters.get("context_types", ()) or ())
        if context_types and hit_context_type not in context_types:
            return False
        hit_adapter = str(metadata.get("adapter_id") or connect.get("adapter_id") or "")
        exact_adapter = filters.get("adapter_id")
        if exact_adapter not in (None, "") and hit_adapter != str(exact_adapter):
            return False
        adapter_access = filters.get("adapter_access_id")
        if adapter_access not in (None, "") and hit_adapter not in {"", str(adapter_access)}:
            if hit_context_type not in {"session", "resource", "skill"}:
                return False
        source_kinds = tuple(str(item) for item in filters.get("source_kinds", ()) or ())
        hit_source_kind = str(metadata.get("source_kind") or connect.get("source_kind") or "")
        if source_kinds and hit_source_kind not in source_kinds:
            return False
        for filter_name, metadata_name in (
            ("record_kinds", "record_kind"),
            ("document_ids", "document_id"),
        ):
            allowed = {str(item) for item in filters.get(filter_name, ()) or ()}
            if allowed and str(metadata.get(metadata_name) or "") not in allowed:
                return False
        allowed_document_kinds = {str(item) for item in filters.get("document_kinds", ()) or ()}
        document_kind_is_scoped = bool(filters.get("document_kinds_apply_to_documents_only"))
        if (
            allowed_document_kinds
            and (not document_kind_is_scoped or str(metadata.get("record_kind") or "") in _DOCUMENT_RECORD_KINDS)
            and str(metadata.get("document_kind") or "") not in allowed_document_kinds
        ):
            return False
        connect_filters = filters.get("connect_filters")
        if isinstance(connect_filters, Mapping):
            for key, value in connect_filters.items():
                if value in (None, ""):
                    continue
                actual = metadata.get(str(key)) if str(key) in {"adapter_id", "source_kind"} else None
                actual = actual if actual not in (None, "") else connect.get(str(key))
                if actual != value:
                    return False
        allowed_views = {str(item) for item in filters.get("retrieval_views", ()) or ()}
        item_views = {str(item) for item in metadata.get("retrieval_views", ()) or ()}
        # An explicit compatibility view is a narrowing security constraint.
        # A backend that cannot prove a candidate's view must not turn missing
        # metadata into permission to cross workspace boundaries.
        if allowed_views and (
            (item_views and not allowed_views.intersection(item_views))
            or not item_views
        ):
            return False
        target_uris = {str(item) for item in filters.get("target_uris", ()) or ()}
        hit_identity_uris = {hit.uri}
        if target_uris and not target_uris.intersection(hit_identity_uris):
            return False
        if filters.get("require_unscoped"):
            try:
                raw_scope = metadata.get("scope", {}) or {}
                raw_applicability = (raw_scope.get("applicability", {}) if isinstance(raw_scope, Mapping) else {}) or {}
                if not isinstance(raw_applicability, Mapping):
                    return False
                if metadata.get("scope_keys", ()):
                    return False
                if scope_keys_from_payloads(raw_applicability.get("all_of", ())):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        return True

    def _from_hit(self, hit: IndexHit) -> RetrievalCandidate:
        metadata = dict(hit.metadata or {})
        record_key = str(metadata.get("catalog_record_key") or hit.uri)
        getter = getattr(self.index_store, "get_catalog", None)
        if callable(getter):
            record = getter(
                tenant_id=str(metadata.get("tenant_id") or "default"),
                record_key=record_key,
            )
            if isinstance(record, CatalogRecord):
                scores = dict(metadata.get("retrieval_scores", {}) or {})
                score = float(scores.get("lexical") or scores.get("identity") or hit.score or 0.0)
                branch = "exact" if float(scores.get("identity") or 0.0) > 0 else "lexical"
                return self._from_record(record, branch=branch, score=score, extra_metadata=metadata)
        scores = dict(metadata.get("retrieval_scores", {}) or {})
        lexical = float(scores.get("lexical") or scores.get("identity") or hit.score or 0.0)
        record_kind = str(metadata.get("record_kind") or CatalogRecordKind.CONTEXT.value)
        return RetrievalCandidate(
            record_key=record_key,
            uri=hit.uri,
            title=hit.title,
            context_type=hit.context_type,
            text=hit.title,
            source_uri=hit.uri,
            record_kind=record_kind,
            source_kind=str(metadata.get("source_kind") or "context"),
            tenant_id=str(metadata.get("tenant_id") or "default"),
            owner_user_id=str(metadata.get("owner_user_id") or ""),
            session_id=str(metadata.get("session_id") or ""),
            workspace_id=str(metadata.get("workspace_id") or metadata.get("project_id") or ""),
            document_id=str(metadata.get("document_id") or ""),
            block_id=str(metadata.get("block_id") or ""),
            document_kind=str(metadata.get("document_kind") or ""),
            document_revision=int(metadata.get("document_revision") or 0),
            projection_generation=int(metadata.get("projection_generation") or 0),
            archive_digest=str(metadata.get("archive_digest") or ""),
            manifest_digest=str(metadata.get("manifest_digest") or ""),
            source_digest=str(metadata.get("source_digest") or ""),
            event_time=str(metadata.get("event_time") or ""),
            metadata=metadata,
            branch_scores={"lexical": lexical},
        )

    @staticmethod
    def _from_record(
        record: CatalogRecord,
        *,
        branch: str,
        score: float,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> RetrievalCandidate:
        metadata = {
            **dict(record.metadata),
            **dict(extra_metadata or {}),
            "tenant_id": record.tenant_id,
            "owner_user_id": record.owner_user_id,
            "workspace_id": record.workspace_id,
            "document_id": record.document_id,
            "block_id": record.block_id,
            "document_kind": record.document_kind,
            "document_revision": record.document_revision,
            "projection_generation": record.projection_generation,
            "projection_effect_hash": record.projection_effect_hash,
            "catalog_record_key": record.record_key,
            "record_kind": record.record_kind,
            "serving_tier": record.serving_tier,
        }
        return RetrievalCandidate(
            record_key=record.record_key,
            uri=record.uri,
            title=record.title,
            context_type=record.context_type,
            source_kind=record.source_kind,
            record_kind=record.record_kind,
            text="",
            l0_text=record.l0_text,
            l1_text=record.l1_text,
            l2_uri=record.l2_uri,
            source_uri=record.source_uri,
            source_digest=record.source_digest,
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            session_id=record.session_id,
            workspace_id=record.workspace_id,
            document_id=record.document_id,
            block_id=record.block_id,
            document_kind=record.document_kind,
            document_revision=record.document_revision,
            projection_generation=record.projection_generation,
            archive_digest=str(record.metadata.get("archive_digest") or ""),
            manifest_digest=str(record.metadata.get("manifest_digest") or ""),
            event_time=record.event_time,
            hotness=max(record.hotness, record.semantic_hotness, record.behavior_support_hotness),
            metadata=metadata,
            branch_scores={branch: score},
        )

    def _vector_candidates(
        self,
        candidates: Sequence[RetrievalCandidate],
        plan: RetrievalQueryPlan,
    ) -> tuple[tuple[RetrievalCandidate, ...], str, int, int]:
        if self.vector_store is None or self.embedding_provider is None or not plan.semantic_query:
            return (), "", 0, 0
        vector_limit = min(plan.candidate_limit, self.MAX_VECTOR_OVERFETCH)
        capabilities = vector_capabilities(self.vector_store)
        native_filtering = all(
            (
                capabilities.supports_metadata_filtering,
                capabilities.supports_namespace_filtering,
                capabilities.supports_time_filtering,
            )
        )
        if native_filtering:
            return self._native_filtered_vector_candidates(plan, vector_limit=vector_limit)
        by_row_id: dict[str, list[RetrievalCandidate]] = {}
        for item in candidates:
            by_row_id.setdefault(vector_row_id(plan.tenant_id or "default", item.record_key), []).append(item)
        bounded = tuple(by_row_id)[:vector_limit]
        if not bounded:
            return (), "vector_requires_structured_candidates", 0, 0
        candidate_search = getattr(self.vector_store, "search_vector_candidates", None)
        if not callable(candidate_search):
            return (), "vector_backend_lacks_bounded_candidates", 0, 0
        try:
            embedding = self.embedding_provider.embed(self._provider_query(plan.semantic_query))
            raw_hits: Any = candidate_search(embedding, bounded, limit=vector_limit)
        except Exception as exc:  # Vector is explicitly eventual and may fall back to FTS.
            return (), f"vector_fallback:{type(exc).__name__}", 0, len(bounded)
        if (
            not isinstance(raw_hits, Sequence)
            or isinstance(raw_hits, (str, bytes, bytearray))
            or any(not isinstance(hit, VectorHit) for hit in raw_hits)
        ):
            return (), "vector_fallback:InvalidResponse", 0, len(bounded)
        hits = list(raw_hits)
        result: list[RetrievalCandidate] = []
        for hit in hits[:vector_limit]:
            for item in by_row_id.get(str(hit.uri), ()):
                result.append(item.with_branch("vector", float(hit.score), len(result) + 1))
                if len(result) >= vector_limit:
                    break
        degraded = "" if capabilities.supports_metadata_filtering else "bounded_vector_candidate_fallback"
        return tuple(result), degraded, 0, len(bounded)

    def _native_filtered_vector_candidates(
        self,
        plan: RetrievalQueryPlan,
        *,
        vector_limit: int,
    ) -> tuple[tuple[RetrievalCandidate, ...], str, int, int]:
        """Ask a production backend to apply trusted filters before Top-K.

        The returned identifiers are still re-bound through the SQL Catalog,
        which independently enforces the same tenant/ACL/path/time contract
        before any hit can enter fusion.
        """

        filtered_search = getattr(self.vector_store, "search_vector_filtered", None)
        lister = getattr(self.index_store, "list_catalog", None)
        embedding_provider = self.embedding_provider
        if not callable(filtered_search) or not callable(lister) or embedding_provider is None:
            return (), "vector_filtered_contract_missing", 0, 0
        filters = self._filters(plan)
        try:
            embedding = embedding_provider.embed(self._provider_query(plan.semantic_query))
            raw_hits: Any = filtered_search(
                embedding,
                namespace=plan.tenant_id or "default",
                filters=filters,
                limit=vector_limit,
            )
        except Exception as exc:
            return (), f"vector_fallback:{type(exc).__name__}", 0, vector_limit
        if (
            not isinstance(raw_hits, Sequence)
            or isinstance(raw_hits, (str, bytes, bytearray))
            or any(not isinstance(hit, VectorHit) for hit in raw_hits)
        ):
            return (), "vector_fallback:InvalidResponse", 0, vector_limit
        hits = tuple(raw_hits[:vector_limit])
        if not hits:
            return (), "", 0, vector_limit
        scores: dict[str, float] = {}
        ordered_record_keys: list[str] = []
        for hit in hits:
            metadata = dict(hit.metadata or {})
            record_key = str(metadata.get("catalog_record_key") or "")
            tenant_id = str(metadata.get("tenant_id") or "")
            expected_row_id = vector_row_id(tenant_id, record_key) if tenant_id and record_key else ""
            if (
                not record_key
                or tenant_id != str(plan.tenant_id or "default")
                or str(hit.uri) != expected_row_id
            ):
                continue
            if record_key not in scores:
                ordered_record_keys.append(record_key)
            scores[record_key] = max(scores.get(record_key, 0.0), self._finite_score(hit.score))
        if not ordered_record_keys:
            return (), "", 0, vector_limit
        raw_records: Any = lister(
            tenant_id=plan.tenant_id or "default",
            filters={**filters, "record_keys": tuple(ordered_record_keys)},
            limit=min(vector_limit, len(ordered_record_keys)),
        )
        records = raw_records if isinstance(raw_records, Sequence) else ()
        by_record_key = {
            record.record_key: record
            for record in records
            if isinstance(record, CatalogRecord) and record.record_key in scores
        }
        result = tuple(
            self._from_record(
                by_record_key[record_key],
                branch="vector",
                score=scores[record_key],
            ).with_branch("vector", scores[record_key], rank)
            for rank, record_key in enumerate(ordered_record_keys, start=1)
            if record_key in by_record_key
        )
        return result, "", 0, vector_limit

    def _provider_query(self, query: str) -> str:
        """Return the fail-closed projection allowed to leave retrieval.

        Exact and lexical retrieval intentionally continue to use the trusted
        caller's original query.  Only the copy sent to an embedding provider
        crosses this egress boundary, so credentials and sensitive absolute
        paths are removed without weakening deterministic local matching.
        """

        projection = self.sanitizer.sanitize(
            title="retrieval query",
            l1_text=query,
            metadata={},
            source_kind="retrieval_query",
        )
        return projection.l1_text

    def _relation_candidates(
        self,
        seeds: Sequence[RetrievalCandidate],
        plan: RetrievalQueryPlan,
    ) -> tuple[RetrievalCandidate, ...]:
        if self.relation_store is None or not plan.relation_expansion:
            return ()
        lister = getattr(self.index_store, "list_catalog", None)
        if not callable(lister):
            return ()
        result: list[RetrievalCandidate] = []
        seen_record_keys = {seed.record_key for seed in seeds}
        for seed in seeds[: min(self.MAX_RELATION_SEEDS, plan.candidate_limit)]:
            kwargs = {
                "tenant_id": plan.tenant_id or "default",
                "owner_user_id": plan.owner_user_id,
            }
            relation_rows_remaining = self.MAX_RELATIONS_PER_SEED
            for seed_identity in self._relation_seed_identities(seed):
                if relation_rows_remaining <= 0:
                    break
                relations = self.relation_store.relations_of(
                    seed_identity,
                    tenant_id=str(kwargs["tenant_id"]),
                    owner_user_id=(
                        str(kwargs["owner_user_id"])
                        if kwargs["owner_user_id"] is not None
                        else None
                    ),
                    limit=relation_rows_remaining,
                )
                bounded_relations = relations[:relation_rows_remaining]
                relation_rows_remaining -= len(bounded_relations)
                for relation in bounded_relations:
                    target_uri = (
                        relation.target_uri
                        if relation.source_uri == seed_identity
                        else relation.source_uri
                    )
                    target_filters = self._target_identity_filters(
                        self._filters(plan),
                        (target_uri,),
                        limit=self.MAX_RECORDS_PER_RELATION_TARGET,
                    )
                    raw_records: Any = lister(
                        tenant_id=plan.tenant_id or "default",
                        filters=target_filters,
                        limit=min(
                            self.MAX_RECORDS_PER_RELATION_TARGET,
                            plan.candidate_limit - len(result),
                        ),
                    )
                    records = raw_records if isinstance(raw_records, Sequence) else ()
                    for record in records:
                        if not isinstance(record, CatalogRecord) or record.record_key in seen_record_keys:
                            continue
                        result.append(
                            self._from_record(
                                record,
                                branch="relation",
                                score=max(0.0, min(1.0, float(relation.weight))),
                            )
                        )
                        seen_record_keys.add(record.record_key)
                        if len(result) >= plan.candidate_limit:
                            return tuple(result)
        return tuple(result)

    @classmethod
    def _relation_seed_identities(cls, seed: RetrievalCandidate) -> tuple[str, ...]:
        return (seed.uri,)

    @staticmethod
    def _target_identity_filters(
        filters: Mapping[str, Any],
        target_uris: Sequence[str],
        *,
        limit: int,
    ) -> dict[str, Any]:
        """Bind stable serving identities to one exact SQL lookup."""

        exact_filters = dict(filters)
        exact_filters.pop("target_uris", None)
        exact_filters["target_identity_uris"] = tuple(target_uris)
        exact_filters["_identity_candidate_limit"] = int(limit)
        return exact_filters

    @staticmethod
    def _supported_kwargs(function: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return {}
        return {key: value for key, value in kwargs.items() if key in signature.parameters}


__all__ = ["CandidateGenerationResult", "CandidateGenerator"]
