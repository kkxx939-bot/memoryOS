"""The single online retrieval orchestration chain."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.candidate_generator import CandidateGenerator
from memoryos.contextdb.retrieval.canonical_resolver import (
    SOURCE_READ_BOUND_ALLOWANCE,
    BoundedCanonicalResolver,
)
from memoryos.contextdb.retrieval.errors import CatalogCandidateBoundExceeded
from memoryos.contextdb.retrieval.fusion import FusionRanker, RetrievalCandidate
from memoryos.contextdb.retrieval.packing import ContextPacker
from memoryos.contextdb.retrieval.query_plan import (
    CanonicalResolutionMode,
    RetrievalQueryIntent,
    RetrievalQueryPlan,
)
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import (
    IndexStore,
    SourceStore,
    is_canonical_memory_object,
)
from memoryos.memory.canonical.projection_state import ProjectionRecordStore
from memoryos.memory.canonical.visibility import committed_content, read_committed_pending
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.providers.rerank import Reranker
from memoryos.security.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)


class RetrievalUnavailableError(RuntimeError):
    """The bounded serving chain cannot safely distinguish empty from unavailable."""

    def __init__(self, reason: str, *, degraded_modes: Sequence[str] = ()) -> None:
        self.reason = str(reason)
        self.degraded_modes = tuple(dict.fromkeys(str(item) for item in degraded_modes if str(item)))
        suffix = f" ({','.join(self.degraded_modes)})" if self.degraded_modes else ""
        super().__init__(f"context retrieval unavailable: {self.reason}{suffix}")


class _LegacyCatalogAdapter:
    """Expose the independent flat Catalog reader to CandidateGenerator."""

    def __init__(self, store: Any) -> None:
        self.store = store

    @property
    def fts_enabled(self) -> bool:
        return bool(getattr(self.store, "fts_enabled", False))

    def list_catalog(self, *, filters: Mapping[str, Any], limit: int) -> Any:
        return self.store.list_legacy_catalog(filters=filters, limit=limit)

    def search_catalog(
        self,
        query: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> Any:
        return self.store.search_legacy_catalog(query, filters=filters, limit=limit)

    def get_catalog(self, record_key: str, *, tenant_id: str | None = None) -> Any:
        return self.store.get_catalog(record_key, tenant_id=tenant_id)


@dataclass(frozen=True)
class RetrievalMetrics:
    structured_candidates: int = 0
    exact_candidates: int = 0
    fts_candidates: int = 0
    vector_candidates: int = 0
    relation_candidates: int = 0
    fusion_candidates: int = 0
    rerank_count: int = 0
    canonical_candidates: int = 0
    canonical_validated: int = 0
    source_reads: int = 0
    selected_count: int = 0
    dropped_count: int = 0
    vector_overfetch: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class UnifiedRetrievalResult:
    plan: RetrievalQueryPlan
    contexts: tuple[dict[str, Any], ...]
    dropped_contexts: tuple[dict[str, Any], ...]
    load_plan: tuple[dict[str, Any], ...]
    metrics: RetrievalMetrics
    degraded_modes: tuple[str, ...] = ()
    reranker_fallback: bool = False
    total_budget: int = 0
    used_tokens: int = 0
    remaining_tokens: int = 0

    def search_payload(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for item in self.contexts:
            payload = dict(item)
            metadata = dict(payload.get("metadata") or {})
            if (
                str(payload.get("canonical_validation_status") or "").startswith("validated")
                and str(metadata.get("record_kind") or "") == "current_slot"
            ):
                # Keep the historical SDK marker while ``selected_layer``
                # continues to report the actual L0/L1/L2 packing choice.
                payload["layer"] = "canonical_source"
            projection_record = metadata.get("projection_record")
            if isinstance(projection_record, Mapping):
                # Preserve the pre-unification SDK response shape without
                # trusting arbitrary Catalog metadata: the resolver only adds
                # this field after exact canonical proof validation.
                payload["projection_record"] = dict(projection_record)
            payloads.append(payload)
        return payloads

    def assemble_payload(self) -> dict[str, Any]:
        return {
            "contexts": self.search_payload(),
            "dropped_contexts": [dict(item) for item in self.dropped_contexts],
            "load_plan": [dict(item) for item in self.load_plan],
            "total_budget": self.total_budget,
            "used_tokens": self.used_tokens,
            "remaining_tokens": self.remaining_tokens,
            "metrics": self.metrics.to_dict(),
            "degraded_modes": list(self.degraded_modes),
            "query_plan": self.plan.to_dict(),
            "reranker_fallback": self.reranker_fallback,
        }


class UnifiedRetrievalOrchestrator:
    """Plan-bound structured→exact→FTS→vector→relation→fusion chain."""

    def __init__(
        self,
        context_db: ContextDB,
        *,
        vector_store: Any = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        projection_store: ProjectionRecordStore | None = None,
    ) -> None:
        self.context_db = context_db
        self.reranker = reranker
        self.sanitizer = ContextProjectionSanitizer()
        index_store = cast(IndexStore, getattr(context_db, "index_store", context_db))
        self.index_store = index_store
        relation_store = getattr(context_db, "relation_store", None)
        source_store = cast(SourceStore, getattr(context_db, "source_store", None))
        self.generator = CandidateGenerator(
            index_store,
            relation_store=relation_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            sanitizer=self.sanitizer,
        )
        self.legacy_generator: CandidateGenerator | None = None
        if callable(getattr(index_store, "list_legacy_catalog", None)) and callable(
            getattr(index_store, "search_legacy_catalog", None)
        ):
            self.legacy_generator = CandidateGenerator(
                cast(IndexStore, _LegacyCatalogAdapter(index_store))
            )
        self.fusion = FusionRanker()
        self.resolver = BoundedCanonicalResolver(
            source_store,
            relation_store,
            projection_store=projection_store or getattr(context_db, "projection_store", None),
        )
        self.packer = ContextPacker()

    def execute(self, plan: RetrievalQueryPlan) -> UnifiedRetrievalResult:
        serving_lock = getattr(self.context_db, "serving_lock", None)
        if serving_lock is None:
            return self._execute_locked(plan)
        with serving_lock:
            return self._execute_locked(plan)

    def _execute_locked(self, plan: RetrievalQueryPlan) -> UnifiedRetrievalResult:
        require_ready = getattr(self.context_db, "_require_ready", None)
        if callable(require_ready):
            require_ready()
        serving_generation = self._serving_generation_token()
        self._require_derived_rebuild_available()
        self._require_current_projection_ready(plan)
        session_lag_modes = self._session_projection_lag_modes(plan)
        migration_mode = self._migration_mode()
        migration_route = self._migration_read_route()
        active_generator = self.generator
        if migration_route in {"LEGACY", "SHADOW"}:
            if self.legacy_generator is None:
                raise RetrievalUnavailableError(
                    "legacy compatibility reader is unavailable",
                    degraded_modes=(migration_mode,),
                )
            active_generator = self.legacy_generator
        try:
            generated = active_generator.generate(plan)
        except CatalogCandidateBoundExceeded as exc:
            raise RetrievalUnavailableError(
                "structured candidate generation exceeded its online scan bound",
                degraded_modes=("structured_candidate_bound_exhausted",),
            ) from exc
        if migration_route == "SHADOW":
            try:
                unified_shadow = self.generator.generate(plan)
            except CatalogCandidateBoundExceeded as exc:
                raise RetrievalUnavailableError(
                    "unified shadow candidate generation exceeded its online scan bound",
                    degraded_modes=("shadow_unified_candidate_bound_exhausted",),
                ) from exc
            self._record_shadow_comparison(plan, legacy=generated, unified=unified_shadow)
        self._require_explicit_current_slot_candidate(plan, generated.branches.get("exact", ()))
        vector_failures = tuple(mode for mode in generated.degraded_modes if str(mode).startswith("vector_fallback:"))
        non_vector_candidates = sum(len(items) for branch, items in generated.branches.items() if branch != "vector")
        if vector_failures and non_vector_candidates == 0:
            raise RetrievalUnavailableError(
                "vector backend failed and no bounded exact, FTS, structured, or relation fallback exists",
                degraded_modes=vector_failures,
            )
        fused = self.fusion.fuse(generated.branches, plan=plan)
        reranked, reranker_fallback = self._rerank(plan, fused)

        if plan.canonical_resolution_mode == CanonicalResolutionMode.DISABLED:
            ordinary = tuple(item for item in reranked if not item.canonical_slot_id and not item.canonical_claim_id)
            canonical_dropped = tuple(
                {
                    "record_key": item.record_key,
                    "uri": item.uri,
                    "drop_reason": "canonical_resolution_disabled",
                    "canonical_validation_status": "disabled",
                }
                for item in reranked
                if item.canonical_slot_id or item.canonical_claim_id
            )
            resolved_candidates = ordinary
            canonical_candidates = len(canonical_dropped)
            canonical_validated = 0
            canonical_source_reads = 0
        else:
            resolution = self.resolver.resolve(reranked, plan=plan)
            resolved_candidates = resolution.candidates
            canonical_dropped = resolution.dropped
            canonical_candidates = resolution.canonical_candidates
            canonical_validated = resolution.canonical_validated
            canonical_source_reads = resolution.source_reads

        blocking_current_drops = tuple(
            item
            for item in canonical_dropped
            if item.get("canonical_validation_status") in {"stale", "unavailable"}
        )
        if (
            plan.query_intent is RetrievalQueryIntent.CURRENT
            and plan.canonical_resolution_mode is not CanonicalResolutionMode.DISABLED
            and blocking_current_drops
        ):
            raise RetrievalUnavailableError(
                "Canonical Current candidate failed bounded authoritative validation",
                degraded_modes=("stale_canonical_current_projection",),
            )

        l2_hydration_keys = self.packer.l2_hydration_record_keys(
            resolved_candidates,
            plan=plan,
        )
        hydrated, hydration_reads, hydration_modes = self._hydrate_legacy(
            resolved_candidates,
            plan=plan,
            l2_hydration_keys=frozenset(l2_hydration_keys),
            source_read_budget=max(
                0,
                plan.candidate_limit
                + SOURCE_READ_BOUND_ALLOWANCE
                - canonical_source_reads
                - generated.source_reads,
            ),
        )
        degraded_modes = tuple(
            dict.fromkeys(
                (
                    *generated.degraded_modes,
                    *session_lag_modes,
                    *hydration_modes,
                    *((migration_mode,) if migration_mode else ()),
                    *(
                        ("canonical_validation_bound",)
                        if any(
                            item.get("canonical_validation_status") == "not_validated_bound"
                            for item in canonical_dropped
                        )
                        else ()
                    ),
                    *(("reranker_fallback",) if reranker_fallback else ()),
                )
            )
        )
        if degraded_modes:
            hydrated = tuple(
                replace(
                    item,
                    metadata={
                        **dict(item.metadata),
                        "degraded_mode": self._merge_degraded_modes(
                            str(item.metadata.get("degraded_mode") or ""),
                            *degraded_modes,
                        ),
                    },
                )
                for item in hydrated
            )
        packed = self.packer.pack(hydrated, plan=plan)
        unavailable_modes = tuple(
            mode
            for mode in generated.degraded_modes
            if str(mode).startswith("vector_fallback:") or str(mode) == "fts_unavailable"
        )
        if unavailable_modes and not packed["contexts"]:
            raise RetrievalUnavailableError(
                "retrieval backends failed and no validated bounded fallback result remains",
                degraded_modes=unavailable_modes,
            )
        migration_gate = getattr(self.context_db, "migration_gate", None)
        empty_requirement = getattr(migration_gate, "empty_result_requires_unavailable", False)
        empty_requires_unavailable = bool(empty_requirement() if callable(empty_requirement) else empty_requirement)
        if not packed["contexts"] and empty_requires_unavailable:
            raise RetrievalUnavailableError(
                "legacy migration read is incomplete and an empty result cannot be proved",
                degraded_modes=(migration_mode,),
            )
        if session_lag_modes and not packed["contexts"]:
            raise RetrievalUnavailableError(
                "Session projection is lagging or failed and an empty result cannot be proved",
                degraded_modes=session_lag_modes,
            )
        dropped = (*canonical_dropped, *tuple(packed["dropped_contexts"]))
        metrics = RetrievalMetrics(
            structured_candidates=generated.structured_candidates,
            exact_candidates=generated.exact_candidates,
            fts_candidates=generated.fts_candidates,
            vector_candidates=generated.vector_candidates,
            relation_candidates=generated.relation_candidates,
            fusion_candidates=len(fused),
            rerank_count=len(reranked) if self.reranker is not None else 0,
            canonical_candidates=canonical_candidates,
            canonical_validated=canonical_validated,
            source_reads=(generated.source_reads + canonical_source_reads + hydration_reads),
            selected_count=int(packed["selected_count"]),
            dropped_count=len(dropped),
            vector_overfetch=generated.vector_overfetch,
        )
        if metrics.canonical_validated > plan.candidate_limit:
            raise RuntimeError("canonical validation exceeded candidate_limit")
        if metrics.source_reads > plan.candidate_limit + SOURCE_READ_BOUND_ALLOWANCE:
            raise RuntimeError("source reads exceeded candidate_limit plus bounded allowance")
        self._assert_serving_generation_unchanged(serving_generation)
        return UnifiedRetrievalResult(
            plan=plan,
            contexts=tuple(packed["contexts"]),
            dropped_contexts=tuple(dropped),
            load_plan=tuple(packed["load_plan"]),
            metrics=metrics,
            degraded_modes=degraded_modes,
            reranker_fallback=reranker_fallback,
            total_budget=int(packed["total_budget"]),
            used_tokens=int(packed["used_tokens"]),
            remaining_tokens=int(packed["remaining_tokens"]),
        )

    @staticmethod
    def _require_explicit_current_slot_candidate(
        plan: RetrievalQueryPlan,
        exact_candidates: Sequence[RetrievalCandidate],
    ) -> None:
        """Fail closed when an explicitly addressed CurrentSlot is absent.

        A healthy projection queue cannot prove that one particular stable
        Slot was projected.  For an explicit Canonical Slot identity, an empty
        exact branch is therefore "unavailable", not a proved empty CURRENT
        result.  Ordinary targets and semantic misses retain normal empty
        result semantics.
        """

        if (
            plan.query_intent is not RetrievalQueryIntent.CURRENT
            or plan.canonical_resolution_mode is CanonicalResolutionMode.DISABLED
        ):
            return
        requested = {
            str(uri)
            for uri in plan.target_uris
            if "/memories/canonical/slots/" in str(uri) and "/claims/" not in str(uri)
        }
        if not requested:
            return
        matched: set[str] = set()
        for candidate in exact_candidates:
            metadata = dict(candidate.metadata or {})
            matched.update(
                value
                for value in (
                    candidate.uri,
                    str(metadata.get("canonical_slot_uri") or ""),
                )
                if value
            )
        missing = requested.difference(matched)
        if missing:
            raise RetrievalUnavailableError(
                "explicit Canonical Current Slot projection is missing",
                degraded_modes=("missing_canonical_current_projection",),
            )

    def _require_current_projection_ready(self, plan: RetrievalQueryPlan) -> None:
        """Never serve a potentially stale/missing CURRENT row behind projection lag."""

        if (
            plan.query_intent is not RetrievalQueryIntent.CURRENT
            or plan.canonical_resolution_mode is CanonicalResolutionMode.DISABLED
        ):
            return
        queue_store = getattr(self.context_db, "queue_store", None)
        stats = getattr(queue_store, "stats", None)
        if not callable(stats):
            return
        scoped_stats = getattr(queue_store, "stats_for_scope", None)
        if plan.owner_user_id and callable(scoped_stats):
            workspace_ids: tuple[str, ...] | None = None
            if plan.workspace_ids:
                workspace_ids = (
                    ("",)
                    if plan.workspace_ids == ("__memoryos_principal_only__",)
                    else tuple(dict.fromkeys(("", *plan.workspace_ids)))
                )
            values = scoped_stats(
                queue_name="memory_projection",
                tenant_id=str(plan.tenant_id or "default"),
                owner_user_id=plan.owner_user_id,
                workspace_ids=workspace_ids,
            )
        elif plan.owner_user_id:
            raise RetrievalUnavailableError("owner-scoped canonical projection health is unavailable")
        else:
            # A non-principal service/public query cannot safely attribute a
            # tenant-global pending count to visible state.  Check globally,
            # but never expose the cross-owner count in the mode label.
            values = stats(queue_name="memory_projection")
        if not isinstance(values, Mapping):
            raise RetrievalUnavailableError("canonical projection queue health is unavailable")
        try:
            unresolved = {
                status: int(values.get(status, 0) or 0)
                for status in ("pending", "leased", "dead_letter", "quarantine")
                if int(values.get(status, 0) or 0) > 0
            }
        except (TypeError, ValueError):
            raise RetrievalUnavailableError("canonical projection queue health is invalid") from None
        if unresolved:
            modes = tuple(
                (
                    f"canonical_projection_{status}:{count}"
                    if plan.owner_user_id
                    else f"canonical_projection_{status}"
                )
                for status, count in sorted(unresolved.items())
            )
            raise RetrievalUnavailableError(
                "Canonical Current projection is lagging or failed",
                degraded_modes=modes,
            )

    def _session_projection_lag_modes(self, plan: RetrievalQueryPlan) -> tuple[str, ...]:
        """Expose durable Session Catalog lag without silently returning empty."""

        if plan.context_types and not {
            ContextType.SESSION,
            ContextType.RESOURCE,
        }.intersection(plan.context_types):
            return ()
        if plan.target_paths and all(
            path.startswith(("memories/", "skills/", "agents/")) for path in plan.target_paths
        ):
            return ()
        index_store = getattr(self.context_db, "index_store", None)
        summary_reader = getattr(index_store, "get_session_projection_frontier_summary", None)
        if not callable(summary_reader):
            return ()
        workspace_ids: tuple[str, ...] | None = None
        if plan.workspace_ids:
            workspace_ids = (
                ("",)
                if plan.workspace_ids == ("__memoryos_principal_only__",)
                else tuple(dict.fromkeys(("", *plan.workspace_ids)))
            )
        values = summary_reader(
            tenant_id=str(plan.tenant_id or "default"),
            owner_user_id=plan.owner_user_id if plan.owner_user_id is not None else "",
            workspace_ids=workspace_ids,
        )
        if not isinstance(values, Mapping):
            raise RetrievalUnavailableError("Session projection frontier health is unavailable")
        try:
            unresolved = {
                status.lower(): int(values.get(status, 0) or 0)
                for status in ("PENDING", "FAILED")
                if int(values.get(status, 0) or 0) > 0
            }
        except (TypeError, ValueError):
            raise RetrievalUnavailableError("Session projection frontier health is invalid") from None
        return tuple(
            (
                f"session_projection_{status}:{count}"
                if plan.owner_user_id
                else f"session_projection_{status}"
            )
            for status, count in sorted(unresolved.items())
        )

    def _migration_mode(self) -> str:
        coordinator = getattr(self.context_db, "migration_gate", None)
        if coordinator is None:
            return ""
        feature_gate = getattr(coordinator, "feature_gate", None)
        route = str(getattr(getattr(feature_gate, "read_route", None), "value", ""))
        state = str(getattr(getattr(feature_gate, "state", None), "value", ""))
        if route == "LEGACY":
            # The catalog schema evolves the existing contexts table in place, so
            # the Unified planner can compatibly read pre-backfill rows without
            # restoring the forbidden filesystem/source scan.
            return f"migration_legacy_compatible_read:{state or 'UNKNOWN'}"
        if route == "SHADOW":
            return f"migration_shadow_read:{state or 'UNKNOWN'}"
        return ""

    def _require_derived_rebuild_available(self) -> None:
        coordinator = getattr(self.context_db, "migration_gate", None)
        raw = getattr(coordinator, "derived_rebuild_requires_unavailable", False)
        blocked = bool(raw() if callable(raw) else raw)
        if blocked:
            raise RetrievalUnavailableError(
                "derived serving rebuild is incomplete",
                degraded_modes=("derived_serving_rebuild_incomplete",),
            )

    def _serving_generation_token(self) -> str:
        coordinator = getattr(self.context_db, "migration_gate", None)
        raw = getattr(coordinator, "serving_generation_token", "")
        value = raw() if callable(raw) else raw
        return str(value or "")

    def _assert_serving_generation_unchanged(self, expected: str) -> None:
        self._require_derived_rebuild_available()
        current = self._serving_generation_token()
        if current != expected:
            raise RetrievalUnavailableError(
                "derived serving generation changed during retrieval",
                degraded_modes=("derived_serving_generation_changed",),
            )

    def _migration_read_route(self) -> str:
        coordinator = getattr(self.context_db, "migration_gate", None)
        feature_gate = getattr(coordinator, "feature_gate", None)
        return str(getattr(getattr(feature_gate, "read_route", None), "value", "UNIFIED"))

    def _record_shadow_comparison(
        self,
        plan: RetrievalQueryPlan,
        *,
        legacy: Any,
        unified: Any,
    ) -> None:
        """Journal bounded old/new fused identities without persisting query text."""

        coordinator = getattr(self.context_db, "migration_gate", None)
        recorder = getattr(coordinator, "record_shadow_read_comparison", None)
        if not callable(recorder):
            raise RetrievalUnavailableError("shadow read result journal is unavailable")

        def identities(generated: Any) -> tuple[str, ...]:
            fused = self.fusion.fuse(generated.branches, plan=plan)
            return tuple(
                self.sanitizer.digest(
                    {
                        "record_key": item.record_key,
                        "uri": item.uri,
                        "slot_id": item.canonical_slot_id,
                        "claim_id": item.canonical_claim_id,
                        "revision": item.canonical_revision,
                    }
                )
                for item in fused[: plan.final_limit]
            )

        legacy_ids = identities(legacy)
        unified_ids = identities(unified)
        legacy_digest = self.sanitizer.digest(list(legacy_ids))
        unified_digest = self.sanitizer.digest(list(unified_ids))
        recorder(
            {
                "plan_digest": self.sanitizer.digest(plan.to_dict()),
                "legacy_count": len(legacy_ids),
                "unified_count": len(unified_ids),
                "overlap_count": len(set(legacy_ids).intersection(unified_ids)),
                "legacy_digest": legacy_digest,
                "unified_digest": unified_digest,
                # The state store recomputes this; keep it for trace/debug
                # compatibility but never trust the caller's assertion.
                "matched": legacy_ids == unified_ids,
            }
        )

    def _rerank(
        self,
        plan: RetrievalQueryPlan,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[list[RetrievalCandidate], bool]:
        bounded = list(candidates[: plan.candidate_limit])
        if self.reranker is None or not bounded:
            return bounded, False
        payload = [
            {
                "record_key": item.record_key,
                "uri": item.uri,
                "title": item.title,
                "text": item.l1_text or item.l0_text or item.title,
                "score": item.score.final_score,
                "metadata": dict(item.metadata),
            }
            for item in bounded
        ]
        try:
            safe_query = self.sanitizer.sanitize(
                title="retrieval query",
                l1_text=plan.semantic_query,
                metadata={},
                source_kind="retrieval_query",
            ).l1_text
            safe_payload = self.sanitizer.sanitize_trace(payload)
            if not isinstance(safe_payload, list):
                raise TypeError("sanitized reranker payload must be a list")
            reranked = self.reranker.rerank(safe_query, safe_payload)
            if not isinstance(reranked, list):
                raise TypeError("reranker must return a list")
            positions: dict[str, int] = {}
            for index, item in enumerate(reranked):
                if not isinstance(item, Mapping):
                    raise TypeError("reranker item must be a mapping")
                key = str(item.get("record_key") or "")
                if key and key not in positions:
                    positions[key] = index
            if not positions:
                raise ValueError("reranker returned no candidate identities")
            total = max(1, len(positions))
            scores = {key: 1.0 - (position / total) for key, position in positions.items()}
            return self.fusion.apply_rerank(bounded, scores), False
        except Exception:
            # Deterministic Fusion order is a complete, observable fallback.
            return bounded, True

    def _hydrate_legacy(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
        l2_hydration_keys: frozenset[str],
        source_read_budget: int,
    ) -> tuple[tuple[RetrievalCandidate, ...], int, tuple[str, ...]]:
        result: list[RetrievalCandidate] = []
        reads = 0
        degraded_modes: list[str] = []
        for item in candidates:
            if item.canonical_slot_id or item.canonical_claim_id:
                result.append(item)
                continue
            if item.record_key in l2_hydration_keys:
                hydrated, used_reads, mode = self._hydrate_resource_l2(
                    item,
                    plan=plan,
                    source_read_budget=max(0, source_read_budget - reads),
                )
                reads += used_reads
                if mode:
                    degraded_modes.append(mode)
                result.append(hydrated)
                continue
            if item.l0_text or item.l1_text:
                if (
                    reads < source_read_budget
                    and item.record_kind in {"session_root", "semantic_segment"}
                    and item.l2_uri
                ):
                    reads += 1
                    full = self._read_session_l2(item)
                    if full is not None:
                        result.append(replace(item, text=full))
                        continue
                result.append(item)
                continue
            # Atomic Session/Tool nodes are serving projections of immutable
            # SessionArchive evidence.  Never reinterpret their archive URI as
            # a generic SourceStore object or read raw message/tool payloads.
            if item.context_type == ContextType.SESSION.value:
                result.append(item)
                continue
            if reads + 1 > source_read_budget:
                mode = "source_read_bound"
                degraded_modes.append(mode)
                result.append(self._mark_degraded(item, mode))
                continue
            source_store = getattr(self.context_db, "source_store", None)
            if source_store is None:
                result.append(item)
                continue
            reads += 1
            committed_pending = None
            try:
                if (
                    plan.query_intent == RetrievalQueryIntent.OPTIONS
                    and plan.metadata_filters.get("include_candidates")
                    and str(item.metadata.get("canonical_kind") or "") == "pending_proposal"
                ):
                    committed_pending = read_committed_pending(
                        source_store,
                        item.uri,
                        getattr(self.context_db, "relation_store", None),
                    )
                    obj = committed_pending.object
                else:
                    obj = source_store.read_object(item.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            review_candidate = bool(
                plan.query_intent == RetrievalQueryIntent.OPTIONS
                and plan.metadata_filters.get("include_candidates")
                and str(item.metadata.get("canonical_kind") or "") == "pending_proposal"
                and obj.lifecycle_state
                in {
                    LifecycleState.PENDING,
                    LifecycleState.RETRYABLE,
                    LifecycleState.CONFIRMED,
                }
            )
            if obj.lifecycle_state != LifecycleState.ACTIVE and not review_candidate:
                continue
            if str(obj.tenant_id or "default") != str(plan.tenant_id or "default"):
                continue
            if plan.owner_user_id and str(obj.owner_user_id or "") not in {"", plan.owner_user_id}:
                continue
            if committed_pending is not None:
                content = committed_content(committed_pending)
            elif reads + 1 > source_read_budget:
                content = obj.title
            else:
                reads += 1
                try:
                    content = source_store.read_content(obj.layers.l2_uri or obj.uri)
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    content = obj.title
            safe = self.sanitizer.sanitize(
                title=obj.title,
                l0_text=obj.title,
                l1_text=content,
                metadata=obj.metadata,
                source_kind=str(obj.metadata.get("source_kind") or "context"),
            )
            result.append(
                replace(
                    item,
                    title=safe.title,
                    l0_text=safe.l0_text,
                    l1_text=safe.l1_text,
                    source_uri=item.source_uri or item.uri,
                    metadata={**dict(item.metadata), **safe.metadata},
                )
            )
        return tuple(result), reads, tuple(dict.fromkeys(degraded_modes))

    def _hydrate_resource_l2(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        source_read_budget: int,
    ) -> tuple[RetrievalCandidate, int, str]:
        """Read one exact ordinary Resource L2 after bounded final preselection."""

        if source_read_budget < 2:
            mode = "l2_source_read_bound"
            return self._mark_degraded(item, mode), 0, mode
        source_store = getattr(self.context_db, "source_store", None)
        if source_store is None:
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), 0, mode

        reads = 1
        try:
            obj = source_store.read_object(item.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, RuntimeError, ValueError):
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), reads, mode

        try:
            source_record = CatalogRecord.from_context_object(obj)
        except (TypeError, ValueError):
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        candidate_owner = str(item.metadata.get("owner_user_id") or "")
        candidate_connect = item.metadata.get("connect")
        candidate_adapter = str(
            item.metadata.get("adapter_id")
            or (candidate_connect.get("adapter_id") if isinstance(candidate_connect, Mapping) else "")
            or ""
        )
        if (
            is_canonical_memory_object(obj)
            or obj.uri != item.uri
            or obj.context_type != ContextType.RESOURCE
            or obj.lifecycle_state != LifecycleState.ACTIVE
            or source_record.record_kind != CatalogRecordKind.CONTEXT.value
            or source_record.canonical_slot_id
            or source_record.canonical_claim_id
            or str(obj.tenant_id or "default") != str(plan.tenant_id or "default")
            or source_record.owner_user_id != candidate_owner
            or source_record.workspace_id != item.workspace_id
            or source_record.adapter_id != candidate_adapter
            or source_record.source_kind != item.source_kind
            or source_record.source_uri != (item.source_uri or item.uri)
        ):
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        if plan.owner_user_id is None:
            owner_allowed = not source_record.owner_user_id
        else:
            owner_allowed = source_record.owner_user_id in {"", plan.owner_user_id}
        if not owner_allowed:
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        if plan.workspace_ids:
            allowed_workspaces = (
                {""}
                if plan.workspace_ids == ("__memoryos_principal_only__",)
                else {"", *plan.workspace_ids}
            )
            if source_record.workspace_id not in allowed_workspaces:
                mode = "l2_source_authority_mismatch"
                return self._mark_degraded(item, mode), reads, mode

        source_l2_uri = str(obj.layers.l2_uri or source_record.source_uri or obj.uri)
        candidate_l2_uri = str(item.l2_uri or item.source_uri or item.uri)
        if source_l2_uri != candidate_l2_uri:
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode

        reads += 1
        try:
            content = source_store.read_content(source_l2_uri)
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
            RuntimeError,
            UnicodeError,
            ValueError,
        ):
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), reads, mode
        expected_digest = str(item.source_digest or "")
        actual_digest = self.sanitizer.digest(content or obj.to_dict())
        if not expected_digest or not hmac.compare_digest(expected_digest, actual_digest):
            mode = "l2_source_revision_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        try:
            safe = self.sanitizer.sanitize(
                title=item.title,
                l0_text=item.l0_text,
                l1_text=content,
                metadata={},
                source_kind=ContextType.RESOURCE.value,
            )
        except ContextProjectionSanitizationError:
            mode = "l2_sanitization_failed"
            return self._mark_degraded(item, mode), reads, mode
        if not safe.l1_text:
            mode = "l2_source_empty"
            return self._mark_degraded(item, mode), reads, mode
        return (
            replace(
                item,
                text=safe.l1_text,
                metadata={
                    **dict(item.metadata),
                    "l2_hydrated": True,
                    "l2_projection_redacted": safe.redacted,
                    "l2_projection_truncated": safe.truncated,
                },
            ),
            reads,
            "",
        )

    @staticmethod
    def _mark_degraded(item: RetrievalCandidate, mode: str) -> RetrievalCandidate:
        return replace(
            item,
            metadata={
                **dict(item.metadata),
                "degraded_mode": UnifiedRetrievalOrchestrator._merge_degraded_modes(
                    str(item.metadata.get("degraded_mode") or ""),
                    mode,
                ),
            },
        )

    @staticmethod
    def _merge_degraded_modes(existing: str, *modes: str) -> str:
        return ",".join(
            dict.fromkeys(
                value
                for raw in (existing, *modes)
                for value in str(raw).split(",")
                if value
            )
        )

    def _read_session_l2(self, item: RetrievalCandidate) -> str | None:
        service = getattr(self.context_db, "session_commit_service", None)
        archive_store = getattr(service, "archive_store", None)
        reader = getattr(archive_store, "read_archive", None)
        if not callable(reader):
            return None
        archive_uri = str(item.metadata.get("archive_uri") or "")
        if not archive_uri:
            return None
        try:
            archive = cast(
                SessionArchive,
                reader(
                    archive_uri,
                    tenant_id=str(item.metadata.get("tenant_id") or "") or None,
                ),
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
            return None
        payload = {
            "messages": archive.messages,
            "tool_results": archive.tool_results,
            "observations": archive.observations,
            "action_results": archive.action_results,
            "used_contexts": archive.used_contexts,
            "used_skills": archive.used_skills,
        }
        safe = self.sanitizer.sanitize(
            title=item.title,
            l0_text=item.l0_text,
            l1_text=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            metadata={},
            source_kind="session",
        )
        return safe.l1_text


__all__ = [
    "RetrievalMetrics",
    "RetrievalUnavailableError",
    "UnifiedRetrievalOrchestrator",
    "UnifiedRetrievalResult",
]
