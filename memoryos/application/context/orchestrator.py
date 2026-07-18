"""The single bounded public context retrieval application chain."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from memoryos.application.context.candidate_generator import CandidateGenerator
from memoryos.application.context.packing import ContextPacker
from memoryos.application.context.reranking import Reranker
from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.retrieval.errors import CatalogCandidateBoundExceeded
from memoryos.contextdb.retrieval.fusion import FusionRanker, RetrievalCandidate
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryPlan
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.queue_store import QueueJob
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.ids import stable_hash
from memoryos.memory.documents.context_overlay import MemoryDocumentContextOverlay
from memoryos.memory.documents.store import DocumentConflictError, DocumentNotFoundError, DocumentUnsafeError
from memoryos.security.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)

SOURCE_READ_BOUND_ALLOWANCE = 8


class RetrievalUnavailableError(RuntimeError):
    """The bounded serving chain cannot safely distinguish empty from unavailable."""

    def __init__(self, reason: str, *, degraded_modes: Sequence[str] = ()) -> None:
        self.reason = str(reason)
        self.degraded_modes = tuple(dict.fromkeys(str(item) for item in degraded_modes if str(item)))
        suffix = f" ({','.join(self.degraded_modes)})" if self.degraded_modes else ""
        super().__init__(f"context retrieval unavailable: {self.reason}{suffix}")


@dataclass(frozen=True)
class RetrievalMetrics:
    structured_candidates: int = 0
    exact_candidates: int = 0
    fts_candidates: int = 0
    vector_candidates: int = 0
    relation_candidates: int = 0
    fusion_candidates: int = 0
    rerank_count: int = 0
    memory_candidates: int = 0
    memory_validated: int = 0
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
        return [dict(item) for item in self.contexts]

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
    """structured → exact → FTS → vector → relation → fusion → source hydration."""

    def __init__(
        self,
        context_db: ContextDB,
        *,
        vector_store: Any = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        document_overlay: MemoryDocumentContextOverlay | None = None,
    ) -> None:
        self.context_db = context_db
        self.reranker = reranker
        self.sanitizer = ContextProjectionSanitizer()
        index_store = cast(IndexStore, getattr(context_db, "index_store", context_db))
        self.index_store = index_store
        self.generator = CandidateGenerator(
            index_store,
            relation_store=getattr(context_db, "relation_store", None),
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            sanitizer=self.sanitizer,
        )
        self.fusion = FusionRanker()
        self.packer = ContextPacker()
        self.document_overlay = document_overlay or getattr(context_db, "memory_document_overlay", None)

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
        generation_before = self._serving_generation_token()
        try:
            generated = self.generator.generate(plan)
        except CatalogCandidateBoundExceeded as exc:
            raise RetrievalUnavailableError(
                "structured candidate generation exceeded its online bound",
                degraded_modes=("structured_candidate_bound_exhausted",),
            ) from exc
        vector_failures = tuple(mode for mode in generated.degraded_modes if str(mode).startswith("vector_fallback:"))
        non_vector_candidates = sum(len(items) for branch, items in generated.branches.items() if branch != "vector")
        if vector_failures and non_vector_candidates == 0:
            raise RetrievalUnavailableError(
                "vector backend failed and no bounded SQL/FTS/relation fallback exists",
                degraded_modes=vector_failures,
            )
        fused = self.fusion.fuse(generated.branches, plan=plan)
        reranked, reranker_fallback = self._rerank(plan, fused)
        hydrated, source_reads, hydration_modes, hydration_drops, memory_validated = self._hydrate(
            reranked,
            plan=plan,
            source_read_budget=max(0, plan.candidate_limit + SOURCE_READ_BOUND_ALLOWANCE - generated.source_reads),
        )
        degraded_modes = tuple(
            dict.fromkeys(
                (
                    *generated.degraded_modes,
                    *hydration_modes,
                    *(self._session_projection_lag_modes(plan)),
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
                "retrieval backends failed and no validated bounded fallback remains",
                degraded_modes=unavailable_modes,
            )
        dropped = (*hydration_drops, *tuple(packed["dropped_contexts"]))
        memory_candidates = sum(1 for item in reranked if item.document_id)
        metrics = RetrievalMetrics(
            structured_candidates=generated.structured_candidates,
            exact_candidates=generated.exact_candidates,
            fts_candidates=generated.fts_candidates,
            vector_candidates=generated.vector_candidates,
            relation_candidates=generated.relation_candidates,
            fusion_candidates=len(fused),
            rerank_count=len(reranked) if self.reranker is not None else 0,
            memory_candidates=memory_candidates,
            memory_validated=memory_validated,
            source_reads=generated.source_reads + source_reads,
            selected_count=int(packed["selected_count"]),
            dropped_count=len(dropped),
            vector_overfetch=generated.vector_overfetch,
        )
        if metrics.source_reads > plan.candidate_limit + SOURCE_READ_BOUND_ALLOWANCE:
            raise RuntimeError("source reads exceeded candidate_limit plus bounded allowance")
        self._assert_serving_generation_unchanged(generation_before)
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
            return bounded, True

    def _hydrate(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
        source_read_budget: int,
    ) -> tuple[
        tuple[RetrievalCandidate, ...],
        int,
        tuple[str, ...],
        tuple[dict[str, Any], ...],
        int,
    ]:
        result: list[RetrievalCandidate] = []
        dropped: list[dict[str, Any]] = []
        degraded_modes: list[str] = []
        reads = 0
        memory_validated = 0
        l2_resource_keys = frozenset(self.packer.l2_hydration_record_keys(candidates, plan=plan))
        memory_l2_remaining = self.packer.policy.max_l2_items
        for item in candidates:
            if item.document_id:
                if reads >= source_read_budget:
                    degraded_modes.append("memory_source_read_bound")
                    dropped.append(self._drop(item, "memory_source_read_bound"))
                    continue
                reads += 1
                hydrated, drop = self._hydrate_memory_document(
                    item,
                    plan=plan,
                    include_l2=memory_l2_remaining > 0,
                )
                if drop is not None:
                    dropped.append(drop)
                    self._schedule_document_rescan(item, plan=plan)
                    continue
                memory_validated += 1
                if hydrated.text:
                    memory_l2_remaining -= 1
                result.append(hydrated)
                continue
            if item.record_key in l2_resource_keys:
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
            if item.context_type == ContextType.SESSION.value:
                if item.record_kind in {"session_root", "semantic_segment", "session_l1"} and item.l2_uri:
                    if reads < source_read_budget:
                        reads += 1
                        full = self._read_session_l2(item, plan=plan)
                        if full is not None:
                            result.append(replace(item, text=full))
                            continue
                result.append(item)
                continue
            if item.l0_text or item.l1_text:
                result.append(item)
                continue
            if reads >= source_read_budget:
                degraded_modes.append("source_read_bound")
                result.append(self._mark_degraded(item, "source_read_bound"))
                continue
            ordinary_hydrated, used_reads = self._hydrate_ordinary(
                item,
                plan=plan,
                remaining=source_read_budget - reads,
            )
            reads += used_reads
            if ordinary_hydrated is not None:
                result.append(ordinary_hydrated)
        return (
            tuple(result),
            reads,
            tuple(dict.fromkeys(degraded_modes)),
            tuple(dropped),
            memory_validated,
        )

    def _hydrate_memory_document(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        include_l2: bool,
    ) -> tuple[RetrievalCandidate, dict[str, Any] | None]:
        overlay = self.document_overlay
        relative_path = str(item.metadata.get("relative_path") or "")
        if overlay is None or not relative_path or not item.source_digest:
            return item, self._drop(item, "memory_source_unavailable")
        try:
            view = overlay.read(
                tenant_id=str(plan.tenant_id or "default"),
                owner_user_id=str(plan.owner_user_id or item.owner_user_id),
                document_uri=item.source_uri or item.uri,
                relative_path=relative_path,
                expected_source_digest=item.source_digest,
            )
        except (DocumentConflictError, DocumentNotFoundError, DocumentUnsafeError, PermissionError, ValueError):
            return item, self._drop(item, "stale_memory_document_projection")
        try:
            safe = self.sanitizer.sanitize(
                title=item.title,
                l0_text=item.l0_text,
                l1_text=view.markdown if include_l2 else item.l1_text,
                metadata=dict(item.metadata),
                source_kind="memory_document",
            )
        except ContextProjectionSanitizationError:
            return item, self._drop(item, "memory_document_sanitization_failed")
        return (
            replace(
                item,
                text=safe.l1_text if include_l2 else "",
                l0_text=safe.l0_text,
                l1_text=item.l1_text or safe.l1_text,
                metadata={
                    **safe.metadata,
                    "relative_path": relative_path,
                    "source_validation_status": "live_digest_verified",
                },
            ),
            None,
        )

    def _hydrate_ordinary(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        remaining: int,
    ) -> tuple[RetrievalCandidate | None, int]:
        source_store = cast(SourceStore | None, getattr(self.context_db, "source_store", None))
        if source_store is None or remaining < 1:
            return item, 0
        reads = 1
        try:
            obj = source_store.read_object(item.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, RuntimeError, ValueError):
            return None, reads
        if (
            obj.lifecycle_state != LifecycleState.ACTIVE
            or str(obj.tenant_id or "default") != str(plan.tenant_id or "default")
            or (plan.owner_user_id and str(obj.owner_user_id or "") not in {"", plan.owner_user_id})
        ):
            return None, reads
        content = obj.title
        if remaining >= 2:
            reads += 1
            try:
                content = source_store.read_content(obj.layers.l2_uri or obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                pass
        safe = self.sanitizer.sanitize(
            title=obj.title,
            l0_text=obj.title,
            l1_text=content,
            metadata=obj.metadata,
            source_kind=str(obj.metadata.get("source_kind") or "context"),
        )
        return (
            replace(
                item,
                title=safe.title,
                l0_text=safe.l0_text,
                l1_text=safe.l1_text,
                metadata={**dict(item.metadata), **safe.metadata},
            ),
            reads,
        )

    def _hydrate_resource_l2(
        self,
        item: RetrievalCandidate,
        *,
        plan: RetrievalQueryPlan,
        source_read_budget: int,
    ) -> tuple[RetrievalCandidate, int, str]:
        if source_read_budget < 2:
            mode = "l2_source_read_bound"
            return self._mark_degraded(item, mode), 0, mode
        source_store = cast(SourceStore | None, getattr(self.context_db, "source_store", None))
        if source_store is None:
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), 0, mode
        reads = 1
        try:
            obj = source_store.read_object(item.uri)
            source_record = CatalogRecord.from_context_object(obj)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, RuntimeError, ValueError):
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), reads, mode
        if (
            obj.uri != item.uri
            or obj.context_type != ContextType.RESOURCE
            or obj.lifecycle_state != LifecycleState.ACTIVE
            or source_record.record_kind != CatalogRecordKind.CONTEXT.value
            or str(obj.tenant_id or "default") != str(plan.tenant_id or "default")
            or source_record.owner_user_id != item.owner_user_id
            or source_record.workspace_id != item.workspace_id
            or source_record.source_uri != (item.source_uri or item.uri)
        ):
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        source_l2_uri = str(obj.layers.l2_uri or source_record.source_uri or obj.uri)
        if source_l2_uri != str(item.l2_uri or item.source_uri or item.uri):
            mode = "l2_source_authority_mismatch"
            return self._mark_degraded(item, mode), reads, mode
        reads += 1
        try:
            content = source_store.read_content(source_l2_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, RuntimeError, UnicodeError, ValueError):
            mode = "l2_source_unavailable"
            return self._mark_degraded(item, mode), reads, mode
        actual_digest = self.sanitizer.digest(content or obj.to_dict())
        if not item.source_digest or not hmac.compare_digest(item.source_digest, actual_digest):
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
        return replace(item, text=safe.l1_text), reads, ""

    def _read_session_l2(self, item: RetrievalCandidate, *, plan: RetrievalQueryPlan) -> str | None:
        service = getattr(self.context_db, "session_commit_service", None)
        archive_store = getattr(service, "archive_store", None)
        reader = getattr(archive_store, "read_archive", None)
        if not callable(reader):
            return None
        archive_uri = str(item.metadata.get("archive_uri") or "")
        if not archive_uri:
            return None
        try:
            archive = cast(SessionArchive, reader(archive_uri, tenant_id=str(plan.tenant_id or "default")))
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
            return None
        if plan.owner_user_id and archive.user_id != plan.owner_user_id:
            return None
        manifest_digest = str(item.manifest_digest or item.metadata.get("manifest_digest") or "")
        if manifest_digest and archive.manifest_digest != manifest_digest:
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

    def _schedule_document_rescan(self, item: RetrievalCandidate, *, plan: RetrievalQueryPlan) -> None:
        queue_store = getattr(self.context_db, "queue_store", None)
        enqueue = getattr(queue_store, "enqueue", None)
        if not callable(enqueue):
            scanner = getattr(self.context_db, "memory_document_scanner", None)
            notify = getattr(scanner, "notify", None)
            if callable(notify):
                notify(str(plan.tenant_id or "default"), str(plan.owner_user_id or item.owner_user_id))
            return
        enqueue(
            QueueJob(
                job_id=f"memory_rescan_{stable_hash((plan.tenant_id, item.owner_user_id, item.document_id, item.source_digest), 32)}",
                queue_name="memory_document_scan",
                action="rescan",
                target_uri=item.source_uri or item.uri,
                payload={
                    "tenant_id": str(plan.tenant_id or "default"),
                    "owner_user_id": str(plan.owner_user_id or item.owner_user_id),
                    "document_id": item.document_id,
                    "observed_source_digest": item.source_digest,
                },
            )
        )

    def _session_projection_lag_modes(self, plan: RetrievalQueryPlan) -> tuple[str, ...]:
        if plan.context_types and ContextType.SESSION not in plan.context_types:
            return ()
        reader = getattr(self.index_store, "get_projection_journal_summary", None)
        if not callable(reader):
            return ()
        values = reader(
            tenant_id=str(plan.tenant_id or "default"),
            projector_kind="session",
            owner_user_id=plan.owner_user_id or "",
        )
        if not isinstance(values, Mapping):
            return ("session_projection_health_unavailable",)
        result: list[str] = []
        for status in ("PENDING", "FAILED"):
            count = int(values.get(status, 0) or 0)
            if count:
                result.append(f"session_projection_{status.lower()}:{count}")
        return tuple(result)

    def _serving_generation_token(self) -> str:
        value = getattr(self.context_db, "serving_generation_token", "")
        return str(value() if callable(value) else value or "")

    def _assert_serving_generation_unchanged(self, expected: str) -> None:
        if self._serving_generation_token() != expected:
            raise RetrievalUnavailableError(
                "derived serving generation changed during retrieval",
                degraded_modes=("derived_serving_generation_changed",),
            )

    @staticmethod
    def _drop(item: RetrievalCandidate, reason: str) -> dict[str, Any]:
        return {
            "record_key": item.record_key,
            "uri": item.uri,
            "source_uri": item.source_uri or item.uri,
            "drop_reason": reason,
            "source_validation_status": "stale" if reason.startswith("stale") else "unavailable",
        }

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


__all__ = [
    "RetrievalMetrics",
    "RetrievalUnavailableError",
    "UnifiedRetrievalOrchestrator",
    "UnifiedRetrievalResult",
]
