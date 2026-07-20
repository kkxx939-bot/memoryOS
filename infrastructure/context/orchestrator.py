"""统一且有界的上下文检索流水线。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from foundation.readiness import RuntimeReadiness
from infrastructure.context.candidate import CandidateGenerator
from infrastructure.context.hydration import ContextHydrator
from infrastructure.context.layers.memory_document_overlay import MemoryDocumentContextOverlay
from infrastructure.context.reranking import Reranker
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.fusion import FusionRanker, RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalQueryPlan
from infrastructure.context.selection import ContextSelector
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.queue import QueueStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.session_archive import SessionArchiveStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.query import CatalogCandidateBoundExceeded
from sanitization.context_projection import ContextProjectionSanitizer

SOURCE_READ_BOUND_ALLOWANCE = 8


class RetrievalUnavailableError(RuntimeError):
    """有界 Serving 链无法安全地区分空结果和后端不可用。"""

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

    def search_payload(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.contexts]

    def assemble_payload(self) -> dict[str, Any]:
        return {
            "contexts": self.search_payload(),
            "dropped_contexts": [dict(item) for item in self.dropped_contexts],
            "load_plan": [dict(item) for item in self.load_plan],
            "metrics": self.metrics.to_dict(),
            "degraded_modes": list(self.degraded_modes),
            "query_plan": self.plan.to_dict(),
            "reranker_fallback": self.reranker_fallback,
        }


class UnifiedRetrievalOrchestrator:
    """依次编排结构化、精确、FTS、向量、关系、融合和内容回源。"""

    def __init__(
        self,
        index_store: IndexStore,
        *,
        source_store: SourceStore | None,
        relation_store: RelationStore | None,
        queue_store: QueueStore | None,
        session_archive_store: SessionArchiveStore | None,
        readiness: RuntimeReadiness | None = None,
        serving_lock: RLock | None = None,
        serving_generation_token: Callable[[], str] | None = None,
        vector_store: Any = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        document_overlay: MemoryDocumentContextOverlay | None = None,
    ) -> None:
        self.readiness = readiness
        self.serving_lock = serving_lock
        self._serving_generation_provider = serving_generation_token
        self.reranker = reranker
        self.sanitizer = ContextProjectionSanitizer()
        self.index_store = index_store
        self.generator = CandidateGenerator(
            index_store,
            relation_store=relation_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            sanitizer=self.sanitizer,
        )
        self.fusion = FusionRanker()
        self.selector = ContextSelector()
        self.document_overlay = document_overlay
        self.hydrator = ContextHydrator(
            source_store=source_store,
            session_archive_store=session_archive_store,
            queue_store=queue_store,
            selector=self.selector,
            sanitizer=self.sanitizer,
            document_overlay=self.document_overlay,
        )

    def execute(self, plan: RetrievalQueryPlan) -> UnifiedRetrievalResult:
        if self.serving_lock is None:
            return self._execute_locked(plan)
        with self.serving_lock:
            return self._execute_locked(plan)

    def _execute_locked(self, plan: RetrievalQueryPlan) -> UnifiedRetrievalResult:
        if self.readiness is not None:
            self.readiness.require_ready()
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
        hydrated, source_reads, hydration_modes, hydration_drops, memory_validated = self.hydrator.hydrate(
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
        selection = self.selector.select(hydrated, plan=plan)
        unavailable_modes = tuple(
            mode
            for mode in generated.degraded_modes
            if str(mode).startswith("vector_fallback:") or str(mode) == "fts_unavailable"
        )
        if unavailable_modes and not selection["contexts"]:
            raise RetrievalUnavailableError(
                "retrieval backends failed and no validated bounded fallback remains",
                degraded_modes=unavailable_modes,
            )
        dropped = (*hydration_drops, *tuple(selection["dropped_contexts"]))
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
            selected_count=int(selection["selected_count"]),
            dropped_count=len(dropped),
            vector_overfetch=generated.vector_overfetch,
        )
        if metrics.source_reads > plan.candidate_limit + SOURCE_READ_BOUND_ALLOWANCE:
            raise RuntimeError("source reads exceeded candidate_limit plus bounded allowance")
        self._assert_serving_generation_unchanged(generation_before)
        return UnifiedRetrievalResult(
            plan=plan,
            contexts=tuple(selection["contexts"]),
            dropped_contexts=tuple(dropped),
            load_plan=tuple(selection["load_plan"]),
            metrics=metrics,
            degraded_modes=degraded_modes,
            reranker_fallback=reranker_fallback,
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
        if self._serving_generation_provider is None:
            return ""
        return str(self._serving_generation_provider())

    def _assert_serving_generation_unchanged(self, expected: str) -> None:
        if self._serving_generation_token() != expected:
            raise RetrievalUnavailableError(
                "derived serving generation changed during retrieval",
                degraded_modes=("derived_serving_generation_changed",),
            )

    @staticmethod
    def _merge_degraded_modes(existing: str, *modes: str) -> str:
        return ",".join(dict.fromkeys(value for raw in (existing, *modes) for value in str(raw).split(",") if value))


__all__ = [
    "RetrievalMetrics",
    "RetrievalUnavailableError",
    "UnifiedRetrievalOrchestrator",
    "UnifiedRetrievalResult",
]
