"""Intent-aware L0/L1/L2 selection and bounded context packing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memoryos.contextdb.catalog import CatalogRecordKind
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.fusion import RetrievalCandidate
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from memoryos.contextdb.retrieval.token_budget import HeuristicTokenCounter, TokenCounter


@dataclass(frozen=True)
class PackedContext:
    record_key: str
    uri: str
    tenant_id: str
    content: str
    selected_layer: str
    source_uri: str
    token_estimate: int
    source_validation_status: str
    projection_lag: int
    degraded_mode: str
    score: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "layer": self.selected_layer,
            "text": self.content,
            "drop_reason": "",
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextPackingPolicy:
    """Safe bounded defaults; deployments may lower, never disable, quotas."""

    max_per_session: int = 5
    max_per_document: int = 3
    max_blocks_per_document: int = 3
    max_per_resource_branch: int = 3
    max_l2_items: int = 3

    def __post_init__(self) -> None:
        for name in (
            "max_per_session",
            "max_per_document",
            "max_blocks_per_document",
            "max_per_resource_branch",
            "max_l2_items",
        ):
            value = int(getattr(self, name))
            if value < 0 or value > 100:
                raise ValueError(f"{name} must be between 0 and 100")
            object.__setattr__(self, name, value)


class LayerSelector:
    """Select the richest layer that fits without reading an entire source tree."""

    def __init__(self, token_counter: TokenCounter | None = None, *, max_l2_items: int = 3) -> None:
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.max_l2_items = max(0, int(max_l2_items))

    def candidates(
        self,
        item: RetrievalCandidate,
        *,
        l2_allowed: bool,
    ) -> tuple[tuple[str, str], ...]:
        layers: list[tuple[str, str]] = []
        if l2_allowed and item.text:
            layers.append(("L2", item.text))
        if item.l1_text:
            layers.append(("L1", item.l1_text))
        if item.l0_text:
            layers.append(("L0", item.l0_text))
        layers.append(("URI", item.source_uri or item.uri))
        return tuple(dict.fromkeys(layers))

    def select(
        self,
        item: RetrievalCandidate,
        *,
        remaining_tokens: int,
        l2_allowed: bool,
    ) -> tuple[str, str, int] | None:
        for layer, content in self.candidates(item, l2_allowed=l2_allowed):
            estimate = self.token_counter.count(content)
            if estimate <= remaining_tokens:
                return layer, content, estimate
        return None


class ContextPacker:
    """Pack one fused stream with type/session/document/resource diversity quotas."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        *,
        policy: ContextPackingPolicy | None = None,
    ) -> None:
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.policy = policy or ContextPackingPolicy()
        self.layers = LayerSelector(self.token_counter, max_l2_items=self.policy.max_l2_items)

    def l2_hydration_record_keys(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
    ) -> tuple[str, ...]:
        """Choose the bounded ordinary Resource prefix that may receive L2.

        L2 is hydrated after fusion and rerank. A
        preview pack reuses the exact final-limit, intent, token, session,
        type, and resource-branch quotas instead of turning every generated
        candidate into a Source read. Memory document projections and Session
        evidence nodes are deliberately ineligible: their source-specific
        archive-specific readers own those paths.
        """

        if self.policy.max_l2_items < 1:
            return ()
        by_key = {item.record_key: item for item in candidates}
        preview = self.pack(candidates, plan=plan)
        selected: list[str] = []
        for packed in preview["contexts"]:
            item = by_key.get(str(packed.get("record_key") or ""))
            if item is None or not self._ordinary_resource_l2_eligible(item):
                continue
            selected.append(item.record_key)
            if len(selected) >= self.policy.max_l2_items:
                break
        return tuple(selected)

    def pack(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
    ) -> dict[str, Any]:
        remaining = plan.token_budget
        selected: list[PackedContext] = []
        dropped: list[dict[str, Any]] = []
        type_counts: Counter[str] = Counter()
        session_counts: Counter[str] = Counter()
        document_counts: Counter[tuple[str, str, str]] = Counter()
        block_counts: Counter[tuple[str, str, str]] = Counter()
        resource_counts: Counter[str] = Counter()
        l2_count = 0
        type_quotas = self._type_quotas(plan)

        ordered = sorted(
            candidates,
            key=lambda item: (
                self._priority(item, plan),
                -item.score.final_score,
                item.record_key,
            ),
        )
        for item in ordered:
            if len(selected) >= plan.final_limit:
                dropped.append(self._drop(item, "final_limit"))
                continue
            session_key = item.manifest_digest or item.archive_digest or item.session_id
            if session_key and session_counts[session_key] >= self.policy.max_per_session:
                dropped.append(self._drop(item, "session_quota"))
                continue
            document_key = (item.tenant_id, item.owner_user_id, item.document_id)
            if item.document_id and document_counts[document_key] >= self.policy.max_per_document:
                dropped.append(self._drop(item, "document_quota"))
                continue
            if item.block_id and block_counts[document_key] >= self.policy.max_blocks_per_document:
                dropped.append(self._drop(item, "document_block_quota"))
                continue
            quota = type_quotas.get(item.context_type, type_quotas.get("*", plan.final_limit))
            if type_counts[item.context_type] >= quota:
                dropped.append(self._drop(item, "context_type_quota"))
                continue
            resource_branch = str(item.metadata.get("resource_location") or "")
            if resource_branch and resource_counts[resource_branch] >= self.policy.max_per_resource_branch:
                dropped.append(self._drop(item, "resource_branch_quota"))
                continue

            choice = self.layers.select(
                item,
                remaining_tokens=remaining,
                l2_allowed=l2_count < self.layers.max_l2_items,
            )
            if choice is None:
                dropped.append(self._drop(item, "token_budget"))
                continue
            layer, content, estimate = choice
            validation = str(item.metadata.get("source_validation_status") or "verified")
            lag = self._integer(item.metadata.get("projection_lag"))
            degraded = str(item.metadata.get("degraded_mode") or item.metadata.get("vector_degraded_mode") or "")
            selected.append(
                PackedContext(
                    record_key=item.record_key,
                    uri=item.uri,
                    tenant_id=plan.tenant_id or "default",
                    content=content,
                    selected_layer=layer,
                    source_uri=item.source_uri or item.uri,
                    token_estimate=estimate,
                    source_validation_status=validation,
                    projection_lag=lag,
                    degraded_mode=degraded,
                    score=item.score.final_score,
                    metadata={
                        **dict(item.metadata),
                        "score_components": item.score.to_dict(),
                        "source_digest": item.source_digest,
                        "record_kind": item.record_kind,
                        "source_kind": item.source_kind,
                    },
                )
            )
            remaining -= estimate
            type_counts[item.context_type] += 1
            if session_key:
                session_counts[session_key] += 1
            if item.document_id:
                document_counts[document_key] += 1
            if item.block_id:
                block_counts[document_key] += 1
            if resource_branch:
                resource_counts[resource_branch] += 1
            if layer == "L2":
                l2_count += 1

        return {
            "contexts": [item.to_dict() for item in selected],
            "dropped_contexts": dropped,
            "total_budget": plan.token_budget,
            "used_tokens": plan.token_budget - remaining,
            "remaining_tokens": remaining,
            "selected_count": len(selected),
            "dropped_count": len(dropped),
            "load_plan": [
                {
                    "record_key": item.record_key,
                    "uri": item.uri,
                    "selected_layer": item.selected_layer,
                    "source_uri": item.source_uri,
                    "token_estimate": item.token_estimate,
                    "source_validation_status": item.source_validation_status,
                    "projection_lag": item.projection_lag,
                    "degraded_mode": item.degraded_mode,
                }
                for item in selected
            ],
        }

    @staticmethod
    def _priority(item: RetrievalCandidate, plan: RetrievalQueryPlan) -> int:
        document_kind = item.document_kind or str(item.metadata.get("document_kind") or "")
        record_kind = item.record_kind
        source_kind = item.source_kind
        if ContextPacker._is_coding_agent(plan):
            if document_kind in {"preferences", "profile"}:
                return 0
            if document_kind in {"entity", "topic"}:
                return 1
            if record_kind == "resource_reference" or source_kind in {"resource", "resource_reference"}:
                return 2
            if record_kind in {"session_root", "session_l0", "session_l1", "semantic_segment"}:
                return 3 if item.session_id and item.session_id in plan.session_ids else 5
            if document_kind == "experience":
                return 4
        if plan.query_intent == RetrievalQueryIntent.CURRENT:
            order = {
                "memory_document": 0,
                "memory_block": 1,
                "resource": 5,
            }
            return order.get(record_kind, order.get(source_kind, 8))
        if plan.query_intent in {RetrievalQueryIntent.OPEN_RECALL, RetrievalQueryIntent.HISTORY}:
            order = {
                "session_root": 0,
                "session_l0": 0,
                "session_l1": 0,
                "semantic_segment": 0,
                "memory_block": 1,
                "memory_document": 1,
                "event": 1,
                "resource": 2,
                "resource_reference": 2,
                "tool_result": 3,
            }
            return order.get(record_kind, order.get(source_kind, 6))
        return 4

    @staticmethod
    def _ordinary_resource_l2_eligible(item: RetrievalCandidate) -> bool:
        return bool(
            item.context_type == ContextType.RESOURCE.value
            and item.record_kind == CatalogRecordKind.CONTEXT.value
            and item.source_kind in {"context", "resource"}
            and not item.document_id
            and (item.l2_uri or item.source_uri or item.uri)
        )

    @staticmethod
    def _is_coding_agent(plan: RetrievalQueryPlan) -> bool:
        if "coding_agent" in plan.source_kinds:
            return True
        connect = plan.metadata_filters.get("connect_filters", plan.metadata_filters.get("connect"))
        return isinstance(connect, Mapping) and str(connect.get("source_kind") or "") == "coding_agent"

    @staticmethod
    def _type_quotas(plan: RetrievalQueryPlan) -> dict[str, int]:
        limit = max(1, plan.final_limit)
        if plan.query_intent == RetrievalQueryIntent.CURRENT:
            return {"memory": max(1, limit // 2), "resource": max(1, limit // 3), "*": max(1, limit // 4)}
        if plan.query_intent in {RetrievalQueryIntent.OPEN_RECALL, RetrievalQueryIntent.HISTORY}:
            return {"session": max(1, limit // 2), "resource": max(1, limit // 3), "*": max(1, limit // 3)}
        return {"*": limit}

    @staticmethod
    def _drop(item: RetrievalCandidate, reason: str) -> dict[str, Any]:
        return {
            "record_key": item.record_key,
            "uri": item.uri,
            "source_uri": item.source_uri or item.uri,
            "selected_layer": "",
            "drop_reason": reason,
            "token_estimate": 0,
            "source_validation_status": str(item.metadata.get("source_validation_status") or "verified"),
            "projection_lag": ContextPacker._integer(item.metadata.get("projection_lag")),
            "degraded_mode": str(item.metadata.get("degraded_mode") or ""),
        }

    @staticmethod
    def _integer(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0


__all__ = ["ContextPacker", "ContextPackingPolicy", "LayerSelector", "PackedContext"]
