"""依据查询意图、相关性和数量配额选择最终上下文。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from infrastructure.context.retrieval.fusion import RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from infrastructure.store.model.catalog import CatalogRecordKind
from infrastructure.store.model.context.context_type import ContextType


@dataclass(frozen=True)
class SelectedContext:
    """一个已经完成层级选择、可以直接返回给调用者的上下文。"""

    record_key: str
    uri: str
    tenant_id: str
    content: str
    selected_layer: str
    source_uri: str
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
class ContextSelectionPolicy:
    """用条目数量限制单一来源占比，避免检索结果被某一分支淹没。"""

    max_per_session: int = 5
    max_per_resource_branch: int = 3
    max_l2_items: int = 3

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = int(getattr(self, name))
            if value < 0 or value > 100:
                raise ValueError(f"{name} must be between 0 and 100")
            object.__setattr__(self, name, value)


class ContextSelector:
    """从排序后的候选中按 final_limit 和多样性配额选择上下文。

    这里不估算 token，也不根据 token 预算截断内容。上下文规模只由公开的
    ``candidate_limit``、``final_limit`` 和本类的来源配额控制。
    """

    def __init__(self, *, policy: ContextSelectionPolicy | None = None) -> None:
        self.policy = policy or ContextSelectionPolicy()

    def l2_hydration_record_keys(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
    ) -> tuple[str, ...]:
        """只为最终选择结果中的少量普通资源预取 L2 正文。"""

        if self.policy.max_l2_items < 1:
            return ()
        by_key = {item.record_key: item for item in candidates}
        preview = self.select(candidates, plan=plan)
        selected: list[str] = []
        for payload in preview["contexts"]:
            item = by_key.get(str(payload.get("record_key") or ""))
            if item is None or not self._ordinary_resource_l2_eligible(item):
                continue
            selected.append(item.record_key)
            if len(selected) >= self.policy.max_l2_items:
                break
        return tuple(selected)

    def select(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
    ) -> dict[str, Any]:
        selected: list[SelectedContext] = []
        dropped: list[dict[str, Any]] = []
        type_counts: Counter[str] = Counter()
        session_counts: Counter[str] = Counter()
        resource_counts: Counter[str] = Counter()
        l2_count = 0
        type_quotas = self._type_quotas(plan)

        ordered = sorted(
            candidates,
            key=lambda item: (
                self._tier_priority(item, plan),
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
            quota = type_quotas.get(item.context_type, type_quotas.get("*", plan.final_limit))
            if type_counts[item.context_type] >= quota:
                dropped.append(self._drop(item, "context_type_quota"))
                continue
            resource_branch = str(item.metadata.get("resource_location") or "")
            if resource_branch and resource_counts[resource_branch] >= self.policy.max_per_resource_branch:
                dropped.append(self._drop(item, "resource_branch_quota"))
                continue

            layer, content = self._select_layer(item, l2_allowed=l2_count < self.policy.max_l2_items)
            selected.append(
                SelectedContext(
                    record_key=item.record_key,
                    uri=item.uri,
                    tenant_id=plan.tenant_id or "default",
                    content=content,
                    selected_layer=layer,
                    source_uri=item.source_uri or item.uri,
                    source_validation_status=str(item.metadata.get("source_validation_status") or "projected"),
                    projection_lag=self._integer(item.metadata.get("projection_lag")),
                    degraded_mode=str(
                        item.metadata.get("degraded_mode") or item.metadata.get("vector_degraded_mode") or ""
                    ),
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
            type_counts[item.context_type] += 1
            if session_key:
                session_counts[session_key] += 1
            if resource_branch:
                resource_counts[resource_branch] += 1
            if layer == "L2":
                l2_count += 1

        return {
            "contexts": [item.to_dict() for item in selected],
            "dropped_contexts": dropped,
            "selected_count": len(selected),
            "dropped_count": len(dropped),
            "load_plan": [
                {
                    "record_key": item.record_key,
                    "uri": item.uri,
                    "selected_layer": item.selected_layer,
                    "source_uri": item.source_uri,
                    "source_validation_status": item.source_validation_status,
                    "projection_lag": item.projection_lag,
                    "degraded_mode": item.degraded_mode,
                }
                for item in selected
            ],
        }

    @staticmethod
    def _select_layer(item: RetrievalCandidate, *, l2_allowed: bool) -> tuple[str, str]:
        if l2_allowed and item.text:
            return "L2", item.text
        if item.l1_text:
            return "L1", item.l1_text
        if item.l0_text:
            return "L0", item.l0_text
        return "URI", item.source_uri or item.uri

    @staticmethod
    def _priority(item: RetrievalCandidate, plan: RetrievalQueryPlan) -> int:
        record_kind = item.record_kind
        source_kind = item.source_kind
        if ContextSelector._is_coding_agent(plan):
            if record_kind == "resource_reference" or source_kind in {"resource", "resource_reference"}:
                return 0
            if record_kind in {"session_root", "session_l0", "session_l1", "semantic_segment"}:
                return 1 if item.session_id and item.session_id in plan.session_ids else 3
        if plan.query_intent == RetrievalQueryIntent.CURRENT:
            order = {"resource": 0, "session_root": 1, "semantic_segment": 1}
            return order.get(record_kind, order.get(source_kind, 8))
        if plan.query_intent in {RetrievalQueryIntent.OPEN_RECALL, RetrievalQueryIntent.HISTORY}:
            order = {
                "session_root": 0,
                "session_l0": 0,
                "session_l1": 0,
                "semantic_segment": 0,
                "event": 1,
                "resource": 2,
                "resource_reference": 2,
                "tool_result": 3,
            }
            return order.get(record_kind, order.get(source_kind, 6))
        return 4

    @staticmethod
    def _tier_priority(item: RetrievalCandidate, plan: RetrievalQueryPlan) -> int:
        if plan.query_intent is not RetrievalQueryIntent.CURRENT:
            return 0
        tier = str(item.metadata.get("serving_tier") or "")
        if tier in {"HOT", "WARM"}:
            return 0
        if tier in {"COLD", "ARCHIVED"}:
            return 1
        return 2

    @staticmethod
    def _ordinary_resource_l2_eligible(item: RetrievalCandidate) -> bool:
        return bool(
            item.context_type == ContextType.RESOURCE.value
            and item.record_kind == CatalogRecordKind.CONTEXT.value
            and item.source_kind in {"context", "resource"}
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
            return {"resource": max(1, limit // 2), "session": max(1, limit // 3), "*": max(1, limit // 3)}
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
            "source_validation_status": str(item.metadata.get("source_validation_status") or "projected"),
            "projection_lag": ContextSelector._integer(item.metadata.get("projection_lag")),
            "degraded_mode": str(item.metadata.get("degraded_mode") or ""),
        }

    @staticmethod
    def _integer(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0


__all__ = ["ContextSelectionPolicy", "ContextSelector", "SelectedContext"]
