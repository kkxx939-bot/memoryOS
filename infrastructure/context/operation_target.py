"""使用 Context 索引为事务操作寻找候选目标。

事务内核只校验显式目标；这里负责查询索引、限制用户和类型范围，并根据绝对
相关度与候选间距决定是否自动绑定。低置信度或歧义结果始终进入人工复核。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from infrastructure.store.contracts.index import IndexHit, IndexStore
from infrastructure.store.contracts.source import SourceStore
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_status import OperationStatus
from transaction.resolver.target_resolver import ResolveResult, TargetResolver


class ContextOperationTargetResolver(TargetResolver):
    """通过有租户边界的 Context 索引解析缺失的普通对象目标。"""

    def __init__(
        self,
        index_store: IndexStore,
        source_store: SourceStore,
        *,
        absolute_threshold: float = 0.75,
        margin_threshold: float = 0.10,
    ) -> None:
        super().__init__(source_store=source_store)
        self.index_store = index_store
        self.absolute_threshold = self._threshold(absolute_threshold, "absolute_threshold")
        self.margin_threshold = self._threshold(margin_threshold, "margin_threshold")

    def _resolve_missing(
        self,
        operation: ContextOperation,
        *,
        commit_user: str,
        tenant_id: str,
        limit: int,
    ) -> ResolveResult:
        query = self._query_for(operation)
        if not query:
            return self._pending(operation, "target_review_required")
        filters: dict[str, object] = {
            "owner_user_id": commit_user,
            "context_type": operation.context_type.value,
        }
        project_id = operation.payload.get("project_id") or operation.payload.get("workspace_id")
        if isinstance(project_id, str) and project_id:
            filters["project_id"] = project_id
        try:
            raw_hits = self.index_store.search(
                query,
                tenant_id=tenant_id,
                filters=filters,
                limit=max(2, limit),
            )
        except (TypeError, ValueError):
            return self._pending(operation, "target_review_required")
        candidates = [
            hit
            for hit in raw_hits
            if isinstance(hit, IndexHit) and self._candidate_is_valid(hit, operation, commit_user, tenant_id)
        ]
        candidates.sort(key=lambda hit: (-self._relevance(hit), hit.uri))
        if not candidates:
            return self._pending(operation, "target_review_required")
        top = self._relevance(candidates[0])
        second = self._relevance(candidates[1]) if len(candidates) > 1 else 0.0
        if top < self.absolute_threshold:
            return self._pending(operation, "target_review_required", candidates)
        if len(candidates) > 1 and top - second < self.margin_threshold:
            return self._pending(operation, "target_ambiguous", candidates)
        operation.target_uri = candidates[0].uri
        operation.status = OperationStatus.RESOLVED
        return ResolveResult(operation, True, "target resolved by scoped index match", candidates)

    def _candidate_is_valid(
        self,
        hit: IndexHit,
        operation: ContextOperation,
        commit_user: str,
        tenant_id: str,
    ) -> bool:
        if self._relevance(hit) <= 0 or self._validate_uri(hit.uri, commit_user):
            return False
        assert self.source_store is not None
        try:
            target = self.source_store.read_object(hit.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
            return False
        return not self._validate_target(
            target,
            operation,
            commit_user,
            tenant_id,
            expected_uri=hit.uri,
        )

    @staticmethod
    def _query_for(operation: ContextOperation) -> str:
        for key in ("query", "title", "content", "support_anchor_uri"):
            value = operation.payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        values = [operation.payload.get("scene_key"), operation.payload.get("action")]
        return " ".join(str(value) for value in values if value)

    @staticmethod
    def _relevance(hit: IndexHit) -> float:
        if not isinstance(hit.metadata, Mapping):
            return 0.0
        scores = hit.metadata.get("retrieval_scores")
        if not isinstance(scores, Mapping):
            return 0.0
        values: list[float] = []
        for name in ("lexical", "vector", "identity"):
            try:
                value = float(scores.get(name, 0.0))
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(value) or value < 0:
                return 0.0
            values.append(min(1.0, value))
        return max(values, default=0.0)

    @staticmethod
    def _threshold(value: float, label: str) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or not 0 <= resolved <= 1:
            raise ValueError(f"{label} must be a finite number between 0 and 1")
        return resolved


__all__ = ["ContextOperationTargetResolver"]
