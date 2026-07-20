"""上下文召回轨迹的生成、清洗和读取语义。"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from foundation.clock import utc_now
from infrastructure.context.orchestrator import UnifiedRetrievalResult
from infrastructure.store.trace import RecallTraceRepository
from sanitization.context_projection import ContextProjectionSanitizer


class RecallTraceService:
    """把召回结果转换为安全轨迹，持久化细节交给 Store。"""

    def __init__(self, repository: RecallTraceRepository) -> None:
        self.repository = repository
        self.sanitizer = ContextProjectionSanitizer()

    def record_unified(self, result: UnifiedRetrievalResult) -> str:
        """记录统一召回结果，不保存查询原文。"""

        plan = result.plan
        metrics = result.metrics.to_dict()
        query_plan = plan.to_dict()
        query_plan.pop("semantic_query", None)
        return self.record(
            plan.semantic_query,
            scope={
                "tenant_id": plan.tenant_id,
                "user_id": plan.owner_user_id,
                "project_id": plan.workspace_ids[0] if len(plan.workspace_ids) == 1 else "",
                "workspace_ids": list(plan.workspace_ids),
                "session_ids": list(plan.session_ids),
                "adapter_id": plan.adapter_id,
                "search_scope": plan.legacy_search_scope,
            },
            query_plan=query_plan,
            retrieval_views=plan.legacy_retrieval_views,
            metadata_filters=plan.metadata_filters,
            selected=result.contexts,
            dropped=result.dropped_contexts,
            details={
                **metrics,
                "candidate_count": metrics["fusion_candidates"],
                "degraded_modes": list(result.degraded_modes),
                "reranker_fallback": result.reranker_fallback,
            },
        )

    def record(
        self,
        query: str,
        *,
        scope: Mapping[str, Any] | None = None,
        query_plan: Mapping[str, Any] | None = None,
        retrieval_views: Sequence[str] = (),
        metadata_filters: Mapping[str, Any] | None = None,
        selected: Sequence[Mapping[str, Any]] = (),
        dropped: Sequence[Mapping[str, Any]] = (),
        details: Mapping[str, Any] | None = None,
    ) -> str:
        """生成规范轨迹并在清洗成功后交给 Store 保存。"""

        trace_id = str(uuid.uuid4())
        selected_rows = [
            {
                "uri": item.get("uri"),
                "source_uri": item.get("source_uri"),
                "score": item.get("score"),
                "layer": item.get("selected_layer") or item.get("layer"),
                "source_validation_status": item.get("source_validation_status"),
                "projection_lag": item.get("projection_lag"),
                "degraded_mode": item.get("degraded_mode"),
                "metadata": dict(item.get("metadata", {}) or {}),
            }
            for item in selected
        ]
        trace = {
            "trace_id": trace_id,
            "created_at": utc_now(),
            "query_digest": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "query_utf8_bytes": len(query.encode("utf-8")),
            "query_plan": dict(query_plan or {}),
            "scope": dict(scope or {}),
            "retrieval_views": list(retrieval_views),
            "metadata_filters": dict(metadata_filters or {}),
            "selected": selected_rows,
            "dropped": [dict(item) for item in dropped],
            **dict(details or {}),
        }
        safe_trace = self.sanitizer.sanitize_trace(trace)
        if not isinstance(safe_trace, dict) or safe_trace.get("trace_id") != trace_id:
            raise ValueError("recall trace sanitization produced an invalid payload")
        self.repository.save(trace_id, safe_trace)
        return trace_id

    def read(self, trace_id: str) -> dict[str, Any]:
        """读取轨迹并再次执行 Context 出口安全校验。"""

        value = self.repository.read(trace_id)
        self.sanitizer.assert_safe(value)
        return value


__all__ = ["RecallTraceService"]
