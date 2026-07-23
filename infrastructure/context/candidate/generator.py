"""从统一上下文目录中有界生成检索候选。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from foundation.scope import scope_keys_from_payloads
from infrastructure.context.candidate.mapper import CandidateMapper
from infrastructure.context.candidate.relation import RelationCandidateSource, target_identity_filters
from infrastructure.context.candidate.vector import VectorCandidateSource
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.fusion import RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from infrastructure.store.contracts.index import IndexHit, IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.vector import VectorStore
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind, ServingTier
from sanitization.context_projection import ContextProjectionSanitizer

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
    """在统一硬上限内编排精确、FTS、向量和关系候选。"""

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
        self.mapper = CandidateMapper(index_store)
        self.vector_source = VectorCandidateSource(
            index_store=index_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            sanitizer=self.sanitizer,
            filters_for_plan=self._filters,
            from_record=self.mapper.from_record,
            finite_score=self._finite_score,
        )
        self.relation_source = RelationCandidateSource(
            index_store=index_store,
            relation_store=relation_store,
            filters_for_plan=self._filters,
            from_record=self.mapper.from_record,
            target_identity_filters=target_identity_filters,
        )

    def generate(self, plan: RetrievalQueryPlan) -> CandidateGenerationResult:
        filters = self._filters(plan)
        exact_records = self._exact_records(plan, filters)
        record_exact = tuple(self.mapper.from_record(record, branch="exact", score=1.0) for record in exact_records)

        lexical_hits = self._search_catalog(plan.semantic_query, filters, plan.candidate_limit)
        hit_candidates = tuple(self.mapper.from_hit(hit) for hit in lexical_hits)
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
            self.mapper.from_record(record, branch="structured", score=0.0) for record in structured_records
        )

        seeds = (*structured_candidates, *exact, *lexical)
        vector, vector_degraded, vector_source_reads, vector_overfetch = self.vector_source.generate(seeds, plan)
        relation = self.relation_source.generate(seeds, plan)
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

    def generate_cold_lexical(
        self,
        plan: RetrievalQueryPlan,
        *,
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        """CURRENT 主阶段不足时，只从冷层执行一次有界词法查询。"""

        bounded_limit = min(max(0, int(limit)), plan.candidate_limit)
        if (
            plan.query_intent is not RetrievalQueryIntent.CURRENT
            or not plan.semantic_query
            or bounded_limit == 0
        ):
            return ()
        cold_tiers = {ServingTier.COLD.value, ServingTier.ARCHIVED.value}
        filters = self._filters(plan)
        filters["serving_tier"] = tuple(sorted(cold_tiers))
        results: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for hit in self._search_catalog(plan.semantic_query, filters, bounded_limit):
            candidate = self.mapper.from_hit(hit)
            if (
                candidate.record_key in seen
                or str(candidate.metadata.get("serving_tier") or "") not in cold_tiers
            ):
                continue
            seen.add(candidate.record_key)
            results.append(
                replace(
                    candidate,
                    branch_scores={"lexical": self._finite_score(hit.score)},
                    branch_ranks={},
                )
            )
            if len(results) >= bounded_limit:
                break
        return tuple(results)

    def _structured_records(
        self,
        plan: RetrievalQueryPlan,
        filters: Mapping[str, Any],
        *,
        has_bounded_text_candidates: bool = False,
    ) -> tuple[CatalogRecord, ...]:
        """为确定性列表计划返回经过 SQL 过滤的有界记录流。

        空查询的 CURRENT/HISTORY 只是受支持的列表视图，不代表允许枚举 Source。
        Catalog 必须在硬性候选 LIMIT 之前应用租户、ACL、类型、路径和时间过滤。

        语义查询通常禁用该分支；唯一例外是意图和索引时间约束足以确定候选集的
        时间计划，即事件时间 OPEN_RECALL 或事务时间 HISTORY。精确/FTS 优先执行，
        有界文本命中已足够时会抑制更宽的时间列表；没有文本命中时，时间 SQL 在
        LIMIT 前继续应用时间、路径和 ACL 条件，作为非词法降级。
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
        """只允许完整且与查询意图匹配的索引时间约束。"""

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
        """只有文本召回足够明确时，才跳过更宽的时间列表。

        日历数字或通用日期词可能制造弱 FTS 命中，即使自然语言问题与存储内容没有
        语义重合。精确身份始终满足查询；词法候选只有达到足够词项覆盖率，形成选择性
        有界结果时，才能抑制时间降级。
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
            # 可信工作区也可读取工作区被有意留空的所有者记录；保留的 principal-only
            # 值会拒绝所有绑定工作区的记录。
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
                # 项目/用户共享视图有意允许跨适配器；Connect 元数据描述可信调用者，
                # 不能变成隐藏租户共享状态的内容过滤器。
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
            # Session 与 Markdown 混合查询只用 document_kind 缩小文档投影；普通
            # Session 记录没有该字段，仍应保留候选资格。
            filters["document_kinds_apply_to_documents_only"] = True
        if metadata.get("minimum_lexical_relevance") is not None:
            try:
                minimum_lexical_relevance = float(metadata["minimum_lexical_relevance"])
            except (TypeError, ValueError) as exc:
                raise ValueError("minimum_lexical_relevance must be a finite score") from exc
            if not 0.0 <= minimum_lexical_relevance <= 1.0:
                raise ValueError("minimum_lexical_relevance must be between 0 and 1")
            filters["minimum_lexical_relevance"] = minimum_lexical_relevance
        if plan.query_intent is RetrievalQueryIntent.CURRENT:
            filters["serving_tier"] = (ServingTier.HOT.value, ServingTier.WARM.value)
        elif plan.query_intent in {
            RetrievalQueryIntent.HISTORY,
            RetrievalQueryIntent.OPEN_RECALL,
            RetrievalQueryIntent.EXACT,
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
        exact_filters = target_identity_filters(
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
                # IndexStore 精确/搜索契约必须携带租户。无法在搜索前绑定租户的后端
                # 不能先查后过滤，否则既可能扫描其他租户，也会在隔离前错误应用 LIMIT。
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
            shared_record = bool(
                metadata.get("workspace_shared") is True and hit_workspace in shared_workspaces
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
        allowed_tiers = {str(item) for item in filters.get("serving_tier", ()) or ()}
        if allowed_tiers and str(metadata.get("serving_tier") or "") not in allowed_tiers:
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
        # 显式召回视图是收窄型安全约束；无法证明候选视图的后端，不能把元数据缺失
        # 解释成跨工作区权限。
        if allowed_views and ((item_views and not allowed_views.intersection(item_views)) or not item_views):
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


__all__ = ["CandidateGenerationResult", "CandidateGenerator"]
