"""关系图扩展产生的有界候选召回。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from infrastructure.context.retrieval.fusion import RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalQueryPlan
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.model.catalog import CatalogRecord


class RelationCandidateSource:
    """从已授权种子出发，按固定上限扩展关系目标。"""

    MAX_SEEDS = 20
    MAX_RELATIONS_PER_SEED = 5
    MAX_RECORDS_PER_TARGET = 2

    def __init__(
        self,
        *,
        index_store: IndexStore,
        relation_store: RelationStore | None,
        filters_for_plan: Callable[[RetrievalQueryPlan], dict[str, Any]],
        from_record: Callable[..., RetrievalCandidate],
        target_identity_filters: Callable[..., dict[str, Any]],
    ) -> None:
        self.index_store = index_store
        self.relation_store = relation_store
        self.filters_for_plan = filters_for_plan
        self.from_record = from_record
        self.target_identity_filters = target_identity_filters

    def generate(
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
        for seed in seeds[: min(self.MAX_SEEDS, plan.candidate_limit)]:
            relation_rows_remaining = self.MAX_RELATIONS_PER_SEED
            for seed_identity in self._seed_identities(seed):
                if relation_rows_remaining <= 0:
                    break
                relations = self.relation_store.relations_of(
                    seed_identity,
                    tenant_id=plan.tenant_id or "default",
                    owner_user_id=plan.owner_user_id,
                    limit=relation_rows_remaining,
                )
                bounded_relations = relations[:relation_rows_remaining]
                relation_rows_remaining -= len(bounded_relations)
                for relation in bounded_relations:
                    target_uri = relation.target_uri if relation.source_uri == seed_identity else relation.source_uri
                    target_filters = self.target_identity_filters(
                        self.filters_for_plan(plan),
                        (target_uri,),
                        limit=self.MAX_RECORDS_PER_TARGET,
                    )
                    raw_records: Any = lister(
                        tenant_id=plan.tenant_id or "default",
                        filters=target_filters,
                        limit=min(self.MAX_RECORDS_PER_TARGET, plan.candidate_limit - len(result)),
                    )
                    records = raw_records if isinstance(raw_records, Sequence) else ()
                    for record in records:
                        if not isinstance(record, CatalogRecord) or record.record_key in seen_record_keys:
                            continue
                        result.append(
                            self.from_record(
                                record,
                                branch="relation",
                                score=max(0.0, min(1.0, float(relation.weight))),
                            )
                        )
                        seen_record_keys.add(record.record_key)
                        if len(result) >= plan.candidate_limit:
                            return tuple(result)
        return tuple(result)

    @staticmethod
    def _seed_identities(seed: RetrievalCandidate) -> tuple[str, ...]:
        return (seed.uri,)


def target_identity_filters(
    filters: Mapping[str, Any],
    target_uris: Sequence[str],
    *,
    limit: int,
) -> dict[str, Any]:
    """把稳定 Serving 身份约束到一次精确 SQL 查询。"""

    exact_filters = dict(filters)
    exact_filters.pop("target_uris", None)
    exact_filters["target_identity_uris"] = tuple(target_uris)
    exact_filters["_identity_candidate_limit"] = int(limit)
    return exact_filters


__all__ = ["RelationCandidateSource", "target_identity_filters"]
