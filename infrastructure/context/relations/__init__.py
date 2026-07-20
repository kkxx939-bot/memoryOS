"""上下文关系生成与检索投影规则。"""

from infrastructure.context.relations.ordinary import (
    NoRelationDomainPolicy,
    OrdinaryRelationEligibility,
    RelationDomainPolicy,
    ordinary_relation_serving_eligibility,
    ordinary_relation_specs_for_object,
)

__all__ = [
    "NoRelationDomainPolicy",
    "OrdinaryRelationEligibility",
    "RelationDomainPolicy",
    "ordinary_relation_serving_eligibility",
    "ordinary_relation_specs_for_object",
]
