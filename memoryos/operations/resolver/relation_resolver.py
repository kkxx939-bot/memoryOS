from __future__ import annotations

from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.operations.model.context_operation import ContextOperation


class RelationResolver:
    def extract_relations(self, operation: ContextOperation) -> list[ContextRelation]:
        relations = []
        source_uri = operation.target_uri
        if not source_uri:
            return []
        for item in operation.payload.get("relations", []):
            if not isinstance(item, dict) or not item.get("target_uri"):
                continue
            relations.append(
                ContextRelation(
                    source_uri=source_uri,
                    relation_type=str(item.get("type", item.get("relation_type", "related_to"))),
                    target_uri=str(item["target_uri"]),
                    weight=float(item.get("weight", 1.0)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return relations
