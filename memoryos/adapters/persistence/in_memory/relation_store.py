"""In-memory RelationStore adapter."""

from __future__ import annotations

from collections.abc import Sequence

from memoryos.contextdb.extensions import ContextDomainClassifier, NoDomainOverlay
from memoryos.contextdb.model.context_relation import ContextRelation


class InMemoryRelationStore:
    def __init__(self, *, domain_classifier: ContextDomainClassifier | None = None) -> None:
        self.relations: list[ContextRelation] = []
        self.domain_classifier = domain_classifier or NoDomainOverlay()

    def add_relation(self, relation: ContextRelation) -> None:
        if relation not in self.relations:
            self.relations.append(relation)

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
        limit: int | None = None,
    ) -> list[ContextRelation]:
        rows = [relation for relation in self.relations if relation.source_uri == uri or relation.target_uri == uri]
        if tenant_id is not None:
            rows = [relation for relation in rows if relation.metadata.get("tenant_id", "default") == tenant_id]
        if owner_user_id is not None:
            rows = [
                relation
                for relation in rows
                if relation.metadata.get("owner_user_id") in {None, "", owner_user_id}
                or relation.target_uri.startswith(("memoryos://resources/", "memoryos://skills/"))
            ]
        rows.sort(key=lambda item: (-item.weight, item.created_at, item.source_uri, item.target_uri))
        return rows[: max(0, int(limit))] if limit is not None else rows

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        matching_tenants = {
            str(relation.metadata.get("tenant_id") or "default")
            for relation in self.relations
            if relation.source_uri == source_uri
            and relation.relation_type == relation_type
            and relation.target_uri == target_uri
        }
        if tenant_id is None and len(matching_tenants) > 1:
            raise ValueError("tenant_id is required for an ambiguous relation identity")
        selected_tenant = tenant_id or next(iter(matching_tenants), None)
        self.relations = [
            relation
            for relation in self.relations
            if not (
                relation.source_uri == source_uri
                and relation.relation_type == relation_type
                and relation.target_uri == target_uri
                and (
                    selected_tenant is None
                    or str(relation.metadata.get("tenant_id") or "default") == selected_tenant
                )
            )
        ]

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int:
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if (relation.source_uri == uri or relation.target_uri == uri)
            and str(relation.metadata.get("tenant_id") or "default") == tenant_id
            and str(relation.metadata.get("catalog_record_key") or "") in {"", catalog_record_key}
        ][:maximum]
        identities = {
            (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            for relation in selected
        }
        self.relations = [
            relation
            for relation in self.relations
            if (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            not in identities
        ]
        return len(identities)

    def delete_uri_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> int:
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if (relation.source_uri == uri or relation.target_uri == uri)
            and str(relation.metadata.get("tenant_id") or "default") == tenant_id
        ][:maximum]
        identities = {
            (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            for relation in selected
        }
        self.relations = [
            relation
            for relation in self.relations
            if (
                str(relation.metadata.get("tenant_id") or "default"),
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
            )
            not in identities
        ]
        return len(identities)

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int:
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if str(relation.metadata.get("tenant_id") or "default") == tenant_id
            and not self.domain_classifier.owns_uri(relation.source_uri)
        ][:maximum]
        identities = {
            (relation.source_uri, relation.relation_type, relation.target_uri)
            for relation in selected
        }
        self.relations = [
            relation
            for relation in self.relations
            if not (
                str(relation.metadata.get("tenant_id") or "default") == tenant_id
                and (relation.source_uri, relation.relation_type, relation.target_uri) in identities
            )
        ]
        return len(identities)

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]:
        values = tuple(relations)
        if len(values) > 1_000:
            raise ValueError("ordinary relation reconcile batch exceeds 1000")
        prepared: dict[tuple[str, str, str], ContextRelation] = {}
        for relation in values:
            relation_tenant = str(relation.metadata.get("tenant_id") or "default")
            if relation_tenant != tenant_id:
                raise ValueError("ordinary relation tenant differs from reconcile tenant")
            if self.domain_classifier.owns_uri(relation.source_uri):
                raise ValueError("ordinary relation reconcile cannot mutate a canonical Source")
            if str(relation.metadata.get("catalog_record_key") or ""):
                raise ValueError("ordinary Source relation cannot claim Catalog projection ownership")
            identity = (relation.source_uri, relation.relation_type, relation.target_uri)
            prior = prepared.get(identity)
            if prior is not None and not self._ordinary_projection_equal(prior, relation):
                raise ValueError("ordinary relation batch contains a conflicting identity")
            prepared[identity] = relation
        written = 0
        skipped = 0
        for identity in sorted(prepared):
            relation = prepared[identity]
            existing = next(
                (
                    item
                    for item in self.relations
                    if item.source_uri == relation.source_uri
                    and item.relation_type == relation.relation_type
                    and item.target_uri == relation.target_uri
                    and str(item.metadata.get("tenant_id") or "default") == tenant_id
                ),
                None,
            )
            if existing is not None and self._ordinary_projection_equal(existing, relation):
                skipped += 1
                continue
            if existing is not None:
                self.delete_relation(
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    tenant_id=tenant_id,
                )
            self.add_relation(relation)
            written += 1
        return {"processed": len(prepared), "written": written, "skipped": skipped}

    @staticmethod
    def _ordinary_projection_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
        )

    def all_relations(self) -> list[ContextRelation]:
        return list(self.relations)
