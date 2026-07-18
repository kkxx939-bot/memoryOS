"""Tenant-qualified in-memory RelationStore adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memoryos.contextdb.extensions import ContextDomainClassifier, NoDomainOverlay
from memoryos.contextdb.model.context_relation import ContextRelation


class InMemoryRelationStore:
    def __init__(self, *, domain_classifier: ContextDomainClassifier | None = None) -> None:
        self.relations: list[ContextRelation] = []
        self.domain_classifier = domain_classifier or NoDomainOverlay()

    @staticmethod
    def _require_tenant(tenant_id: str) -> str:
        resolved = str(tenant_id or "").strip()
        if not resolved:
            raise ValueError("tenant_id is required")
        return resolved

    def add_relation(self, relation: ContextRelation, *, tenant_id: str) -> None:
        resolved_tenant = self._require_tenant(tenant_id)
        metadata = self._metadata_for_tenant(relation.metadata, resolved_tenant)
        stored = relation
        if metadata != dict(relation.metadata or {}):
            stored = ContextRelation(
                source_uri=relation.source_uri,
                relation_type=relation.relation_type,
                target_uri=relation.target_uri,
                weight=relation.weight,
                metadata=metadata,
                created_at=relation.created_at,
            )
        identity = self._identity(stored)
        self.relations = [
            item
            for item in self.relations
            if (self._relation_tenant(item), *self._identity(item))
            != (resolved_tenant, *identity)
        ]
        self.relations.append(stored)

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        limit: int | None = None,
    ) -> list[ContextRelation]:
        resolved_tenant = self._require_tenant(tenant_id)
        rows = [
            relation
            for relation in self.relations
            if self._relation_tenant(relation) == resolved_tenant
            and (relation.source_uri == uri or relation.target_uri == uri)
        ]
        if owner_user_id is not None:
            rows = [
                relation
                for relation in rows
                if str(relation.metadata.get("owner_user_id") or "") in {"", str(owner_user_id)}
            ]
        rows.sort(
            key=lambda item: (
                -item.weight,
                item.created_at,
                item.source_uri,
                item.relation_type,
                item.target_uri,
            )
        )
        return rows[: max(0, min(int(limit), 1_000))] if limit is not None else rows

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str,
    ) -> None:
        resolved_tenant = self._require_tenant(tenant_id)
        identity = (str(source_uri), str(relation_type), str(target_uri))
        self.relations = [
            relation
            for relation in self.relations
            if (self._relation_tenant(relation), *self._identity(relation))
            != (resolved_tenant, *identity)
        ]

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int:
        resolved_tenant = self._require_tenant(tenant_id)
        resolved_record_key = str(catalog_record_key or "").strip()
        if not resolved_record_key:
            raise ValueError("catalog_record_key is required")
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if self._relation_tenant(relation) == resolved_tenant
            and (relation.source_uri == uri or relation.target_uri == uri)
            and str(relation.metadata.get("catalog_record_key") or "") == resolved_record_key
        ][:maximum]
        identities = {(resolved_tenant, *self._identity(relation)) for relation in selected}
        self.relations = [
            relation
            for relation in self.relations
            if (self._relation_tenant(relation), *self._identity(relation)) not in identities
        ]
        return len(identities)

    def delete_memory_document_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        owner_user_id: str,
        limit: int,
    ) -> int:
        """Delete one bounded exact-owner batch touching a memory document URI."""

        resolved_tenant = self._require_tenant(tenant_id)
        resolved_owner = str(owner_user_id or "").strip()
        if not resolved_owner:
            raise ValueError("owner_user_id is required")
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if self._relation_tenant(relation) == resolved_tenant
            and str(relation.metadata.get("owner_user_id") or "") == resolved_owner
            and (relation.source_uri == uri or relation.target_uri == uri)
        ][:maximum]
        identities = {(resolved_tenant, *self._identity(relation)) for relation in selected}
        self.relations = [
            relation
            for relation in self.relations
            if (self._relation_tenant(relation), *self._identity(relation)) not in identities
        ]
        return len(identities)

    def delete_uri_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> int:
        resolved_tenant = self._require_tenant(tenant_id)
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if self._relation_tenant(relation) == resolved_tenant
            and (relation.source_uri == uri or relation.target_uri == uri)
        ][:maximum]
        identities = {(resolved_tenant, *self._identity(relation)) for relation in selected}
        self.relations = [
            relation
            for relation in self.relations
            if (self._relation_tenant(relation), *self._identity(relation)) not in identities
        ]
        return len(identities)

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int:
        resolved_tenant = self._require_tenant(tenant_id)
        maximum = max(1, min(int(limit), 1_000))
        selected = [
            relation
            for relation in self.relations
            if self._relation_tenant(relation) == resolved_tenant
            and not self.domain_classifier.owns_uri(relation.source_uri)
        ][:maximum]
        identities = {(resolved_tenant, *self._identity(relation)) for relation in selected}
        self.relations = [
            relation
            for relation in self.relations
            if (self._relation_tenant(relation), *self._identity(relation)) not in identities
        ]
        return len(identities)

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]:
        resolved_tenant = self._require_tenant(tenant_id)
        values = tuple(relations)
        if len(values) > 1_000:
            raise ValueError("ordinary relation reconcile batch exceeds 1000")
        prepared: dict[tuple[str, str, str], ContextRelation] = {}
        for relation in values:
            metadata = self._metadata_for_tenant(relation.metadata, resolved_tenant)
            normalized = relation
            if metadata != dict(relation.metadata or {}):
                normalized = ContextRelation(
                    source_uri=relation.source_uri,
                    relation_type=relation.relation_type,
                    target_uri=relation.target_uri,
                    weight=relation.weight,
                    metadata=metadata,
                    created_at=relation.created_at,
                )
            if self.domain_classifier.owns_uri(normalized.source_uri):
                raise ValueError("ordinary relation reconcile cannot mutate a domain-owned Source")
            if str(normalized.metadata.get("catalog_record_key") or ""):
                raise ValueError("ordinary Source relation cannot claim Catalog projection ownership")
            identity = self._identity(normalized)
            prior = prepared.get(identity)
            if prior is not None and not self._ordinary_projection_equal(prior, normalized):
                raise ValueError("ordinary relation batch contains a conflicting identity")
            prepared[identity] = normalized

        written = 0
        skipped = 0
        for identity in sorted(prepared):
            relation = prepared[identity]
            existing = next(
                (
                    item
                    for item in self.relations
                    if self._relation_tenant(item) == resolved_tenant
                    and self._identity(item) == identity
                ),
                None,
            )
            if existing is not None and self._ordinary_projection_equal(existing, relation):
                skipped += 1
                continue
            self.add_relation(relation, tenant_id=resolved_tenant)
            written += 1
        return {"processed": len(prepared), "written": written, "skipped": skipped}

    def all_relations(self, *, tenant_id: str) -> list[ContextRelation]:
        resolved_tenant = self._require_tenant(tenant_id)
        return [
            relation
            for relation in self.relations
            if self._relation_tenant(relation) == resolved_tenant
        ]

    @staticmethod
    def _identity(relation: ContextRelation) -> tuple[str, str, str]:
        return relation.source_uri, relation.relation_type, relation.target_uri

    @staticmethod
    def _relation_tenant(relation: ContextRelation) -> str:
        tenant_id = str(relation.metadata.get("tenant_id") or "").strip()
        if not tenant_id:
            raise ValueError("stored relation is missing tenant_id")
        return tenant_id

    @staticmethod
    def _metadata_for_tenant(metadata: Mapping[str, object], tenant_id: str) -> dict[str, object]:
        result = dict(metadata or {})
        supplied = str(result.get("tenant_id") or "")
        if supplied and supplied != tenant_id:
            raise ValueError("relation tenant differs from explicit tenant_id")
        result["tenant_id"] = tenant_id
        return result

    @staticmethod
    def _ordinary_projection_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
            and left.created_at == right.created_at
        )


__all__ = ["InMemoryRelationStore"]
