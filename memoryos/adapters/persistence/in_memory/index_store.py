"""Tenant-safe in-memory IndexStore used by tests and embedded runtimes."""

from __future__ import annotations

import json

from memoryos.contextdb.extensions import ContextDomainClassifier, NoDomainOverlay
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.lexical import lexical_match_count, lexical_relevance, lexical_terms
from memoryos.contextdb.store.index_store import IndexHit
from memoryos.core.integrity import canonical_digest
from memoryos.core.types import scope_keys_from_payloads
from memoryos.security.context_projection import ContextProjectionSanitizer


class InMemoryIndexStore:
    def __init__(self, *, domain_classifier: ContextDomainClassifier | None = None) -> None:
        self.rows: dict[tuple[str, str], tuple[ContextObject, str]] = {}
        self.domain_classifier = domain_classifier or NoDomainOverlay()

    @staticmethod
    def _tenant(tenant_id: str) -> str:
        resolved = str(tenant_id or "").strip()
        if not resolved:
            raise ValueError("tenant_id is required")
        return resolved

    def upsert_index(self, obj: ContextObject, content: str = "", *, tenant_id: str) -> None:
        resolved = self._tenant(tenant_id)
        if str(obj.tenant_id or "default") != resolved:
            raise ValueError("ContextObject tenant does not match tenant_id")
        self.rows[(resolved, obj.uri)] = (obj, content)

    def delete_index(self, uri: str, *, tenant_id: str) -> None:
        self.rows.pop((self._tenant(tenant_id), str(uri)), None)

    def indexed_uris(self, *, tenant_id: str) -> list[str]:
        resolved = self._tenant(tenant_id)
        return [uri for row_tenant, uri in self.rows if row_tenant == resolved]

    def get_index_metadata(self, uri: str, *, tenant_id: str) -> dict | None:
        resolved = self._tenant(tenant_id)
        row = self.rows.get((resolved, str(uri)))
        if row is None:
            return None
        obj, content = row
        metadata = dict(obj.metadata or {})
        return {
            **metadata,
            "tenant_id": resolved,
            "owner_user_id": str(obj.owner_user_id or ""),
            "context_type": obj.context_type.value,
            "document_id": str(metadata.get("document_id") or ""),
            "block_id": str(metadata.get("block_id") or ""),
            "document_kind": str(metadata.get("document_kind") or ""),
            "document_revision": int(metadata.get("document_revision") or 0),
            "projection_generation": int(metadata.get("projection_generation") or 0),
            "index_content_digest": canonical_digest(content),
        }

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        del session_id
        row = self.rows.get((self._tenant(tenant_id), str(uri)))
        if row is None:
            return "missing"
        if row[0].lifecycle_state in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            return "inactive"
        return "active"

    def clear(self, *, tenant_id: str) -> None:
        resolved = self._tenant(tenant_id)
        for key in tuple(self.rows):
            if key[0] == resolved:
                self.rows.pop(key, None)

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        filters: dict | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        resolved = self._tenant(tenant_id)
        normalized = dict(filters or {})
        supplied_tenant = normalized.get("tenant_id")
        if supplied_tenant is not None and str(supplied_tenant) != resolved:
            raise ValueError("structured filters cannot cross tenant boundary")
        hits: list[IndexHit] = []
        sanitizer = ContextProjectionSanitizer()
        for (row_tenant, _uri), (obj, content) in self.rows.items():
            if row_tenant != resolved:
                continue
            metadata = dict(obj.metadata or {})
            if not self._matches(obj, metadata, normalized):
                continue
            text = " ".join((obj.title, content, json.dumps(metadata, ensure_ascii=False))).casefold()
            lexical_matches = lexical_match_count(query, text)
            lexical = lexical_relevance(query, text)
            identity = 1.0 if any(
                str(metadata.get(field) or "") == str(query).strip()
                for field in (
                    "scene_key",
                    "action",
                    "support_anchor_uri",
                    "document_id",
                    "block_id",
                )
            ) else 0.0
            base_relevance = max(lexical, identity)
            if base_relevance <= 0:
                continue
            hotness = (obj.hotness + obj.semantic_hotness + obj.behavior_support_hotness) / 3.0
            score = max(float(lexical_matches), identity) + 0.05 * hotness
            serving_metadata = {
                key: metadata[key]
                for key in (
                    "adapter_id",
                    "connect",
                    "document_id",
                    "block_id",
                    "document_kind",
                    "document_revision",
                    "projection_generation",
                    "project_id",
                    "record_kind",
                    "retrieval_views",
                    "scope",
                    "scope_keys",
                    "session_id",
                    "source_adapter_id",
                    "source_kind",
                    "workspace_id",
                )
                if key in metadata
            }
            serving_metadata.setdefault("record_kind", "context")
            serving_metadata.setdefault("source_kind", "context")
            serving_metadata.setdefault("source_uri", str(metadata.get("source_uri") or obj.uri))
            serving_metadata.setdefault(
                "source_digest",
                str(
                    metadata.get("source_digest")
                    or sanitizer.digest(content or obj.to_dict())
                ),
            )
            hits.append(
                IndexHit(
                    uri=obj.uri,
                    score=score,
                    context_type=obj.context_type.value,
                    title=obj.title,
                    metadata={
                        **serving_metadata,
                        "tenant_id": resolved,
                        "owner_user_id": str(obj.owner_user_id or ""),
                        "context_type": obj.context_type.value,
                        "retrieval_scores": {
                            "lexical": lexical,
                            "vector": 0.0,
                            "identity": identity,
                            "base_relevance": base_relevance,
                            "hotness": hotness,
                            "score": score,
                        },
                    },
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.uri))
        return hits[: max(0, int(limit))]

    @staticmethod
    def _matches(obj: ContextObject, metadata: dict, filters: dict) -> bool:
        if "allowed_uris" in filters and obj.uri not in set(filters.get("allowed_uris", ()) or ()):
            return False
        if filters.get("lifecycle_state") is None and obj.lifecycle_state in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            return False
        if filters.get("lifecycle_state") and obj.lifecycle_state.value != str(filters["lifecycle_state"]):
            return False
        principal = filters.get("principal_owner_id")
        if principal is not None:
            scope = metadata.get("scope")
            visibility = scope.get("visibility") if isinstance(scope, dict) else None
            allowed_principals = (
                {str(item) for item in visibility.get("allowed_principal_ids", ()) or ()}
                if isinstance(visibility, dict)
                else set()
            )
            allowed_services = (
                {str(item) for item in visibility.get("allowed_service_ids", ()) or ()}
                if isinstance(visibility, dict)
                else set()
            )
            tenant_public = bool(
                isinstance(visibility, dict)
                and str(visibility.get("tenant_id") or "") == str(obj.tenant_id or "default")
                and visibility.get("private") is False
                and not allowed_principals
                and not allowed_services
            )
            public_resource = not obj.owner_user_id and obj.context_type in {
                ContextType.RESOURCE,
                ContextType.SKILL,
            }
            service_allowed = bool(
                filters.get("service_access_id")
                and str(filters["service_access_id"]) in allowed_services
            )
            if not (
                str(obj.owner_user_id or "") == str(principal)
                or public_resource
                or str(principal) in allowed_principals
                or service_allowed
                or tenant_public
            ):
                return False
        owner = filters.get("owner_user_id")
        if owner is not None and str(obj.owner_user_id or "") != str(owner):
            return False
        if filters.get("context_type") and obj.context_type.value != str(filters["context_type"]):
            return False
        context_types = filters.get("context_types")
        if context_types is not None and obj.context_type.value not in {str(value) for value in context_types}:
            return False
        source_kinds = filters.get("source_kinds")
        if source_kinds is not None and str(metadata.get("source_kind") or "context") not in {
            str(value) for value in source_kinds
        }:
            return False
        record_kinds = filters.get("record_kinds")
        if record_kinds is not None and str(metadata.get("record_kind") or "context") not in {
            str(value) for value in record_kinds
        }:
            return False
        for name in ("document_id", "block_id", "document_kind", "document_revision", "projection_generation"):
            if name in filters and str(metadata.get(name) or "") != str(filters[name]):
                return False
        raw_scope = metadata.get("scope", {}) or {}
        if not isinstance(raw_scope, dict):
            return False
        applicability = raw_scope.get("applicability", {}) or {}
        if not isinstance(applicability, dict):
            return False
        try:
            actual_scope_keys = set(scope_keys_from_payloads(applicability.get("all_of", [])))
        except (KeyError, TypeError, ValueError):
            return False
        required_scopes = set(filters.get("applicability_scope_keys", ()) or ())
        if required_scopes and not actual_scope_keys.issubset(required_scopes):
            return False
        return not (filters.get("require_unscoped") and actual_scope_keys)

    @staticmethod
    def _lexical_terms(query: str) -> tuple[str, ...]:
        return lexical_terms(query)


__all__ = ["InMemoryIndexStore"]
