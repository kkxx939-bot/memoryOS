"""In-memory IndexStore adapter used by tests and embedded runtimes."""

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


class InMemoryIndexStore:
    def __init__(self, *, domain_classifier: ContextDomainClassifier | None = None) -> None:
        self.rows: dict[str, tuple[ContextObject, str]] = {}
        self.domain_classifier = domain_classifier or NoDomainOverlay()

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        self.rows[obj.uri] = (obj, content)

    def delete_index(self, uri: str) -> None:
        self.rows.pop(uri, None)

    def indexed_uris(self) -> list[str]:
        return list(self.rows)

    def get_index_metadata(self, uri: str) -> dict | None:
        row = self.rows.get(uri)
        return (
            {
                **dict(row[0].metadata or {}),
                "tenant_id": str(row[0].tenant_id or ""),
                "owner_user_id": str(row[0].owner_user_id or ""),
                "context_type": row[0].context_type.value,
                "claim_state": str(
                    dict(row[0].metadata or {}).get("state") or dict(row[0].metadata or {}).get("claim_state") or ""
                ),
                "index_content_digest": canonical_digest(row[1]),
            }
            if row is not None
            else None
        )

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        del session_id
        row = self.rows.get(uri)
        if row is None:
            return "missing"
        obj = row[0]
        if str(obj.tenant_id or "default") != str(tenant_id):
            return "missing"
        if obj.lifecycle_state in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            return "inactive"
        return "active"

    def clear(self) -> None:
        self.rows.clear()

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        filters = filters or {}
        hits = []
        for obj, content in self.rows.values():
            if "allowed_uris" in filters and obj.uri not in set(filters.get("allowed_uris", []) or []):
                continue
            if filters.get("lifecycle_state") is None and obj.lifecycle_state in {
                LifecycleState.DELETED,
                LifecycleState.ARCHIVED,
                LifecycleState.OBSOLETE,
            }:
                continue
            if filters.get("lifecycle_state") and obj.lifecycle_state.value != filters["lifecycle_state"]:
                continue
            if filters.get("principal_owner_id") is not None:
                expected_owner = str(filters["principal_owner_id"])
                metadata = dict(obj.metadata or {})
                raw_scope = dict(metadata.get("scope", {}) or {})
                raw_applicability = dict(raw_scope.get("applicability", {}) or {})
                raw_visibility = dict(raw_scope.get("visibility", {}) or {})
                workspace = str(
                    metadata.get("workspace_id")
                    or metadata.get("project_id")
                    or next(
                        (
                            str(item.get("id"))
                            for item in raw_applicability.get("all_of", []) or []
                            if isinstance(item, dict) and item.get("kind") == "workspace"
                        ),
                        "",
                    )
                )
                shared_workspaces = {
                    str(value)
                    for value in filters.get("workspace_access_ids", ()) or ()
                    if str(value) not in {"", "__memoryos_principal_only__"}
                }
                record_kind = str(metadata.get("record_kind") or "")
                if not record_kind and str(metadata.get("canonical_kind") or "") == "claim":
                    record_kind = "claim_revision"
                canonical_shared = bool(
                    obj.context_type == ContextType.MEMORY
                    and record_kind in {"current_slot", "claim_revision"}
                    and str(metadata.get("slot_id") or metadata.get("canonical_slot_id") or "")
                    and str(metadata.get("claim_id") or metadata.get("canonical_claim_id") or "")
                    and str(raw_visibility.get("tenant_id") or "") == str(obj.tenant_id or "default")
                    and (
                        expected_owner
                        in {str(item) for item in raw_visibility.get("allowed_principal_ids", ()) or ()}
                        or (
                            raw_visibility.get("private") is False
                            and raw_visibility.get("allowed_principal_ids") in ([], ())
                            and raw_visibility.get("allowed_service_ids") in ([], ())
                            and workspace in shared_workspaces
                        )
                    )
                )
                if (
                    obj.owner_user_id != expected_owner
                    and not (
                        obj.owner_user_id in {None, ""}
                        and obj.context_type in {ContextType.RESOURCE, ContextType.SKILL}
                    )
                    and not canonical_shared
                ):
                    continue
            elif filters.get("owner_user_id") is not None:
                expected_owner = str(filters["owner_user_id"])
                if expected_owner:
                    if obj.context_type not in {ContextType.RESOURCE, ContextType.SKILL}:
                        if obj.owner_user_id != expected_owner:
                            continue
                    elif obj.owner_user_id not in {None, "", expected_owner}:
                        continue
                elif obj.owner_user_id not in {None, ""}:
                    continue
            if filters.get("tenant_id") and str(obj.tenant_id or "default") != str(filters["tenant_id"]):
                continue
            if filters.get("context_type") and obj.context_type.value != filters["context_type"]:
                continue
            metadata = dict(obj.metadata or {})
            if filters.get("adapter_access_id") is not None:
                actual_adapter = str(
                    metadata.get("source_adapter_id") or dict(metadata.get("connect", {}) or {}).get("adapter_id") or ""
                )
                if actual_adapter not in {"", str(filters["adapter_access_id"])}:
                    record_kind = str(metadata.get("record_kind") or "")
                    if obj.context_type not in {ContextType.SESSION, ContextType.RESOURCE, ContextType.SKILL} and (
                        record_kind != "current_slot"
                    ):
                        continue
            context_types = filters.get("context_types")
            if context_types is not None and obj.context_type.value not in {str(value) for value in context_types}:
                continue
            source_kinds = filters.get("source_kinds")
            if source_kinds is not None and str(metadata.get("source_kind") or "context") not in {
                str(value) for value in source_kinds
            }:
                continue
            record_kinds = filters.get("record_kinds")
            if record_kinds is not None:
                actual_record_kind = str(
                    metadata.get("record_kind")
                    or ("claim_revision" if metadata.get("canonical_kind") == "claim" else "context")
                )
                if actual_record_kind not in {str(value) for value in record_kinds}:
                    continue
            try:
                raw_scope = metadata.get("scope", {}) or {}
                if not isinstance(raw_scope, dict):
                    continue
                raw_applicability = raw_scope.get("applicability", {}) or {}
                if not isinstance(raw_applicability, dict):
                    continue
                actual_scope_keys = set(scope_keys_from_payloads(raw_applicability.get("all_of", [])))
            except (KeyError, TypeError, ValueError):
                continue
            admission = dict(metadata.get("admission", {}) or {})
            excluded_admission = {"restricted", "archive_only", "reject"}
            if not filters.get("include_candidates"):
                excluded_admission.add("pending")
            if filters.get("admission_status") is None and admission.get("decision") in excluded_admission:
                continue
            if filters.get("project_id"):
                scope = raw_scope
                fields = dict(metadata.get("fields", {}) or {})
                applicability = raw_applicability
                workspace = next(
                    (
                        str(item.get("id"))
                        for item in applicability.get("all_of", []) or []
                        if isinstance(item, dict) and item.get("kind") == "workspace"
                    ),
                    "",
                )
                project_id = str(
                    scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or workspace
                )
                memory_type = str(metadata.get("memory_type") or "")
                if memory_type in {"project_rule", "project_decision", "agent_experience"} and project_id != str(
                    filters["project_id"]
                ):
                    continue
            workspace_access = filters.get("workspace_access_ids")
            if workspace_access is not None:
                scope = raw_scope
                fields = dict(metadata.get("fields", {}) or {})
                applicability = raw_applicability
                workspace = str(
                    metadata.get("workspace_id")
                    or metadata.get("project_id")
                    or scope.get("project_id")
                    or fields.get("project_id")
                    or next(
                        (
                            str(item.get("id"))
                            for item in applicability.get("all_of", []) or []
                            if isinstance(item, dict) and item.get("kind") == "workspace"
                        ),
                        "",
                    )
                )
                if workspace not in {str(value) for value in workspace_access}:
                    continue
            metadata_matches = True
            for field in ("adapter_id", "admission_status", "claim_state", "slot_id", "memory_type"):
                expected = filters.get(field)
                if expected is None:
                    continue
                values = set(expected) if isinstance(expected, list | tuple | set | frozenset) else {expected}
                actual = {
                    "adapter_id": metadata.get("source_adapter_id")
                    or dict(metadata.get("connect", {}) or {}).get("adapter_id"),
                    "admission_status": dict(metadata.get("admission", {}) or {}).get("decision"),
                    "claim_state": metadata.get("state") or metadata.get("claim_state"),
                    "slot_id": metadata.get("slot_id"),
                    "memory_type": metadata.get("memory_type"),
                }[field]
                if actual not in values:
                    metadata_matches = False
                    break
            if not metadata_matches:
                continue
            required_scopes = set(filters.get("applicability_scope_keys", []) or [])
            if required_scopes and not actual_scope_keys.issubset(required_scopes):
                continue
            if filters.get("require_unscoped") and actual_scope_keys:
                continue
            text = " ".join([obj.title, content, json.dumps(obj.metadata, ensure_ascii=False)]).casefold()
            lexical_matches = lexical_match_count(query, text)
            lexical = lexical_relevance(query, text)
            identity = (
                1.0
                if any(
                    str(metadata.get(field, "")) == str(query).strip()
                    for field in {"scene_key", "action", "memory_anchor_uri"}
                )
                else 0.0
            )
            base_relevance = max(lexical, identity)
            if base_relevance <= 0:
                continue
            hotness = (obj.hotness + obj.semantic_hotness + obj.behavior_support_hotness) / 3.0
            score = max(float(lexical_matches), identity) + 0.05 * hotness
            canonical_projection = bool(
                self.domain_classifier.owns_object(obj)
                or str(metadata.get("canonical_kind") or "")
                in {"slot", "claim", "pending_proposal", "current_slot_projection"}
                or str(metadata.get("schema_version") or "").startswith("canonical_")
            )
            ordinary_serving_metadata = {
                key: metadata[key]
                for key in (
                    "adapter_id",
                    "admission",
                    "connect",
                    "memory_type",
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
            hit_metadata = {
                # Preserve the pre-existing ordinary-context fallback contract:
                # Behavior/Prediction consumers re-read Source for business
                # metadata. Only controlled serving/scope fields cross the
                # legacy IndexHit boundary; arbitrary reward/event payloads
                # do not affect Behavior selection. Canonical offline
                # validation requires the full projector proof envelope.
                **(metadata if canonical_projection else ordinary_serving_metadata),
                "tenant_id": str(obj.tenant_id or "default"),
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
            }
            hits.append(
                IndexHit(
                    uri=obj.uri,
                    score=score,
                    context_type=obj.context_type.value,
                    title=obj.title,
                    metadata=hit_metadata,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _lexical_terms(self, query: str) -> tuple[str, ...]:
        return lexical_terms(query)
