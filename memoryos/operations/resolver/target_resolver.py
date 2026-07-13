"""操作提交里的目标解析器。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexHit,
    IndexStore,
    SourceStore,
    is_canonical_memory_object,
)
from memoryos.memory.canonical.scope import MemoryScope, scope_key_from_payload
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


@dataclass(frozen=True)
class ResolveResult:
    operation: ContextOperation
    resolved: bool
    reason: str = ""
    candidates: list[IndexHit] = field(default_factory=list)


@dataclass(frozen=True)
class _TargetBoundary:
    user_id: str
    tenant_id: str
    context_type: str
    memory_type: str = ""
    workspace_id: str = ""
    exact_scope_keys: tuple[str, ...] = ()
    exact_scope_declared: bool = False
    available_scope_keys: tuple[str, ...] = ()
    scope_keys_declared: bool = False


class TargetResolver:
    def __init__(
        self,
        index_store: IndexStore | None = None,
        source_store: SourceStore | None = None,
        *,
        absolute_threshold: float = 0.75,
        margin_threshold: float = 0.10,
    ) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.absolute_threshold = self._validated_threshold(absolute_threshold, "absolute_threshold")
        self.margin_threshold = self._validated_threshold(margin_threshold, "margin_threshold")

    def resolve(self, operation: ContextOperation, user_id: str | None = None, limit: int = 5) -> ResolveResult:
        commit_user_id = str(user_id or operation.user_id)
        boundary, boundary_error = self._operation_boundary(operation, commit_user_id)
        if boundary_error:
            return self._reject(operation, boundary_error)

        if operation.target_uri:
            return self._resolve_explicit(operation, boundary, creating=operation.action == OperationAction.ADD)

        if operation.action != OperationAction.ADD:
            declared_payload_target = operation.payload.get("target_uri")
            object_payload = operation.payload.get("context_object")
            declared_object_target = (
                object_payload.get("uri")
                if operation.action != OperationAction.SUPERSEDE and isinstance(object_payload, dict)
                else None
            )
            for declared_target in (declared_payload_target, declared_object_target):
                if declared_target is None or declared_target == "":
                    continue
                if self._invalid_boundary_value(declared_target):
                    return self._reject(operation, "invalid_payload_target_uri")
                candidate_uri = str(declared_target)
                if operation.target_uri is not None and operation.target_uri != candidate_uri:
                    return self._reject(operation, "payload_target_uri_mismatch")
                operation.target_uri = candidate_uri
            if operation.target_uri:
                return self._resolve_explicit(operation, boundary, creating=False)

        if operation.action == OperationAction.ADD:
            object_payload = operation.payload.get("context_object")
            object_uri = ""
            if isinstance(object_payload, dict) and "uri" in object_payload:
                raw_object_uri = object_payload.get("uri")
                if self._invalid_boundary_value(raw_object_uri) or not raw_object_uri:
                    return self._reject(operation, "invalid_payload_target_uri")
                object_uri = str(raw_object_uri)
                error = self._validate_target_uri(object_uri, boundary)
                if error:
                    return self._reject(operation, error)
            elif isinstance(object_payload, dict) and object_payload:
                return self._reject(operation, "payload_target_uri_missing")
            for field_name in ("target_uri", "policy_uri"):
                declared_target = operation.payload.get(field_name)
                if declared_target is None or declared_target == "":
                    continue
                if self._invalid_boundary_value(declared_target):
                    return self._reject(operation, "invalid_payload_target_uri")
                declared_uri = str(declared_target)
                error = self._validate_target_uri(declared_uri, boundary)
                if error:
                    return self._reject(operation, error)
                if object_uri and declared_uri != object_uri:
                    return self._reject(operation, "payload_target_uri_mismatch")
            operation.status = OperationStatus.RESOLVED
            return ResolveResult(operation=operation, resolved=True, reason="add operation creates its target")

        payload_target = operation.payload.get("policy_uri") or operation.payload.get("target_uri")
        if payload_target and operation.action in {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }:
            operation.target_uri = str(payload_target)
            return self._resolve_explicit(operation, boundary, creating=False)

        if self.source_store is None:
            return self._pending(operation, "target_validation_unavailable")
        candidates = self._candidate_targets(operation, boundary=boundary, limit=max(10, 2, limit))
        candidates.sort(key=lambda hit: (self._base_relevance(hit), self._bounded_score(hit.score)), reverse=True)
        if candidates:
            top_score = self._base_relevance(candidates[0])
            second_score = self._base_relevance(candidates[1]) if len(candidates) > 1 else 0.0
            weak_single_cjk = self._single_cjk_lexical_only(
                self._query_for(operation),
                candidates[0],
            )
            if not weak_single_cjk and top_score >= self.absolute_threshold and (
                len(candidates) == 1 or top_score - second_score >= self.margin_threshold
            ):
                operation.target_uri = candidates[0].uri
                operation.status = OperationStatus.RESOLVED
                return ResolveResult(
                    operation=operation,
                    resolved=True,
                    reason="target resolved by relevant scoped index match",
                    candidates=candidates,
                )
            if top_score >= self.absolute_threshold and len(candidates) > 1:
                return self._pending(operation, "target_ambiguous", candidates)
            return self._pending(operation, "target_review_required", candidates)
        return self._pending(operation, "target_review_required")

    def _resolve_explicit(
        self,
        operation: ContextOperation,
        boundary: _TargetBoundary,
        *,
        creating: bool,
    ) -> ResolveResult:
        target_uri = str(operation.target_uri or "")
        error = self._validate_target_uri(target_uri, boundary)
        if error:
            return self._reject(operation, error)
        for field_name in ("target_uri", "policy_uri"):
            declared_target = operation.payload.get(field_name)
            if declared_target is None or declared_target == "":
                continue
            if self._invalid_boundary_value(declared_target):
                return self._reject(operation, "invalid_payload_target_uri")
            if str(declared_target) != target_uri:
                return self._reject(operation, "payload_target_uri_mismatch")
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict) and "uri" in object_payload:
            declared_payload_uri = object_payload.get("uri")
            if self._invalid_boundary_value(declared_payload_uri) or not declared_payload_uri:
                return self._reject(operation, "invalid_payload_target_uri")
            payload_uri = str(declared_payload_uri)
            payload_uri_error = self._validate_target_uri(payload_uri, boundary)
            if payload_uri_error:
                return self._reject(operation, f"payload_{payload_uri_error}")
            if operation.action == OperationAction.SUPERSEDE and payload_uri == target_uri:
                return self._reject(operation, "supersede_replacement_matches_target")
            if operation.action != OperationAction.SUPERSEDE and payload_uri != target_uri:
                return self._reject(operation, "payload_target_uri_mismatch")
        elif isinstance(object_payload, dict) and object_payload and operation.action in {
            OperationAction.ADD,
            OperationAction.UPDATE,
            OperationAction.MERGE,
        }:
            return self._reject(operation, "payload_target_uri_missing")
        if not creating:
            if self.source_store is None:
                if operation.action == OperationAction.SUPERSEDE:
                    return self._pending(operation, "target_validation_unavailable")
                return self._reject(operation, "target_validation_unavailable")
            try:
                target = self.source_store.read_object(target_uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return self._reject(operation, "target_not_found")
            except (TypeError, ValueError):
                return self._reject(operation, "target_object_invalid")
            try:
                error = self._validate_target_object(target_uri, target, boundary)
            except (TypeError, ValueError):
                return self._reject(operation, "target_object_invalid")
            if error:
                return self._reject(operation, error)
        operation.status = OperationStatus.RESOLVED
        return ResolveResult(operation=operation, resolved=True, reason="explicit target validated")

    def _candidate_targets(
        self,
        operation: ContextOperation,
        *,
        boundary: _TargetBoundary,
        limit: int,
    ) -> list[IndexHit]:
        if self.index_store is None or self.source_store is None:
            return []
        query = self._query_for(operation)
        if not query:
            return []
        filters: dict[str, Any] = {
            "owner_user_id": boundary.user_id,
            "tenant_id": boundary.tenant_id,
            "context_type": boundary.context_type,
        }
        if boundary.workspace_id:
            filters["project_id"] = boundary.workspace_id
        if boundary.memory_type:
            filters["memory_type"] = boundary.memory_type
        scope_keys = boundary.available_scope_keys or boundary.exact_scope_keys
        if scope_keys:
            filters["applicability_scope_keys"] = scope_keys
        try:
            hits = self.index_store.search(query, filters=filters, limit=limit)
        except (TypeError, ValueError):
            return []
        if not isinstance(hits, list | tuple):
            return []
        validated = []
        object_payload = operation.payload.get("context_object")
        payload_uri = (
            str(object_payload.get("uri"))
            if (
                operation.action != OperationAction.SUPERSEDE
                and isinstance(object_payload, dict)
                and object_payload.get("uri")
            )
            else ""
        )
        for hit in hits:
            if not isinstance(hit, IndexHit):
                continue
            if payload_uri and hit.uri != payload_uri:
                continue
            if self._finite_nonnegative(hit.score) is None or self._base_relevance(hit) <= 0:
                continue
            if self._validate_target_uri(hit.uri, boundary):
                continue
            try:
                target = self.source_store.read_object(hit.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
                continue
            if not isinstance(target, ContextObject) or not isinstance(target.metadata, Mapping):
                continue
            target_metadata = dict(target.metadata)
            raw_target_admission = target_metadata.get("admission", {})
            if raw_target_admission is not None and not isinstance(raw_target_admission, Mapping):
                continue
            target_admission = dict(raw_target_admission or {})
            if (
                target.lifecycle_state != LifecycleState.ACTIVE
                or target_metadata.get("canonical_kind") == "pending_proposal"
                or target_admission.get("decision") in {"pending", "restricted", "archive_only", "reject"}
            ):
                continue
            if self._validate_target_object(hit.uri, target, boundary):
                continue
            validated.append(hit)
        return validated

    def _operation_boundary(
        self,
        operation: ContextOperation,
        commit_user_id: str,
    ) -> tuple[_TargetBoundary, str]:
        if not commit_user_id or operation.user_id != commit_user_id:
            return self._empty_boundary(operation, commit_user_id), "operation_user_mismatch"
        if not isinstance(operation.payload, dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_operation_payload"
        payload = dict(operation.payload)
        raw_object_payload = payload.get("context_object")
        if raw_object_payload is not None and not isinstance(raw_object_payload, dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_context_object"
        object_payload = dict(raw_object_payload or {})
        raw_metadata = object_payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_metadata"
        metadata = dict(raw_metadata or {})
        owner_values = (payload.get("owner_user_id"), object_payload.get("owner_user_id"))
        if any(self._invalid_boundary_value(value) for value in owner_values):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_owner"
        payload_owners = {
            str(value)
            for value in owner_values
            if value is not None and value != ""
        }
        if payload_owners and payload_owners != {commit_user_id}:
            return self._empty_boundary(operation, commit_user_id), "payload_owner_mismatch"
        object_context_type = str(object_payload.get("context_type") or "")
        if object_context_type and object_context_type != operation.context_type.value:
            return self._empty_boundary(operation, commit_user_id), "payload_context_type_mismatch"

        tenant_inputs = (payload.get("tenant_id"), object_payload.get("tenant_id"))
        if any(self._invalid_boundary_value(value) for value in tenant_inputs):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_tenant"
        tenant_values = {
            str(value)
            for value in tenant_inputs
            if value is not None and value != ""
        }
        if payload.get("scope") is not None and not isinstance(payload.get("scope"), dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        if metadata.get("scope") is not None and not isinstance(metadata.get("scope"), dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        payload_scope = dict(payload.get("scope", {}) or {})
        metadata_scope = dict(metadata.get("scope", {}) or {})
        if not self._valid_scope_shape(payload_scope) or not self._valid_scope_shape(metadata_scope):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        if payload.get("scope") is not None and metadata.get("scope") is not None and payload_scope != metadata_scope:
            return self._empty_boundary(operation, commit_user_id), "payload_scope_mismatch"
        if metadata.get("fields") is not None and not isinstance(metadata.get("fields"), dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_metadata"
        scope = payload_scope if payload.get("scope") is not None else metadata_scope
        raw_visibility = scope.get("visibility", {})
        if raw_visibility is not None and not isinstance(raw_visibility, dict):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        visibility_tenant = dict(raw_visibility or {}).get("tenant_id")
        if self._invalid_boundary_value(visibility_tenant):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        if visibility_tenant not in {None, ""}:
            tenant_values.add(str(visibility_tenant))
        if len(tenant_values) > 1:
            return self._empty_boundary(operation, commit_user_id), "payload_tenant_mismatch"
        tenant_id = next(iter(tenant_values), str(getattr(self.source_store, "tenant_id", "default") or "default"))

        memory_type_inputs = (payload.get("memory_type"), metadata.get("memory_type"))
        if any(self._invalid_boundary_value(value) for value in memory_type_inputs):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_memory_type"
        memory_types = {
            str(value)
            for value in memory_type_inputs
            if value is not None and value != ""
        }
        if len(memory_types) > 1:
            return self._empty_boundary(operation, commit_user_id), "payload_memory_type_mismatch"
        memory_type = next(iter(memory_types), "")

        scoped_workspaces = self._workspace_values(scope, metadata)
        if scoped_workspaces is None:
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_workspace"
        workspace_inputs = (
            payload.get("workspace_id"),
            payload.get("project_id"),
            *scoped_workspaces,
        )
        if any(self._invalid_boundary_value(value) for value in workspace_inputs):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_workspace"
        workspaces = {
            str(value)
            for value in workspace_inputs
            if value is not None and value != ""
        }
        if len(workspaces) > 1:
            return self._empty_boundary(operation, commit_user_id), "payload_workspace_mismatch"
        workspace_id = next(iter(workspaces), "")
        structured_scope_keys = self._scope_keys(scope)
        structured_scope_declared = "applicability" in scope
        raw_available_scopes = payload.get("applicability_scope_keys", []) or []
        if not isinstance(raw_available_scopes, list | tuple):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        if any(not self._valid_scope_key(item) for item in raw_available_scopes):
            return self._empty_boundary(operation, commit_user_id), "invalid_payload_scope"
        declared_available_scope_keys = tuple(dict.fromkeys(str(item) for item in raw_available_scopes))
        declared_scope_workspaces = tuple(
            dict.fromkeys(
                workspace
                for scope_key in declared_available_scope_keys
                if (workspace := self._workspace_from_scope_key(scope_key))
            )
        )
        if len(declared_scope_workspaces) > 1:
            return self._empty_boundary(operation, commit_user_id), "payload_workspace_mismatch"
        declared_scope_workspace = next(iter(declared_scope_workspaces), "")
        if workspace_id and declared_scope_workspace and workspace_id != declared_scope_workspace:
            return self._empty_boundary(operation, commit_user_id), "payload_workspace_mismatch"
        if not workspace_id:
            workspace_id = declared_scope_workspace
        principal_scope_key = f"memoryos:principal:{commit_user_id}"
        declared_exact_scope_keys = tuple(
            dict.fromkeys((principal_scope_key, *declared_available_scope_keys))
        )
        if structured_scope_declared and declared_available_scope_keys and set(structured_scope_keys) != set(
            declared_exact_scope_keys
        ):
            return self._empty_boundary(operation, commit_user_id), "payload_scope_mismatch"
        exact_scope_declared = bool(structured_scope_declared or declared_available_scope_keys)
        exact_scope_keys = structured_scope_keys if structured_scope_declared else declared_exact_scope_keys
        available_scope_keys = tuple(
            dict.fromkeys(exact_scope_keys if exact_scope_declared else (principal_scope_key,))
        )
        return (
            _TargetBoundary(
                user_id=commit_user_id,
                tenant_id=tenant_id,
                context_type=operation.context_type.value,
                memory_type=memory_type,
                workspace_id=workspace_id,
                exact_scope_keys=exact_scope_keys,
                exact_scope_declared=exact_scope_declared,
                available_scope_keys=available_scope_keys,
                scope_keys_declared=exact_scope_declared,
            ),
            "",
        )

    def _empty_boundary(self, operation: ContextOperation, user_id: str) -> _TargetBoundary:
        return _TargetBoundary(
            user_id=user_id,
            tenant_id="default",
            context_type=operation.context_type.value,
        )

    def _validate_target_uri(self, uri: str, boundary: _TargetBoundary) -> str:
        try:
            parsed = ContextURI.parse(uri)
        except (TypeError, ValueError):
            return "invalid_target_uri"
        if parsed.authority == "user":
            if parsed.user_id != boundary.user_id:
                return "target_owner_mismatch"
        elif parsed.authority == "resources":
            if boundary.context_type != "resource":
                return "target_context_type_mismatch"
        elif parsed.authority == "skills" and boundary.context_type != "skill":
            return "target_context_type_mismatch"
        return ""

    def _validate_target_object(
        self,
        uri: str,
        target: ContextObject,
        boundary: _TargetBoundary,
    ) -> str:
        if target.uri != uri:
            return "target_uri_mismatch"
        try:
            parsed = ContextURI.parse(uri)
        except (TypeError, ValueError):
            return "invalid_target_uri"
        if parsed.authority == "user" and target.owner_user_id != boundary.user_id:
            return "target_owner_mismatch"
        if (
            parsed.authority != "user"
            and target.owner_user_id is not None
            and target.owner_user_id != ""
            and target.owner_user_id != boundary.user_id
        ):
            return "target_owner_mismatch"
        if str(target.tenant_id or "default") != boundary.tenant_id:
            return "target_tenant_mismatch"
        if target.context_type.value != boundary.context_type:
            return "target_context_type_mismatch"
        if not isinstance(target.metadata, Mapping):
            return "target_object_invalid"
        metadata = dict(target.metadata)
        target_memory_type = str(metadata.get("memory_type") or "")
        if boundary.memory_type and not target_memory_type:
            return "target_memory_type_unverified"
        if target_memory_type and not boundary.memory_type:
            return "target_memory_type_unverified"
        if target_memory_type != boundary.memory_type:
            return "target_memory_type_mismatch"
        raw_target_scope = metadata.get("scope", {})
        if raw_target_scope is not None and not isinstance(raw_target_scope, dict):
            return "target_scope_invalid"
        if metadata.get("fields") is not None and not isinstance(metadata.get("fields"), dict):
            return "target_scope_invalid"
        target_scope = dict(raw_target_scope or {})
        if is_canonical_memory_object(target):
            try:
                canonical_scope = MemoryScope.from_dict(target_scope)
            except (KeyError, TypeError, ValueError):
                return "target_scope_invalid"
            if canonical_scope.canonical_subject is None:
                return "target_scope_invalid"
            if canonical_scope.visibility.tenant_id != boundary.tenant_id:
                return "target_tenant_mismatch"
            if canonical_scope.authority.inferred:
                return "target_scope_invalid"
            asserted_by = str(
                metadata.get("asserted_by")
                or (target.owner_user_id if metadata.get("canonical_kind") == "pending_proposal" else "")
                or ""
            )
            asserted_by_service = str(metadata.get("asserted_by_service") or "")
            if (
                canonical_scope.authority.principal_ids
                or canonical_scope.authority.service_ids
            ) and not (
                asserted_by in set(canonical_scope.authority.principal_ids)
                or asserted_by_service in set(canonical_scope.authority.service_ids)
            ):
                return "target_scope_invalid"
        if not self._valid_scope_shape(target_scope):
            return "target_scope_invalid"
        target_workspaces = self._workspace_values(target_scope, metadata)
        if target_workspaces is None or len(target_workspaces) > 1:
            return "target_scope_invalid"
        target_workspace = next(iter(target_workspaces), "")
        if boundary.workspace_id and not target_workspace:
            return "target_workspace_unverified"
        if target_workspace and not boundary.workspace_id:
            return "target_workspace_unverified"
        if target_workspace != boundary.workspace_id:
            return "target_workspace_mismatch"
        target_scope_keys = set(self._scope_keys(target_scope))
        if boundary.exact_scope_declared:
            if target_scope_keys != set(boundary.exact_scope_keys):
                return "target_scope_mismatch"
        elif boundary.scope_keys_declared and not target_scope_keys:
            return "target_scope_unverified"
        elif target_scope_keys and not target_scope_keys.issubset(set(boundary.available_scope_keys)):
            return "target_scope_mismatch" if boundary.scope_keys_declared else "target_scope_unverified"
        raw_visibility = target_scope.get("visibility", {})
        if raw_visibility is not None and not isinstance(raw_visibility, dict):
            return "target_scope_invalid"
        visibility_tenant = dict(raw_visibility or {}).get("tenant_id")
        if self._invalid_boundary_value(visibility_tenant):
            return "target_scope_invalid"
        if visibility_tenant is not None and visibility_tenant != "" and str(visibility_tenant) != boundary.tenant_id:
            return "target_tenant_mismatch"
        return ""

    def _valid_scope_shape(self, scope: dict[str, Any]) -> bool:
        if not isinstance(scope, dict):
            return False
        for field_name in ("workspace_id", "project_id"):
            if self._invalid_boundary_value(scope.get(field_name)):
                return False
        visibility = scope.get("visibility")
        if visibility is not None and not isinstance(visibility, dict):
            return False
        if isinstance(visibility, dict) and self._invalid_boundary_value(visibility.get("tenant_id")):
            return False
        if "applicability" not in scope:
            return True
        applicability = scope.get("applicability")
        if not isinstance(applicability, dict):
            return False
        all_of = applicability.get("all_of", [])
        if not isinstance(all_of, list | tuple):
            return False
        for item in all_of:
            if not isinstance(item, dict):
                return False
            if not self._nonempty_string(item.get("kind")) or not self._nonempty_string(item.get("id")):
                return False
            if item.get("namespace") is not None and not self._nonempty_string(item.get("namespace")):
                return False
            try:
                scope_key_from_payload(item)
            except (KeyError, TypeError, ValueError):
                return False
        return True

    def _invalid_boundary_value(self, value: Any) -> bool:
        return value is not None and value != "" and not self._nonempty_string(value)

    def _scope_keys(self, scope: dict[str, Any]) -> tuple[str, ...]:
        if not self._valid_scope_shape(scope):
            return ()
        raw_applicability = scope.get("applicability")
        if not isinstance(raw_applicability, dict):
            return ()
        applicability = dict(raw_applicability)
        keys = [
            scope_key_from_payload(item)
            for item in applicability.get("all_of", []) or []
            if isinstance(item, dict) and item.get("kind") and item.get("id")
        ]
        return tuple(dict.fromkeys(keys))

    def _workspace_values(
        self,
        scope: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[str, ...] | None:
        if not self._valid_scope_shape(scope) or not isinstance(metadata, dict):
            return None
        raw_fields = metadata.get("fields", {})
        if raw_fields is not None and not isinstance(raw_fields, dict):
            return None
        fields = dict(raw_fields or {})
        raw_applicability = scope.get("applicability", {})
        if raw_applicability is not None and not isinstance(raw_applicability, dict):
            return None
        applicability = dict(raw_applicability or {})
        candidates: list[Any] = [
            scope.get("workspace_id"),
            scope.get("project_id"),
            fields.get("workspace_id"),
            fields.get("project_id"),
            metadata.get("workspace_id"),
            metadata.get("project_id"),
        ]
        candidates.extend(
            item.get("id")
            for item in applicability.get("all_of", []) or []
            if isinstance(item, dict) and item.get("kind") == "workspace"
        )
        if any(self._invalid_boundary_value(value) for value in candidates):
            return None
        return tuple(dict.fromkeys(str(value) for value in candidates if value not in {None, ""}))

    def _valid_scope_key(self, value: Any) -> bool:
        if not self._nonempty_string(value) or any(character.isspace() for character in value):
            return False
        parts = value.split(":")
        return len(parts) >= 3 and all(parts[:3])

    def _workspace_from_scope_key(self, scope_key: str) -> str:
        parts = scope_key.split(":", 3)
        if len(parts) < 3 or parts[1] != "workspace":
            return ""
        return parts[3] if parts[2] == "path" and len(parts) == 4 else parts[2]

    def _nonempty_string(self, value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _base_relevance(self, hit: IndexHit) -> float:
        if not isinstance(hit.metadata, Mapping):
            return 0.0
        raw_scores = hit.metadata.get("retrieval_scores", {})
        if not isinstance(raw_scores, Mapping):
            return 0.0
        scores = dict(raw_scores)
        values = []
        for name in ("lexical", "vector", "identity"):
            value = self._finite_nonnegative(scores.get(name, 0.0))
            if value is None:
                return 0.0
            values.append(self._bounded_score(value))
        return max(values, default=0.0)

    def _single_cjk_lexical_only(self, query: str, hit: IndexHit) -> bool:
        normalized = str(query).strip()
        if not re.fullmatch(r"[\u4e00-\u9fff]", normalized):
            return False
        if not isinstance(hit.metadata, Mapping):
            return True
        raw_scores = hit.metadata.get("retrieval_scores", {})
        if not isinstance(raw_scores, Mapping):
            return True
        scores = dict(raw_scores)
        vector = self._finite_nonnegative(scores.get("vector", 0.0))
        identity = self._finite_nonnegative(scores.get("identity", 0.0))
        return not ((vector is not None and vector > 0) or (identity is not None and identity > 0))

    def _bounded_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))

    def _finite_nonnegative(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score) or score < 0:
            return None
        return score

    def _validated_threshold(self, value: Any, field_name: str) -> float:
        threshold = self._finite_nonnegative(value)
        if threshold is None or threshold > 1.0:
            raise ValueError(f"{field_name} must be a finite number between 0 and 1")
        return threshold

    def _pending(
        self,
        operation: ContextOperation,
        reason: str,
        candidates: list[IndexHit] | None = None,
    ) -> ResolveResult:
        operation.status = OperationStatus.PENDING
        candidates = list(candidates or [])
        payload = dict(operation.payload) if isinstance(operation.payload, dict) else {}
        operation.payload = {**payload, "target_resolution_reason": reason}
        if candidates:
            operation.payload["target_candidates"] = [hit.__dict__ for hit in candidates]
        return ResolveResult(operation=operation, resolved=False, reason=reason, candidates=candidates)

    def _reject(self, operation: ContextOperation, reason: str) -> ResolveResult:
        operation.status = OperationStatus.REJECTED
        payload = dict(operation.payload) if isinstance(operation.payload, dict) else {}
        operation.payload = {**payload, "target_resolution_reason": reason}
        return ResolveResult(operation=operation, resolved=False, reason=reason)

    def _query_for(self, operation: ContextOperation) -> str:
        for key in ("query", "title", "content", "memory_anchor_uri"):
            value = operation.payload.get(key)
            if value:
                return str(value)
        scene_key = operation.payload.get("scene_key")
        action = operation.payload.get("action")
        if scene_key or action:
            return " ".join(str(item) for item in (scene_key, action) if item)
        return ""
