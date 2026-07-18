"""Tenant-safe target resolution for ordinary Context operations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.index_store import IndexHit, IndexStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


@dataclass(frozen=True)
class ResolveResult:
    operation: ContextOperation
    resolved: bool
    reason: str = ""
    candidates: list[IndexHit] = field(default_factory=list)


class TargetResolver:
    """Resolve only ordinary SourceStore objects.

    A Markdown document URI is never a legal target here, even when a custom
    SourceStore implementation would otherwise accept it.
    """

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
        self.absolute_threshold = self._threshold(absolute_threshold, "absolute_threshold")
        self.margin_threshold = self._threshold(margin_threshold, "margin_threshold")

    def resolve(
        self,
        operation: ContextOperation,
        user_id: str | None = None,
        limit: int = 5,
    ) -> ResolveResult:
        commit_user = str(user_id or operation.user_id)
        if operation.user_id != commit_user:
            return self._reject(operation, "operation_user_mismatch")
        if not isinstance(operation.payload, dict):
            return self._reject(operation, "invalid_operation_payload")
        tenant_id = self._tenant(operation)
        if not tenant_id:
            return self._reject(operation, "invalid_payload_tenant")
        payload_error = self._validate_object_payload(operation, commit_user, tenant_id)
        if payload_error:
            return self._reject(operation, payload_error)

        if not operation.target_uri:
            declared = self._declared_target(operation)
            if declared:
                operation.target_uri = declared
        if operation.target_uri:
            return self._resolve_explicit(
                operation,
                commit_user=commit_user,
                tenant_id=tenant_id,
                creating=operation.action == OperationAction.ADD,
            )

        if operation.action == OperationAction.ADD:
            return self._reject(operation, "payload_target_uri_missing")
        if self.source_store is None or self.index_store is None:
            return self._pending(operation, "target_validation_unavailable")
        query = self._query_for(operation)
        if not query:
            return self._pending(operation, "target_review_required")
        filters: dict[str, object] = {
            "owner_user_id": commit_user,
            "context_type": operation.context_type.value,
        }
        project_id = operation.payload.get("project_id") or operation.payload.get("workspace_id")
        if isinstance(project_id, str) and project_id:
            filters["project_id"] = project_id
        try:
            raw_hits = self.index_store.search(
                query,
                tenant_id=tenant_id,
                filters=filters,
                limit=max(2, limit),
            )
        except (TypeError, ValueError):
            return self._pending(operation, "target_review_required")
        candidates = [
            hit
            for hit in raw_hits
            if isinstance(hit, IndexHit)
            and self._candidate_is_valid(hit, operation, commit_user, tenant_id)
        ]
        candidates.sort(key=lambda hit: (-self._relevance(hit), hit.uri))
        if not candidates:
            return self._pending(operation, "target_review_required")
        top = self._relevance(candidates[0])
        second = self._relevance(candidates[1]) if len(candidates) > 1 else 0.0
        if top < self.absolute_threshold:
            return self._pending(operation, "target_review_required", candidates)
        if len(candidates) > 1 and top - second < self.margin_threshold:
            return self._pending(operation, "target_ambiguous", candidates)
        operation.target_uri = candidates[0].uri
        operation.status = OperationStatus.RESOLVED
        return ResolveResult(operation, True, "target resolved by scoped index match", candidates)

    def _resolve_explicit(
        self,
        operation: ContextOperation,
        *,
        commit_user: str,
        tenant_id: str,
        creating: bool,
    ) -> ResolveResult:
        uri = str(operation.target_uri or "")
        error = self._validate_uri(uri, commit_user)
        if error:
            return self._reject(operation, error)
        declared = self._declared_target(operation)
        if operation.action != OperationAction.SUPERSEDE and declared and declared != uri:
            return self._reject(operation, "payload_target_uri_mismatch")
        if operation.action == OperationAction.SUPERSEDE:
            replacement = self._object_uri(operation)
            if not replacement:
                return self._reject(operation, "payload_target_uri_missing")
            if replacement == uri:
                return self._reject(operation, "supersede_replacement_matches_target")
            replacement_error = self._validate_uri(replacement, commit_user)
            if replacement_error:
                return self._reject(operation, f"payload_{replacement_error}")
        if not creating:
            if self.source_store is None:
                return self._pending(operation, "target_validation_unavailable")
            try:
                target = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return self._reject(operation, "target_not_found")
            except (TypeError, ValueError):
                return self._reject(operation, "target_object_invalid")
            object_error = self._validate_target(target, operation, commit_user, tenant_id)
            if object_error:
                return self._reject(operation, object_error)
        operation.status = OperationStatus.RESOLVED
        return ResolveResult(operation, True, "explicit target validated")

    def _candidate_is_valid(
        self,
        hit: IndexHit,
        operation: ContextOperation,
        commit_user: str,
        tenant_id: str,
    ) -> bool:
        if self._relevance(hit) <= 0 or self._validate_uri(hit.uri, commit_user):
            return False
        assert self.source_store is not None
        try:
            target = self.source_store.read_object(hit.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
            return False
        return not self._validate_target(
            target,
            operation,
            commit_user,
            tenant_id,
            expected_uri=hit.uri,
        )

    @staticmethod
    def _object_uri(operation: ContextOperation) -> str:
        payload = operation.payload.get("context_object")
        return str(payload.get("uri") or "") if isinstance(payload, dict) else ""

    def _declared_target(self, operation: ContextOperation) -> str:
        if operation.action == OperationAction.ADD:
            return self._object_uri(operation)
        for value in (
            operation.payload.get("target_uri"),
            operation.payload.get("policy_uri"),
            self._object_uri(operation) if operation.action != OperationAction.SUPERSEDE else "",
        ):
            if isinstance(value, str) and value:
                return value
        return ""

    def _validate_object_payload(
        self,
        operation: ContextOperation,
        commit_user: str,
        tenant_id: str,
    ) -> str:
        raw = operation.payload.get("context_object")
        if raw is None:
            return ""
        if not isinstance(raw, dict):
            return "invalid_payload_context_object"
        if raw.get("owner_user_id") not in {None, "", commit_user}:
            return "payload_owner_mismatch"
        if raw.get("tenant_id") not in {None, "", tenant_id}:
            return "payload_tenant_mismatch"
        if raw.get("context_type") not in {None, "", operation.context_type.value}:
            return "payload_context_type_mismatch"
        uri = raw.get("uri")
        if uri is not None and not isinstance(uri, str):
            return "invalid_payload_target_uri"
        if isinstance(uri, str) and uri:
            error = self._validate_uri(uri, commit_user)
            if error:
                return f"payload_{error}"
        return ""

    @staticmethod
    def _validate_uri(uri: str, commit_user: str) -> str:
        try:
            parsed = ContextURI.parse(uri)
        except (TypeError, ValueError):
            return "invalid_target_uri"
        if parsed.authority == "user" and parsed.user_id != commit_user:
            return "target_owner_mismatch"
        if parsed.authority == "user" and parsed.segments[1:3] == ("memory", "documents"):
            return "document_target_forbidden"
        return ""

    @staticmethod
    def _validate_target(
        target: ContextObject,
        operation: ContextOperation,
        commit_user: str,
        tenant_id: str,
        *,
        expected_uri: str | None = None,
    ) -> str:
        if target.uri != (expected_uri or operation.target_uri):
            return "target_uri_mismatch"
        if target.owner_user_id not in {None, "", commit_user}:
            return "target_owner_mismatch"
        if str(target.tenant_id or "default") != tenant_id:
            return "target_tenant_mismatch"
        if target.context_type != operation.context_type:
            return "target_context_type_mismatch"
        if target.lifecycle_state in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            return "target_not_active"
        return ""

    def _tenant(self, operation: ContextOperation) -> str:
        values = []
        raw = operation.payload.get("tenant_id")
        if raw not in {None, ""}:
            values.append(raw)
        obj = operation.payload.get("context_object")
        if isinstance(obj, dict) and obj.get("tenant_id") not in {None, ""}:
            values.append(obj["tenant_id"])
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return ""
        if len(set(values)) > 1:
            return ""
        return str(values[0]) if values else str(getattr(self.source_store, "tenant_id", "default") or "default")

    @staticmethod
    def _query_for(operation: ContextOperation) -> str:
        for key in ("query", "title", "content", "support_anchor_uri"):
            value = operation.payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        values = [operation.payload.get("scene_key"), operation.payload.get("action")]
        return " ".join(str(value) for value in values if value)

    @staticmethod
    def _relevance(hit: IndexHit) -> float:
        if not isinstance(hit.metadata, Mapping):
            return 0.0
        scores = hit.metadata.get("retrieval_scores")
        if not isinstance(scores, Mapping):
            return 0.0
        values: list[float] = []
        for name in ("lexical", "vector", "identity"):
            try:
                value = float(scores.get(name, 0.0))
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(value) or value < 0:
                return 0.0
            values.append(min(1.0, value))
        return max(values, default=0.0)

    @staticmethod
    def _threshold(value: float, label: str) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or not 0 <= resolved <= 1:
            raise ValueError(f"{label} must be a finite number between 0 and 1")
        return resolved

    @staticmethod
    def _pending(
        operation: ContextOperation,
        reason: str,
        candidates: list[IndexHit] | None = None,
    ) -> ResolveResult:
        operation.status = OperationStatus.PENDING
        operation.payload["target_resolution_reason"] = reason
        selected = list(candidates or [])
        if selected:
            operation.payload["target_candidates"] = [hit.__dict__ for hit in selected]
        return ResolveResult(operation, False, reason, selected)

    @staticmethod
    def _reject(operation: ContextOperation, reason: str) -> ResolveResult:
        operation.status = OperationStatus.REJECTED
        operation.payload["target_resolution_reason"] = reason
        return ResolveResult(operation, False, reason)


__all__ = ["ResolveResult", "TargetResolver"]
