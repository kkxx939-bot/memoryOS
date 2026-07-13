"""Immutable pre-write planning and pending-intent proofs.

These artifacts are published before redo/Source mutation.  Receipts only
reference their digests, preserving the one-way dependency DAG.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.memory.canonical.event import canonical_digest
from memoryos.operations.commit.effect_marker import atomic_create_json
from memoryos.operations.model.context_operation import ContextOperation

try:  # pragma: no cover - production Unix platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


DIRECT_PLANNING_PROOF_SCHEMA_VERSION = "memory_direct_planning_proof_v1"
PENDING_PREPARED_INTENT_SCHEMA_VERSION = "memory_pending_prepared_intent_v1"
CANONICAL_PREPARED_INTENT_SCHEMA_VERSION = "memory_canonical_prepared_intent_v1"


class PlanningProofIntegrityError(RuntimeError):
    """An immutable planning or prepared-intent artifact is invalid."""


def normalized_operation_set(operations: Sequence[Any]) -> list[dict[str, Any]]:
    """Return the stable semantic operation set used by pre-write proofs."""

    normalized: list[dict[str, Any]] = []
    for operation in operations:
        raw = operation.to_dict() if callable(getattr(operation, "to_dict", None)) else dict(operation)
        item = deepcopy(raw)
        item.pop("status", None)
        item.pop("created_at", None)
        payload = dict(item.get("payload", {}) or {})
        payload.pop("planning_digest", None)
        context_object = payload.get("context_object")
        if isinstance(context_object, dict):
            relations = context_object.get("relations")
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, dict):
                        relation.pop("created_at", None)
        item["payload"] = payload
        normalized.append(item)
    return sorted(normalized, key=lambda item: str(item.get("operation_id") or ""))


def operation_set_digest(operations: Sequence[Any]) -> str:
    if not operations:
        raise ValueError("planning proof requires at least one operation")
    return canonical_digest(normalized_operation_set(operations))


def _context_operations(operations: Sequence[Any]) -> list[ContextOperation]:
    return [
        operation if isinstance(operation, ContextOperation) else ContextOperation.from_dict(dict(operation))
        for operation in operations
    ]


class ImmutablePlanningProofStore:
    """Create-only store for direct plans and pending prepared intents."""

    def __init__(self, artifact_root: str | Path, *, tenant_id: str) -> None:
        self.artifact_root = Path(artifact_root)
        self.tenant_id = require_safe_path_segment(tenant_id, "planning proof tenant_id")

    def direct_path(self, proof_id: str) -> Path:
        safe = require_safe_path_segment(proof_id, "direct planning proof id")
        return self.artifact_root / "system" / "direct-planning-proofs" / f"{safe}.json"

    def intent_path(self, operation_id: str) -> Path:
        safe = require_safe_path_segment(operation_id, "pending prepared intent operation_id")
        return self.artifact_root / "system" / "prepared-intents" / f"{safe}.json"

    def canonical_intent_path(self, transaction_id: str) -> Path:
        safe = require_safe_path_segment(
            transaction_id,
            "canonical prepared intent transaction_id",
        )
        return self.artifact_root / "system" / "canonical-prepared-intents" / f"{safe}.json"

    def ensure_direct(
        self,
        operations: Sequence[Any],
        *,
        kind: str,
        transaction_id: str,
        idempotency_key: str,
        user_id: str,
        commit_group_id: str,
    ) -> dict[str, Any]:
        payload = self.build_direct(
            operations,
            kind=kind,
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            user_id=user_id,
            commit_group_id=commit_group_id,
        )
        proof_id = str(payload["proof_id"])
        path = self.direct_path(proof_id)
        with self._artifact_lock(path):
            self._create_or_match(path, payload, self.validate_direct)
        return self.validate_direct(self._read(path), operations=operations)

    def build_direct(
        self,
        operations: Sequence[Any],
        *,
        kind: str,
        transaction_id: str,
        idempotency_key: str,
        user_id: str,
        commit_group_id: str,
    ) -> dict[str, Any]:
        """Build and validate a direct proof without publishing an artifact."""

        proof_id = require_safe_path_segment(transaction_id, "direct planning transaction_id")
        if not isinstance(commit_group_id, str) or not commit_group_id:
            raise PlanningProofIntegrityError("direct planning proof requires a commit group identity")
        core: dict[str, Any] = {
            "schema_version": DIRECT_PLANNING_PROOF_SCHEMA_VERSION,
            "proof_id": proof_id,
            "kind": kind,
            "transaction_id": proof_id,
            "idempotency_key": require_safe_path_segment(idempotency_key, "direct planning idempotency_key"),
            "tenant_id": self.tenant_id,
            "user_id": require_safe_path_segment(user_id, "direct planning user_id"),
            "commit_group_id": commit_group_id,
            "operation_ids": sorted(
                str(item.get("operation_id") or "") for item in normalized_operation_set(operations)
            ),
            "operation_set_digest": operation_set_digest(operations),
            "proposal_fingerprints": sorted(
                {
                    str(value)
                    for operation in normalized_operation_set(operations)
                    for value in dict(operation.get("payload", {}) or {}).get("proposal_fingerprints", []) or []
                }
            ),
        }
        core["planning_digest"] = canonical_digest(core)
        payload = {**core, "proof_digest": canonical_digest(core)}
        declared = {
            str(
                dict(
                    (item.to_dict() if callable(getattr(item, "to_dict", None)) else dict(item)).get("payload", {})
                    or {}
                ).get("planning_digest")
                or ""
            )
            for item in operations
        } - {""}
        if len(declared) > 1:
            raise PlanningProofIntegrityError("direct operation set declares multiple planning digests")
        if declared and declared != {str(payload["planning_digest"])}:
            raise PlanningProofIntegrityError("direct operation set declares a different planning digest")
        return self.validate_direct(payload, operations=operations)

    def validate_direct(
        self,
        payload: object,
        *,
        operations: Sequence[Any] | None = None,
        transaction_id: str | None = None,
        planning_digest: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != DIRECT_PLANNING_PROOF_SCHEMA_VERSION:
            raise PlanningProofIntegrityError("direct planning proof schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "proof_digest"}
        expected_planning = canonical_digest({key: value for key, value in core.items() if key != "planning_digest"})
        if (
            payload.get("proof_digest") != canonical_digest(core)
            or payload.get("planning_digest") != expected_planning
            or payload.get("tenant_id") != self.tenant_id
            or not isinstance(payload.get("commit_group_id"), str)
            or not payload.get("commit_group_id")
        ):
            raise PlanningProofIntegrityError("direct planning proof digest is corrupt")
        if transaction_id is not None and payload.get("transaction_id") != transaction_id:
            raise PlanningProofIntegrityError("direct planning proof transaction identity differs")
        if planning_digest is not None and payload.get("planning_digest") != planning_digest:
            raise PlanningProofIntegrityError("direct planning proof digest differs from receipt")
        if operations is not None:
            if payload.get("operation_set_digest") != operation_set_digest(operations):
                raise PlanningProofIntegrityError("direct planning proof operation set differs")
            operation_ids = sorted(str(item.get("operation_id") or "") for item in normalized_operation_set(operations))
            if payload.get("operation_ids") != operation_ids:
                raise PlanningProofIntegrityError("direct planning proof membership differs")
        return payload

    def load_direct(
        self,
        transaction_id: str,
        *,
        operations: Sequence[Any] | None = None,
        planning_digest: str | None = None,
    ) -> dict[str, Any]:
        path = self.direct_path(transaction_id)
        if not path.exists() or path.is_symlink():
            raise PlanningProofIntegrityError("immutable direct planning proof is missing")
        return self.validate_direct(
            self._read(path),
            operations=operations,
            transaction_id=transaction_id,
            planning_digest=planning_digest,
        )

    def ensure_pending_intent(self, operation: Any, *, relation_manifest: object) -> dict[str, Any]:
        normalized = normalized_operation_set([operation])
        operation_id = require_safe_path_segment(str(normalized[0].get("operation_id") or ""), "operation_id")
        payload_data = dict(normalized[0].get("payload", {}) or {})
        commit_group_id = payload_data.get("commit_group_id")
        if not isinstance(commit_group_id, str) or not commit_group_id:
            raise PlanningProofIntegrityError("pending prepared-intent requires a commit group identity")
        core: dict[str, Any] = {
            "schema_version": PENDING_PREPARED_INTENT_SCHEMA_VERSION,
            "operation_id": operation_id,
            "transaction_id": str(payload_data.get("transaction_id") or operation_id),
            "idempotency_key": str(payload_data.get("idempotency_key") or operation_id),
            "tenant_id": self.tenant_id,
            "user_id": str(normalized[0].get("user_id") or ""),
            "commit_group_id": commit_group_id,
            "operation_set_digest": operation_set_digest([operation]),
            "relation_manifest_digest": canonical_digest(relation_manifest),
        }
        payload = {**core, "prepared_intent_digest": canonical_digest(core)}
        path = self.intent_path(operation_id)
        with self._artifact_lock(path):
            self._create_or_match(path, payload, self.validate_pending_intent)
        return self.validate_pending_intent(
            self._read(path),
            operation=operation,
            relation_manifest=relation_manifest,
        )

    def validate_pending_intent(
        self,
        payload: object,
        *,
        operation: Any | None = None,
        relation_manifest: object | None = None,
        prepared_intent_digest: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != PENDING_PREPARED_INTENT_SCHEMA_VERSION:
            raise PlanningProofIntegrityError("pending prepared-intent schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "prepared_intent_digest"}
        if (
            payload.get("prepared_intent_digest") != canonical_digest(core)
            or payload.get("tenant_id") != self.tenant_id
            or not isinstance(payload.get("commit_group_id"), str)
            or not payload.get("commit_group_id")
        ):
            raise PlanningProofIntegrityError("pending prepared-intent digest is corrupt")
        if prepared_intent_digest is not None and payload.get("prepared_intent_digest") != prepared_intent_digest:
            raise PlanningProofIntegrityError("pending prepared-intent differs from receipt")
        if operation is not None and payload.get("operation_set_digest") != operation_set_digest([operation]):
            raise PlanningProofIntegrityError("pending prepared-intent operation differs")
        if relation_manifest is not None and payload.get("relation_manifest_digest") != canonical_digest(
            relation_manifest
        ):
            raise PlanningProofIntegrityError("pending prepared-intent relation manifest differs")
        return payload

    def load_pending_intent(
        self,
        operation_id: str,
        *,
        operation: Any | None = None,
        relation_manifest: object | None = None,
        prepared_intent_digest: str | None = None,
    ) -> dict[str, Any]:
        path = self.intent_path(operation_id)
        if not path.exists() or path.is_symlink():
            raise PlanningProofIntegrityError("pending prepared-intent artifact is missing")
        return self.validate_pending_intent(
            self._read(path),
            operation=operation,
            relation_manifest=relation_manifest,
            prepared_intent_digest=prepared_intent_digest,
        )

    def ensure_canonical_intent(
        self,
        outbox: dict[str, Any],
        *,
        operations: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Publish the canonical transaction intent once, before redo begins."""

        payload = self.build_canonical_intent(outbox, operations=operations)
        path = self.canonical_intent_path(str(payload["transaction_id"]))
        with self._artifact_lock(path):
            self._create_or_match(path, payload, self.validate_canonical_intent)
        return self.validate_canonical_intent(
            self._read(path),
            operations=operations,
            prepared_intent_digest=str(payload["prepared_intent_digest"]),
        )

    def build_canonical_intent(
        self,
        outbox: dict[str, Any],
        *,
        operations: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        from memoryos.operations.commit.outbox_envelope import (
            prepared_intent_payload,
            validate_outbox,
        )

        validated = validate_outbox(outbox)
        if validated.get("tenant_id") != self.tenant_id:
            raise PlanningProofIntegrityError("canonical prepared intent crosses its tenant boundary")
        if operations is not None:
            validated = validate_outbox(
                validated,
                operations=_context_operations(operations),
            )
        immutable_intent = prepared_intent_payload(validated)
        core: dict[str, Any] = {
            "schema_version": CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
            "transaction_id": str(validated["transaction_id"]),
            "idempotency_key": str(validated["idempotency_key"]),
            "tenant_id": str(validated["tenant_id"]),
            "user_id": str(validated["user_id"]),
            "commit_group_id": str(validated["commit_group_id"]),
            "operation_ids": list(validated["operation_ids"]),
            "prepared_intent": immutable_intent,
            "prepared_intent_digest": str(validated["prepared_intent_digest"]),
        }
        payload = {**core, "artifact_digest": canonical_digest(core)}
        return self.validate_canonical_intent(payload, operations=operations)

    def validate_canonical_intent(
        self,
        payload: object,
        *,
        operations: Sequence[Any] | None = None,
        transaction_id: str | None = None,
        prepared_intent_digest: str | None = None,
    ) -> dict[str, Any]:
        from memoryos.operations.commit.outbox_envelope import (
            OutboxIntegrityError,
            validate_outbox,
        )

        if not isinstance(payload, dict) or payload.get("schema_version") != CANONICAL_PREPARED_INTENT_SCHEMA_VERSION:
            raise PlanningProofIntegrityError("canonical prepared-intent schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "artifact_digest"}
        immutable_intent = payload.get("prepared_intent")
        if (
            payload.get("artifact_digest") != canonical_digest(core)
            or not isinstance(immutable_intent, dict)
            or payload.get("tenant_id") != self.tenant_id
            or payload.get("prepared_intent_digest") != canonical_digest(immutable_intent)
        ):
            raise PlanningProofIntegrityError("canonical prepared-intent digest is corrupt")
        reconstructed_core = {
            **immutable_intent,
            "status": "prepared",
            "prepared_intent_digest": payload["prepared_intent_digest"],
            "receipt_path": "",
            "receipt_digest": "",
        }
        reconstructed = {
            **reconstructed_core,
            "outbox_digest": canonical_digest(reconstructed_core),
        }
        try:
            validated = validate_outbox(
                reconstructed,
                transaction_id=str(payload.get("transaction_id") or ""),
                idempotency_key=str(payload.get("idempotency_key") or ""),
                tenant_id=self.tenant_id,
                user_id=str(payload.get("user_id") or ""),
                operations=_context_operations(operations) if operations is not None else None,
                allowed_statuses={"prepared"},
            )
        except (OutboxIntegrityError, TypeError, ValueError) as exc:
            raise PlanningProofIntegrityError("canonical prepared-intent payload is invalid") from exc
        if (
            payload.get("commit_group_id") != validated.get("commit_group_id")
            or payload.get("operation_ids") != validated.get("operation_ids")
            or payload.get("prepared_intent_digest") != validated.get("prepared_intent_digest")
        ):
            raise PlanningProofIntegrityError("canonical prepared-intent identity is inconsistent")
        if transaction_id is not None and payload.get("transaction_id") != transaction_id:
            raise PlanningProofIntegrityError("canonical prepared-intent transaction identity differs")
        if prepared_intent_digest is not None and payload.get("prepared_intent_digest") != prepared_intent_digest:
            raise PlanningProofIntegrityError("canonical prepared-intent differs from receipt")
        return payload

    def load_canonical_intent(
        self,
        transaction_id: str,
        *,
        operations: Sequence[Any] | None = None,
        prepared_intent_digest: str | None = None,
    ) -> dict[str, Any]:
        path = self.canonical_intent_path(transaction_id)
        if not path.exists() or path.is_symlink():
            raise PlanningProofIntegrityError("immutable canonical prepared-intent artifact is missing")
        return self.validate_canonical_intent(
            self._read(path),
            operations=operations,
            transaction_id=transaction_id,
            prepared_intent_digest=prepared_intent_digest,
        )

    def validate_all(self) -> dict[str, int]:
        direct_count = 0
        direct_root = self.artifact_root / "system" / "direct-planning-proofs"
        for path in sorted(direct_root.glob("*.json")) if direct_root.exists() else ():
            payload = self.validate_direct(self._read(path))
            if path.is_symlink() or self.direct_path(str(payload.get("proof_id") or "")).resolve() != path.resolve():
                raise PlanningProofIntegrityError("direct planning proof has an invalid artifact path")
            direct_count += 1
        intent_count = 0
        intent_root = self.artifact_root / "system" / "prepared-intents"
        for path in sorted(intent_root.glob("*.json")) if intent_root.exists() else ():
            payload = self.validate_pending_intent(self._read(path))
            if (
                path.is_symlink()
                or self.intent_path(str(payload.get("operation_id") or "")).resolve() != path.resolve()
            ):
                raise PlanningProofIntegrityError("pending prepared intent has an invalid artifact path")
            intent_count += 1
        canonical_intent_count = 0
        canonical_intent_root = self.artifact_root / "system" / "canonical-prepared-intents"
        for path in sorted(canonical_intent_root.glob("*.json")) if canonical_intent_root.exists() else ():
            payload = self.validate_canonical_intent(self._read(path))
            if (
                path.is_symlink()
                or self.canonical_intent_path(str(payload.get("transaction_id") or "")).resolve() != path.resolve()
            ):
                raise PlanningProofIntegrityError("canonical prepared intent has an invalid artifact path")
            canonical_intent_count += 1
        return {
            "direct_plans": direct_count,
            "pending_intents": intent_count,
            "canonical_intents": canonical_intent_count,
        }

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanningProofIntegrityError(f"immutable planning artifact is unreadable: {path.name}") from exc
        if not isinstance(payload, dict):
            raise PlanningProofIntegrityError("immutable planning artifact is not an object")
        return payload

    def _create_or_match(self, path: Path, payload: dict[str, Any], validator: Any) -> None:
        if path.is_symlink():
            raise PlanningProofIntegrityError("immutable planning artifact path cannot be a symbolic link")
        if path.exists():
            existing = validator(ImmutablePlanningProofStore._read(path))
            if existing != payload:
                raise PlanningProofIntegrityError("immutable planning artifact conflicts with an existing identity")
            return
        atomic_create_json(path, payload, artifact_root=self.artifact_root)

    @contextmanager
    def _artifact_lock(self, path: Path) -> Iterator[None]:
        lock_path = path.parent / ".locks" / f"{path.name}.lock"
        descriptor = open_private_lock(lock_path, root=self.artifact_root)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
