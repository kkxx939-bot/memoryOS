"""Create-only idempotency records for committed pending-memory reviews."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from memoryos.core.clock import utc_now
from memoryos.core.durable_io import atomic_write_json
from memoryos.core.file_lock import open_private_lock
from memoryos.core.integrity import canonical_digest

PENDING_REVIEW_COMMAND_SCHEMA_VERSION = "pending_review_command_v1"


class PendingReviewCommandIntegrityError(RuntimeError):
    pass


class PendingReviewIdempotencyConflict(ValueError):
    pass


class PendingReviewCommandStore:
    def __init__(self, root: str | Path, *, tenant_id: str) -> None:
        if not tenant_id or tenant_id in {".", ".."} or "/" in tenant_id or "\\" in tenant_id:
            raise ValueError("invalid pending review tenant_id")
        shared = Path(root)
        artifact_root = shared if tenant_id == "default" else shared / "tenants" / tenant_id
        self.artifact_root = artifact_root
        self.root = artifact_root / "system" / "pending-review-commands"
        self.tenant_id = tenant_id

    def path(self, command_id: str) -> Path:
        if not command_id:
            raise ValueError("pending review command_id is required")
        return self.root / f"{hashlib.sha256(command_id.encode('utf-8')).hexdigest()}.json"

    def begin(
        self,
        command_id: str,
        *,
        owner_user_id: str,
        pending_uri: str,
        decision: str,
        expected_lifecycle_revision: int,
        expected_proposal_fingerprint: str,
        reason: str,
        correction_proposal_digest: str = "",
    ) -> dict[str, Any]:
        request = {
            "tenant_id": self.tenant_id,
            "owner_user_id": owner_user_id,
            "pending_uri": pending_uri,
            "decision": decision,
            "expected_lifecycle_revision": expected_lifecycle_revision,
            "expected_proposal_fingerprint": expected_proposal_fingerprint,
            "reason": reason,
        }
        if correction_proposal_digest:
            request["correction_proposal_digest"] = correction_proposal_digest
        request_digest = canonical_digest(request)
        path = self.path(command_id)
        with self._command_lock(command_id):
            if path.is_symlink():
                raise PendingReviewCommandIntegrityError("pending review command cannot be a symbolic link")
            if path.exists():
                existing = self.load(command_id)
                if existing["request_digest"] != request_digest:
                    raise PendingReviewIdempotencyConflict(
                        "pending review command_id is already bound to a different decision or effect"
                    )
                return existing
            now = utc_now()
            core: dict[str, Any] = {
                "schema_version": PENDING_REVIEW_COMMAND_SCHEMA_VERSION,
                "command_id": command_id,
                "request": request,
                "request_digest": request_digest,
                "status": "running",
                "result": {},
                "error": {},
                "created_at": now,
                "updated_at": now,
            }
            atomic_write_json(
                path,
                {**core, "record_digest": canonical_digest(core)},
                artifact_root=self.artifact_root,
            )
            return self.load(command_id)

    def complete(self, command_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        with self._command_lock(command_id):
            current = self.load(command_id)
            normalized = dict(result)
            if current["status"] == "completed":
                if dict(current["result"]) != normalized:
                    raise PendingReviewIdempotencyConflict(
                        "completed pending review command conflicts with a different result"
                    )
                return current
            if current["status"] == "failed":
                raise PendingReviewIdempotencyConflict("failed pending review command cannot be rewritten as completed")
            core = {
                key: value
                for key, value in current.items()
                if key not in {"record_digest", "status", "result", "error", "updated_at"}
            }
            core.update(
                {
                    "status": "completed",
                    "result": normalized,
                    "error": {},
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(
                self.path(command_id),
                {**core, "record_digest": canonical_digest(core)},
                artifact_root=self.artifact_root,
            )
            return self.load(command_id)

    def fail(self, command_id: str, error: BaseException) -> dict[str, Any]:
        """Persist a terminal request conflict; process crashes remain resumable running commands."""

        with self._command_lock(command_id):
            current = self.load(command_id)
            if current["status"] in {"completed", "failed"}:
                return current
            error_payload = {
                "type": type(error).__name__,
                "message": str(error)[:500],
            }
            core = {
                key: value
                for key, value in current.items()
                if key not in {"record_digest", "status", "result", "error", "updated_at"}
            }
            core.update(
                {
                    "status": "failed",
                    "result": {},
                    "error": error_payload,
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(
                self.path(command_id),
                {**core, "record_digest": canonical_digest(core)},
                artifact_root=self.artifact_root,
            )
            return self.load(command_id)

    def load(self, command_id: str) -> dict[str, Any]:
        path = self.path(command_id)
        try:
            if path.is_symlink():
                raise OSError("pending review command cannot be a symbolic link")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PendingReviewCommandIntegrityError("pending review command record is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != PENDING_REVIEW_COMMAND_SCHEMA_VERSION:
            raise PendingReviewCommandIntegrityError("pending review command schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "record_digest"}
        if payload.get("record_digest") != canonical_digest(core):
            raise PendingReviewCommandIntegrityError("pending review command record was modified")
        request = payload.get("request")
        if not isinstance(request, dict) or request.get("tenant_id") != self.tenant_id:
            raise PendingReviewCommandIntegrityError("pending review command crosses tenant boundary")
        if payload.get("request_digest") != canonical_digest(request):
            raise PendingReviewCommandIntegrityError("pending review command request was modified")
        if payload.get("command_id") != command_id or payload.get("status") not in {
            "running",
            "completed",
            "failed",
        }:
            raise PendingReviewCommandIntegrityError("pending review command identity is invalid")
        if not isinstance(payload.get("result"), dict) or not isinstance(payload.get("error", {}), dict):
            raise PendingReviewCommandIntegrityError("pending review command result is invalid")
        required_request = {
            "tenant_id",
            "owner_user_id",
            "pending_uri",
            "decision",
            "expected_lifecycle_revision",
            "expected_proposal_fingerprint",
            "reason",
        }
        request_fields = set(request)
        if request_fields != required_request and request_fields != {
            *required_request,
            "correction_proposal_digest",
        }:
            raise PendingReviewCommandIntegrityError("pending review command request fields are invalid")
        if (
            not isinstance(request.get("owner_user_id"), str)
            or not request.get("owner_user_id")
            or not isinstance(request.get("pending_uri"), str)
            or "/memories/pending/" not in str(request.get("pending_uri"))
            or str(request.get("decision") or "").upper()
            not in {"CONFIRM", "CONFIRM_AND_APPLY", "CORRECT", "REJECT", "EXPIRE", "RETRY"}
            or not isinstance(request.get("expected_lifecycle_revision"), int)
            or int(request.get("expected_lifecycle_revision", 0)) < 1
            or not isinstance(request.get("expected_proposal_fingerprint"), str)
            or not request.get("expected_proposal_fingerprint")
            or not isinstance(request.get("reason"), str)
        ):
            raise PendingReviewCommandIntegrityError("pending review command request semantics are invalid")
        if payload["status"] == "completed" and not payload["result"]:
            raise PendingReviewCommandIntegrityError("completed pending review command has no result")
        if payload["status"] == "failed" and not payload["error"]:
            raise PendingReviewCommandIntegrityError("failed pending review command has no terminal error")
        return {**payload, "error": dict(payload.get("error", {}) or {})}

    def iter_records(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")) if self.root.exists() else ():
            if path.is_symlink():
                raise PendingReviewCommandIntegrityError("pending review command has an invalid artifact path")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PendingReviewCommandIntegrityError(f"pending review command is unreadable: {path.name}") from exc
            command_id = str(raw.get("command_id") or "") if isinstance(raw, dict) else ""
            if not command_id or path.is_symlink() or self.path(command_id).resolve() != path.resolve():
                raise PendingReviewCommandIntegrityError("pending review command has an invalid artifact path")
            records.append(self.load(command_id))
        return tuple(records)

    @contextmanager
    def _command_lock(self, command_id: str) -> Iterator[None]:
        lock_path = self.path(command_id).with_suffix(".lock")
        descriptor = open_private_lock(lock_path, root=self.root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def validate_pending_review_record(record: Mapping[str, Any], pending: Any) -> None:
    """Bind one mutable command status to receipt-backed immutable lifecycle history."""

    request = dict(record.get("request", {}) or {})
    command_id = str(record.get("command_id") or "")
    decision = str(request.get("decision") or "").upper()
    request_digest = str(record.get("request_digest") or "")
    if pending.uri != request.get("pending_uri") or pending.proposal.fingerprint != request.get(
        "expected_proposal_fingerprint"
    ):
        raise PendingReviewCommandIntegrityError("pending review command targets another committed proposal")
    bound = [
        dict(item) for item in pending.lifecycle_history if str(dict(item).get("review_command_id") or "") == command_id
    ]
    if any(
        str(item.get("review_decision") or "").upper() != decision
        or str(item.get("review_request_digest") or "") != request_digest
        for item in bound
    ):
        raise PendingReviewCommandIntegrityError("pending review command conflicts with receipt-backed history")
    if bound and int(bound[0].get("from_revision", 0) or 0) != int(request.get("expected_lifecycle_revision", 0) or 0):
        raise PendingReviewCommandIntegrityError("pending review command expected revision differs from history")
    status = str(record.get("status") or "")
    if status == "completed":
        if not bound:
            raise PendingReviewCommandIntegrityError(
                "completed pending review command has no committed lifecycle proof"
            )
        latest = max(bound, key=lambda item: int(item.get("to_revision", 0) or 0))
        result = dict(record.get("result", {}) or {})
        if (
            result.get("uri") != pending.uri
            or int(result.get("lifecycle_revision", 0) or 0) != int(latest.get("to_revision", 0) or 0)
            or str(result.get("status") or "") != str(latest.get("to") or "")
        ):
            raise PendingReviewCommandIntegrityError("completed pending review result differs from committed history")
    elif status == "failed" and bound:
        raise PendingReviewCommandIntegrityError("failed pending review command already has a committed effect")


def _recovered_completed_result(
    record: Mapping[str, Any],
    pending: Any,
    committed: Any,
) -> dict[str, Any] | None:
    """Reconstruct a response only when the command's full requested effect is current.

    A CONFIRM_AND_APPLY command whose CONFIRM step alone is durable remains
    running and resumable.  Simpler terminal decisions can be completed from
    their receipt-backed lifecycle history during startup recovery.
    """

    if str(record.get("status") or "") != "running":
        return None
    request = dict(record.get("request", {}) or {})
    command_id = str(record.get("command_id") or "")
    decision = str(request.get("decision") or "").upper()
    expected_state = {
        "CONFIRM": "confirmed",
        "CONFIRM_AND_APPLY": "resolved",
        "CORRECT": "rejected",
        "REJECT": "rejected",
        "EXPIRE": "expired",
        "RETRY": "retryable",
    }.get(decision)
    if expected_state is None:
        return None
    bound = [
        dict(item) for item in pending.lifecycle_history if str(dict(item).get("review_command_id") or "") == command_id
    ]
    if not bound:
        return None
    latest = max(bound, key=lambda item: int(item.get("to_revision", 0) or 0))
    lifecycle_revision = int(latest.get("to_revision", 0) or 0)
    if (
        str(latest.get("to") or "").casefold() != expected_state
        or pending.lifecycle_state.value.casefold() != expected_state
        or pending.lifecycle_revision != lifecycle_revision
    ):
        return None
    receipt = dict(getattr(committed, "receipt", {}) or {})
    diff_id = str(dict(receipt.get("diff", {}) or {}).get("diff_id") or "")
    result: dict[str, Any] = {
        "uri": pending.uri,
        "status": pending.lifecycle_state.value,
        "lifecycle_revision": lifecycle_revision,
        "diff_id": diff_id,
    }
    for raw_operation in receipt.get("operations", []) or []:
        if not isinstance(raw_operation, dict) or raw_operation.get("target_uri") != pending.uri:
            continue
        payload = dict(raw_operation.get("payload", {}) or {})
        resolved_claims = [str(item) for item in payload.get("resolved_claim_uris", []) or []]
        corrected_claims = [str(item) for item in payload.get("corrected_claim_uris", []) or []]
        if resolved_claims:
            result["resolved_claim_uris"] = list(dict.fromkeys(resolved_claims))
        if corrected_claims:
            result["corrected_claim_uris"] = list(dict.fromkeys(corrected_claims))
            result["corrected_proposal_fingerprint"] = str(payload.get("corrected_proposal_fingerprint") or "")
        break
    return result


def validate_pending_review_commands(
    root: str | Path,
    *,
    tenant_id: str,
    source_store: Any,
    relation_store: Any = None,
) -> dict[str, int]:
    """Validate command records and globally unique receipt-backed command bindings."""

    from memoryos.memory.canonical.proposal import PendingMemoryProposal
    from memoryos.memory.canonical.visibility import list_committed_canonical, read_committed_pending

    store = PendingReviewCommandStore(root, tenant_id=tenant_id)
    history_bindings: dict[str, tuple[str, str, str]] = {}
    pending_by_uri: dict[str, tuple[Any, str, Any]] = {}
    for committed in list_committed_canonical(
        source_store,
        relation_store,
        kinds=("pending_proposal",),
    ):
        pending = PendingMemoryProposal.from_context_object(committed.object)
        pending_by_uri[pending.uri] = (
            pending,
            str(committed.object.owner_user_id or ""),
            committed,
        )
        for raw in pending.lifecycle_history:
            item = dict(raw)
            command_id = str(item.get("review_command_id") or "")
            if not command_id:
                continue
            decision = str(item.get("review_decision") or "").upper()
            digest = str(item.get("review_request_digest") or "")
            if not decision or len(digest) != 64:
                raise PendingReviewCommandIntegrityError(
                    f"pending lifecycle has an incomplete review binding: {pending.uri}"
                )
            binding = (pending.uri, decision, digest)
            if history_bindings.setdefault(command_id, binding) != binding:
                raise PendingReviewCommandIntegrityError(
                    f"pending review command is reused by conflicting committed effects: {command_id}"
                )

    records = store.iter_records()
    record_ids = {str(record.get("command_id") or "") for record in records}
    missing_records = set(history_bindings) - record_ids
    if missing_records:
        raise PendingReviewCommandIntegrityError(
            "pending lifecycle history has no durable review command record: " + ",".join(sorted(missing_records))
        )
    for record in records:
        request = dict(record["request"])
        pending_uri = str(request["pending_uri"])
        pending_record = pending_by_uri.get(pending_uri)
        if pending_record is None:
            committed = read_committed_pending(source_store, pending_uri, relation_store)
            pending = PendingMemoryProposal.from_context_object(committed.object)
            owner_user_id = committed.object.owner_user_id
        else:
            pending, owner_user_id, committed = pending_record
        if request.get("owner_user_id") != owner_user_id:
            raise PendingReviewCommandIntegrityError("pending review command crosses proposal owner boundary")
        recovered_result = _recovered_completed_result(record, pending, committed)
        if recovered_result is not None:
            record = store.complete(str(record["command_id"]), recovered_result)
        validate_pending_review_record(record, pending)
    return {"records": len(records), "history_bindings": len(history_bindings)}
