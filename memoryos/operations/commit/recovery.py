"""Durable commit recovery owned by the operation plane."""

from __future__ import annotations

import json
from dataclasses import dataclass

from memoryos.contextdb.store.lock_store import LockLostError
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.outbox_envelope import OutboxIntegrityError, validate_outbox
from memoryos.operations.commit.planning_proof import PlanningProofIntegrityError
from memoryos.operations.commit.receipt import load_transaction_receipt
from memoryos.operations.commit.redo_log import (
    RedoControlFileError,
    RedoIntegrityError,
    RedoLog,
)
from memoryos.operations.model.context_operation import ContextOperation


@dataclass(frozen=True)
class RecoveryResult:
    recovered_count: int
    operation_ids: list[str]
    failed_count: int = 0
    quarantine_count: int = 0
    last_error: str = ""


class RecoveryService:
    def __init__(self, redo_log: RedoLog, committer: OperationCommitter) -> None:
        self.redo_log = redo_log
        self.committer = committer

    def recover(self, user_id: str) -> RecoveryResult:
        try:
            entries = self.redo_log.pending_entries()
        except RedoControlFileError as exc:
            return RecoveryResult(
                recovered_count=0,
                operation_ids=[],
                failed_count=len(exc.records),
                quarantine_count=len(exc.records),
                last_error=type(exc).__name__,
            )
        if not entries:
            return RecoveryResult(recovered_count=0, operation_ids=[])
        recovered: list[str] = []
        failed_count = 0
        quarantine_count = 0
        last_error = ""
        canonical_by_transaction: dict[str, list] = {}
        regular_entries = []
        for entry in entries:
            if entry.operation.user_id != user_id:
                continue
            if entry.operation.payload.get("canonical_memory") is True:
                transaction_id = str(entry.operation.payload.get("transaction_id", ""))
                canonical_by_transaction.setdefault(transaction_id, []).append(entry)
            else:
                regular_entries.append(entry)
        for transaction_entries in canonical_by_transaction.values():
            try:
                recovered.extend(self.committer.resume_canonical_batch(user_id, transaction_entries))
            except RedoIntegrityError as exc:
                failed_count += len(transaction_entries)
                last_error = self._describe_failure(exc)
                quarantine_count += self._quarantine_canonical(user_id, transaction_entries, exc)
                for entry in transaction_entries:
                    self._record_failure(user_id, entry, exc, terminal="quarantine")
            except (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                LockLostError,
            ) as exc:
                failed_count += len(transaction_entries)
                last_error = self._describe_failure(exc)
                for entry in transaction_entries:
                    self._record_failure(user_id, entry, exc, terminal="retryable")
        for entry in regular_entries:
            operation = entry.operation
            try:
                if self.committer.resume(
                    user_id,
                    operation,
                    entry.phase,
                    source_effect=entry.source_effect,
                    relation_manifest=entry.relation_manifest,
                ):
                    recovered.append(operation.operation_id)
            except RedoIntegrityError as exc:
                failed_count += 1
                last_error = self._describe_failure(exc)
                quarantine_count += self._quarantine_regular(user_id, entry, exc)
                self._record_failure(user_id, entry, exc, terminal="quarantine")
            except (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                LockLostError,
            ) as exc:
                failed_count += 1
                last_error = self._describe_failure(exc)
                self._record_failure(user_id, entry, exc, terminal="retryable")
        return RecoveryResult(
            recovered_count=len(recovered),
            operation_ids=recovered,
            failed_count=failed_count,
            quarantine_count=quarantine_count,
            last_error=last_error,
        )

    def recover_outboxes(self) -> RecoveryResult:
        """Seed missing redo records from complete, integrity-checked outbox envelopes."""

        outbox_root = self.committer.artifact_root / "system" / "outbox"
        if not outbox_root.exists():
            return RecoveryResult(0, [])
        recovered: list[str] = []
        failed = quarantined = 0
        last_error = ""
        for path in sorted(outbox_root.glob("*.json")):
            try:
                envelope = validate_outbox(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                quarantine_control_file(
                    self.committer.artifact_root,
                    path,
                    kind="outbox",
                    error=exc,
                    identifiers={"transaction_id": path.stem},
                )
                failed += 1
                quarantined += 1
                last_error = self._describe_failure(exc)
                continue
            status = str(envelope["status"])
            operations = [ContextOperation.from_dict(item) for item in envelope["operations"]]
            try:
                self.committer.planning_proofs.load_canonical_intent(
                    str(envelope["transaction_id"]),
                    operations=operations,
                    prepared_intent_digest=str(envelope["prepared_intent_digest"]),
                )
            except PlanningProofIntegrityError as exc:
                quarantine_control_file(
                    self.committer.artifact_root,
                    path,
                    kind="outbox",
                    error=exc,
                    identifiers={"transaction_id": envelope["transaction_id"]},
                )
                failed += max(1, len(operations))
                quarantined += 1
                last_error = self._describe_failure(exc)
                continue
            if status == "aborted":
                continue
            if status == "committed":
                try:
                    marker = self.committer._transaction_marker(str(envelope["idempotency_key"]))
                    if not marker.exists():
                        raise RedoIntegrityError("committed outbox has no marker")
                    self.committer._validate_transaction_marker(marker, operations)
                    self.committer._validate_head_published_receipt(
                        marker,
                        load_transaction_receipt(marker),
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    # The outbox has already passed its own digest, tenant,
                    # operation-set and immutable-intent checks.  A missing or
                    # corrupt earlier receipt/head binding must fail closed,
                    # but does not prove the later outbox itself is corrupt.
                    # Preserve every artifact for deterministic repair and
                    # let startup remain NOT_READY.
                    failed += 1
                    last_error = self._describe_failure(exc)
                continue
            if status not in {"prepared", "source_committed"}:
                continue
            try:
                self._seed_redo_from_outbox(envelope, operations)
                transaction_entries = [
                    entry
                    for entry in self.redo_log.pending_entries()
                    if str(entry.operation.payload.get("transaction_id") or "") == str(envelope["transaction_id"])
                ]
                recovered.extend(self.committer.resume_canonical_batch(str(envelope["user_id"]), transaction_entries))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                entries = [
                    entry
                    for entry in self._safe_pending_entries()
                    if str(entry.operation.payload.get("transaction_id") or "") == str(envelope["transaction_id"])
                ]
                for entry in entries:
                    self._quarantine_regular(str(envelope["user_id"]), entry, exc)
                if path.exists():
                    quarantine_control_file(
                        self.committer.artifact_root,
                        path,
                        kind="outbox",
                        error=exc,
                        identifiers={"transaction_id": envelope["transaction_id"]},
                    )
                failed += max(1, len(operations))
                quarantined += 1 + len(entries)
                last_error = self._describe_failure(exc)
        return RecoveryResult(
            recovered_count=len(recovered),
            operation_ids=recovered,
            failed_count=failed,
            quarantine_count=quarantined,
            last_error=last_error,
        )

    @staticmethod
    def _describe_failure(exc: BaseException) -> str:
        """Keep the failing artifact identity visible without unbounded output."""

        message = " ".join(str(exc).split())[:500]
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    def _seed_redo_from_outbox(
        self,
        envelope: dict,
        operations: list[ContextOperation],
    ) -> None:
        effects = {
            str(item["operation_id"]): item for item in envelope.get("effect_manifests", []) if isinstance(item, dict)
        }
        before = {str(item["uri"]): item for item in envelope.get("before_images", []) if isinstance(item, dict)}
        for operation in operations:
            path = self.redo_log.redo_dir / f"{operation.operation_id}.json"
            if path.exists():
                continue
            effect = effects[operation.operation_id]
            relation_manifest = dict(effect.get("relation_manifest", {}) or {})
            self.committer._validate_canonical_relation_manifest(operation, relation_manifest)
            try:
                source_effect = self.committer._capture_canonical_source_effect(
                    operation,
                    relation_manifest,
                )
            except (FileNotFoundError, RuntimeError, ValueError):
                snapshot = before.get(str(effect["uri"]))
                if not self._matches_before_image(snapshot):
                    raise RedoIntegrityError(
                        "orphan outbox Source/Relation state matches neither before nor planned effect"
                    ) from None
                if envelope["status"] == "source_committed":
                    raise RedoIntegrityError("source_committed outbox is missing its Source effect") from None
                self.redo_log.begin(
                    operation,
                    phase="started",
                    relation_manifest=relation_manifest,
                )
            else:
                self.redo_log.begin(
                    operation,
                    phase="source_written",
                    source_effect=source_effect,
                    relation_manifest=relation_manifest,
                )

    def _matches_before_image(self, snapshot: object) -> bool:
        if not isinstance(snapshot, dict):
            return False
        uri = str(snapshot.get("uri") or "")
        try:
            obj = self.committer.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return snapshot.get("exists") is False and not snapshot.get("relations")
        if snapshot.get("exists") is not True or not isinstance(snapshot.get("object"), dict):
            return False
        if canonical_json(obj.to_dict()) != canonical_json(snapshot["object"]):
            return False
        try:
            content = self.committer.source_store.read_content(obj.layers.l2_uri or obj.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            content = ""
        if content != str(snapshot.get("content") or ""):
            return False
        expected_relations = sorted(
            (dict(item) for item in snapshot.get("relations", []) if isinstance(item, dict)),
            key=canonical_json,
        )
        if self.committer.relation_store is None:
            return not expected_relations
        actual_relations = sorted(
            (
                self.committer._relation_effect_spec(relation)
                for relation in self.committer.relation_store.relations_of(
                    uri,
                    tenant_id=self.committer.tenant_id,
                )
            ),
            key=canonical_json,
        )
        return canonical_json(actual_relations) == canonical_json(expected_relations)

    def _safe_pending_entries(self) -> list:
        try:
            return self.redo_log.pending_entries()
        except RedoControlFileError:
            return []

    def _record_failure(
        self,
        user_id: str,
        entry,  # noqa: ANN001
        exc: BaseException,
        *,
        terminal: str,
    ) -> None:
        operation = entry.operation
        self.committer.audit.record(
            user_id,
            "recovery_failed",
            {
                "operation_id": operation.operation_id,
                "target_uri": operation.target_uri,
                "redo_phase": entry.phase,
                "error_type": type(exc).__name__,
                "terminal": terminal,
            },
        )

    def _quarantine_regular(self, user_id: str, entry, exc: BaseException) -> int:  # noqa: ANN001
        path = self.redo_log.redo_dir / f"{entry.operation_id}.json"
        if not path.exists():
            return 0
        quarantine_control_file(
            self.redo_log.root,
            path,
            kind="redo",
            error=exc,
            identifiers={"operation_id": entry.operation_id, "user_id": user_id},
        )
        return 1

    def _quarantine_canonical(self, user_id: str, entries: list, exc: BaseException) -> int:  # noqa: ANN001
        count = 0
        transaction_id = str(entries[0].operation.payload.get("transaction_id") or "")
        for entry in entries:
            count += self._quarantine_regular(user_id, entry, exc)
        if transaction_id:
            outbox = self.committer._outbox_path(transaction_id)
            if outbox.exists():
                quarantine_control_file(
                    self.redo_log.root,
                    outbox,
                    kind="outbox",
                    error=exc,
                    identifiers={"transaction_id": transaction_id, "user_id": user_id},
                )
                count += 1
        return count
