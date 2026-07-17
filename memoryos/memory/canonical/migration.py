"""Idempotent migration into immutable receipts, current heads, and bundles."""

from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, NoReturn

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.durable_io import atomic_create_json, atomic_write_json
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.current_head import (
    iter_current_head_uris,
    load_current_head,
    publish_current_head_sets,
    receipt_history_contains_uri,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
    ProjectionRecordStore,
)
from memoryos.memory.canonical.visibility import read_committed_canonical
from memoryos.memory.integration.planning_envelope import (
    PLANNING_ENVELOPE_ANCHOR_SCHEMA_VERSION,
    PLANNING_ENVELOPE_SCHEMA_VERSION,
    PlanningEnvelopeIntegrityError,
    PlanningEnvelopeStore,
    validate_planning_envelope_payload,
)
from memoryos.operations.commit.effect_marker import (
    EFFECT_MARKER_SCHEMA_VERSION,
)
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    validate_outbox,
)
from memoryos.operations.commit.planning_proof import (
    CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
    ImmutablePlanningProofStore,
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    build_transaction_receipt,
    load_transaction_receipt,
    validate_transaction_receipt,
)
from memoryos.operations.commit.redo_log import RedoControlFileError, RedoLog
from memoryos.operations.model.context_operation import ContextOperation

MIGRATION_SCHEMA_VERSION = "memory_closure_migration_v1"
PROJECTION_MIGRATION_SCHEMA_VERSION = "memory_projection_migration_v5"
PLANNING_MIGRATION_SCHEMA_VERSION = "memory_planning_migration_v2"
PREPARED_INTENT_MIGRATION_SCHEMA_VERSION = "memory_prepared_intent_migration_v1"


class MemoryClosureMigrationError(RuntimeError):
    pass


class MemoryClosureMigration:
    """Converts only memory artifacts; regular operation markers stay v1."""

    def __init__(
        self,
        root: str | Path,
        *,
        tenant_id: str,
        source_store: SourceStore,
        relation_store: RelationStore | None = None,
    ) -> None:
        shared = Path(root)
        self.shared_root = shared
        self.artifact_root = shared if tenant_id == "default" else shared / "tenants" / tenant_id
        self.source_store = source_store
        self.relation_store = relation_store
        self.tenant_id = tenant_id
        self.receipt_path = self.artifact_root / "system" / "migrations" / "memory-closure-v1.json"
        self.failure_path = self.artifact_root / "system" / "migrations" / "memory-closure-v1.failed.json"
        self.projection_receipt_path = self.artifact_root / "system" / "migrations" / "memory-projection-v5.json"
        self.planning_receipt_path = self.artifact_root / "system" / "migrations" / "memory-planning-v2.json"
        self.planning_failure_path = self.artifact_root / "system" / "migrations" / "memory-planning-v2.failed.json"

    def run(self, *, allow_inflight: bool = False) -> dict[str, Any]:
        for path, label in (
            (self.receipt_path, "memory closure migration receipt"),
            (self.failure_path, "memory closure migration failure receipt"),
            (self.projection_receipt_path, "projection migration receipt"),
            (self.planning_receipt_path, "planning migration receipt"),
            (self.planning_failure_path, "planning migration failure receipt"),
        ):
            self._reject_control_symlink(path, label)
        if self.failure_path.exists():
            self._raise_recorded_failure()
        if self.planning_failure_path.exists():
            self._raise_planning_migration_failure()
        self._migrate_planning_envelopes()
        self._migrate_canonical_prepared_intents()
        migrated_projection_records = self._migrate_projection_state()
        if self.receipt_path.exists():
            receipt = self._load_migration_receipt()
            self._audit_no_legacy_memory_markers()
            self._validate_migrated_state(allow_inflight=allow_inflight)
            return receipt
        migrated_bundles = self._migrate_bundles()
        migrated_markers: list[str] = []
        legacy_digests: dict[str, str] = {}
        receipt_paths: list[Path] = []
        roots = (
            self.artifact_root / "system" / "transactions",
            self.artifact_root / "system" / "operations",
        )
        for marker_root in roots:
            for path in sorted(marker_root.glob("*.json")) if marker_root.exists() else []:
                try:
                    if path.is_symlink():
                        raise OSError("legacy marker cannot be a symbolic link")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    self._quarantine_unproved(path, exc)
                if not isinstance(payload, dict):
                    self._quarantine_unproved(path, ValueError("marker is not an object"))
                if payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
                    receipt = load_transaction_receipt(path)
                    if self._memory_receipt(receipt):
                        receipt_paths.append(path)
                    continue
                if payload.get("schema_version") != EFFECT_MARKER_SCHEMA_VERSION:
                    continue
                if not self._legacy_memory_marker(payload):
                    continue
                try:
                    receipt = self._convert_legacy_marker(payload)
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    self._quarantine_unproved(path, exc)
                archive = self.artifact_root / "system" / "migrations" / "legacy-markers" / path.parent.name / path.name
                self._reject_control_symlink(archive, "legacy marker archive")
                if archive.exists():
                    archived = json.loads(archive.read_text(encoding="utf-8"))
                    if canonical_digest(archived) != canonical_digest(payload):
                        raise MemoryClosureMigrationError("legacy marker archive conflicts with source")
                else:
                    atomic_create_json(
                        archive,
                        payload,
                        artifact_root=self.artifact_root,
                    )
                atomic_write_json(path, receipt, artifact_root=self.artifact_root)
                migrated_markers.append(str(path.relative_to(self.artifact_root)))
                legacy_digests[str(path.relative_to(self.artifact_root))] = str(payload["marker_digest"])
                receipt_paths.append(path)

        # Publishing in immutable revision order yields the latest legal
        # head while retaining every historical receipt unchanged.  Revision
        # order is authoritative; timestamps are only a deterministic
        # tie-breaker because legacy clocks can collide or move backwards.
        ordered = self._ordered_receipt_paths(receipt_paths)
        published_heads: list[str] = []
        head_published_transactions, pre_head_transactions = self._head_publication_states()
        # Convert and publish the legacy history first.  A current-schema
        # historical receipt can legitimately be superseded by a later legacy
        # receipt whose digest changed during conversion; validating that
        # historical receipt against the stale pre-migration head would reject
        # a provable mixed-version upgrade before the later head is rebuilt.
        for path in ordered:
            receipt = load_transaction_receipt(path)
            if not receipt.get("migration_source_marker_digest"):
                continue
            for published in publish_current_head_sets(self.artifact_root, path, receipt):
                published_heads.append(str(published.relative_to(self.artifact_root)))
        for path in ordered:
            receipt = load_transaction_receipt(path)
            transaction_id = str(receipt["transaction_id"])
            migrated_legacy = bool(receipt.get("migration_source_marker_digest"))
            if migrated_legacy:
                continue
            must_have_published_head = transaction_id in head_published_transactions or (
                transaction_id not in pre_head_transactions
            )
            if must_have_published_head:
                self._validate_already_published_heads(receipt)
                continue
            for published in publish_current_head_sets(self.artifact_root, path, receipt):
                published_heads.append(str(published.relative_to(self.artifact_root)))

        head_digests = self._validate_migrated_state(allow_inflight=allow_inflight)
        receipt_digests = {
            str(path.relative_to(self.artifact_root)): str(load_transaction_receipt(path)["receipt_digest"])
            for path in ordered
        }

        core: dict[str, Any] = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "status": "completed",
            "migrated_markers": migrated_markers,
            "legacy_marker_digests": legacy_digests,
            "migrated_bundles": migrated_bundles,
            "quarantined_legacy_projection_records": migrated_projection_records,
            "published_heads": sorted(dict.fromkeys(published_heads)),
            "receipt_digests": receipt_digests,
            "head_set_digests": head_digests,
            "consistency_check": "passed",
        }
        receipt = {**core, "migration_digest": canonical_digest(core)}
        self._reject_control_symlink(self.receipt_path, "memory closure migration receipt")
        atomic_create_json(
            self.receipt_path,
            receipt,
            artifact_root=self.artifact_root,
        )
        return receipt

    def _ordered_receipt_paths(self, receipt_paths: list[Path]) -> list[Path]:
        """Topologically order immutable receipts by every URI revision edge.

        A transaction can advance several objects whose revision numbers are
        unrelated.  Sorting by a transaction's maximum revision (or by its
        timestamp/file name) can therefore publish a dependent revision before
        its predecessor.  Build the actual artifact DAG instead: for every
        ``uri: before -> after`` effect, the receipt that produced ``before``
        must precede this receipt.  Cycles, gaps and same-revision forks are
        unprovable migration inputs and fail closed.
        """

        paths = sorted(set(receipt_paths), key=str)
        receipts = {path: load_transaction_receipt(path) for path in paths}
        dependencies: dict[Path, set[Path]] = {path: set() for path in paths}
        by_uri_revision: dict[tuple[str, int], Path] = {}

        for path, receipt in receipts.items():
            snapshots = receipt.get("effect_snapshots", [])
            if not isinstance(snapshots, list) or not snapshots:
                raise MemoryClosureMigrationError(
                    f"memory receipt has no effect snapshots: {receipt.get('transaction_id')}"
                )
            for snapshot in snapshots:
                if not isinstance(snapshot, dict) or not snapshot.get("uri"):
                    raise MemoryClosureMigrationError("memory receipt has an invalid revision effect")
                uri = str(snapshot["uri"])
                before = snapshot.get("before_revision")
                after = snapshot.get("after_revision")
                if (
                    isinstance(before, bool)
                    or not isinstance(before, int)
                    or isinstance(after, bool)
                    or not isinstance(after, int)
                    or before < 0
                    or after != before + 1
                ):
                    raise MemoryClosureMigrationError(f"memory receipt has a non-contiguous revision effect: {uri}")
                identity = (uri, after)
                existing = by_uri_revision.setdefault(identity, path)
                if existing != path:
                    raise MemoryClosureMigrationError(f"memory receipt history has a same-revision fork: {uri}#{after}")

        for path, receipt in receipts.items():
            for snapshot in receipt["effect_snapshots"]:
                uri = str(snapshot["uri"])
                before = int(snapshot["before_revision"])
                if before == 0:
                    continue
                predecessor = by_uri_revision.get((uri, before))
                if predecessor is None:
                    raise MemoryClosureMigrationError(
                        f"memory receipt history has a missing predecessor: {uri}#{before}"
                    )
                dependencies[path].add(predecessor)

        def key(path: Path) -> tuple[str, str, str]:
            receipt = receipts[path]
            return (
                str(receipt.get("created_at") or ""),
                str(receipt.get("transaction_id") or ""),
                str(path),
            )

        dependents: dict[Path, set[Path]] = {path: set() for path in paths}
        for path, required in dependencies.items():
            for predecessor in required:
                dependents[predecessor].add(path)
        ready: list[tuple[tuple[str, str, str], Path]] = [
            (key(path), path) for path, required in dependencies.items() if not required
        ]
        heapq.heapify(ready)
        ordered: list[Path] = []
        while ready:
            _sort_key, path = heapq.heappop(ready)
            ordered.append(path)
            for dependent in sorted(dependents[path], key=key):
                dependencies[dependent].discard(path)
                if not dependencies[dependent]:
                    heapq.heappush(ready, (key(dependent), dependent))
        if len(ordered) != len(paths):
            raise MemoryClosureMigrationError("memory receipt revision dependency graph contains a cycle")
        return ordered

    def _head_publication_states(self) -> tuple[set[str], set[str]]:
        """Classify current receipts without guessing from a missing pointer.

        A current-schema receipt with no redo is a completed transaction whose
        head must already exist.  Only a legacy marker converted by this
        migration, or an explicitly pre-head redo, may authorize initial head
        publication.  A committed outbox is an additional post-head proof.
        """

        try:
            entries = RedoLog(self.artifact_root).pending_entries()
        except RedoControlFileError as exc:
            raise MemoryClosureMigrationError("migration cannot classify corrupt in-flight redo artifacts") from exc
        published = {
            str(entry.operation.payload.get("transaction_id") or entry.operation.operation_id)
            for entry in entries
            if entry.phase == "head_published"
        }
        pre_head = {
            str(entry.operation.payload.get("transaction_id") or entry.operation.operation_id)
            for entry in entries
            if entry.phase != "head_published"
        }
        outbox_root = self.artifact_root / "system" / "outbox"
        for path in sorted(outbox_root.glob("*.json")) if outbox_root.exists() else ():
            self._reject_control_symlink(path, "canonical outbox publication proof")
            try:
                outbox = validate_outbox(
                    json.loads(path.read_text(encoding="utf-8")),
                    tenant_id=self.tenant_id,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                raise MemoryClosureMigrationError(
                    f"migration cannot classify canonical outbox publication: {path.name}"
                ) from exc
            transaction_id = str(outbox["transaction_id"])
            if path.stem != transaction_id:
                raise MemoryClosureMigrationError("canonical outbox publication path does not match its transaction")
            if outbox["status"] == "committed":
                published.add(transaction_id)
        # A canonical head-set is published atomically before the committer
        # advances each per-operation redo file.  A process exit between those
        # advances can therefore leave mixed redo phases for one transaction;
        # one head_published member proves the whole head-set crossed the
        # publication boundary.
        pre_head.difference_update(published)
        return published, pre_head

    def _validate_already_published_heads(self, receipt: dict[str, Any]) -> None:
        transaction_id = str(receipt["transaction_id"])
        receipt_digest = str(receipt["receipt_digest"])
        for snapshot in receipt.get("effect_snapshots", []) or []:
            if not isinstance(snapshot, dict) or not snapshot.get("uri"):
                raise MemoryClosureMigrationError(
                    f"head-published transaction has an invalid effect snapshot: {transaction_id}"
                )
            uri = str(snapshot["uri"])
            try:
                head, bound_receipt, _bound_snapshot = load_current_head(
                    self.artifact_root,
                    uri,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise MemoryClosureMigrationError(
                    f"head-published transaction {transaction_id} is missing its current head for {uri}"
                ) from exc
            if str(head.get("current_transaction_id") or "") != transaction_id:
                object_payload = dict(snapshot.get("object", {}) or {})
                metadata = dict(object_payload.get("metadata", {}) or {})
                prior_revision = metadata.get(
                    "lifecycle_revision",
                    metadata.get("revision", 0),
                )
                current_revision = head.get("current_revision", 0)
                if (
                    isinstance(prior_revision, bool)
                    or not isinstance(prior_revision, int)
                    or isinstance(current_revision, bool)
                    or not isinstance(current_revision, int)
                    or current_revision <= prior_revision
                ):
                    raise MemoryClosureMigrationError(
                        f"head-published transaction {transaction_id} has no legal later head for {uri}"
                    )
                continue
            if (
                str(head.get("receipt_digest") or "") != receipt_digest
                or str(bound_receipt.get("receipt_digest") or "") != receipt_digest
            ):
                raise MemoryClosureMigrationError(
                    f"head-published transaction {transaction_id} has a detached current head for {uri}"
                )

    def _migrate_canonical_prepared_intents(self) -> dict[str, int]:
        """Backfill create-only canonical intents from a still provable outbox.

        The mutable outbox is used only as migration evidence.  A per-
        transaction receipt records the exact immutable digest, so an
        interrupted sweep can resume without rewriting any completed proof.
        """

        outbox_root = self.artifact_root / "system" / "outbox"
        proof_store = ImmutablePlanningProofStore(
            self.artifact_root,
            tenant_id=self.tenant_id,
        )
        receipt_by_transaction: dict[str, dict[str, Any]] = {}
        transaction_root = self.artifact_root / "system" / "transactions"
        for receipt_path in sorted(transaction_root.glob("*.json")) if transaction_root.exists() else ():
            self._reject_control_symlink(
                receipt_path,
                "canonical transaction receipt migration source",
            )
            try:
                raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise MemoryClosureMigrationError(
                    f"canonical receipt is unreadable during prepared-intent migration: {receipt_path.name}"
                ) from exc
            if (
                not isinstance(raw_receipt, dict)
                or raw_receipt.get("schema_version") != TRANSACTION_RECEIPT_SCHEMA_VERSION
            ):
                continue
            receipt = load_transaction_receipt(receipt_path)
            transaction_id = str(receipt["transaction_id"])
            existing_receipt = receipt_by_transaction.setdefault(
                transaction_id,
                receipt,
            )
            if existing_receipt.get("receipt_digest") != receipt.get("receipt_digest"):
                raise MemoryClosureMigrationError(
                    "canonical transaction has conflicting immutable receipts during prepared-intent migration"
                )
        migrated = 0
        validated = 0
        for path in sorted(outbox_root.glob("*.json")) if outbox_root.exists() else ():
            self._reject_control_symlink(path, "canonical outbox migration source")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                outbox = validate_outbox(raw, tenant_id=self.tenant_id)
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                raise MemoryClosureMigrationError(
                    f"canonical prepared-intent migration cannot prove {path.name}"
                ) from exc
            transaction_id = str(outbox["transaction_id"])
            expected_path = outbox_root / f"{transaction_id}.json"
            if path != expected_path:
                raise MemoryClosureMigrationError("canonical outbox path does not match its transaction identity")
            intent_path = proof_store.canonical_intent_path(transaction_id)
            existed = intent_path.exists() or intent_path.is_symlink()
            bound_receipt = receipt_by_transaction.get(transaction_id)
            if (
                not existed
                and bound_receipt is not None
                and not bound_receipt.get("migration_source_marker_digest")
                and bound_receipt.get("prepared_intent_schema_version") == CANONICAL_PREPARED_INTENT_SCHEMA_VERSION
            ):
                raise MemoryClosureMigrationError(
                    f"a current-schema canonical receipt lost its immutable prepared intent: {transaction_id}"
                )
            try:
                intent = proof_store.ensure_canonical_intent(outbox)
            except PlanningProofIntegrityError as exc:
                raise MemoryClosureMigrationError(
                    f"canonical prepared-intent migration conflicts for {transaction_id}"
                ) from exc
            migration_path = (
                self.artifact_root / "system" / "migrations" / "canonical-prepared-intents" / f"{transaction_id}.json"
            )
            self._reject_control_symlink(
                migration_path,
                "canonical prepared-intent migration receipt",
            )
            core = {
                "schema_version": PREPARED_INTENT_MIGRATION_SCHEMA_VERSION,
                "tenant_id": self.tenant_id,
                "transaction_id": transaction_id,
                "prepared_intent_digest": str(intent["prepared_intent_digest"]),
                "prepared_intent_artifact_digest": str(intent["artifact_digest"]),
            }
            migration_receipt = {**core, "migration_digest": canonical_digest(core)}
            if migration_path.exists():
                try:
                    stored = json.loads(migration_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise MemoryClosureMigrationError(
                        "canonical prepared-intent migration receipt is unreadable"
                    ) from exc
                if stored != migration_receipt:
                    raise MemoryClosureMigrationError("canonical prepared-intent migration receipt conflicts")
            else:
                atomic_create_json(
                    migration_path,
                    migration_receipt,
                    artifact_root=self.artifact_root,
                )
            if existed:
                validated += 1
            else:
                migrated += 1
        return {"migrated": migrated, "validated": validated}

    def _migrate_projection_state(self) -> list[str]:
        """Preserve and detach legacy derived records so v5 can be rebuilt.

        Projection state is not canonical fact.  An unsupported record must
        never be adopted, but retaining the original in quarantine gives the
        migration an auditable and interruption-safe outcome.
        """

        state_root = self.artifact_root / "system" / "projection-state"
        quarantined: list[str] = []
        for path in sorted(state_root.glob("**/attempt-*.json")) if state_root.exists() else ():
            try:
                if path.is_symlink():
                    raise ProjectionIntegrityError("projection migration attempt cannot be a symbolic link")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ProjectionIntegrityError("projection attempt is not an object")
                ProjectionRecord.from_dict(payload)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                ProjectionIntegrityError,
            ) as exc:
                quarantine_record = quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="legacy_projection_record",
                    error=exc,
                    identifiers={"record_id": path.stem},
                )
                quarantined.append(quarantine_record.original_relative_path)
        store = ProjectionRecordStore(self.artifact_root)
        for path in sorted(state_root.glob("**/current.json")) if state_root.exists() else ():
            try:
                if path.is_symlink():
                    raise ProjectionIntegrityError("projection migration pointer cannot be a symbolic link")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ProjectionIntegrityError("projection pointer is not an object")
                pointer_core = {key: value for key, value in payload.items() if key != "pointer_digest"}
                if payload.get("schema_version") != "canonical_projection_current_v5" or payload.get(
                    "pointer_digest"
                ) != canonical_digest(pointer_core):
                    raise ProjectionIntegrityError("unsupported or corrupt projection pointer")
                claim_uri = str(payload.get("claim_uri") or "")
                source_revision = int(payload.get("source_revision", 0) or 0)
                attempt_id = str(payload.get("projection_attempt_id") or "")
                expected_record = store.attempt_path(claim_uri, source_revision, attempt_id)
                if (
                    not claim_uri
                    or not attempt_id
                    or expected_record.is_symlink()
                    or not expected_record.exists()
                    or str(payload.get("record_path") or "") != str(expected_record)
                ):
                    raise ProjectionIntegrityError("projection pointer has no matching attempt record")
                projection_record = ProjectionRecord.from_dict(json.loads(expected_record.read_text(encoding="utf-8")))
                if (
                    projection_record.claim_uri != claim_uri
                    or projection_record.source_revision != source_revision
                    or projection_record.projection_attempt_id != attempt_id
                    or projection_record.publish_token != str(payload.get("publish_token") or "")
                    or str(projection_record.to_dict()["record_digest"]) != str(payload.get("record_digest") or "")
                ):
                    raise ProjectionIntegrityError("projection pointer disagrees with its attempt record")
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                ProjectionIntegrityError,
            ) as exc:
                quarantine_record = quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="legacy_projection_record",
                    error=exc,
                    identifiers={"record_id": path.stem},
                )
                quarantined.append(quarantine_record.original_relative_path)
        self._publish_projection_migration_receipt(quarantined)
        return quarantined

    def _publish_projection_migration_receipt(self, quarantined: list[str]) -> dict[str, Any]:
        self._reject_control_symlink(
            self.projection_receipt_path,
            "projection migration receipt",
        )
        if self.projection_receipt_path.exists():
            try:
                payload = json.loads(self.projection_receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise MemoryClosureMigrationError("projection migration receipt is unreadable") from exc
            if not isinstance(payload, dict):
                raise MemoryClosureMigrationError("projection migration receipt is corrupt")
            stored_core = {key: value for key, value in payload.items() if key != "migration_digest"}
            if (
                payload.get("schema_version") != PROJECTION_MIGRATION_SCHEMA_VERSION
                or payload.get("tenant_id") != self.tenant_id
                or payload.get("status") != "completed"
                or payload.get("migration_digest") != canonical_digest(stored_core)
            ):
                raise MemoryClosureMigrationError("projection migration receipt integrity check failed")
            return payload
        receipt_core: dict[str, Any] = {
            "schema_version": PROJECTION_MIGRATION_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "status": "completed",
            "target_record_schema": "canonical_projection_v5",
            "target_pointer_schema": "canonical_projection_current_v5",
            "quarantined_legacy_artifacts": sorted(dict.fromkeys(quarantined)),
            "rebuild_required": bool(quarantined),
        }
        receipt = {**receipt_core, "migration_digest": canonical_digest(receipt_core)}
        self._reject_control_symlink(
            self.projection_receipt_path,
            "projection migration receipt",
        )
        atomic_create_json(
            self.projection_receipt_path,
            receipt,
            artifact_root=self.artifact_root,
        )
        return receipt

    def _migrate_planning_envelopes(self) -> dict[str, Any]:
        """Validate v2 planning artifacts and fail closed on unproved older state."""

        store = PlanningEnvelopeStore(self.shared_root, tenant_id=self.tenant_id)
        quarantined: list[str] = []
        paths = [
            *(sorted(store.root.glob("*.json")) if store.root.exists() else ()),
            *(
                sorted((self.artifact_root / "system" / "planning-envelope-anchors").glob("*.json"))
                if (self.artifact_root / "system" / "planning-envelope-anchors").exists()
                else ()
            ),
        ]
        for path in paths:
            try:
                if path.is_symlink():
                    raise PlanningEnvelopeIntegrityError("planning migration artifact cannot be a symbolic link")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise PlanningEnvelopeIntegrityError("planning migration artifact is not an object")
                task_id = str(payload.get("task_id") or "")
                if path.parent == store.root:
                    validate_planning_envelope_payload(
                        payload,
                        tenant_id=self.tenant_id,
                        task_id=task_id,
                    )
                    if payload.get("schema_version") != PLANNING_ENVELOPE_SCHEMA_VERSION:
                        raise PlanningEnvelopeIntegrityError("unsupported planning envelope schema")
                    if not task_id or path.is_symlink() or path.resolve() != store.path(task_id).resolve():
                        raise PlanningEnvelopeIntegrityError("planning envelope path identity is invalid")
                else:
                    if payload.get("schema_version") != PLANNING_ENVELOPE_ANCHOR_SCHEMA_VERSION:
                        raise PlanningEnvelopeIntegrityError("unsupported planning envelope anchor schema")
                    store._load_anchor(path)
                    if not task_id or path.is_symlink() or path.resolve() != store.anchor_path(task_id).resolve():
                        raise PlanningEnvelopeIntegrityError("planning envelope anchor path identity is invalid")
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                PlanningEnvelopeIntegrityError,
            ) as exc:
                record = quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="legacy_planning_envelope",
                    error=exc,
                    identifiers={"record_id": path.stem},
                )
                quarantined.append(record.original_relative_path)
        if quarantined:
            self._publish_planning_migration_failure(
                "unsupported or corrupt planning artifacts were quarantined",
                quarantined,
            )
        try:
            validation = store.validate_all()
        except PlanningEnvelopeIntegrityError as exc:
            self._publish_planning_migration_failure(str(exc), quarantined)
        self._reject_control_symlink(self.planning_receipt_path, "planning migration receipt")
        if self.planning_receipt_path.exists():
            return self._load_planning_migration_receipt()
        core: dict[str, Any] = {
            "schema_version": PLANNING_MIGRATION_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "status": "completed",
            "target_envelope_schema": PLANNING_ENVELOPE_SCHEMA_VERSION,
            "target_anchor_schema": PLANNING_ENVELOPE_ANCHOR_SCHEMA_VERSION,
            "initial_validation": validation,
        }
        receipt = {**core, "migration_digest": canonical_digest(core)}
        self._reject_control_symlink(self.planning_receipt_path, "planning migration receipt")
        atomic_create_json(
            self.planning_receipt_path,
            receipt,
            artifact_root=self.artifact_root,
        )
        return receipt

    def _load_planning_migration_receipt(self) -> dict[str, Any]:
        self._reject_control_symlink(self.planning_receipt_path, "planning migration receipt")
        try:
            payload = json.loads(self.planning_receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MemoryClosureMigrationError("planning migration receipt is unreadable") from exc
        if not isinstance(payload, dict):
            raise MemoryClosureMigrationError("planning migration receipt is corrupt")
        core = {key: value for key, value in payload.items() if key != "migration_digest"}
        if (
            payload.get("schema_version") != PLANNING_MIGRATION_SCHEMA_VERSION
            or payload.get("tenant_id") != self.tenant_id
            or payload.get("status") != "completed"
            or payload.get("migration_digest") != canonical_digest(core)
        ):
            raise MemoryClosureMigrationError("planning migration receipt integrity check failed")
        return payload

    def _publish_planning_migration_failure(
        self,
        reason: str,
        quarantined: list[str],
    ) -> NoReturn:
        self._reject_control_symlink(
            self.planning_failure_path,
            "planning migration failure receipt",
        )
        if self.planning_failure_path.exists():
            self._raise_planning_migration_failure()
        core: dict[str, Any] = {
            "schema_version": PLANNING_MIGRATION_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "status": "failed",
            "reason": str(reason)[:1000],
            "quarantined_artifacts": sorted(dict.fromkeys(quarantined)),
        }
        atomic_create_json(
            self.planning_failure_path,
            {**core, "migration_digest": canonical_digest(core)},
            artifact_root=self.artifact_root,
        )
        raise MemoryClosureMigrationError(f"planning envelope migration failed: {reason}")

    def _raise_planning_migration_failure(self) -> NoReturn:
        self._reject_control_symlink(
            self.planning_failure_path,
            "planning migration failure receipt",
        )
        try:
            payload = json.loads(self.planning_failure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MemoryClosureMigrationError("planning migration failure receipt is unreadable") from exc
        if not isinstance(payload, dict):
            raise MemoryClosureMigrationError("planning migration failure receipt is corrupt")
        core = {key: value for key, value in payload.items() if key != "migration_digest"}
        if (
            payload.get("schema_version") != PLANNING_MIGRATION_SCHEMA_VERSION
            or payload.get("tenant_id") != self.tenant_id
            or payload.get("status") != "failed"
            or payload.get("migration_digest") != canonical_digest(core)
        ):
            raise MemoryClosureMigrationError("planning migration failure receipt is corrupt")
        raise MemoryClosureMigrationError(
            f"planning envelope migration previously failed: {payload.get('reason', 'unknown reason')}"
        )

    def _migrate_bundles(self) -> list[str]:
        migrated: list[str] = []
        for obj in self.source_store.list_objects():
            if not self._canonical_object(obj):
                continue
            object_dir = self._object_dir(obj.uri)
            bundle_pointer = object_dir / ".bundle-current.json"
            self._reject_control_symlink(bundle_pointer, "canonical bundle current pointer")
            if bundle_pointer.exists():
                continue
            try:
                content = self.source_store.read_content(obj.layers.l2_uri or obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = ""
            self.source_store.write_object(obj, content=content)
            migrated.append(obj.uri)
        return migrated

    def _convert_legacy_marker(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_legacy_core(payload)
        marker_digest = str(payload["marker_digest"])
        operations = [ContextOperation.from_dict(dict(item)) for item in payload["operations"]]
        effects = {
            str(item.get("uri") or ""): dict(item) for item in payload["object_effects"] if isinstance(item, dict)
        }
        if len(effects) != len(operations):
            raise MemoryClosureMigrationError("legacy marker operation/effect set is incomplete")
        for operation in operations:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict):
                raise MemoryClosureMigrationError("legacy memory marker has no object snapshot")
            uri = str(object_payload.get("uri") or "")
            effect = effects.get(uri)
            if effect is None or effect.get("expected_exists") is not True:
                raise MemoryClosureMigrationError("legacy memory marker does not prove object presence")
            if effect.get("object_digest") != canonical_digest(object_payload):
                raise MemoryClosureMigrationError("legacy object snapshot does not match marker digest")
            if effect.get("content_digest") != canonical_digest(str(operation.payload.get("content", ""))):
                raise MemoryClosureMigrationError("legacy content snapshot does not match marker digest")
        commit_groups = {
            str(operation.payload.get("commit_group_id") or "")
            for operation in operations
            if operation.payload.get("commit_group_id")
        }
        if len(commit_groups) > 1:
            raise MemoryClosureMigrationError("legacy memory marker crosses commit groups")
        commit_group_id = next(
            iter(commit_groups),
            f"migrated_commit_group_{canonical_digest([marker_digest])[:32]}",
        )
        planning_digest = canonical_digest(
            {"schema_version": "migrated_planning_proof_v1", "legacy_marker_digest": marker_digest}
        )
        prepared_intent_digest = canonical_digest(
            {"schema_version": "migrated_prepared_intent_v1", "legacy_marker_digest": marker_digest}
        )
        for operation in operations:
            operation.payload["transaction_id"] = str(payload["transaction_id"])
            operation.payload["idempotency_key"] = str(payload["idempotency_key"])
            operation.payload["tenant_id"] = str(payload["tenant_id"])
            operation.payload["commit_group_id"] = commit_group_id
            operation.payload["planning_digest"] = planning_digest
        diff = dict(payload["diff"])
        diff["user_id"] = str(payload["user_id"])
        diff["operations"] = [operation.to_dict() for operation in operations]
        diff.setdefault("pending_operations", [])
        diff.setdefault("rejected_operations", [])
        if not diff.get("diff_id"):
            diff["diff_id"] = f"migrated_diff_{payload['transaction_id']}"
        self._persist_migrated_diff(diff)
        receipt = build_transaction_receipt(
            transaction_id=str(payload["transaction_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            tenant_id=str(payload["tenant_id"]),
            user_id=str(payload["user_id"]),
            commit_group_id=commit_group_id,
            operations=operations,
            diff=diff,
            planning_digest=planning_digest,
            prepared_intent_digest=prepared_intent_digest,
            relation_effects=[dict(item) for item in payload["relation_effects"]],
            created_at=str(payload.get("committed_at") or ""),
        )
        core = {
            **{key: value for key, value in receipt.items() if key != "receipt_digest"},
            "migration_source_marker_digest": marker_digest,
        }
        return validate_transaction_receipt({**core, "receipt_digest": canonical_digest(core)})

    def _persist_migrated_diff(self, diff: dict[str, Any]) -> None:
        diff_id = str(diff.get("diff_id") or "")
        if not diff_id or diff_id in {".", ".."} or "/" in diff_id or "\\" in diff_id:
            raise MemoryClosureMigrationError("legacy memory marker has an unsafe diff identity")
        path = self.artifact_root / "system" / "diffs" / f"{diff_id}.json"
        self._reject_control_symlink(path, "migrated diff artifact")
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise MemoryClosureMigrationError("migrated diff artifact is unreadable") from exc
            if canonical_digest(existing) != canonical_digest(diff):
                raise MemoryClosureMigrationError("migrated diff artifact conflicts with marker")
            return
        atomic_create_json(path, diff, artifact_root=self.artifact_root)

    def _audit_no_legacy_memory_markers(self) -> None:
        for directory_name in ("transactions", "operations"):
            marker_root = self.artifact_root / "system" / directory_name
            for path in sorted(marker_root.glob("*.json")) if marker_root.exists() else ():
                try:
                    if path.is_symlink():
                        raise OSError("memory marker cannot be a symbolic link")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    self._quarantine_unproved(path, exc)
                if (
                    isinstance(payload, dict)
                    and payload.get("schema_version") == EFFECT_MARKER_SCHEMA_VERSION
                    and self._legacy_memory_marker(payload)
                ):
                    self._quarantine_unproved(
                        path,
                        MemoryClosureMigrationError("legacy canonical marker appeared after migration completed"),
                    )

    def _validate_legacy_core(self, payload: dict[str, Any]) -> None:
        core = {key: value for key, value in payload.items() if key != "marker_digest"}
        if payload.get("marker_digest") != canonical_digest(core):
            raise MemoryClosureMigrationError("legacy marker digest is corrupt")
        if (
            payload.get("status") != "committed"
            or payload.get("tenant_id") != self.tenant_id
            or not isinstance(payload.get("operations"), list)
            or not payload["operations"]
            or not isinstance(payload.get("object_effects"), list)
            or not isinstance(payload.get("relation_effects"), list)
            or not isinstance(payload.get("diff"), dict)
        ):
            raise MemoryClosureMigrationError("legacy memory marker is incomplete or crosses tenant")
        ids = [str(item.get("operation_id") or "") for item in payload["operations"] if isinstance(item, dict)]
        if ids != [str(item) for item in payload.get("operation_ids", [])] or len(ids) != len(payload["operations"]):
            raise MemoryClosureMigrationError("legacy marker operation set is inconsistent")

    def _load_migration_receipt(self) -> dict[str, Any]:
        self._reject_control_symlink(self.receipt_path, "memory closure migration receipt")
        try:
            payload = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MemoryClosureMigrationError("migration receipt is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != MIGRATION_SCHEMA_VERSION:
            raise MemoryClosureMigrationError("migration receipt schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "migration_digest"}
        if (
            payload.get("tenant_id") != self.tenant_id
            or payload.get("status") != "completed"
            or payload.get("migration_digest") != canonical_digest(core)
        ):
            raise MemoryClosureMigrationError("migration receipt integrity check failed")
        return payload

    def _quarantine_unproved(self, path: Path, exc: BaseException) -> NoReturn:
        artifact_digest = ""
        if path.exists() and not path.is_symlink():
            artifact_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        failure_core: dict[str, Any] = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "status": "failed",
            "artifact_path": str(path.relative_to(self.artifact_root)),
            "artifact_digest": artifact_digest,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
        self._reject_control_symlink(
            self.failure_path,
            "memory closure migration failure receipt",
        )
        atomic_create_json(
            self.failure_path,
            {**failure_core, "migration_digest": canonical_digest(failure_core)},
            artifact_root=self.artifact_root,
        )
        if path.exists() or path.is_symlink():
            quarantine_control_file(
                self.artifact_root,
                path,
                kind="legacy_memory_marker",
                error=exc,
                identifiers={"path": str(path.relative_to(self.artifact_root))},
            )
        raise MemoryClosureMigrationError(
            f"legacy memory artifact cannot be proved: {path.name}: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc

    def _raise_recorded_failure(self) -> NoReturn:
        self._reject_control_symlink(
            self.failure_path,
            "memory closure migration failure receipt",
        )
        try:
            payload = json.loads(self.failure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MemoryClosureMigrationError("migration failure receipt is unreadable") from exc
        if not isinstance(payload, dict):
            raise MemoryClosureMigrationError("migration failure receipt is corrupt")
        core = {key: value for key, value in payload.items() if key != "migration_digest"}
        if (
            payload.get("schema_version") != MIGRATION_SCHEMA_VERSION
            or payload.get("tenant_id") != self.tenant_id
            or payload.get("status") != "failed"
            or payload.get("migration_digest") != canonical_digest(core)
        ):
            raise MemoryClosureMigrationError("migration failure receipt is corrupt")
        raise MemoryClosureMigrationError(
            f"memory closure migration remains failed for {payload.get('artifact_path')}: {payload.get('error_type')}"
        )

    @staticmethod
    def _reject_control_symlink(path: Path, label: str) -> None:
        if path.is_symlink():
            raise MemoryClosureMigrationError(f"{label} cannot be a symbolic link")

    def validate_current_state(self) -> dict[str, str]:
        """Strict post-recovery validation used before Runtime can become READY."""

        return self._validate_migrated_state(allow_inflight=False)

    def _validate_migrated_state(self, *, allow_inflight: bool = False) -> dict[str, str]:
        head_digests: dict[str, str] = {}
        head_uris = set(
            iter_current_head_uris(
                self.artifact_root,
                kinds=("slot", "claim", "pending_proposal"),
            )
        )
        for uri in sorted(head_uris):
            head, _receipt, _snapshot = load_current_head(self.artifact_root, uri)
            committed = read_committed_canonical(
                self.source_store,
                uri,
                self.relation_store,
            )
            if committed.from_before_image and not allow_inflight:
                raise MemoryClosureMigrationError(f"migrated current head does not match Source bundle: {uri}")
            head_digests[uri] = str(head["head_digest"])
        if not allow_inflight:
            orphaned = sorted(
                obj.uri
                for obj in self.source_store.list_objects()
                if self._canonical_object(obj) and obj.uri not in head_uris
            )
            if orphaned:
                missing_heads = sorted(uri for uri in orphaned if receipt_history_contains_uri(self.artifact_root, uri))
                if missing_heads:
                    raise MemoryClosureMigrationError(
                        "required current head is missing for committed canonical Source objects: "
                        + ",".join(missing_heads)
                    )
                raise MemoryClosureMigrationError(
                    "canonical Source objects have no immutable receipt/current head proof: " + ",".join(orphaned)
                )
        return head_digests

    def _object_dir(self, uri: str) -> Path:
        resolver = getattr(self.source_store, "_object_dir", None)
        if not callable(resolver):
            raise MemoryClosureMigrationError("SourceStore cannot migrate canonical bundles")
        resolved = resolver(uri)
        if not isinstance(resolved, (str, Path)):
            raise MemoryClosureMigrationError("SourceStore returned an invalid object directory")
        return Path(resolved)

    @staticmethod
    def _canonical_object(obj: ContextObject) -> bool:
        return (
            str(dict(obj.metadata or {}).get("canonical_kind") or "") in {"slot", "claim", "pending_proposal"}
            or obj.schema_version in {"canonical_memory_v2", "canonical_pending_proposal_v1"}
            or "/memories/canonical/" in obj.uri
            or "/memories/pending/" in obj.uri
        )

    @staticmethod
    def _legacy_memory_marker(payload: dict[str, Any]) -> bool:
        for item in payload.get("operations", []) or []:
            if not isinstance(item, dict):
                continue
            operation_payload = dict(item.get("payload", {}) or {})
            obj = operation_payload.get("context_object")
            kind = str(dict(obj.get("metadata", {}) or {}).get("canonical_kind") or "") if isinstance(obj, dict) else ""
            if (
                operation_payload.get("canonical_memory") is True
                or operation_payload.get("canonical_pending_proposal") is True
                or kind in {"slot", "claim", "pending_proposal"}
            ):
                return True
        return False

    @staticmethod
    def _memory_receipt(receipt: dict[str, Any]) -> bool:
        return any(
            isinstance(item, dict) and str(item.get("canonical_kind") or "") in {"slot", "claim", "pending_proposal"}
            for item in receipt.get("effect_snapshots", []) or []
        )
