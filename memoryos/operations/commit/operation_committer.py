"""操作提交里的操作提交。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.action_policy.update.action_policy_updater import ActionPolicyUpdater
from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.store.local_stores import InMemoryLockStore
from memoryos.contextdb.store.source_store import (
    IndexStore,
    LockStore,
    QueueJob,
    QueueStore,
    RelationStore,
    SourceStore,
)
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.time import utc_now
from memoryos.memory.canonical.event import canonical_json, resolve_content_path
from memoryos.memory.canonical.evidence import evidence_hash
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.canonical.transaction import RevisionConflictError
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.operation_coalescer import OperationCoalescer
from memoryos.operations.commit.redo_log import RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.conflict_resolver import ConflictResolver
from memoryos.operations.resolver.target_resolver import TargetResolver


class OperationCommitter:
    """负责加锁、版本校验、批量提交、故障恢复和 Outbox 落盘。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str,
        lock_store: LockStore | None = None,
        relation_store: RelationStore | None = None,
        queue_store: QueueStore | None = None,
        target_resolver: TargetResolver | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.root = Path(root)
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.target_resolver = target_resolver or TargetResolver(index_store)
        self.redo = RedoLog(root)
        self.diff_writer = DiffWriter(root)
        self.audit = AuditWriter(root)
        self.path_lock = PathLock(lock_store or InMemoryLockStore())
        self.action_policy_updater = ActionPolicyUpdater()

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        """执行这一步处理，并保持已有状态约束。"""

        canonical = [operation for operation in operations if operation.payload.get("canonical_memory") is True]
        if canonical:
            diffs = []
            regular = [operation for operation in operations if operation.payload.get("canonical_memory") is not True]
            if regular:
                diffs.append(self.commit(user_id, regular))
            grouped: dict[str, list[ContextOperation]] = {}
            for operation in canonical:
                transaction_id = str(operation.payload.get("transaction_id", ""))
                grouped.setdefault(transaction_id, []).append(operation)
            diffs.extend(
                self._commit_canonical_batch(user_id, transaction_operations)
                for transaction_operations in grouped.values()
            )
            return ContextDiff(
                user_id=user_id,
                operations=[operation for diff in diffs for operation in diff.operations],
                pending_operations=[operation for diff in diffs for operation in diff.pending_operations],
                rejected_operations=[operation for diff in diffs for operation in diff.rejected_operations],
            )
        resolved_operations: list[ContextOperation] = []
        pending: list[ContextOperation] = []
        for operation in operations:
            result = self.target_resolver.resolve(operation, user_id=user_id)
            if result.resolved:
                resolved_operations.append(result.operation)
            else:
                result.operation.status = OperationStatus.PENDING
                pending.append(result.operation)
        conflict_result = self.conflicts.resolve(self._coalesce_non_policy_operations(resolved_operations))
        for operation in conflict_result.rejected:
            operation.status = OperationStatus.REJECTED
        committed = []
        pending_redo = {entry.operation_id: entry for entry in self.redo.pending_entries()}
        for operation in conflict_result.accepted:
            if operation.status == OperationStatus.PENDING:
                pending.append(operation)
                continue
            lock_key = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
            with self.path_lock.acquire(lock_key):
                marker = self._operation_marker(operation.operation_id)
                if marker.exists():
                    self._validate_operation_marker(marker, operation)
                    operation.status = OperationStatus.COMMITTED
                    committed.append(operation)
                    continue
                pending_entry = pending_redo.get(operation.operation_id)
                if pending_entry is not None and pending_entry.phase not in {"started", "begin"}:
                    self.resume(user_id, pending_entry.operation, pending_entry.phase)
                    if marker.exists():
                        self._validate_operation_marker(marker, operation)
                        operation.status = OperationStatus.COMMITTED
                        committed.append(operation)
                        continue
                self.redo.begin(operation, phase="started")
                self._apply_source(operation)
                self.redo.advance(operation, phase="source_written")
                self._apply_index(operation)
                self.redo.advance(operation, phase="index_written")
                self.audit.record(user_id, "context_operation_committed", operation.to_dict())
                self.redo.advance(operation, phase="audit_written")
                operation.status = OperationStatus.COMMITTED
            committed.append(operation)
        diff = ContextDiff(
            user_id=user_id,
            operations=committed,
            pending_operations=pending,
            rejected_operations=conflict_result.rejected,
        )
        self.diff_writer.write(diff)
        for operation in committed:
            self._write_operation_marker(operation)
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
        return diff

    def _commit_canonical_batch(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        if not operations:
            return ContextDiff(user_id=user_id)
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1 or "" in idempotency_keys:
            raise ValueError("canonical batch requires one transaction_id and idempotency_key")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        completed = self._transaction_marker(idempotency_key)
        if completed.exists():
            self._finalize_canonical_outbox(transaction_id, idempotency_key, operations)
            return self._diff_from_payload(json.loads(completed.read_text(encoding="utf-8")))

        slot_uri = next(
            (
                str(payload.get("uri"))
                for operation in operations
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        with self.path_lock.acquire(f"canonical:{slot_uri}"):
            if completed.exists():
                self._finalize_canonical_outbox(transaction_id, idempotency_key, operations)
                return self._diff_from_payload(json.loads(completed.read_text(encoding="utf-8")))
            self._preflight_canonical_revisions(operations)
            self._validate_authoritative_batch(operations)
            backups = self._capture_canonical_state(operations)
            committed: list[ContextOperation] = []
            self._write_outbox_event(
                transaction_id,
                idempotency_key,
                operations,
                status="prepared",
                before_images=backups,
            )
            for operation in operations:
                self.redo.begin(operation, phase="started")
            try:
                for operation in operations:
                    self._apply_canonical_source(operation)
                    self.redo.advance(operation, phase="source_written")
                    self.audit.record(user_id, "canonical_memory_operation_applied", operation.to_dict())
                    self.redo.advance(operation, phase="audit_written")
                    operation.status = OperationStatus.COMMITTED
                    committed.append(operation)
                self._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    committed,
                    status="source_committed",
                    before_images=backups,
                )
            except Exception:
                self._restore_canonical_state(backups)
                self._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    operations,
                    status="aborted",
                )
                for operation in operations:
                    self.redo.commit(operation.operation_id)
                self.audit.record(
                    user_id,
                    "canonical_memory_transaction_rolled_back",
                    {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in operations]},
                )
                raise
            diff = ContextDiff(
                user_id=user_id,
                operations=committed,
                diff_id=f"diff_{transaction_id}",
            )
            self.diff_writer.write(diff)
            self._write_transaction_marker(completed, diff)
            self.audit.record(
                user_id,
                "canonical_memory_transaction_committed",
                {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in committed]},
            )
            self._finalize_canonical_outbox(transaction_id, idempotency_key, committed, slot_uri=slot_uri)
            for operation in committed:
                self.redo.commit(operation.operation_id)
            return diff

    def _finalize_canonical_outbox(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        slot_uri: str | None = None,
    ) -> Path:
        outbox_path = self._write_outbox_event(
            transaction_id,
            idempotency_key,
            operations,
            status="committed",
        )
        resolved_slot = slot_uri or next(
            (
                str(payload.get("uri"))
                for operation in operations
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        self._enqueue_outbox(transaction_id, resolved_slot, outbox_path, operations)
        return outbox_path

    def _preflight_canonical_revisions(self, operations: list[ContextOperation]) -> None:
        tenants: set[str] = set()
        owners: set[str] = set()
        slot_ids: set[str] = set()
        scope_payloads: set[str] = set()
        for operation in operations:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict) or not object_payload.get("uri"):
                raise ValueError("canonical operation requires a context_object URI")
            uri = str(object_payload["uri"])
            metadata = dict(object_payload.get("metadata", {}) or {})
            if object_payload.get("schema_version") != "canonical_memory_v2":
                raise ValueError("canonical operation requires canonical_memory_v2 object schema")
            if operation.payload.get("schema_version") != "canonical_memory_v2":
                raise ValueError("canonical operation requires canonical_memory_v2 transaction schema")
            if (
                metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
                or operation.payload.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
            ):
                raise ValueError("canonical operation requires Identity V2")
            if "identity_alias_operations" in operation.payload:
                raise ValueError("Identity V2 canonical transactions cannot contain redirects")
            scope = dict(metadata.get("scope", {}) or {})
            subject_payload = scope.get("canonical_subject")
            subject_key = str(metadata.get("canonical_subject") or "")
            if not isinstance(subject_payload, dict) or not subject_key:
                raise ValueError("canonical operation requires an explicit canonical subject")
            if ScopeRef.from_dict(subject_payload).key != subject_key:
                raise ValueError("canonical operation subject payload does not match Identity V2")
            authority = dict(scope.get("authority", {}) or {})
            if not authority or bool(authority.get("inferred", False)):
                raise ValueError("canonical operation requires non-inferred assertion authority")
            object_tenant = str(object_payload.get("tenant_id") or "default")
            operation_tenant = str(operation.payload.get("tenant_id") or "default")
            object_owner = str(object_payload.get("owner_user_id") or operation.user_id)
            asserted_by = str(metadata.get("asserted_by") or operation.user_id)
            if (
                object_tenant != operation_tenant
                or object_owner != operation.user_id
                or asserted_by != operation.user_id
            ):
                raise ValueError("canonical operation tenant or owner does not match its transaction envelope")
            tenants.add(object_tenant)
            owners.add(object_owner)
            slot_ids.add(str(metadata.get("slot_id") or operation.payload.get("slot_id") or ""))
            scope_payloads.add(json.dumps(metadata.get("scope", {}), ensure_ascii=False, sort_keys=True))
            if not operation.evidence or any(
                not item.get("event_id") or not item.get("content_hash") for item in operation.evidence
            ):
                raise ValueError("canonical operation requires durable evidence references")
            self._validate_canonical_evidence(operation)
            expected = int(operation.payload.get("expected_revision", 0))
            try:
                current = self.source_store.read_object(uri)
                actual = int(dict(current.metadata or {}).get("revision", 0))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                actual = 0
            if actual != expected:
                raise RevisionConflictError(f"revision conflict for {uri}: expected {expected}, actual {actual}")
        if len(tenants) != 1 or len(slot_ids - {""}) != 1 or len(scope_payloads) != 1:
            raise ValueError("canonical transaction must preserve tenant, slot, and scope boundaries")

    def _validate_canonical_evidence(self, operation: ContextOperation) -> None:
        store = SessionArchiveStore(
            self.root,
            tenant_id=str(operation.payload.get("tenant_id") or "default"),
        )
        verified_sources: set[str] = set()
        operation_refs = {canonical_json(payload) for payload in operation.evidence}
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            metadata = dict(object_payload.get("metadata", {}) or {})
            for revision in metadata.get("revisions", []) or []:
                if not isinstance(revision, dict):
                    raise ValueError("canonical revision evidence payload must be an object")
                if int(revision.get("revision", 0)) != int(metadata.get("revision", 0)):
                    continue
                field_refs = dict(revision.get("field_evidence_refs", {}) or {})
                for field_name, refs in field_refs.items():
                    if not refs:
                        raise ValueError(f"canonical revision has no field evidence for {field_name}")
                    for ref in refs:
                        if canonical_json(ref) not in operation_refs:
                            raise ValueError(
                                f"canonical field evidence is missing from the transaction envelope: {field_name}"
                            )
        for payload in operation.evidence:
            source_uri = str(payload.get("source_uri") or "")
            if not source_uri:
                raise ValueError("canonical evidence requires a durable source_uri")
            if source_uri not in verified_sources:
                store.current_manifest(
                    source_uri,
                    tenant_id=str(operation.payload.get("tenant_id") or "default"),
                )
                verified_sources.add(source_uri)
            event_digest = str(payload.get("event_digest") or "")
            required = {
                "event_id",
                "event_digest",
                "event_schema_version",
                "tenant_id",
                "episode_id",
                "actor_id",
                "actor_kind",
                "actor_role",
                "actor_id_inferred",
                "actor_role_inferred",
                "subject_refs",
                "content_path",
                "occurred_at",
                "ingested_at",
                "sequence",
                "evidence_strength",
                "content_hash",
            }
            if any(name not in payload or payload[name] is None or payload[name] == "" for name in required):
                raise ValueError("canonical evidence reference is incomplete")
            event = store.read_event(
                source_uri,
                event_digest,
                tenant_id=str(operation.payload.get("tenant_id") or "default"),
            )
            if str(event.get("event_id")) != str(payload["event_id"]):
                raise ValueError("canonical evidence event ID does not match its immutable digest")
            if str(event.get("episode_id")) != str(payload["episode_id"]) or str(payload["episode_id"]) != str(
                operation.source_episode_id
            ):
                raise ValueError("canonical evidence event is not part of the source episode")
            if str(event.get("schema_version")) != str(payload["event_schema_version"]):
                raise ValueError("canonical evidence schema version mismatch")
            tenant_id = str(operation.payload.get("tenant_id") or "default")
            if str(event.get("tenant_id")) != str(payload["tenant_id"]) or str(payload["tenant_id"]) != tenant_id:
                raise ValueError("canonical evidence tenant mismatch")
            actor = dict(event.get("actor", {}) or {})
            for field_name, evidence_name in (
                ("id", "actor_id"),
                ("kind", "actor_kind"),
                ("role", "actor_role"),
                ("id_inferred", "actor_id_inferred"),
                ("role_inferred", "actor_role_inferred"),
            ):
                if actor.get(field_name) != payload[evidence_name]:
                    raise ValueError(f"canonical evidence actor mismatch: {evidence_name}")
            expected_subjects = tuple(canonical_json(item) for item in event.get("subjects", []) or [])
            if tuple(str(item) for item in payload.get("subject_refs", []) or []) != expected_subjects:
                raise ValueError("canonical evidence subject mismatch")
            content_path = str(payload["content_path"])
            if content_path != str(event.get("content_path") or ""):
                raise ValueError("canonical evidence content path mismatch")
            content = resolve_content_path(event.get("content"), content_path)
            text = content if isinstance(content, str) else canonical_json(content)
            if evidence_hash(text) != str(payload["content_hash"]):
                raise ValueError("canonical evidence content hash no longer matches the archive")
            if not self._same_evidence_time(event.get("occurred_at"), payload["occurred_at"]):
                raise ValueError("canonical evidence occurred_at mismatch")
            if not self._same_evidence_time(event.get("ingested_at"), payload["ingested_at"]):
                raise ValueError("canonical evidence ingested_at mismatch")
            if int(event.get("sequence", 0)) != int(payload["sequence"]):
                raise ValueError("canonical evidence sequence mismatch")
            inference = dict(event.get("inference", {}) or {})
            expected_strength = "INFERRED" if any(bool(value) for value in inference.values()) else "EXPLICIT"
            if str(payload["evidence_strength"]) != expected_strength:
                raise ValueError("canonical evidence strength mismatch")
            span_start = payload.get("span_start")
            span_end = payload.get("span_end")
            if (span_start is None) != (span_end is None):
                raise ValueError("canonical evidence span is incomplete")
            if span_start is None or span_end is None:
                continue
            start, end = int(span_start), int(span_end)
            if start < 0 or end <= start or end > len(text):
                raise ValueError("canonical evidence span is invalid")
            quoted_hash = payload.get("quoted_text_hash")
            quoted_text = text[start:end]
            if not quoted_hash or evidence_hash(quoted_text) != str(quoted_hash):
                raise ValueError("canonical evidence quote hash no longer matches the archive")
            if payload.get("quoted_text") != quoted_text:
                raise ValueError("canonical evidence quoted text no longer matches the archive")

    def _same_evidence_time(self, left: object, right: object) -> bool:
        from datetime import datetime, timezone

        try:
            left_time = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
            right_time = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
        except ValueError:
            return False
        if left_time.tzinfo is None:
            left_time = left_time.replace(tzinfo=timezone.utc)
        if right_time.tzinfo is None:
            right_time = right_time.replace(tzinfo=timezone.utc)
        return left_time.astimezone(timezone.utc) == right_time.astimezone(timezone.utc)

    def _validate_authoritative_batch(self, operations: list[ContextOperation]) -> None:
        slot_active: dict[str, str | None] = {}
        active_by_slot: dict[str, list[str]] = {}
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "slot":
                self._validate_existing_slot_invariant(str(payload.get("uri", "")))
                slot_active[str(metadata.get("slot_id", ""))] = (
                    str(metadata["active_claim_id"]) if metadata.get("active_claim_id") else None
                )
            elif (
                metadata.get("canonical_kind") == "claim"
                and metadata.get("transition_profile") == "AUTHORITATIVE_STATE"
                and metadata.get("state") == "ACTIVE"
            ):
                active_by_slot.setdefault(str(metadata.get("slot_id", "")), []).append(
                    str(metadata.get("claim_id", ""))
                )
        for slot_id, active_claims in active_by_slot.items():
            if len(active_claims) > 1:
                raise ValueError("authoritative slot cannot commit more than one ACTIVE claim")
            declared = slot_active.get(slot_id)
            if declared and active_claims and declared != active_claims[0]:
                raise ValueError("slot active_claim_id does not match active claim revision")

    def _validate_existing_slot_invariant(self, slot_uri: str) -> None:
        if not slot_uri:
            return
        try:
            slot = self.source_store.read_object(slot_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return
        metadata = dict(slot.metadata or {})
        claim_ids = [str(item) for item in metadata.get("claim_ids", []) or []]
        active: list[str] = []
        for claim_id in claim_ids:
            try:
                claim = self.source_store.read_object(f"{slot_uri}/claims/{claim_id}")
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            claim_metadata = dict(claim.metadata or {})
            if str(claim_metadata.get("state", "")) == "ACTIVE":
                active.append(str(claim_metadata.get("claim_id", claim_id)))
        if len(active) > 1:
            raise ValueError(f"canonical slot invariant violation: multiple ACTIVE claims for {slot_uri}")
        pointer = str(metadata.get("active_claim_id") or "")
        if pointer and active and pointer != active[0]:
            raise ValueError(f"canonical slot invariant violation: active_claim_id mismatch for {slot_uri}")

    def _apply_canonical_source(self, operation: ContextOperation) -> None:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical operation requires context_object")
        obj = ContextObject.from_dict(payload)
        self.source_store.write_object(obj, content=str(operation.payload.get("content", "")))
        metadata = dict(obj.metadata or {})
        relation_metadata = {
            "tenant_id": obj.tenant_id or "default",
            "owner_user_id": obj.owner_user_id,
            "canonical_transaction_id": operation.payload.get("transaction_id"),
            "canonical_idempotency_key": operation.payload.get("idempotency_key"),
            "source_revision": metadata.get("revision"),
            "commit_group_id": operation.payload.get("commit_group_id"),
        }
        if self.relation_store is not None:
            for relation in obj.relations:
                self.relation_store.add_relation(
                    ContextRelation(
                        source_uri=relation.source_uri,
                        relation_type=relation.relation_type,
                        target_uri=relation.target_uri,
                        weight=relation.weight,
                        metadata={**dict(relation.metadata or {}), **relation_metadata},
                    )
                )
        if self.relation_store is not None and metadata.get("canonical_kind") == "claim":
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            self._add_relation(obj.uri, "belongs_to_slot", slot_uri, relation_metadata)
            self._add_relation(slot_uri, "has_claim", obj.uri, relation_metadata)

    def _write_outbox_event(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        status: str = "committed",
        before_images: list[dict] | None = None,
    ) -> Path:
        path = self.root / "system" / "outbox" / f"{transaction_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        claim_revisions = []
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "claim":
                claim_revisions.append(
                    {
                        "uri": payload.get("uri"),
                        "claim_id": metadata.get("claim_id"),
                        "revision": metadata.get("revision"),
                    }
                )
        event: dict = {
            "event_type": "MemoryCommitted",
            "transaction_id": transaction_id,
            "idempotency_key": idempotency_key,
            "claim_revisions": claim_revisions,
            "operation_ids": [operation.operation_id for operation in operations],
            "operations": [operation.to_dict() for operation in operations],
            "status": status,
            "before_images": [self._before_image_payload(item) for item in (before_images or [])],
            "commit_group_id": next(
                (
                    str(operation.payload.get("commit_group_id"))
                    for operation in operations
                    if operation.payload.get("commit_group_id")
                ),
                "",
            ),
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            merged_claims = {
                str(item.get("uri")): item for item in existing.get("claim_revisions", []) or [] if item.get("uri")
            }
            for item in claim_revisions:
                current = merged_claims.get(str(item.get("uri")))
                if current is None or int(item.get("revision") or 0) >= int(current.get("revision") or 0):
                    merged_claims[str(item.get("uri"))] = item
            event["claim_revisions"] = list(merged_claims.values())
            event["operation_ids"] = list(
                dict.fromkeys(
                    [
                        *[str(item) for item in existing.get("operation_ids", []) or []],
                        *[operation.operation_id for operation in operations],
                    ]
                )
            )
            merged_operations = {
                str(item.get("operation_id")): item
                for item in existing.get("operations", []) or []
                if isinstance(item, dict) and item.get("operation_id")
            }
            for item in event["operations"]:
                if isinstance(item, dict) and item.get("operation_id"):
                    merged_operations[str(item["operation_id"])] = item
            event["operations"] = list(merged_operations.values())
            if not event["before_images"]:
                event["before_images"] = list(existing.get("before_images", []) or [])
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def _before_image_payload(self, snapshot: dict) -> dict:
        obj = snapshot.get("object")
        return {
            "uri": str(snapshot.get("uri", "")),
            "exists": bool(snapshot.get("exists")),
            "object": obj.to_dict() if isinstance(obj, ContextObject) else None,
            "content": str(snapshot.get("content", "")),
        }

    def _capture_canonical_state(self, operations: list[ContextOperation]) -> list[dict]:
        snapshots = []
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            uri = str(payload["uri"])
            try:
                obj = self.source_store.read_object(uri)
                exists = True
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                obj = None
                exists = False
            if obj is not None:
                try:
                    content = self.source_store.read_content(obj.layers.l2_uri or uri)
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    content = ""
            else:
                content = ""
            relations = (
                self.relation_store.relations_of(
                    uri,
                    tenant_id=str(payload.get("tenant_id") or "default"),
                    owner_user_id=payload.get("owner_user_id"),
                )
                if self.relation_store is not None
                else []
            )
            snapshots.append({"uri": uri, "exists": exists, "object": obj, "content": content, "relations": relations})
        return snapshots

    def _restore_canonical_state(self, snapshots: list[dict]) -> None:
        for snapshot in reversed(snapshots):
            uri = str(snapshot["uri"])
            if snapshot["exists"]:
                self.source_store.write_object(snapshot["object"], content=str(snapshot["content"]))
                if snapshot["content"] == "":
                    obj = snapshot["object"]
                    self.source_store.write_content(obj.layers.l2_uri or uri, "")
            else:
                delete = getattr(self.source_store, "delete_object", None)
                if not callable(delete):
                    raise RuntimeError("SourceStore must support delete_object for canonical rollback")
                delete(uri)
            if self.relation_store is None:
                continue
            original = list(snapshot["relations"])
            current = self.relation_store.relations_of(uri)
            for relation in current:
                if relation not in original:
                    self.relation_store.delete_relation(
                        relation.source_uri,
                        relation.relation_type,
                        relation.target_uri,
                    )
            for relation in original:
                self.relation_store.add_relation(relation)

    def _enqueue_outbox(
        self,
        transaction_id: str,
        slot_uri: str,
        outbox_path: Path,
        operations: list[ContextOperation],
    ) -> None:
        if self.queue_store is None:
            return
        try:
            self.queue_store.enqueue(
                QueueJob(
                    job_id=f"outbox_{transaction_id}",
                    queue_name="memory_projection",
                    action="project_memory_committed",
                    target_uri=slot_uri,
                    payload={
                        "transaction_id": transaction_id,
                        "outbox_path": str(outbox_path),
                        "operation_ids": [operation.operation_id for operation in operations],
                    },
                )
            )
        except Exception as exc:
            self.audit.record(
                operations[0].user_id,
                "canonical_memory_outbox_enqueue_failed",
                {"transaction_id": transaction_id, "error_type": type(exc).__name__},
            )

    def _transaction_marker(self, idempotency_key: str) -> Path:
        return self.root / "system" / "transactions" / f"{idempotency_key}.json"

    def _write_transaction_marker(self, path: Path, diff: ContextDiff) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(diff.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _diff_from_payload(self, payload: dict) -> ContextDiff:
        return ContextDiff(
            user_id=str(payload["user_id"]),
            operations=[ContextOperation.from_dict(item) for item in payload.get("operations", [])],
            pending_operations=[ContextOperation.from_dict(item) for item in payload.get("pending_operations", [])],
            rejected_operations=[ContextOperation.from_dict(item) for item in payload.get("rejected_operations", [])],
            diff_id=str(payload.get("diff_id", "")),
            created_at=str(payload.get("created_at", "")),
            schema_version=str(payload.get("schema_version", "context_diff_v1")),
        )

    def resume(self, user_id: str, operation: ContextOperation, phase: str) -> bool:
        """处理 resume 这一步。"""

        if phase in {"committed"}:
            if operation.payload.get("canonical_memory") is not True:
                self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return False
        if phase in {"started", "begin"}:
            diff = self.commit(user_id, [operation])
            return any(op.operation_id == operation.operation_id for op in diff.operations)
        if operation.payload.get("canonical_memory") is True:
            return self._resume_canonical(user_id, operation, phase)
        if phase == "source_written":
            self._apply_index(operation)
            self.redo.advance(operation, phase="index_written")
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        if phase == "index_written":
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        if phase == "audit_written":
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        if phase == "diff_written":
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        return False

    def _resume_canonical(self, user_id: str, operation: ContextOperation, phase: str) -> bool:
        if phase == "source_written":
            self.audit.record(user_id, "canonical_memory_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
        transaction_id = str(operation.payload.get("transaction_id", ""))
        idempotency_key = str(operation.payload.get("idempotency_key", ""))
        object_payload = operation.payload.get("context_object")
        slot_uri = operation.target_uri or transaction_id
        if isinstance(object_payload, dict):
            metadata = dict(object_payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "claim":
                slot_uri = str(object_payload.get("uri", slot_uri)).rsplit("/claims/", 1)[0]
        outbox_path = self._write_outbox_event(transaction_id, idempotency_key, [operation])
        self._enqueue_outbox(transaction_id, slot_uri, outbox_path, [operation])
        self._write_recovery_diff(user_id, operation)
        self.redo.advance(operation, phase="diff_written")
        self.redo.commit(operation.operation_id)
        return True

    def resume_canonical_batch(self, user_id: str, entries: list) -> list[str]:  # noqa: ANN001
        """从事务日志记录的阶段继续完成整批写入。"""

        operations = [entry.operation for entry in entries]
        if not operations:
            return []
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1:
            raise ValueError("canonical recovery requires one complete transaction")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        outbox_path = self.root / "system" / "outbox" / f"{transaction_id}.json"
        prepared = json.loads(outbox_path.read_text(encoding="utf-8"))
        expected_operation_ids = [str(item) for item in prepared.get("operation_ids", []) or []]
        by_id = {operation.operation_id: operation for operation in operations}
        for payload in prepared.get("operations", []) or []:
            operation = ContextOperation.from_dict(payload)
            by_id.setdefault(operation.operation_id, operation)
        if set(expected_operation_ids) != set(by_id):
            raise RuntimeError("canonical recovery outbox is missing transaction operations")
        ordered = [by_id[operation_id] for operation_id in expected_operation_ids]
        slot_uri = next(
            (
                str(payload.get("uri"))
                for operation in ordered
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        with self.path_lock.acquire(f"canonical:{slot_uri}"):
            marker = self._transaction_marker(idempotency_key)
            if marker.exists():
                self._finalize_canonical_outbox(
                    transaction_id,
                    idempotency_key,
                    ordered,
                    slot_uri=slot_uri,
                )
                for operation in ordered:
                    self.redo.commit(operation.operation_id)
                return [operation.operation_id for operation in ordered]
            for operation in ordered:
                payload = operation.payload.get("context_object")
                if not isinstance(payload, dict):
                    raise ValueError("canonical recovery requires context_object")
                uri = str(payload["uri"])
                expected = int(operation.payload.get("expected_revision", 0))
                desired = int(dict(payload.get("metadata", {}) or {}).get("revision", 0))
                try:
                    actual = int(dict(self.source_store.read_object(uri).metadata or {}).get("revision", 0))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    actual = 0
                if actual == expected:
                    self._apply_canonical_source(operation)
                elif actual != desired:
                    raise RevisionConflictError(
                        f"canonical recovery conflict for {uri}: expected {expected} or {desired}, actual {actual}"
                    )
                self.audit.record(user_id, "canonical_memory_operation_applied_during_recovery", operation.to_dict())
                operation.status = OperationStatus.COMMITTED
            diff = ContextDiff(user_id=user_id, operations=ordered, diff_id=f"diff_{transaction_id}")
            self.diff_writer.write(diff)
            self._write_transaction_marker(marker, diff)
            self.audit.record(
                user_id,
                "canonical_memory_transaction_recovered",
                {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in ordered]},
            )
            self._finalize_canonical_outbox(
                transaction_id,
                idempotency_key,
                ordered,
                slot_uri=slot_uri,
            )
            for operation in ordered:
                self.redo.commit(operation.operation_id)
            return [operation.operation_id for operation in ordered]

    def recover_pending_canonical(self, user_id: str) -> list[str]:
        """恢复卡在准备阶段或源数据已写入阶段的记忆事务。"""

        grouped: dict[str, list] = {}
        for entry in self.redo.pending_entries():
            if entry.operation.payload.get("canonical_memory") is not True:
                continue
            transaction_id = str(entry.operation.payload.get("transaction_id", ""))
            grouped.setdefault(transaction_id, []).append(entry)
        recovered = []
        for entries in grouped.values():
            recovered.extend(self.resume_canonical_batch(user_id, entries))
        return recovered

    def _write_recovery_diff(self, user_id: str, operation: ContextOperation) -> None:
        operation.status = OperationStatus.COMMITTED
        self.diff_writer.write(
            ContextDiff(user_id=user_id, operations=[operation], diff_id=f"diff_{operation.operation_id}")
        )

    def _operation_marker(self, operation_id: str) -> Path:
        return self.root / "system" / "operations" / f"{operation_id}.json"

    def _write_operation_marker(self, operation: ContextOperation) -> None:
        if operation.payload.get("canonical_memory") is True:
            return
        path = self._operation_marker(operation.operation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "context_type": operation.context_type.value,
            "target_uri": operation.target_uri,
            "commit_group_id": operation.payload.get("commit_group_id"),
            "commit_consumer": operation.payload.get("commit_consumer"),
            "status": "committed",
        }
        if path.exists():
            self._validate_operation_marker(path, operation)
            return
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            self._validate_operation_marker(path, operation)
        finally:
            tmp.unlink(missing_ok=True)

    def _validate_operation_marker(self, path: Path, operation: ContextOperation) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "context_type": operation.context_type.value,
            "target_uri": operation.target_uri,
            "commit_group_id": operation.payload.get("commit_group_id"),
            "commit_consumer": operation.payload.get("commit_consumer"),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("operation idempotency marker conflicts with the requested operation")

    def _coalesce_non_policy_operations(self, operations: list[ContextOperation]) -> list[ContextOperation]:
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        policy_ops = [operation for operation in operations if operation.action in policy_actions]
        other_ops = [operation for operation in operations if operation.action not in policy_actions]
        return [*self.coalescer.coalesce(other_ops), *policy_ops]

    def _apply_source(self, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_source(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                content = str(operation.payload.get("content", ""))
                self.source_store.write_object(obj, content=content)
                if content:
                    LayerRefresher(self.source_store).refresh(obj, content)
                    operation.payload["context_object"] = obj.to_dict()
                self._apply_relations(obj, operation)
            return
        if (
            operation.action
            in {
                OperationAction.REWARD,
                OperationAction.PENALIZE,
                OperationAction.COOLDOWN,
                OperationAction.SUPPRESS,
                OperationAction.DISABLE,
            }
            and operation.target_uri
        ):
            if operation.context_type == ContextType.ACTION_POLICY:
                policy = self._read_action_policy(operation.target_uri)
                policy = self._apply_action_policy_mutation(policy, operation)
                self._write_action_policy(policy)
            elif operation.action == OperationAction.DISABLE:
                self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return
        if operation.action == OperationAction.COMPRESS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            LayerRefresher(self.source_store).refresh(
                obj, content, bullets=[operation.payload.get("reason", "compressed")]
            )
            obj.lifecycle_state = LifecycleState.COLD
            obj.metadata = {
                **obj.metadata,
                "compressed_at": utc_now(),
                "compression_reason": operation.payload.get("reason", ""),
            }
            self.source_store.write_object(obj)
            return
        if operation.action == OperationAction.REFRESH_LAYERS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            LayerRefresher(self.source_store).refresh(obj, content)
            return
        if operation.action == OperationAction.ARCHIVE and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            obj.lifecycle_state = LifecycleState.ARCHIVED
            obj.metadata = {
                **obj.metadata,
                "archived_at": utc_now(),
                "archive_reason": operation.payload.get("reason", ""),
            }
            content = self._read_content_or_empty(operation.target_uri)
            self.source_store.write_object(obj, content=content)
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return

    def _apply_index(self, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_index(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                self.index_store.upsert_index(obj, content=str(operation.payload.get("content", "")))
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.index_store.delete_index(operation.target_uri)
            return
        if operation.target_uri and operation.action in {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.ARCHIVE,
            OperationAction.REINDEX,
        }:
            if operation.action == OperationAction.DISABLE and operation.context_type != ContextType.ACTION_POLICY:
                self.index_store.delete_index(operation.target_uri)
                return
            obj = self.source_store.read_object(operation.target_uri)
            self.index_store.upsert_index(obj, content=self._read_content_or_empty(operation.target_uri))

    def _apply_action_policy_mutation(self, policy: ActionPolicy, operation: ContextOperation) -> ActionPolicy:
        if operation.action == OperationAction.REWARD:
            return self.action_policy_updater.reward(
                policy, RewardSignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.PENALIZE:
            return self.action_policy_updater.penalize(
                policy, PenaltySignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.COOLDOWN:
            return self.action_policy_updater.cooldown(
                policy, operation.payload.get("cooldown_until"), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.SUPPRESS:
            return self.action_policy_updater.suppress(policy, operation_id=operation.operation_id)
        if operation.action == OperationAction.DISABLE:
            return self.action_policy_updater.disable_auto_execute(policy, operation_id=operation.operation_id)
        return policy

    def _apply_supersede_source(self, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        object_payload = operation.payload.get("context_object")
        if not isinstance(object_payload, dict):
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        old_content = self._read_content_or_empty(operation.target_uri)
        new_obj = ContextObject.from_dict(object_payload)
        new_obj.lifecycle_state = LifecycleState.ACTIVE
        superseded_at = utc_now()
        reason = str(operation.payload.get("reason") or operation.payload.get("supersede_reason") or "")
        old_obj.lifecycle_state = LifecycleState.OBSOLETE
        old_obj.metadata = {
            **old_obj.metadata,
            "superseded_at": superseded_at,
            "superseded_by": new_obj.uri,
            "supersede_reason": reason,
        }
        new_obj.metadata = {
            **new_obj.metadata,
            "supersedes": old_obj.uri,
            "superseded_at": superseded_at,
            "supersede_reason": reason,
        }
        self.source_store.write_object(old_obj, content=old_content)
        self.source_store.write_object(new_obj, content=str(operation.payload.get("content", "")))
        self._apply_relations(new_obj, operation)
        self._add_supersede_relations(old_obj, new_obj)

    def _apply_supersede_index(self, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        self.index_store.upsert_index(old_obj, content=self._read_content_or_empty(operation.target_uri))
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            new_uri = object_payload.get("uri")
            if not new_uri:
                return
            new_obj = self.source_store.read_object(str(new_uri))
            self.index_store.upsert_index(new_obj, content=str(operation.payload.get("content", "")))

    def _add_supersede_relations(self, old_obj: ContextObject, new_obj: ContextObject) -> None:
        metadata = {
            "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
            "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
        }
        self._add_relation(new_obj.uri, "supersedes", old_obj.uri, metadata)
        self._add_relation(old_obj.uri, "superseded_by", new_obj.uri, metadata)

    def _read_action_policy(self, uri: str) -> ActionPolicy:
        obj = self.source_store.read_object(uri)
        data = dict(obj.metadata)
        if not data:
            content = self._read_content_or_empty(uri)
            data = json.loads(content) if content else {}
        return ActionPolicy(**data)

    def _write_action_policy(self, policy: ActionPolicy) -> None:
        obj = policy.to_context_object()
        self.source_store.write_object(
            obj,
            content=json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
        )
        self._apply_relations(
            obj,
            ContextOperation(
                user_id=policy.user_id,
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.UPDATE,
                target_uri=policy.uri,
                payload={},
            ),
        )

    def _apply_relations(self, obj: ContextObject, operation: ContextOperation) -> None:
        if self.relation_store is None:
            return
        metadata = dict(obj.metadata)
        relation_metadata = {"tenant_id": obj.tenant_id or "default", "owner_user_id": obj.owner_user_id}
        if obj.context_type == ContextType.ACTION_POLICY:
            self._add_relation(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("required_resource_uris", []) or []:
                self._add_relation(obj.uri, "requires_resource", str(uri), relation_metadata)
            for uri in metadata.get("required_skill_uris", []) or []:
                self._add_relation(obj.uri, "requires_skill", str(uri), relation_metadata)
            for uri in metadata.get("supported_behavior_pattern_uris", []) or []:
                self._add_relation(obj.uri, "supported_by", str(uri), relation_metadata)
            for uri in metadata.get("constrained_by_memory_uris", []) or []:
                self._add_relation(obj.uri, "constrained_by", str(uri), relation_metadata)
        elif obj.context_type in {ContextType.BEHAVIOR_PATTERN, ContextType.BEHAVIOR_CLUSTER}:
            self._add_relation(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("case_refs", []) or []:
                self._add_relation(obj.uri, "aggregated_from", str(uri), relation_metadata)
            for uri in metadata.get("related_policy_uris", []) or metadata.get("policy_uris", []) or []:
                self._add_relation(str(uri), "supported_by", obj.uri, relation_metadata)
        elif obj.context_type == ContextType.MEMORY:
            for policy_uri in metadata.get("constrains_policy_uris", []) or []:
                self._add_relation(str(policy_uri), "constrained_by", obj.uri, relation_metadata)
            for behavior_uri in metadata.get("supporting_behavior_uris", []) or []:
                self._add_relation(obj.uri, "evidence_for", str(behavior_uri), relation_metadata)
        for relation in obj.relations:
            if self.relation_store is not None:
                self.relation_store.add_relation(relation)

    def _add_relation(self, source_uri: str, relation_type: str, target_uri: str, metadata: dict) -> None:
        if self.relation_store is None or not target_uri:
            return
        self.relation_store.add_relation(
            ContextRelation(
                source_uri=source_uri,
                relation_type=relation_type,
                target_uri=target_uri,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    def _read_content_or_empty(self, uri: str) -> str:
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
