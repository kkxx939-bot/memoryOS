"""Implementation component for CommitOutbox.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.queue_store import QueueJob
from memoryos.core.durable_io import atomic_write_json
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    assert_transition,
    build_outbox,
    planned_effect_manifest,
    projection_workspace_id,
    validate_outbox,
)
from memoryos.operations.commit.planning_proof import (
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    load_transaction_receipt,
)
from memoryos.operations.model.context_operation import ContextOperation


class CommitOutbox:
    """Own the CommitOutbox responsibility of a commit."""

    @staticmethod
    def _finalize_canonical_outbox(
        committer,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        slot_uri: str | None = None,
    ) -> Path:
        require_safe_path_segment(idempotency_key, "canonical idempotency_key")
        receipt_file = committer._transaction_marker(idempotency_key)
        committer._reject_control_symlink(receipt_file, "canonical transaction receipt")
        if not receipt_file.exists():
            raise ValueError("canonical outbox cannot commit before its immutable receipt")
        receipt = load_transaction_receipt(receipt_file)
        try:
            receipt_reference = str(receipt_file.resolve().relative_to(committer.artifact_root.resolve()))
        except ValueError as exc:
            raise ValueError("canonical receipt is outside the tenant artifact root") from exc
        outbox_path = committer._outbox_path(transaction_id)
        committer._reject_control_symlink(outbox_path, "canonical outbox")
        outbox_complete = False
        if outbox_path.exists():
            try:
                existing = validate_outbox(
                    json.loads(outbox_path.read_text(encoding="utf-8")),
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    tenant_id=committer.tenant_id,
                    user_id=operations[0].user_id,
                    operations=operations,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                raise ValueError("canonical committed outbox is unreadable") from exc
            if existing.get("status") == "committed":
                existing_operations = [
                    ContextOperation.from_dict(item)
                    for item in existing.get("operations", []) or []
                    if isinstance(item, dict)
                ]
                if operations:
                    committer._validate_and_bind_operations(operations[0].user_id, existing_operations)
                if committer._canonical_transaction_request_fingerprint(
                    existing_operations
                ) != committer._canonical_transaction_request_fingerprint(
                    operations
                ) or committer._canonical_transaction_effect_fingerprint(
                    existing_operations
                ) != committer._canonical_transaction_effect_fingerprint(operations):
                    raise ValueError("canonical committed outbox conflicts with its transaction marker")
                outbox_complete = True
        if not outbox_complete:
            outbox_path = committer._write_outbox_event(
                transaction_id,
                idempotency_key,
                operations,
                status="committed",
                receipt_path=receipt_reference,
                receipt_digest=str(receipt["receipt_digest"]),
            )
        # This hook proves the immutable committed outbox is durable while the
        # projection queue is still untouched.  It is intentionally emitted
        # for idempotent replay of an already-committed outbox as well.
        committer._notify("after_committed_outbox", transaction_id)
        resolved_slot = slot_uri or next(
            (
                str(payload.get("uri"))
                for operation in operations
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        committer._enqueue_outbox(transaction_id, resolved_slot, outbox_path, operations)
        committer._notify("after_projection_enqueue", transaction_id)
        return outbox_path

    @staticmethod
    def _write_outbox_event(
        committer,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        status: str = "committed",
        before_images: list[dict] | None = None,
        relation_manifests: dict[str, dict] | None = None,
        receipt_path: str = "",
        receipt_digest: str = "",
    ) -> Path:
        require_safe_path_segment(idempotency_key, "canonical idempotency_key")
        path = committer._outbox_path(transaction_id)
        committer._reject_control_symlink(path, "canonical outbox")
        if not operations:
            raise ValueError("canonical outbox requires transaction operations")
        committer._ensure_canonical_planning_digest(operations)
        committer._validate_and_bind_operations(operations[0].user_id, operations)
        claim_revisions: list[dict] = []
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
        existing: dict | None = None
        if path.exists():
            try:
                existing_payload = json.loads(path.read_text(encoding="utf-8"))
                existing = validate_outbox(
                    existing_payload,
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    tenant_id=committer.tenant_id,
                    user_id=operations[0].user_id,
                    operations=operations,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                raise ValueError("canonical outbox is corrupt or crosses its transaction boundary") from exc
            assert_transition(str(existing["status"]), status)
        before_payloads = (
            [committer._before_image_payload(item) for item in before_images]
            if before_images is not None
            else list((existing or {}).get("before_images", []) or [])
        )
        if relation_manifests is not None:
            effects = [
                planned_effect_manifest(operation, relation_manifests.get(operation.operation_id))
                for operation in operations
            ]
        elif existing is not None:
            effects = list(existing.get("effect_manifests", []) or [])
        else:
            effects = [planned_effect_manifest(operation, None) for operation in operations]
        event = build_outbox(
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            tenant_id=committer.tenant_id,
            user_id=operations[0].user_id,
            operations=operations,
            status=status,
            before_images=before_payloads,
            effect_manifests=effects,
            claim_revisions=claim_revisions,
            commit_group_id=next(
                (
                    str(operation.payload.get("commit_group_id"))
                    for operation in operations
                    if operation.payload.get("commit_group_id")
                ),
                "",
            ),
            receipt_path=receipt_path,
            receipt_digest=receipt_digest,
        )
        # The outbox is a durable transaction boundary, not merely an
        # internal serialization detail.  Re-validate the fully assembled
        # envelope before publication so a builder regression cannot persist
        # a Claim projection set, prepared intent, or receipt binding that is
        # detached from the immutable operation set.
        try:
            event = validate_outbox(
                event,
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                tenant_id=committer.tenant_id,
                user_id=operations[0].user_id,
                operations=operations,
                allowed_statuses={status},
            )
        except OutboxIntegrityError as exc:
            raise ValueError("canonical outbox failed pre-publication validation") from exc
        try:
            if status == "prepared":
                immutable_intent = committer.planning_proofs.ensure_canonical_intent(
                    event,
                    operations=operations,
                )
            else:
                immutable_intent = committer.planning_proofs.load_canonical_intent(
                    transaction_id,
                    operations=operations,
                    prepared_intent_digest=str(event["prepared_intent_digest"]),
                )
        except PlanningProofIntegrityError as exc:
            raise ValueError("canonical outbox transition is detached from its immutable prepared intent") from exc
        if immutable_intent["prepared_intent_digest"] != event["prepared_intent_digest"]:
            raise ValueError("canonical outbox prepared intent digest changed across transition")
        committer._reject_control_symlink(path, "canonical outbox")
        atomic_write_json(path, event, artifact_root=committer.artifact_root)
        return path

    @staticmethod
    def _before_image_payload(committer, snapshot: dict) -> dict:
        obj = snapshot.get("object")
        relations = sorted(
            (
                committer._relation_effect_spec(relation)
                for relation in snapshot.get("relations", []) or []
                if isinstance(relation, ContextRelation)
            ),
            key=canonical_json,
        )
        return {
            "uri": str(snapshot.get("uri", "")),
            "exists": bool(snapshot.get("exists")),
            "object": obj.to_dict() if isinstance(obj, ContextObject) else None,
            "content": str(snapshot.get("content", "")),
            "relations": relations,
            "relations_digest": canonical_digest(relations),
        }

    @staticmethod
    def _capture_canonical_state(committer, operations: list[ContextOperation]) -> list[dict]:
        snapshots = []
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            uri = str(payload["uri"])
            try:
                committed = committer._read_committed_canonical(uri)
                obj = committed.object
                exists = True
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                obj = None
                exists = False
            if obj is not None:
                content = committer._committed_canonical_content(committed)
            else:
                content = ""
            relations = list(committer._committed_canonical_relations(committed)) if obj is not None else []
            snapshots.append({"uri": uri, "exists": exists, "object": obj, "content": content, "relations": relations})
        return snapshots

    @staticmethod
    def _restore_canonical_state(committer, snapshots: list[dict]) -> None:
        for snapshot in reversed(snapshots):
            uri = str(snapshot["uri"])
            if snapshot["exists"]:
                committer.source_store.write_object(snapshot["object"], content=str(snapshot["content"]))
                if snapshot["content"] == "":
                    obj = snapshot["object"]
                    committer.source_store.write_content(obj.layers.l2_uri or uri, "")
            else:
                delete = getattr(committer.source_store, "delete_object", None)
                if not callable(delete):
                    raise RuntimeError("SourceStore must support delete_object for canonical rollback")
                delete(uri)
            if committer.relation_store is None:
                continue
            original = list(snapshot["relations"])
            tenant_id = str(snapshot["object"].tenant_id or "default") if snapshot["exists"] else committer.tenant_id
            current = committer.relation_store.relations_of(uri, tenant_id=tenant_id)
            for relation in current:
                if relation not in original:
                    committer.relation_store.delete_relation(
                        relation.source_uri,
                        relation.relation_type,
                        relation.target_uri,
                        tenant_id=tenant_id,
                    )
            for relation in original:
                committer.relation_store.add_relation(relation)

    @staticmethod
    def _enqueue_outbox(
        committer,
        transaction_id: str,
        slot_uri: str,
        outbox_path: Path,
        operations: list[ContextOperation],
    ) -> None:
        transaction_id = require_safe_path_segment(transaction_id, "canonical transaction_id")
        if committer.queue_store is None:
            return
        try:
            committer.queue_store.enqueue(
                QueueJob(
                    job_id=f"outbox_{transaction_id}",
                    queue_name="memory_projection",
                    action="project_memory_committed",
                    target_uri=slot_uri,
                    payload={
                        "transaction_id": transaction_id,
                        "outbox_path": str(outbox_path),
                        "operation_ids": [operation.operation_id for operation in operations],
                        "tenant_id": committer.tenant_id,
                        "owner_user_id": operations[0].user_id,
                        "workspace_id": projection_workspace_id(operations),
                    },
                )
            )
        except Exception as exc:
            committer.audit.record(
                operations[0].user_id,
                "canonical_memory_outbox_enqueue_failed",
                {"transaction_id": transaction_id, "error_type": type(exc).__name__},
            )
