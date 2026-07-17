from __future__ import annotations

import json

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)
from memoryos.contextdb.store.source_store import QueueStore
from memoryos.memory.canonical.current_head import publish_current_head_sets
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


def _scope(*refs, tenant_id: str = "t1", principals=()):  # noqa: ANN001, ANN202
    scope_refs = [
        {
            "namespace": "memoryos",
            "kind": kind,
            "id": identifier,
            "parent_id": None,
            "attributes": {},
            "confidence": 1.0,
            "source": "explicit",
            "inferred": False,
        }
        for kind, identifier in refs
    ]
    return {
        "canonical_subject": scope_refs[0],
        "applicability": {"all_of": scope_refs},
        "visibility": {
            "tenant_id": tenant_id,
            "allowed_principal_ids": list(principals),
            "allowed_service_ids": [],
            "private": bool(principals),
        },
        "authority": {
            "principal_ids": list(principals or ("u1",)),
            "service_ids": [],
            "inferred": False,
        },
        "origin_refs": [],
    }


def _write_committed_canonical_fixture(
    source: FileSystemSourceStore,
    entries: list[tuple[ContextObject, str]],
    *,
    key: str,
    action: OperationAction = OperationAction.ADD,
    queue_store: QueueStore | None = None,
    finalize_outbox: bool = False,
) -> None:
    """Persist canonical fixtures behind an integrity-valid transaction marker."""

    transaction_id = f"tx-{key}"
    idempotency_key = f"idem-{key}"
    commit_group_id = f"fixture-group-{key}"
    operations: list[ContextOperation] = []
    for index, (obj, content) in enumerate(entries):
        owner_user_id = obj.owner_user_id
        assert isinstance(owner_user_id, str)
        obj.metadata = {
            **dict(obj.metadata or {}),
            "canonical_transaction_id": transaction_id,
            "canonical_idempotency_key": idempotency_key,
        }
        operations.append(
            ContextOperation(
                operation_id=f"op-{key}-{index}",
                user_id=owner_user_id,
                context_type=obj.context_type,
                action=action,
                target_uri=obj.uri,
                status=OperationStatus.COMMITTED,
                payload={
                    "canonical_memory": True,
                    "transaction_id": transaction_id,
                    "idempotency_key": idempotency_key,
                    "commit_group_id": commit_group_id,
                    "tenant_id": obj.tenant_id,
                    "expected_revision": 0
                    if action == OperationAction.ADD
                    else max(0, int(obj.metadata.get("revision", 1)) - 1),
                    "context_object": obj.to_dict(),
                    "content": content,
                },
            )
        )
    assert operations
    assert len({operation.user_id for operation in operations}) == 1
    fixture_relations = InMemoryRelationStore()
    committer = OperationCommitter(
        source,
        InMemoryIndexStore(),
        str(source.root),
        relation_store=fixture_relations,
        queue_store=queue_store,
        tenant_id=source.tenant_id,
    )
    before_images = committer._capture_canonical_state(operations)
    before_by_uri = {str(item["uri"]): item.get("object") for item in before_images}
    relation_manifests = {
        operation.operation_id: committer._build_canonical_relation_manifest(
            operation,
            before_by_uri.get(str(operation.target_uri or "")),
        )
        for operation in operations
    }
    diff = ContextDiff(
        user_id=operations[0].user_id,
        operations=operations,
        diff_id=f"diff-{transaction_id}",
    )
    planning_digest = committer._ensure_canonical_planning_digest(operations)
    for operation in operations:
        operation.payload["planning_digest"] = planning_digest
    committer._write_outbox_event(
        transaction_id,
        idempotency_key,
        operations,
        status="prepared",
        before_images=before_images,
        relation_manifests=relation_manifests,
    )
    for obj, content in entries:
        source.write_object(obj, content=content)
    committer._write_outbox_event(
        transaction_id,
        idempotency_key,
        operations,
        status="source_committed",
        before_images=before_images,
        relation_manifests=relation_manifests,
    )
    marker = committer._transaction_marker(idempotency_key)
    committer._write_transaction_marker(
        marker,
        diff,
        operations,
        relation_manifests=relation_manifests,
    )
    committer._validate_transaction_marker(marker, operations)
    publish_current_head_sets(
        committer.artifact_root,
        marker,
        json.loads(marker.read_text(encoding="utf-8")),
    )
    if finalize_outbox:
        if queue_store is None:
            raise ValueError("finalized canonical fixture requires a QueueStore")
        committer._finalize_canonical_outbox(transaction_id, idempotency_key, operations)
