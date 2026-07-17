from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryRelationStore,
)
from memoryos.memory.canonical.current_head import publish_current_head_sets
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.visibility import (
    CommittedStateIntegrityError,
    read_committed_canonical,
    relation_is_committed,
)
from memoryos.memory.integration.context_overlay import CanonicalMemoryContextOverlay
from memoryos.operations.commit import effect_marker as effect_marker_module
from memoryos.operations.commit.effect_marker import (
    atomic_write_json,
    build_marker,
    object_effect_from_store,
    relation_effects_from_manifest,
    validate_marker,
)
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    build_outbox,
    planned_effect_manifest,
    validate_outbox,
)
from memoryos.operations.commit.receipt import ReceiptIntegrityError, build_transaction_receipt
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction

URI = "memoryos://user/u1/memories/canonical/slots/s1/claims/c1"


def _source_store(root: Path, *, tenant_id: str = "default") -> FileSystemSourceStore:
    return FileSystemSourceStore(
        root,
        tenant_id=tenant_id,
        domain_classifier=CanonicalMemoryContextOverlay(),
    )


def _claim(*, tenant_id: str, title: str = "current") -> ContextObject:
    return ContextObject(
        uri=URI,
        context_type=ContextType.MEMORY,
        title=title,
        owner_user_id="u1",
        tenant_id=tenant_id,
        metadata={
            "canonical_kind": "claim",
            "canonical_idempotency_key": "shared-proof",
            "canonical_transaction_id": "shared-transaction",
        },
    )


def test_receipt_rejects_missing_or_crossed_commit_group_boundary() -> None:
    obj = _claim(tenant_id="t1")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        operation_id="commit-group-proof-operation",
        payload={
            "canonical_memory": True,
            "transaction_id": "commit-group-proof-transaction",
            "idempotency_key": "commit-group-proof",
            "tenant_id": "t1",
            "commit_group_id": "operation-group",
            "planning_digest": canonical_digest(["commit-group-proof"]),
            "context_object": obj.to_dict(),
            "content": "proof",
        },
    )
    diff = {
        "user_id": "u1",
        "operations": [operation.to_dict()],
        "pending_operations": [],
        "rejected_operations": [],
        "diff_id": "commit-group-proof-diff",
    }
    common: dict[str, Any] = {
        "transaction_id": "commit-group-proof-transaction",
        "idempotency_key": "commit-group-proof",
        "tenant_id": "t1",
        "user_id": "u1",
        "operations": [operation],
        "diff": diff,
        "planning_digest": str(operation.payload["planning_digest"]),
        "prepared_intent_digest": canonical_digest(["commit-group-intent"]),
    }

    with pytest.raises(ReceiptIntegrityError, match="commit group"):
        build_transaction_receipt(commit_group_id="", **common)
    with pytest.raises(ReceiptIntegrityError, match="commit group"):
        build_transaction_receipt(commit_group_id="different-group", **common)


@pytest.mark.parametrize("outbox_group", ["", "different-group"])
def test_prepared_outbox_rejects_missing_or_crossed_commit_group_boundary(
    outbox_group: str,
) -> None:
    obj = _claim(tenant_id="t1")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        operation_id="outbox-commit-group-operation",
        payload={
            "canonical_memory": True,
            "transaction_id": "outbox-commit-group-transaction",
            "idempotency_key": "outbox-commit-group",
            "tenant_id": "t1",
            "commit_group_id": "operation-group",
            "context_object": obj.to_dict(),
            "content": "proof",
        },
    )
    before_images = [
        {
            "uri": obj.uri,
            "exists": False,
            "object": None,
            "content": "",
            "relations": [],
            "relations_digest": canonical_digest([]),
        }
    ]
    with pytest.raises(OutboxIntegrityError, match="commit group"):
        payload = build_outbox(
            transaction_id="outbox-commit-group-transaction",
            idempotency_key="outbox-commit-group",
            tenant_id="t1",
            user_id="u1",
            operations=[operation],
            status="prepared",
            before_images=before_images,
            effect_manifests=[planned_effect_manifest(operation, None)],
            claim_revisions=[],
            commit_group_id=outbox_group,
        )
        validate_outbox(payload)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact_root(root: Path, tenant_id: str) -> Path:
    return root if tenant_id == "default" else root / "tenants" / tenant_id


def _write_proof(
    source: FileSystemSourceStore,
    obj: ContextObject,
    *,
    relation: ContextRelation | None = None,
) -> Path:
    relation_effects = relation_effects_from_manifest(
        {"expected": [relation.to_dict()], "remove": []} if relation is not None else None
    )
    if relation is not None and relation not in obj.relations:
        obj.relations.append(relation)
        try:
            content = source.read_content(obj.layers.l2_uri or obj.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            content = ""
        source.write_object(obj, content=content)
    marker = (
        _artifact_root(Path(source.root), source.tenant_id)
        / "system"
        / "transactions"
        / "shared-proof.json"
    )
    try:
        content = source.read_content(obj.layers.l2_uri or obj.uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        content = ""
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        target_uri=obj.uri,
        operation_id="proof-operation",
        payload={
            "canonical_memory": True,
            "transaction_id": "shared-transaction",
            "idempotency_key": "shared-proof",
            "tenant_id": source.tenant_id,
            "commit_group_id": "visibility-proof-group",
            "planning_digest": canonical_digest(["visibility-proof", source.tenant_id]),
            "expected_revision": 0,
            "context_object": obj.to_dict(),
            "content": content,
        },
    )
    diff = {
        "user_id": "u1",
        "operations": [operation.to_dict()],
        "pending_operations": [],
        "rejected_operations": [],
        "diff_id": "proof-diff",
    }
    receipt = build_transaction_receipt(
        transaction_id="shared-transaction",
        idempotency_key="shared-proof",
        tenant_id=source.tenant_id,
        user_id="u1",
        commit_group_id="visibility-proof-group",
        operations=[operation],
        diff=diff,
        planning_digest=str(operation.payload["planning_digest"]),
        prepared_intent_digest=canonical_digest(["visibility-intent", source.tenant_id]),
        relation_effects=relation_effects,
    )
    artifact_root = _artifact_root(Path(source.root), source.tenant_id)
    atomic_write_json(marker, receipt, artifact_root=artifact_root)
    publish_current_head_sets(artifact_root, marker, receipt)
    return marker


def _write_prepared_outbox(
    source: FileSystemSourceStore,
    current: ContextObject,
    before: ContextObject,
) -> None:
    source.write_object(before, content="previous content")
    _write_proof(source, before)
    source.write_object(current, content="current content")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.UPDATE,
        target_uri=current.uri,
        operation_id="proof-operation",
        payload={
            "transaction_id": "shared-transaction",
            "idempotency_key": "shared-proof",
            "tenant_id": source.tenant_id,
            "commit_group_id": "visibility-proof-group",
            "context_object": current.to_dict(),
            "content": "current content",
        },
    )
    event = build_outbox(
        transaction_id="shared-transaction",
        idempotency_key="shared-proof",
        tenant_id=source.tenant_id,
        user_id="u1",
        operations=[operation],
        status="prepared",
        before_images=[
            {
                "uri": before.uri,
                "exists": True,
                "object": before.to_dict(),
                "content": "previous content",
                "relations": [],
                "relations_digest": canonical_digest([]),
            }
        ],
        effect_manifests=[planned_effect_manifest(operation, {})],
        claim_revisions=[],
        commit_group_id="visibility-proof-group",
    )
    atomic_write_json(
        _artifact_root(Path(source.root), source.tenant_id)
        / "system"
        / "outbox"
        / "shared-transaction.json",
        event,
        artifact_root=_artifact_root(Path(source.root), source.tenant_id),
    )


def test_default_tenant_keeps_root_level_commit_proof_compatibility(tmp_path: Path) -> None:
    source = _source_store(tmp_path)
    obj = _claim(tenant_id="default")
    source.write_object(obj)
    _write_proof(source, obj)

    committed = read_committed_canonical(source, URI)

    assert committed.object.title == "current"
    assert committed.from_before_image is False


def test_nondefault_tenant_reads_only_its_scoped_commit_proof(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    obj = _claim(tenant_id="tenant-a")
    source.write_object(obj)
    _write_proof(source, obj)

    assert read_committed_canonical(source, URI).object.title == "current"


def test_other_tenant_or_legacy_root_marker_cannot_prove_nondefault_object(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    source.write_object(_claim(tenant_id="tenant-a"))
    _write_json(tmp_path / "system" / "transactions" / "shared-proof.json", {})
    _write_json(
        tmp_path / "tenants" / "tenant-b" / "system" / "transactions" / "shared-proof.json",
        {},
    )

    with pytest.raises(FileNotFoundError, match="not committed"):
        read_committed_canonical(source, URI)


def test_prepared_outbox_without_redo_cannot_authorize_a_source_ahead_before_image(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    current = _claim(tenant_id="tenant-a")
    source.write_object(current, content="current content")
    before = _claim(tenant_id="tenant-a", title="previous")
    _write_prepared_outbox(source, current, before)

    with pytest.raises(CommittedStateIntegrityError, match="without an in-flight redo proof"):
        read_committed_canonical(source, URI)


def test_other_tenant_or_legacy_root_outbox_cannot_prove_nondefault_object(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    source.write_object(_claim(tenant_id="tenant-a"))
    event = {
        "before_images": [
            {
                "uri": URI,
                "exists": True,
                "object": _claim(tenant_id="tenant-a", title="previous").to_dict(),
                "content": "previous content",
            }
        ]
    }
    _write_json(tmp_path / "system" / "outbox" / "shared-transaction.json", event)
    _write_json(
        tmp_path / "tenants" / "tenant-b" / "system" / "outbox" / "shared-transaction.json",
        event,
    )

    with pytest.raises(FileNotFoundError, match="not committed"):
        read_committed_canonical(source, URI)


def test_relation_commit_proof_is_tenant_scoped(tmp_path: Path) -> None:
    source_a = _source_store(tmp_path, tenant_id="tenant-a")
    source_b = _source_store(tmp_path, tenant_id="tenant-b")
    relation_store = InMemoryRelationStore()
    relation = ContextRelation(
        source_uri=URI,
        relation_type="alternative",
        target_uri=f"{URI}-other",
        metadata={
            "canonical_idempotency_key": "shared-proof",
            "canonical_transaction_id": "shared-transaction",
            "tenant_id": "tenant-b",
            "owner_user_id": "u1",
        },
    )
    obj = _claim(tenant_id="tenant-b")
    source_b.write_object(obj)
    relation_store.add_relation(relation)
    _write_proof(source_b, obj, relation=relation)

    assert relation_is_committed(source_a, relation, relation_store) is False
    assert relation_is_committed(source_b, relation, relation_store) is True

    relation_store.delete_relation(relation.source_uri, relation.relation_type, relation.target_uri)
    assert relation_is_committed(source_b, relation, relation_store) is True


def test_unsafe_marker_key_cannot_borrow_proof_outside_tenant_artifact_root(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    obj = _claim(tenant_id="tenant-a")
    obj.metadata["canonical_idempotency_key"] = "../../../outside-marker"
    source.write_object(obj)
    _write_json(tmp_path / "tenants" / "outside-marker.json", {})

    with pytest.raises(FileNotFoundError, match="no committed transaction proof"):
        read_committed_canonical(source, URI)


def test_unsafe_transaction_id_cannot_borrow_outbox_outside_tenant_artifact_root(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    obj = _claim(tenant_id="tenant-a")
    obj.metadata["canonical_transaction_id"] = "../../../outside-outbox"
    source.write_object(obj)
    before = _claim(tenant_id="tenant-a", title="previous")
    _write_json(
        tmp_path / "tenants" / "outside-outbox.json",
        {
            "before_images": [
                {
                    "uri": URI,
                    "exists": True,
                    "object": before.to_dict(),
                    "content": "previous content",
                }
            ]
        },
    )

    with pytest.raises(FileNotFoundError, match="no committed transaction proof"):
        read_committed_canonical(source, URI)


def test_unsafe_relation_key_cannot_borrow_proof_outside_tenant_artifact_root(tmp_path: Path) -> None:
    source = _source_store(tmp_path, tenant_id="tenant-a")
    relation = ContextRelation(
        source_uri=URI,
        relation_type="alternative",
        target_uri=f"{URI}-other",
        metadata={"canonical_idempotency_key": "../../../outside-relation"},
    )
    _write_json(tmp_path / "tenants" / "outside-relation.json", {})

    assert relation_is_committed(source, relation) is False


@pytest.mark.parametrize("raw", ["", "{broken"])
def test_empty_or_broken_marker_is_never_visible(tmp_path: Path, raw: str) -> None:
    source = _source_store(tmp_path)
    source.write_object(_claim(tenant_id="default"))
    path = tmp_path / "system" / "transactions" / "shared-proof.json"
    path.parent.mkdir(parents=True)
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(CommittedStateIntegrityError, match="immutable receipt history is invalid"):
        read_committed_canonical(source, URI)


def test_marker_identity_digest_missing_and_source_tamper_fail_closed(tmp_path: Path) -> None:
    source = _source_store(tmp_path)
    obj = _claim(tenant_id="default")
    source.write_object(obj, content="current")
    marker = _write_proof(source, obj)
    original_receipt = json.loads(marker.read_text(encoding="utf-8"))

    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["transaction_id"] = "different-transaction"
    core = {key: value for key, value in payload.items() if key != "receipt_digest"}
    payload["receipt_digest"] = canonical_digest(core)
    atomic_write_json(marker, payload, artifact_root=tmp_path)
    with pytest.raises(CommittedStateIntegrityError, match="current proof is invalid"):
        read_committed_canonical(source, URI)

    atomic_write_json(marker, original_receipt, artifact_root=tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["receipt_digest"] = "0" * 64
    atomic_write_json(marker, payload, artifact_root=tmp_path)
    with pytest.raises(CommittedStateIntegrityError, match="current proof is invalid"):
        read_committed_canonical(source, URI)

    atomic_write_json(marker, original_receipt, artifact_root=tmp_path)
    source.delete_object(URI)
    with pytest.raises(CommittedStateIntegrityError, match="without an in-flight redo proof"):
        read_committed_canonical(source, URI)

    source.write_object(obj, content="current")
    atomic_write_json(marker, original_receipt, artifact_root=tmp_path)
    tampered = source.read_object(URI)
    tampered.title = "tampered"
    source.write_object(tampered, content="changed")
    with pytest.raises(CommittedStateIntegrityError, match="without an in-flight redo proof"):
        read_committed_canonical(source, URI)
    assert marker.exists()
    assert not list((tmp_path / "system" / "quarantine").glob("**/*.original"))


def test_delete_marker_proves_logical_absence(tmp_path: Path) -> None:
    source = _source_store(tmp_path)
    obj = _claim(tenant_id="default")
    obj.lifecycle_state = LifecycleState.DELETED
    obj.metadata["delete_reason"] = "delete"
    source.write_object(obj)
    effect = object_effect_from_store(
        source,
        URI,
        operation_type="delete",
        expected_exists=False,
        logical_absence=True,
    )
    marker = tmp_path / "system" / "operations" / "delete-proof.json"
    atomic_write_json(
        marker,
        build_marker(
            transaction_id="delete-proof",
            idempotency_key="delete-proof",
            tenant_id="default",
            user_id="u1",
            operation_ids=["delete-proof"],
            object_effects=[effect],
            relation_effects=[],
            diff={"user_id": "u1", "operations": []},
            operations=[],
        ),
        artifact_root=tmp_path,
    )

    proof = validate_marker(marker, source, tenant_id="default", user_id="u1")
    assert proof["object_effects"][0]["expected_exists"] is False


def test_marker_publish_failure_never_exposes_partial_json(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    source = _source_store(tmp_path)
    obj = _claim(tenant_id="default")
    source.write_object(obj)
    marker = tmp_path / "system" / "transactions" / "shared-proof.json"
    payload = build_marker(
        transaction_id="shared-transaction",
        idempotency_key="shared-proof",
        tenant_id="default",
        user_id="u1",
        operation_ids=["proof-operation"],
        object_effects=[object_effect_from_store(source, URI, operation_type="update")],
        relation_effects=[],
        diff={"user_id": "u1", "operations": []},
        operations=[],
    )

    def fail_replace(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("injected marker publish crash")

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(effect_marker_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="marker publish crash"):
        atomic_write_json(marker, payload, artifact_root=tmp_path)
    assert not marker.exists()
    assert not list(marker.parent.glob("*.tmp"))
