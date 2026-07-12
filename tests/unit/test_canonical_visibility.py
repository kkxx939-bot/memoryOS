from __future__ import annotations

import json
from pathlib import Path

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.memory.canonical.visibility import (
    read_committed_canonical,
    relation_is_committed,
)

URI = "memoryos://user/u1/memories/canonical/slots/s1/claims/c1"


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_tenant_keeps_root_level_commit_proof_compatibility(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path)
    obj = _claim(tenant_id="default")
    source.write_object(obj)
    _write_json(tmp_path / "system" / "transactions" / "shared-proof.json", {})

    committed = read_committed_canonical(source, URI)

    assert committed.object.title == "current"
    assert committed.from_before_image is False


def test_nondefault_tenant_reads_only_its_scoped_commit_proof(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    obj = _claim(tenant_id="tenant-a")
    source.write_object(obj)
    _write_json(
        tmp_path / "tenants" / "tenant-a" / "system" / "transactions" / "shared-proof.json",
        {},
    )

    assert read_committed_canonical(source, URI).object.title == "current"


def test_other_tenant_or_legacy_root_marker_cannot_prove_nondefault_object(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    source.write_object(_claim(tenant_id="tenant-a"))
    _write_json(tmp_path / "system" / "transactions" / "shared-proof.json", {})
    _write_json(
        tmp_path / "tenants" / "tenant-b" / "system" / "transactions" / "shared-proof.json",
        {},
    )

    with pytest.raises(FileNotFoundError, match="not committed"):
        read_committed_canonical(source, URI)


def test_nondefault_before_image_uses_only_tenant_scoped_outbox(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    source.write_object(_claim(tenant_id="tenant-a"))
    before = _claim(tenant_id="tenant-a", title="previous")
    _write_json(
        tmp_path / "tenants" / "tenant-a" / "system" / "outbox" / "shared-transaction.json",
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

    committed = read_committed_canonical(source, URI)

    assert committed.object.title == "previous"
    assert committed.content_override == "previous content"
    assert committed.from_before_image is True


def test_other_tenant_or_legacy_root_outbox_cannot_prove_nondefault_object(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
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
    source_a = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    source_b = FileSystemSourceStore(tmp_path, tenant_id="tenant-b")
    relation = ContextRelation(
        source_uri=URI,
        relation_type="alternative",
        target_uri=f"{URI}-other",
        metadata={"canonical_idempotency_key": "shared-proof"},
    )
    _write_json(
        tmp_path / "tenants" / "tenant-b" / "system" / "transactions" / "shared-proof.json",
        {},
    )

    assert relation_is_committed(source_a, relation) is False
    assert relation_is_committed(source_b, relation) is True


def test_unsafe_marker_key_cannot_borrow_proof_outside_tenant_artifact_root(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    obj = _claim(tenant_id="tenant-a")
    obj.metadata["canonical_idempotency_key"] = "../../../outside-marker"
    source.write_object(obj)
    _write_json(tmp_path / "tenants" / "outside-marker.json", {})

    with pytest.raises(FileNotFoundError, match="no committed transaction proof"):
        read_committed_canonical(source, URI)


def test_unsafe_transaction_id_cannot_borrow_outbox_outside_tenant_artifact_root(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
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
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    relation = ContextRelation(
        source_uri=URI,
        relation_type="alternative",
        target_uri=f"{URI}-other",
        metadata={"canonical_idempotency_key": "../../../outside-relation"},
    )
    _write_json(tmp_path / "tenants" / "outside-relation.json", {})

    assert relation_is_committed(source, relation) is False
