from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore

URI = "memoryos://user/u1/memories/canonical/slots/s1/claims/c1"
BUNDLE_STAGES = (
    "after_meta",
    "after_relations",
    "after_content",
    "after_bundle_manifest",
    "before_bundle_publish",
    "before_current_pointer",
    "after_current_pointer",
    "after_bundle_publish",
)


class _CrashOnce:
    def __init__(self, target: str) -> None:
        self.target = target
        self.crashed = False

    def __call__(self, stage: str, _uri: str, _generation_id: str) -> None:
        if stage == self.target and not self.crashed:
            self.crashed = True
            raise SystemExit(f"bundle crash at {stage}")


def _object(revision: int) -> ContextObject:
    return ContextObject(
        uri=URI,
        context_type=ContextType.MEMORY,
        title=f"revision-{revision}",
        owner_user_id="u1",
        tenant_id="t1",
        metadata={
            "canonical_kind": "claim",
            "revision": revision,
            "state": "ACTIVE",
        },
        relations=[
            ContextRelation(
                source_uri=URI,
                relation_type="belongs_to_slot",
                target_uri="memoryos://user/u1/memories/canonical/slots/s1",
                metadata={"revision": revision},
                created_at=f"2026-01-{revision:02d}T00:00:00+00:00",
            )
        ],
        created_at="2026-01-01T00:00:00+00:00",
        updated_at=f"2026-01-{revision:02d}T00:00:00+00:00",
    )


@pytest.mark.parametrize("crash_stage", BUNDLE_STAGES)
def test_versioned_bundle_never_exposes_torn_object_content_or_relations(
    tmp_path: Path,
    crash_stage: str,
) -> None:
    store = FileSystemSourceStore(tmp_path, tenant_id="t1")
    old = _object(1)
    new = _object(2)
    store.write_object(old, content="content-1")
    store.test_hook = _CrashOnce(crash_stage)

    with pytest.raises(SystemExit, match=crash_stage):
        store.write_object(new, content="content-2")

    restarted = FileSystemSourceStore(tmp_path, tenant_id="t1")
    observed = restarted.read_object(URI)
    observed_content = restarted.read_content(URI)
    observed_relation_revision = observed.relations[0].metadata["revision"]
    state = (
        int(observed.metadata["revision"]),
        observed.title,
        observed_content,
        int(observed_relation_revision),
    )
    if crash_stage in {"after_current_pointer", "after_bundle_publish"}:
        assert state == (2, "revision-2", "content-2", 2)
    else:
        assert state == (1, "revision-1", "content-1", 1)

    restarted.write_object(deepcopy(new), content="content-2")
    completed = restarted.read_object(URI)
    assert (
        int(completed.metadata["revision"]),
        completed.title,
        restarted.read_content(URI),
        int(completed.relations[0].metadata["revision"]),
    ) == (2, "revision-2", "content-2", 2)


def test_canonical_bundle_distinguishes_metadata_only_write_from_explicit_empty_content(
    tmp_path: Path,
) -> None:
    store = FileSystemSourceStore(tmp_path, tenant_id="t1")
    original = _object(1)
    store.write_object(original, content="non-empty")

    metadata_only = _object(2)
    store.write_object(metadata_only)
    assert store.read_content(URI) == "non-empty"

    store.write_content(URI, "")
    assert store.read_object(URI).metadata["revision"] == 2
    assert store.read_content(URI) == ""


def test_canonical_bundle_write_rejects_broken_current_pointer_symlink(
    tmp_path: Path,
) -> None:
    store = FileSystemSourceStore(tmp_path, tenant_id="t1")
    store.write_object(_object(1), content="content-1")
    pointer = store._object_dir(URI) / ".bundle-current.json"
    pointer.unlink()
    missing_target = tmp_path / "missing-bundle-pointer.json"
    pointer.symlink_to(missing_target)

    with pytest.raises(RuntimeError, match="symbolic link"):
        store.write_object(_object(2), content="content-2")

    assert pointer.is_symlink()
    assert not missing_target.exists()


def test_canonical_bundle_rejects_in_root_cross_tenant_directory_symlink(
    tmp_path: Path,
) -> None:
    foreign_memories = tmp_path / "tenants" / "t2" / "users" / "u1" / "memories"
    foreign_memories.mkdir(parents=True)
    local_parent = tmp_path / "tenants" / "t1" / "users" / "u1"
    local_parent.mkdir(parents=True)
    (local_parent / "memories").symlink_to(foreign_memories, target_is_directory=True)
    store = FileSystemSourceStore(tmp_path, tenant_id="t1")

    with pytest.raises(ValueError, match="symbolic link"):
        store.write_object(_object(1), content="must-not-cross-tenants")

    assert not (foreign_memories / "canonical" / "slots" / "s1" / "claims" / "c1").exists()


def test_new_canonical_bundle_fsyncs_its_complete_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSystemSourceStore(tmp_path, tenant_id="t1")
    fsynced: list[Path] = []
    original = FileSystemSourceStore._fsync_directory

    def record(path: Path) -> None:
        fsynced.append(path)
        original(path)

    monkeypatch.setattr(FileSystemSourceStore, "_fsync_directory", staticmethod(record))
    store.write_object(_object(1), content="content-1")

    object_dir = store._object_dir(URI)
    required = {tmp_path, tmp_path / "tenants", tmp_path / "tenants" / "t1", object_dir}
    assert required.issubset(set(fsynced))
