from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from foundation.readiness import RuntimeNotReadyError
from infrastructure.store.contracts.queue import QueueJob
from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory.bootstrap import MemoryDocumentBootstrapper
from infrastructure.store.memory.control_store import MemoryDocumentControlStore
from infrastructure.store.memory.erasure_store import MemoryDocumentEraseStore
from infrastructure.store.memory.layout import (
    RUNTIME_LAYOUT_SCHEMA,
    RuntimeLayout,
    RuntimeResetRequired,
    UnsupportedRuntimeLayout,
    tenant_control_root,
    user_memory_root,
)
from memory.commit.erase import DocumentEraseStatus
from memory.core.model import ABSENT
from memory.core.structure.frontmatter import (
    matches_adopted_source,
    new_document_id,
    parse_front_matter,
    render_new_document,
)
from memory.ports.document_store import DocumentConflictError, DocumentUnsafeError
from foundation.identity import LocalUserContext
from runtime.config import RuntimeConfig
from tests.support.runtime import build_test_runtime


def test_default_control_and_all_user_sources_follow_trusted_tenant_layout(tmp_path: Path) -> None:
    assert tenant_control_root(tmp_path, "default") == tmp_path
    assert tenant_control_root(tmp_path, "tenant-a") == tmp_path / "tenants" / "tenant-a"
    assert user_memory_root(tmp_path, "default", "alice") == (
        tmp_path / "tenants" / "default" / "users" / "alice" / "memory"
    )
    assert user_memory_root(tmp_path, "tenant-a", "alice") == (
        tmp_path / "tenants" / "tenant-a" / "users" / "alice" / "memory"
    )


def test_empty_runtime_initializes_private_marker_and_replay_validates_it(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    layout = RuntimeLayout.open(root, tenant_id="default")

    payload = layout.initialize_or_validate()

    assert payload == {
        "schema": RUNTIME_LAYOUT_SCHEMA,
        "tenant_id": "default",
        "source_layout": "tenants/<tenant_id>/users/<user_id>/memory",
    }
    assert json.loads(layout.marker_path.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(layout.marker_path.stat().st_mode) == 0o600
    assert layout.initialize_or_validate() == payload


def test_filesystem_probe_precedes_all_sqlite_serving_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"

    def reject_probe(
        _store: FileSystemMemoryDocumentStore,
        _tenant_id: str,
        _owner_user_id: str | None = None,
    ) -> None:
        assert not (root / "indexes").exists()
        assert not (root / "queues").exists()
        assert not (root / "system" / "locks.sqlite3").exists()
        raise DocumentUnsafeError("probe rejected")

    monkeypatch.setattr(
        FileSystemMemoryDocumentStore,
        "probe_write_capabilities",
        reject_probe,
    )
    with pytest.raises(DocumentUnsafeError, match="probe rejected"):
        build_test_runtime(RuntimeConfig(root=str(root)))

    assert (root / "system" / "runtime-layout.json").is_file()
    assert not list(root.rglob("*.sqlite3"))


@pytest.mark.parametrize("root_text", ["", "$HOME/runtime", "${HOME}/runtime", "runtime*", "runtime?"])
def test_runtime_root_must_be_one_explicit_path(root_text: str) -> None:
    with pytest.raises(UnsupportedRuntimeLayout, match="explicit path"):
        RuntimeLayout.open(root_text, tenant_id="default")


def test_runtime_root_rejects_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / "existing-runtime").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(UnsupportedRuntimeLayout, match="symbolic link"):
        RuntimeLayout.open(linked / "new-runtime", tenant_id="default")
    with pytest.raises(UnsupportedRuntimeLayout, match="symbolic link"):
        RuntimeLayout.open(linked / "existing-runtime", tenant_id="default")


def test_nonempty_unmarked_runtime_fails_closed_without_deleting_data(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    legacy = root / "legacy.sqlite3"
    legacy.write_bytes(b"do not delete")

    with pytest.raises(RuntimeResetRequired, match="explicit reset"):
        RuntimeLayout.open(root, tenant_id="default").initialize_or_validate()

    assert legacy.read_bytes() == b"do not delete"
    assert not (root / "system" / "runtime-layout.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b'{"schema":"canonical_memory_v2","tenant_id":"default"}\n',
        b'{"schema":"markdown_memory_v1","tenant_id":"other"}\n',
        b"\xef\xbb\xbf{}",
    ],
)
def test_invalid_or_legacy_runtime_marker_fails_closed(tmp_path: Path, payload: bytes) -> None:
    marker = tmp_path / "system" / "runtime-layout.json"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(payload)

    with pytest.raises(UnsupportedRuntimeLayout):
        RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()


def test_runtime_marker_rejects_symlink_hardlink_and_directory(tmp_path: Path) -> None:
    expected = json.dumps(
        {
            "schema": RUNTIME_LAYOUT_SCHEMA,
            "tenant_id": "default",
            "source_layout": "tenants/<tenant_id>/users/<user_id>/memory",
        }
    ).encode()

    symlink_root = tmp_path / "symlink-root"
    symlink_marker = symlink_root / "system" / "runtime-layout.json"
    symlink_marker.parent.mkdir(parents=True)
    target = tmp_path / "marker-target.json"
    target.write_bytes(expected)
    symlink_marker.symlink_to(target)
    with pytest.raises(UnsupportedRuntimeLayout):
        RuntimeLayout.open(symlink_root, tenant_id="default").initialize_or_validate()

    hardlink_root = tmp_path / "hardlink-root"
    hardlink_marker = hardlink_root / "system" / "runtime-layout.json"
    hardlink_marker.parent.mkdir(parents=True)
    hardlink_source = tmp_path / "hardlink-source.json"
    hardlink_source.write_bytes(expected)
    os.link(hardlink_source, hardlink_marker)
    with pytest.raises(UnsupportedRuntimeLayout, match="hard-linked"):
        RuntimeLayout.open(hardlink_root, tenant_id="default").initialize_or_validate()

    directory_root = tmp_path / "directory-root"
    directory_marker = directory_root / "system" / "runtime-layout.json"
    directory_marker.mkdir(parents=True)
    with pytest.raises(UnsupportedRuntimeLayout):
        RuntimeLayout.open(directory_root, tenant_id="default").initialize_or_validate()


def test_first_user_bootstrap_creates_five_unique_managed_templates_once(tmp_path: Path) -> None:
    RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()
    store = FileSystemMemoryDocumentStore(tmp_path)
    bootstrap = MemoryDocumentBootstrapper(
        tmp_path,
        store,
        control_store=MemoryDocumentControlStore(tmp_path),
    )

    scan = bootstrap.ensure_user("default", "alice")
    expected_paths = {
        "MEMORY.md",
        "profile.md",
        "preferences.md",
        "knowledge/MEMORY.md",
        "knowledge/open-loops.md",
    }

    assert scan.complete
    assert {item.relative_path for item in scan.managed} == expected_paths
    assert len({item.document_id for item in scan.managed}) == 5
    memory_root = user_memory_root(tmp_path, "default", "alice")
    for relative_path in expected_paths:
        raw = (memory_root / relative_path).read_bytes()
        assert parse_front_matter(raw, max_header_bytes=32 * 1024).document_id
        assert stat.S_IMODE((memory_root / relative_path).stat().st_mode) == 0o600

    (memory_root / "profile.md").unlink()
    replay = bootstrap.ensure_user("default", "alice")
    assert "profile.md" not in {item.relative_path for item in replay.managed}
    assert not (memory_root / "profile.md").exists()


def test_bootstrap_refuses_preexisting_unmanaged_tree(tmp_path: Path) -> None:
    RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()
    memory_root = user_memory_root(tmp_path, "default", "alice")
    memory_root.mkdir(parents=True)
    user_file = memory_root / "profile.md"
    user_file.write_bytes(b"# Existing user data\n")

    with pytest.raises(RuntimeResetRequired, match="already contains data"):
        MemoryDocumentBootstrapper(
            tmp_path,
            FileSystemMemoryDocumentStore(tmp_path),
            control_store=MemoryDocumentControlStore(tmp_path),
        ).ensure_user("default", "alice")

    assert user_file.read_bytes() == b"# Existing user data\n"


def test_adopt_first_bootstrap_fails_closed_on_other_template_collision(tmp_path: Path) -> None:
    RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()
    store = FileSystemMemoryDocumentStore(tmp_path)
    adopted_id = new_document_id()
    adopted_raw = render_new_document(adopted_id, "# Adopted\n\nbody\n")
    store.create(
        "default",
        "alice",
        "knowledge/topics/adopted.md",
        adopted_raw,
        expected=ABSENT,
    )
    memory_root = user_memory_root(tmp_path, "default", "alice")
    collision = memory_root / "profile.md"
    collision.write_bytes(b"# Unmanaged profile collision\n")

    with pytest.raises(DocumentConflictError, match="template path collision"):
        MemoryDocumentBootstrapper(
            tmp_path,
            store,
            control_store=MemoryDocumentControlStore(tmp_path),
        ).ensure_adopted_user(
            "default",
            "alice",
            "knowledge/topics/adopted.md",
            document_id=adopted_id,
            adopted_raw_sha256=hashlib.sha256(adopted_raw).hexdigest(),
        )

    assert collision.read_bytes() == b"# Unmanaged profile collision\n"
    assert not (tmp_path / "system" / "memory-documents" / "alice" / "bootstrap.json").exists()


def test_adopt_first_bootstrap_resumes_prepared_templates_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()
    store = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    memory_root = user_memory_root(tmp_path, "default", "alice")
    memory_root.mkdir(parents=True)
    profile = memory_root / "profile.md"
    original = b"# User profile\n\nkeep exact body\n"
    profile.write_bytes(original)
    adopted = store.adopt(
        "default",
        "alice",
        "profile.md",
        expected_raw_sha256=hashlib.sha256(original).hexdigest(),
        assigned_document_id=new_document_id(),
    )
    durable_create = store.create
    crashed = False

    def stop_after_first_template(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - crash boundary.
        nonlocal crashed
        result = durable_create(*args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("bootstrap process stopped")
        return result

    monkeypatch.setattr(store, "create", stop_after_first_template)
    bootstrap = MemoryDocumentBootstrapper(tmp_path, store, control_store=controls)
    with pytest.raises(RuntimeError, match="process stopped"):
        bootstrap.ensure_adopted_user(
            "default",
            "alice",
            "profile.md",
            document_id=adopted.document_id,
            adopted_raw_sha256=adopted.raw_sha256,
        )
    marker = tmp_path / "system" / "memory-documents" / "alice" / "bootstrap.json"
    assert '"status":"PREPARED"' in marker.read_text(encoding="utf-8")

    restarted_store = FileSystemMemoryDocumentStore(tmp_path)
    restarted = MemoryDocumentBootstrapper(
        tmp_path,
        restarted_store,
        control_store=controls,
    )
    completed = restarted.ensure_adopted_user(
        "default",
        "alice",
        "profile.md",
        document_id=adopted.document_id,
        adopted_raw_sha256=adopted.raw_sha256,
    )

    assert '"status":"COMPLETED"' in marker.read_text(encoding="utf-8")
    assert profile.read_bytes() == adopted.raw_bytes
    assert {item.relative_path for item in completed.managed} == {
        "MEMORY.md",
        "profile.md",
        "preferences.md",
        "knowledge/MEMORY.md",
        "knowledge/open-loops.md",
    }
    first_ids = {item.relative_path: item.document_id for item in completed.managed}
    replay = restarted.ensure_user("default", "alice")
    assert {item.relative_path: item.document_id for item in replay.managed} == first_ids


def test_bootstrap_publishes_root_identity_before_completed_marker_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()
    store = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    bootstrap = MemoryDocumentBootstrapper(tmp_path, store, control_store=controls)
    durable_ensure = controls.ensure_root_identity

    def stop_after_identity(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - crash boundary.
        durable_ensure(*args, **kwargs)
        raise RuntimeError("bootstrap stopped after root identity publication")

    monkeypatch.setattr(controls, "ensure_root_identity", stop_after_identity)
    with pytest.raises(RuntimeError, match="after root identity publication"):
        bootstrap.ensure_user("default", "alice")

    marker = tmp_path / "system" / "memory-documents" / "alice" / "bootstrap.json"
    assert '"status":"PREPARED"' in marker.read_text(encoding="utf-8")
    identity = controls.load_root_identity("default", "alice")
    assert identity is not None

    monkeypatch.setattr(controls, "ensure_root_identity", durable_ensure)
    replay = MemoryDocumentBootstrapper(
        tmp_path,
        FileSystemMemoryDocumentStore(tmp_path),
        control_store=controls,
    ).ensure_user("default", "alice")

    assert '"status":"COMPLETED"' in marker.read_text(encoding="utf-8")
    assert replay.root_identity == identity.root_identity
    assert controls.load_root_identity("default", "alice") == identity


def test_startup_replays_durable_pending_erasure_before_scan_and_rebuild(tmp_path: Path) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    document_id = new_document_id()
    relative_path = "knowledge/topics/startup-erasure.md"
    created = first.memory.document_store.create(
        "default",
        "u1",
        relative_path,
        render_new_document(document_id, "# Startup erasure\n\nSTARTUP_SECRET\n"),
        expected=ABSENT,
    )
    first.memory.projection_worker.rebuild_owner("default", "u1")
    pending = MemoryDocumentEraseStore(tmp_path).begin(
        tenant_id="default",
        owner_user_id="u1",
        document_id=document_id,
        relative_path=relative_path,
        source_digest=created.raw_sha256,
        document_revision_floor=0,
        projection_generation_floor=0,
        backend_names=("local.live_source",),
        independent_evidence_retained=(),
        started_at="2026-07-18T00:00:00+00:00",
    )
    assert pending.status is DocumentEraseStatus.ERASING

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    recovered = restarted.memory.eraser.erase_store.load("default", "u1", document_id)
    assert recovered is not None and recovered.status is DocumentEraseStatus.ERASED
    assert restarted.memory.document_store.read_state("default", "u1", relative_path) == ABSENT
    assert restarted.readiness.details["memory_document_erasures"]["u1"] == {
        "completed": [document_id],
        "pending": [],
    }
    assert restarted.memory.projection_worker._owner_document_records("default", "u1") == ()


def test_startup_owner_discovery_fails_closed_at_explicit_bound(tmp_path: Path) -> None:
    RuntimeLayout.open(tmp_path, tenant_id="default").initialize_or_validate()
    users_root = tmp_path / "tenants" / "default" / "users"
    users_root.mkdir(parents=True)
    for index in range(1_001):
        (users_root / f"owner-{index:04d}").mkdir()

    container = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    snapshot = container.readiness.snapshot()
    assert snapshot["state"] == "NOT_READY"
    assert any("owner enumeration exceeded its bound" in reason for reason in snapshot["reasons"])
    assert "owners" not in snapshot["details"]


@pytest.mark.parametrize("fault_stage", ["temp_file_fsynced", "atomic_installed"])
def test_runtime_restart_resumes_exact_adoption_receipt_before_scanner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    relative_path = "knowledge/topics/runtime-restart-adopt.md"
    original = b"# Runtime restart\n\ntrusted adopt survives\n"
    path = user_memory_root(tmp_path, "default", "u1") / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    durable_adopt = first.memory.document_store.adopt

    def terminate_during_source_cas(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - crash boundary.
        def stop_at_exact_store_stage(stage: str) -> None:
            if stage == fault_stage:
                raise RuntimeError(f"runtime process terminated at {stage}")

        return durable_adopt(*args, **kwargs, fault_hook=stop_at_exact_store_stage)

    monkeypatch.setattr(first.memory.document_store, "adopt", terminate_during_source_cas)
    with pytest.raises(RuntimeError, match="process terminated"):
        first.memory.command_service.adopt_memory_document(
            relative_path,
            expected,
            caller=caller,
        )
    receipts = first.memory.control_store.adoption_receipts(
        "default",
        "u1",
    )
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.relative_path == relative_path
    assert receipt.expected_raw_sha256 == expected
    assert (
        first.memory.control_store.load_control(
            "default",
            "u1",
            receipt.document_id,
        )
        is None
    )
    if fault_stage == "temp_file_fsynced":
        assert path.read_bytes() == original
        assert len(tuple(path.parent.glob(f".{path.name}.memoryos-*.tmp"))) == 1
    else:
        assert matches_adopted_source(
            path.read_bytes(),
            receipt.document_id,
            receipt.expected_raw_sha256,
        )

    resumed_adopts: list[dict[str, object]] = []
    store_adopt = FileSystemMemoryDocumentStore.adopt

    def observe_resumed_adopt(store, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        resumed_adopts.append(dict(kwargs))
        return store_adopt(store, *args, **kwargs)

    monkeypatch.setattr(FileSystemMemoryDocumentStore, "adopt", observe_resumed_adopt)

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    restarted.readiness.require_ready()
    if fault_stage == "temp_file_fsynced":
        assert resumed_adopts == [
            {
                "expected_raw_sha256": receipt.expected_raw_sha256,
                "assigned_document_id": receipt.document_id,
                "operation_id": receipt.receipt_id,
            }
        ]
    else:
        assert resumed_adopts == []
    assert not tuple(path.parent.glob(f".{path.name}.memoryos-*.tmp"))
    control = restarted.memory.control_store.load_control(
        "default",
        "u1",
        receipt.document_id,
    )
    assert control is not None
    assert control.status == "present"
    assert control.relative_path == receipt.relative_path
    assert matches_adopted_source(
        path.read_bytes(),
        receipt.document_id,
        receipt.expected_raw_sha256,
    )
    binding = restarted.memory.control_store.load_event_binding(
        "default",
        "u1",
        receipt.document_id,
        control.last_event_id,
    )
    assert binding is not None
    _intent, event = binding
    assert event.actor_binding == receipt.actor_binding
    assert event.evidence_reference == receipt.evidence_reference
    assert event.evidence_digest == receipt.evidence_digest
    projected = [
        record
        for record in restarted.memory.projection_worker._owner_document_records(
            "default",
            "u1",
        )
        if record.document_id == receipt.document_id
    ]
    assert len(projected) == 1
    assert projected[0].source_digest == control.raw_sha256
    marker = tmp_path / "system" / "memory-documents" / "u1" / "bootstrap.json"
    assert '"status":"COMPLETED"' in marker.read_text(encoding="utf-8")
    adoption_recovery = restarted.readiness.details["memory_document_adoptions"]
    assert adoption_recovery["published"] == 1
    assert adoption_recovery["resumed_unmanaged" if fault_stage == "temp_file_fsynced" else "resumed_managed"] == 1
    remembered = restarted.memory.command_service.remember(
        "remember immediately after recovered startup",
        target_hint="topic:restarted-runtime",
        caller=caller,
    )
    assert remembered.changed is True


def test_runtime_restart_finishes_committed_adoption_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    relative_path = "knowledge/topics/adoption-control-before-bootstrap.md"
    path = user_memory_root(tmp_path, "default", "u1") / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    unmanaged = b"# Control before bootstrap\n\ncommitted adoption survives\n"
    path.write_bytes(unmanaged)
    interrupted_bootstrapper = first.memory.bootstrapper
    durable_ensure = MemoryDocumentBootstrapper.ensure_adopted_user

    def stop_before_bootstrap(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if self is interrupted_bootstrapper:
            raise RuntimeError("runtime process terminated after adoption control")
        return durable_ensure(self, *args, **kwargs)

    monkeypatch.setattr(
        MemoryDocumentBootstrapper,
        "ensure_adopted_user",
        stop_before_bootstrap,
    )
    with pytest.raises(RuntimeError, match="after adoption control"):
        first.memory.command_service.adopt_memory_document(
            relative_path,
            hashlib.sha256(unmanaged).hexdigest(),
            caller=caller,
        )
    receipt = first.memory.control_store.adoption_receipts("default", "u1")[0]
    control = first.memory.control_store.load_control(
        "default",
        "u1",
        receipt.document_id,
    )
    assert control is not None
    assert (
        first.memory.control_store.load_event_binding(
            "default",
            "u1",
            receipt.document_id,
            control.last_event_id,
        )
        is not None
    )
    marker = tmp_path / "system" / "memory-documents" / "u1" / "bootstrap.json"
    assert not marker.exists()

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    restarted.readiness.require_ready()
    assert '"status":"COMPLETED"' in marker.read_text(encoding="utf-8")
    recovery = restarted.readiness.details["memory_document_adoptions"]
    assert recovery["already_committed"] == 1
    assert recovery["bootstrap_resumed"] == 1
    assert recovery["published"] == 0
    remembered = restarted.memory.command_service.remember(
        "remember works immediately after bootstrap-only recovery",
        target_hint="topic:bootstrap-recovered",
        caller=caller,
    )
    assert remembered.changed is True


@pytest.mark.parametrize(
    ("third_state", "fault_stage", "reason"),
    [
        ("edited", "temp_file_fsynced", "third source state"),
        ("unsafe", "temp_file_fsynced", "absent or unsafe"),
        ("duplicate", "atomic_installed", "unsafe, duplicated, or unregistered"),
    ],
)
def test_runtime_adoption_receipt_preserves_unsafe_or_third_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    third_state: str,
    fault_stage: str,
    reason: str,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    relative_path = "knowledge/topics/adoption-third-state.md"
    path = user_memory_root(tmp_path, "default", "u1") / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b"# Adoption third state\n\noriginal unmanaged bytes\n"
    path.write_bytes(original)
    durable_adopt = first.memory.document_store.adopt

    def terminate_during_source_cas(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        def stop_at_exact_store_stage(stage: str) -> None:
            if stage == fault_stage:
                raise RuntimeError(f"runtime process terminated at {stage}")

        return durable_adopt(*args, **kwargs, fault_hook=stop_at_exact_store_stage)

    monkeypatch.setattr(first.memory.document_store, "adopt", terminate_during_source_cas)
    with pytest.raises(RuntimeError, match="process terminated"):
        first.memory.command_service.adopt_memory_document(
            relative_path,
            hashlib.sha256(original).hexdigest(),
            caller=caller,
        )
    receipt = first.memory.control_store.adoption_receipts("default", "u1")[0]

    duplicate_path: Path | None = None
    outside: Path | None = None
    if third_state == "edited":
        path.write_bytes(b"# Adoption third state\n\nuser changed this after the crash\n")
    elif third_state == "unsafe":
        outside = tmp_path / "outside.md"
        outside.write_bytes(b"outside bytes must remain untouched")
        path.unlink()
        path.symlink_to(outside)
    else:
        duplicate_path = path.with_name("adoption-duplicate.md")
        duplicate_path.write_bytes(path.read_bytes())

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    snapshot = restarted.readiness.snapshot()
    assert snapshot["state"] == "NOT_READY"
    assert any(reason in item for item in snapshot["reasons"])
    assert "memory_full_scan" not in snapshot["details"]
    assert (
        restarted.memory.control_store.load_control(
            "default",
            "u1",
            receipt.document_id,
        )
        is None
    )
    if third_state == "edited":
        assert b"user changed this after the crash" in path.read_bytes()
    elif third_state == "unsafe":
        assert path.is_symlink()
        assert outside is not None and outside.read_bytes() == b"outside bytes must remain untouched"
    else:
        assert duplicate_path is not None
        assert path.read_bytes() == duplicate_path.read_bytes()


def test_runtime_restart_treats_renamed_edited_adoption_receipt_as_history(
    tmp_path: Path,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    original_path = "knowledge/topics/adoption-history.md"
    source = user_memory_root(tmp_path, "default", "u1") / original_path
    source.parent.mkdir(parents=True, exist_ok=True)
    unmanaged = b"# Adoption history\n\noriginal unmanaged bytes\n"
    source.write_bytes(unmanaged)
    adopted = first.memory.command_service.adopt_memory_document(
        original_path,
        hashlib.sha256(unmanaged).hexdigest(),
        caller=caller,
    )
    renamed_path = "knowledge/topics/adoption-history-renamed.md"
    renamed = first.memory.command_service.rename_memory_document(
        adopted.document_uri,
        renamed_path,
        adopted.source_digest,
        edit="# Renamed adoption\n\nthis body legitimately changed later\n",
        caller=caller,
    )

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    restarted.readiness.require_ready()
    control = restarted.memory.control_store.load_control(
        "default",
        "u1",
        adopted.document_id,
    )
    assert control is not None
    assert control.relative_path == renamed_path
    assert control.raw_sha256 == renamed.source_digest
    assert restarted.readiness.details["memory_document_adoptions"]["already_committed"] == 1
    assert restarted.readiness.details["memory_document_adoptions"]["published"] == 0
    assert restarted.memory.document_store.read_state("default", "u1", original_path) == ABSENT
    assert b"legitimately changed later" in restarted.memory.document_store.read_raw(
        "default",
        "u1",
        document_id=adopted.document_id,
    )


def test_runtime_restart_does_not_resurrect_hard_erased_adoption_receipt(
    tmp_path: Path,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    relative_path = "knowledge/topics/adoption-hard-erased.md"
    path = user_memory_root(tmp_path, "default", "u1") / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    unmanaged = b"# Erased adoption\n\nthis body must never return\n"
    path.write_bytes(unmanaged)
    adopted = first.memory.command_service.adopt_memory_document(
        relative_path,
        hashlib.sha256(unmanaged).hexdigest(),
        caller=caller,
    )
    receipt = first.memory.control_store.load_adoption_receipt_for_document(
        "default",
        "u1",
        adopted.document_id,
    )
    assert receipt is not None
    erased = first.memory.eraser.hard_erase(
        tenant_id="default",
        owner_user_id="u1",
        document_id=adopted.document_id,
        expected_source_digest=adopted.source_digest,
        relative_path=relative_path,
    )
    assert erased.completed is True

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    restarted.readiness.require_ready()
    assert restarted.memory.document_store.read_state("default", "u1", relative_path) == ABSENT
    assert (
        restarted.memory.control_store.load_control(
            "default",
            "u1",
            adopted.document_id,
        )
        is None
    )
    assert (
        restarted.memory.control_store.load_adoption_receipt_for_document(
            "default",
            "u1",
            adopted.document_id,
        )
        == receipt
    )
    assert restarted.readiness.details["memory_document_adoptions"]["erasure_blocked"] == 1
    assert restarted.readiness.details["memory_document_adoptions"]["published"] == 0
    with pytest.raises(DocumentConflictError, match="durable erasure epoch"):
        restarted.memory.committer.erasure_store.assert_mutation_allowed(
            "default",
            "u1",
            adopted.document_id,
        )


@pytest.mark.parametrize("offline_change", ["edit", "rename_edit", "delete", "unchanged"])
def test_runtime_restart_seeds_scanner_from_durable_controls(
    tmp_path: Path,
    offline_change: str,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    remembered = first.memory.command_service.remember(
        f"offline {offline_change} before",
        target_hint=f"topic:offline-{offline_change}",
        caller=caller,
    )
    original_path = user_memory_root(tmp_path, "default", "u1") / remembered.relative_path
    expected_path = remembered.relative_path
    expected_digest = remembered.source_digest
    expected_kind = "create"
    expected_status = "present"
    expected_revision = 1
    updated_raw = b""
    if offline_change in {"edit", "rename_edit"}:
        updated_raw = render_new_document(
            remembered.document_id,
            f"# Offline {offline_change}\n\nchanged while stopped\n",
        )
        expected_digest = hashlib.sha256(updated_raw).hexdigest()
        expected_kind = "update" if offline_change == "edit" else "rename"
        expected_revision = 2
        if offline_change == "rename_edit":
            expected_path = "knowledge/topics/offline-renamed-and-edited.md"
            renamed = user_memory_root(tmp_path, "default", "u1") / expected_path
            renamed.parent.mkdir(parents=True, exist_ok=True)
            original_path.rename(renamed)
            renamed.write_bytes(updated_raw)
        else:
            original_path.write_bytes(updated_raw)
    elif offline_change == "delete":
        original_path.unlink()
        # One startup observation is only a pending absence.  Durable control
        # remains live until the scanner worker sees the same absence again.

    event_directory = tmp_path / "system" / "memory-documents" / "u1" / "events" / remembered.document_id
    before_event_count = len(tuple(event_directory.glob("*.json")))
    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    restarted.readiness.require_ready()
    control = restarted.memory.control_store.load_control(
        "default",
        "u1",
        remembered.document_id,
    )
    assert control is not None
    assert control.status == expected_status
    assert control.relative_path == expected_path
    assert control.raw_sha256 == expected_digest
    assert control.logical_revision == expected_revision
    revision = restarted.memory.revision_store.load_revision(
        "default",
        "u1",
        remembered.document_id,
        expected_revision,
    )
    assert revision is not None and revision.edit_kind.value == expected_kind
    after_events = tuple(sorted(event_directory.glob("*.json")))
    if offline_change in {"unchanged", "delete"}:
        assert len(after_events) == before_event_count
    else:
        assert len(after_events) == before_event_count + 1
        event_payload = json.loads(after_events[-1].read_text(encoding="utf-8"))
        assert event_payload["actor_binding"] == "external-editor:stable-full-scan"
        assert str(event_payload["evidence_reference"]).startswith("scan-generation:")
    if offline_change in {"edit", "rename_edit"}:
        assert (
            restarted.memory.revision_store.read_revision_blob(
                "default",
                "u1",
                remembered.document_id,
                2,
            )
            == updated_raw
        )
    if offline_change == "delete":
        assert restarted.readiness.details["memory_full_scan"]["u1"]["pending"] == 1
        verification = restarted.readiness.details["memory_document_verification"]["u1"]
        assert verification.get("pending_missing", 0) in {0, 1}
        assert (
            restarted.memory.control_store.load_publication_barrier("default", "u1", remembered.document_id)
            is None
        )
        restarted.memory.scanner.stability_seconds = 0
        restarted.stores.queue.enqueue(
            QueueJob(
                job_id=f"test_memory_rescan_{remembered.document_id}",
                queue_name="memory_document_scan",
                action="rescan",
                target_uri=remembered.document_uri,
                payload={
                    "tenant_id": "default",
                    "owner_user_id": "u1",
                    "document_id": remembered.document_id,
                    "observed_source_digest": remembered.source_digest,
                },
            )
        )
        scan_run = restarted.memory.scan_worker.process_pending()
        assert scan_run["processed"] == 1
        projection_run = restarted.memory.projection_worker.process_pending(limit=10)
        assert not projection_run.failed
        deleted = restarted.memory.control_store.load_control("default", "u1", remembered.document_id)
        assert deleted is not None
        assert deleted.status == "deleted"
        assert deleted.raw_sha256 == ""
        assert deleted.logical_revision == 2
        deletion_revision = restarted.memory.revision_store.load_revision(
            "default", "u1", remembered.document_id, 2
        )
        assert deletion_revision is not None
        assert deletion_revision.edit_kind.value == "delete"
        assert len(tuple(event_directory.glob("*.json"))) == before_event_count + 1


def test_runtime_restart_rejects_replaced_user_memory_root(tmp_path: Path) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    remembered = first.memory.command_service.remember(
        "root replacement must not become delete authority",
        target_hint="topic:root-replacement",
        caller=caller,
    )
    memory_root = user_memory_root(tmp_path, "default", "u1")
    detached_root = memory_root.with_name("memory-detached")
    original_bytes = (memory_root / remembered.relative_path).read_bytes()
    memory_root.rename(detached_root)
    memory_root.mkdir()

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    with pytest.raises(RuntimeNotReadyError):
        restarted.readiness.require_ready()
    control = restarted.memory.control_store.load_control(
        "default",
        "u1",
        remembered.document_id,
    )
    assert control is not None
    assert control.status == "present"
    assert control.raw_sha256 == remembered.source_digest
    assert control.logical_revision == remembered.document_revision
    assert (detached_root / remembered.relative_path).read_bytes() == original_bytes
    assert not list(memory_root.rglob("*.md"))


def test_runtime_restart_pauses_delete_without_durable_root_identity(tmp_path: Path) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    remembered = first.memory.command_service.remember(
        "missing identity must pause delete",
        target_hint="topic:missing-root-identity",
        caller=caller,
    )
    identity_path = tmp_path / "system" / "memory-documents" / "u1" / "scan-root.json"
    assert identity_path.is_file()
    identity_path.unlink()
    source_path = user_memory_root(tmp_path, "default", "u1") / remembered.relative_path
    source_path.unlink()

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    with pytest.raises(RuntimeNotReadyError):
        restarted.readiness.require_ready()
    control = restarted.memory.control_store.load_control(
        "default",
        "u1",
        remembered.document_id,
    )
    assert control is not None and control.status == "present"
    assert control.raw_sha256 == remembered.source_digest
    assert not identity_path.exists()


def test_completed_bootstrap_never_recreates_a_missing_root_identity(tmp_path: Path) -> None:
    container = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    container.memory.command_service.remember(
        "completed marker cannot recreate root authority",
        target_hint="topic:completed-root-authority",
        caller=caller,
    )
    assert container.memory.control_store.controls("default", "u1")
    identity_path = tmp_path / "system" / "memory-documents" / "u1" / "scan-root.json"
    identity_path.unlink()

    with pytest.raises(RuntimeResetRequired, match="missing its durable document root identity"):
        container.memory.bootstrapper.ensure_user("default", "u1")

    assert not identity_path.exists()


def test_missing_identity_cannot_bless_a_same_bytes_replacement_then_delete(
    tmp_path: Path,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    remembered = first.memory.command_service.remember(
        "same bytes replacement must not gain delete authority",
        target_hint="topic:same-bytes-root-replacement",
        caller=caller,
    )
    control_before = first.memory.control_store.load_control(
        "default",
        "u1",
        remembered.document_id,
    )
    assert control_before is not None and control_before.status == "present"
    identity_path = tmp_path / "system" / "memory-documents" / "u1" / "scan-root.json"
    identity_path.unlink()
    memory_root = user_memory_root(tmp_path, "default", "u1")
    detached_root = memory_root.with_name("memory-detached-same-bytes")
    memory_root.rename(detached_root)
    shutil.copytree(detached_root, memory_root)

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    with pytest.raises(RuntimeNotReadyError):
        restarted.readiness.require_ready()
    assert not identity_path.exists()
    assert (
        restarted.memory.control_store.load_control(
            "default",
            "u1",
            remembered.document_id,
        )
        == control_before
    )
    replacement_path = memory_root / remembered.relative_path
    assert replacement_path.read_bytes() == (detached_root / remembered.relative_path).read_bytes()

    replacement_path.unlink()
    second = restarted.memory.scanner.scan("default", "u1", force_stable=True)

    assert second.deletions_paused is True
    assert second.confirmed_changes == ()
    assert (
        restarted.memory.control_store.load_control(
            "default",
            "u1",
            remembered.document_id,
        )
        == control_before
    )
    assert not identity_path.exists()


@pytest.mark.parametrize("artifact_kind", ["tampered", "symlink"])
def test_runtime_restart_rejects_unsafe_root_identity_artifact(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    first = build_test_runtime(RuntimeConfig(root=str(tmp_path)))
    caller = LocalUserContext(
        user_id="u1",
    )
    remembered = first.memory.command_service.remember(
        "root artifact integrity",
        target_hint="topic:root-artifact-integrity",
        caller=caller,
    )
    identity_path = tmp_path / "system" / "memory-documents" / "u1" / "scan-root.json"
    identity_path.unlink()
    if artifact_kind == "tampered":
        identity_path.write_bytes(b'{"schema":"wrong"}')
    else:
        target = tmp_path / "untrusted-root-identity.json"
        target.write_text("{}", encoding="utf-8")
        identity_path.symlink_to(target)

    restarted = build_test_runtime(RuntimeConfig(root=str(tmp_path)))

    with pytest.raises(RuntimeNotReadyError):
        restarted.readiness.require_ready()
    control = restarted.memory.control_store.load_control(
        "default",
        "u1",
        remembered.document_id,
    )
    assert control is not None and control.status == "present"
