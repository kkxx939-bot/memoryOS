from __future__ import annotations

import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.core.durable_io import atomic_create_json
from memoryos.memory.documents import (
    ABSENT,
    DocumentConflictError,
    DocumentControlIntegrityError,
    DocumentEditKind,
    DocumentEditPlan,
    DocumentIntentStatus,
    ExternalChangeKind,
    ExternalDocumentChange,
    MemoryDocumentCommitter,
    MemoryDocumentControlStore,
    MemoryDocumentRevisionStore,
    MemoryDocumentScanner,
    adoption_document_id,
    adoption_request_digest,
    new_document_id,
    render_new_document,
)
from memoryos.memory.documents.layout import user_memory_root


class _InjectedCrash(RuntimeError):
    pass


class _CrashAt:
    def __init__(self, stage: str) -> None:
        self.stage = stage

    def __call__(self, stage: str, intent) -> None:  # noqa: ANN001
        if stage == self.stage:
            raise _InjectedCrash(intent.intent_id)


class _FileProjectionQueue:
    """Small create-only durable QueueStore test double used across restarts."""

    def __init__(self, root: Path) -> None:
        self.root = root / "system" / "test-memory-projection-queue"

    def enqueue(self, job: QueueJob) -> QueueJob:
        atomic_create_json(
            self.root / f"{job.job_id}.json",
            {
                "job_id": job.job_id,
                "queue_name": job.queue_name,
                "action": job.action,
                "target_uri": job.target_uri,
                "payload": job.payload,
            },
            artifact_root=self.root,
        )
        return job

    def payloads(self) -> tuple[dict, ...]:
        return tuple(json.loads(path.read_text()) for path in sorted(self.root.glob("*.json")))


def _plan(
    key: str,
    document_id: str,
    kind: DocumentEditKind,
    expected,
    path: str,
    after: bytes | None = None,
) -> DocumentEditPlan:
    return DocumentEditPlan(
        idempotency_key=key,
        tenant_id="default",
        owner_user_id="owner-1",
        edit_kind=kind,
        expected_state=expected,
        evidence_digest="e" * 64,
        edit_summary=f"{kind.value} crash test",
        document_id=document_id,
        relative_path=path,
        after_bytes=after,
        expected_registration_document_id=document_id,
    )


def _committer(root: Path, *, hook=None):  # noqa: ANN001, ANN202
    source = FileSystemMemoryDocumentStore(root)
    controls = MemoryDocumentControlStore(root)
    revisions = MemoryDocumentRevisionStore(root)
    queue = _FileProjectionQueue(root)
    return source, controls, revisions, queue, MemoryDocumentCommitter(
        source,
        controls,
        revisions,
        cast(QueueStore, queue),
        test_hook=hook,
    )


@pytest.mark.parametrize(
    "fault_stage",
    [
        "intent_prepared",
        "after_installed",
        "event_appended",
        "revision_recorded",
        "control_recorded",
        "projection_enqueued",
    ],
)
def test_create_crash_windows_roll_forward_from_exact_before_or_after(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    root = (tmp_path / fault_stage).resolve()
    root.mkdir()
    source, controls, _, _, crashing = _committer(root, hook=_CrashAt(fault_stage))
    document_id = new_document_id()
    path = "knowledge/topics/crash.md"
    raw = render_new_document(document_id, "durable exact after bytes")

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("create-crash", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="archive:manifest-digest",
        )
    intents = controls.incomplete_intents("default", "owner-1")
    assert len(intents) == 1

    recovered_source, recovered_controls, revisions, queue, recovered = _committer(root)
    report = recovered.recover_all("default", "owner-1")

    assert report.conflicted_intent_ids == ()
    assert len(report.completed) == 1
    assert recovered_source.read_raw("default", "owner-1", relative_path=path) == raw
    intent = recovered_controls.load_intent("default", "owner-1", intents[0].intent_id)
    assert intent is not None and intent.status == DocumentIntentStatus.COMPLETED
    assert revisions.latest_revision("default", "owner-1", document_id) == 1
    payloads = queue.payloads()
    assert {payload["action"] for payload in payloads} == {
        "memory_committed",
        "recover_document_intent",
    }
    recovery = next(payload for payload in payloads if payload["action"] == "recover_document_intent")
    assert recovery["payload"] == {
        "tenant_id": "default",
        "owner_user_id": "owner-1",
        "document_id": document_id,
        "intent_id": intent.intent_id,
    }
    assert all("durable exact after bytes" not in json.dumps(payload) for payload in payloads)
    assert source.read_state("default", "owner-1", path) == recovered_source.read_state(
        "default", "owner-1", path
    )


def test_initial_create_prebinds_real_root_before_blob_intent_or_install(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, revisions, queue, crashing = _committer(
        root,
        hook=_CrashAt("root_identity_preflighted"),
    )
    document_id = new_document_id()
    path = "knowledge/topics/preflight.md"
    raw = render_new_document(document_id, "preflight before every durable body artifact")

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("preflight-create", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:preflight",
        )

    identity = controls.load_root_identity("default", "owner-1")
    assert identity is not None
    assert identity.root_identity == source.full_scan("default", "owner-1").root_identity
    assert controls.incomplete_intents("default", "owner-1") == ()
    assert revisions.latest_revision("default", "owner-1", document_id) == 0
    assert not tuple(
        (root / "system" / "memory-documents" / "owner-1" / "blobs").rglob("*.blob")
    )
    assert queue.payloads() == ()
    assert source.read_state("default", "owner-1", path) == ABSENT

    recovered_source, recovered_controls, _, _, recovered = _committer(root)
    result = recovered.commit(
        _plan("preflight-create", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:preflight",
    )

    assert result.status == DocumentIntentStatus.COMPLETED
    assert recovered_controls.load_root_identity("default", "owner-1") == identity
    assert recovered_source.read_raw("default", "owner-1", relative_path=path) == raw


def test_prepared_intent_missing_identity_fails_closed_without_backfill(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, _, _, crashing = _committer(root, hook=_CrashAt("intent_prepared"))
    document_id = new_document_id()
    path = "knowledge/topics/missing-prepared-identity.md"
    raw = render_new_document(document_id, "prepared identity cannot be reconstructed")

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("missing-identity", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:missing-identity",
        )
    assert len(controls.incomplete_intents("default", "owner-1")) == 1
    identity_path = root / "system" / "memory-documents" / "owner-1" / "scan-root.json"
    identity_path.unlink()

    with pytest.raises(DocumentControlIntegrityError, match="missing its source root identity"):
        _committer(root)[-1].recover_all("default", "owner-1")

    assert source.read_state("default", "owner-1", path) == ABSENT
    assert not identity_path.exists()


@pytest.mark.parametrize(
    ("fault_stage", "live_after"),
    [("intent_prepared", False), ("after_installed", True)],
)
def test_prepared_intent_root_replacement_fails_closed_in_before_and_after_state(
    tmp_path: Path,
    fault_stage: str,
    live_after: bool,
) -> None:
    root = (tmp_path / fault_stage).resolve()
    root.mkdir()
    source, controls, _, _, crashing = _committer(root, hook=_CrashAt(fault_stage))
    document_id = new_document_id()
    path = "knowledge/topics/replaced-prepared-root.md"
    raw = render_new_document(document_id, f"prepared {fault_stage} replacement")

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("replace-prepared-root", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:replace-prepared-root",
        )
    identity = controls.load_root_identity("default", "owner-1")
    assert identity is not None
    memory_root = user_memory_root(root, "default", "owner-1")
    detached_root = memory_root.with_name(f"memory-detached-{fault_stage}")
    memory_root.rename(detached_root)
    if live_after:
        shutil.copytree(detached_root, memory_root)
    else:
        memory_root.mkdir()

    with pytest.raises(DocumentControlIntegrityError, match="root identity changed"):
        _committer(root)[-1].recover_all("default", "owner-1")

    assert controls.load_root_identity("default", "owner-1") == identity
    if live_after:
        assert (memory_root / path).read_bytes() == raw
        assert (detached_root / path).read_bytes() == raw
    else:
        assert not (memory_root / path).exists()
        assert not (detached_root / path).exists()


def test_create_postinstall_verification_detects_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    source, controls, _, _, committer = _committer(root)
    document_id = new_document_id()
    path = "knowledge/topics/install-root-swap.md"
    raw = render_new_document(document_id, "swap after atomic install")
    durable_create = source.create
    detached_root = user_memory_root(root, "default", "owner-1").with_name(
        "memory-detached-during-install"
    )

    def create_then_swap(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - race injection.
        result = durable_create(*args, **kwargs)
        memory_root = user_memory_root(root, "default", "owner-1")
        memory_root.rename(detached_root)
        shutil.copytree(detached_root, memory_root)
        return result

    monkeypatch.setattr(source, "create", create_then_swap)

    with pytest.raises(DocumentControlIntegrityError, match="root identity changed"):
        committer.commit(
            _plan("install-root-swap", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:install-root-swap",
        )

    identity = controls.load_root_identity("default", "owner-1")
    assert identity is not None
    assert identity.root_identity != source.full_scan("default", "owner-1").root_identity
    assert controls.load_control("default", "owner-1", document_id) is None
    assert len(controls.incomplete_intents("default", "owner-1")) == 1
    assert (user_memory_root(root, "default", "owner-1") / path).read_bytes() == raw
    assert (detached_root / path).read_bytes() == raw


@pytest.mark.parametrize("second_create", ["direct", "external", "adoption"])
def test_missing_identity_intent_cannot_be_backfilled_by_another_document(
    tmp_path: Path,
    second_create: str,
) -> None:
    root = (tmp_path / second_create).resolve()
    root.mkdir()
    source, controls, _, _, crashing = _committer(root, hook=_CrashAt("intent_prepared"))
    first_id = new_document_id()
    first_path = "knowledge/topics/first-prepared.md"
    first_raw = render_new_document(first_id, "first durable intent owns missing authority")
    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("first-prepared", first_id, DocumentEditKind.CREATE, ABSENT, first_path, first_raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:first-prepared",
        )
    first_intents = controls.incomplete_intents("default", "owner-1")
    assert len(first_intents) == 1
    identity_path = root / "system" / "memory-documents" / "owner-1" / "scan-root.json"
    identity_path.unlink()
    blobs_before = tuple(sorted(root.rglob("*.blob")))
    recovered = _committer(root)[-1]
    second_path = f"knowledge/topics/second-{second_create}.md"
    second_raw = b""
    unmanaged = b""
    unmanaged_path = user_memory_root(root, "default", "owner-1") / second_path

    with pytest.raises(DocumentControlIntegrityError, match="document intents"):
        if second_create == "direct":
            second_id = new_document_id()
            second_raw = render_new_document(second_id, "second direct create")
            recovered.commit(
                _plan("second-direct", second_id, DocumentEditKind.CREATE, ABSENT, second_path, second_raw),
                actor_binding="trusted-runtime:owner-1",
                evidence_reference="explicit-command:second-direct",
            )
        elif second_create == "external":
            second_id = new_document_id()
            second_raw = render_new_document(second_id, "second generic external create")
            created = source.create(
                "default",
                "owner-1",
                second_path,
                second_raw,
                expected=ABSENT,
            )
            recovered.record_external_change(
                ExternalDocumentChange(
                    change_kind=ExternalChangeKind.CREATE,
                    tenant_id="default",
                    owner_user_id="owner-1",
                    document_id=second_id,
                    old_relative_path="",
                    new_relative_path=second_path,
                    before_raw_digest="",
                    after_raw_digest=created.raw_sha256,
                    scan_generation_id="scan_second_external",
                )
            )
        else:
            unmanaged = b"# Second adoption\n\nexact unmanaged bytes\n"
            unmanaged_path.parent.mkdir(parents=True, exist_ok=True)
            unmanaged_path.write_bytes(unmanaged)
            request_digest = adoption_request_digest(
                "default",
                "owner-1",
                second_path,
                hashlib.sha256(unmanaged).hexdigest(),
            )
            recovered.preflight_adoption_create(
                "default",
                "owner-1",
                adoption_document_id(request_digest),
            )

    assert not identity_path.exists()
    assert controls.incomplete_intents("default", "owner-1") == first_intents
    assert controls.controls("default", "owner-1") == ()
    assert tuple(sorted(root.rglob("*.blob"))) == blobs_before
    assert source.read_state("default", "owner-1", first_path) == ABSENT
    if second_create == "direct":
        assert source.read_state("default", "owner-1", second_path) == ABSENT
    elif second_create == "external":
        assert source.read_raw("default", "owner-1", relative_path=second_path) == second_raw
    else:
        assert unmanaged_path.read_bytes() == unmanaged
        assert controls.adoption_receipts("default", "owner-1") == ()
    reconciliation = MemoryDocumentScanner(
        source,
        control_store=controls,
        stability_seconds=0,
    ).scan("default", "owner-1", force_stable=True)
    assert reconciliation.deletions_paused is True
    assert reconciliation.confirmed_changes == ()
    assert not identity_path.exists()


def test_concurrent_first_creates_share_one_owner_root_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    start = threading.Barrier(2)
    document_ids = (new_document_id(), new_document_id())

    def create(index: int):  # noqa: ANN202 - compact concurrent worker.
        source, _, _, _, committer = _committer(root)
        document_id = document_ids[index]
        raw = render_new_document(document_id, f"concurrent first create {index}")
        start.wait()
        result = committer.commit(
            _plan(
                f"concurrent-{index}",
                document_id,
                DocumentEditKind.CREATE,
                ABSENT,
                f"knowledge/topics/concurrent-{index}.md",
                raw,
            ),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference=f"explicit-command:concurrent-{index}",
        )
        return source, result

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = tuple(executor.map(create, range(2)))

    controls = MemoryDocumentControlStore(root)
    identity = controls.load_root_identity("default", "owner-1")
    assert identity is not None
    assert all(result.status is DocumentIntentStatus.COMPLETED for _, result in completed)
    assert len(controls.controls("default", "owner-1")) == 2
    assert all(
        source.full_scan("default", "owner-1").root_identity == identity.root_identity
        for source, _ in completed
    )


@pytest.mark.parametrize("bootstrap_status", ["PREPARED", "COMPLETED"])
def test_direct_create_cannot_backfill_missing_bootstrap_identity(
    tmp_path: Path,
    bootstrap_status: str,
) -> None:
    root = (tmp_path / bootstrap_status.lower()).resolve()
    source, controls, _, _, committer = _committer(root)
    source.probe_write_capabilities("default", "owner-1")
    marker = root / "system" / "memory-documents" / "owner-1" / "bootstrap.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema": "memory_document_bootstrap_v1",
                "status": bootstrap_status,
                "tenant_id": "default",
                "owner_user_id": "owner-1",
            }
        ),
        encoding="utf-8",
    )
    document_id = new_document_id()
    path = "knowledge/topics/bootstrap-backfill.md"
    raw = render_new_document(document_id, "bootstrap marker owns initial authority")

    with pytest.raises(DocumentControlIntegrityError, match="bootstrap authority"):
        committer.commit(
            _plan("bootstrap-backfill", document_id, DocumentEditKind.CREATE, ABSENT, path, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:bootstrap-backfill",
        )

    assert controls.load_root_identity("default", "owner-1") is None
    assert controls.incomplete_intents("default", "owner-1") == ()
    assert controls.controls("default", "owner-1") == ()
    assert source.read_state("default", "owner-1", path) == ABSENT
    assert not tuple(root.rglob("*.blob"))


def test_third_state_after_install_is_preserved_and_intent_becomes_conflicted(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, _, _, _, committer = _committer(root)
    document_id = new_document_id()
    path = "knowledge/topics/user-edit.md"
    before = render_new_document(document_id, "before")
    after = render_new_document(document_id, "system after")
    external = render_new_document(document_id, "external editor wins")
    committer.commit(
        _plan("create", document_id, DocumentEditKind.CREATE, ABSENT, path, before),
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:create",
    )
    expected = source.read_state("default", "owner-1", path)
    crashing = MemoryDocumentCommitter(
        source,
        MemoryDocumentControlStore(root),
        MemoryDocumentRevisionStore(root),
        cast(QueueStore, _FileProjectionQueue(root)),
        test_hook=_CrashAt("after_installed"),
    )

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("update-crash", document_id, DocumentEditKind.UPDATE, expected, path, after),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="archive:update",
        )
    installed = source.read_state("default", "owner-1", path)
    source.replace("default", "owner-1", document_id, external, expected_state=installed)

    recovered_source, controls, revisions, _, recovered = _committer(root)
    report = recovered.recover_all("default", "owner-1")

    assert report.completed == ()
    assert len(report.conflicted_intent_ids) == 1
    assert recovered_source.read_raw("default", "owner-1", relative_path=path) == external
    intent = controls.load_intent("default", "owner-1", report.conflicted_intent_ids[0])
    assert intent is not None and intent.status == DocumentIntentStatus.CONFLICTED
    assert revisions.latest_revision("default", "owner-1", document_id) == 1


def test_rename_partial_vector_is_conflicted_without_deleting_either_path(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, _, _, committer = _committer(root)
    document_id = new_document_id()
    old_path = "knowledge/topics/old.md"
    new_path = "knowledge/topics/new.md"
    raw = render_new_document(document_id, "rename body")
    committer.commit(
        _plan("create", document_id, DocumentEditKind.CREATE, ABSENT, old_path, raw),
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:create",
    )
    expected = source.read_state("default", "owner-1", old_path)
    rename_plan = DocumentEditPlan(
        idempotency_key="rename-crash",
        tenant_id="default",
        owner_user_id="owner-1",
        edit_kind=DocumentEditKind.RENAME,
        expected_state=expected,
        evidence_digest="f" * 64,
        edit_summary="rename crash test",
        document_id=document_id,
        relative_path=old_path,
        new_relative_path=new_path,
        expected_registration_document_id=document_id,
    )
    crashing = MemoryDocumentCommitter(
        source,
        controls,
        MemoryDocumentRevisionStore(root),
        cast(QueueStore, _FileProjectionQueue(root)),
        test_hook=_CrashAt("intent_prepared"),
    )
    with pytest.raises(_InjectedCrash):
        crashing.commit(
            rename_plan,
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:rename",
        )

    external_target = root / "tenants" / "default" / "users" / "owner-1" / "memory" / new_path
    external_target.parent.mkdir(parents=True, exist_ok=True)
    external_target.write_bytes(raw)
    report = _committer(root)[-1].recover_all("default", "owner-1")

    assert len(report.conflicted_intent_ids) == 1
    assert source.read_raw("default", "owner-1", relative_path=old_path) == raw
    assert source.read_raw("default", "owner-1", relative_path=new_path) == raw


def test_after_blob_crash_prunes_unreferenced_plaintext_on_recovery(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, _, queue, crashing = _committer(root, hook=_CrashAt("after_blob_fsynced"))
    document_id = new_document_id()
    relative = "knowledge/topics/orphan-blob.md"
    raw = render_new_document(document_id, "ORPHAN_SECRET")

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan("orphan-blob", document_id, DocumentEditKind.CREATE, ABSENT, relative, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:orphan-blob",
        )
    assert controls.incomplete_intents("default", "owner-1") == ()
    assert len(tuple(root.rglob("*.blob"))) == 1
    assert queue.payloads() == ()

    report = _committer(root)[-1].recover_all("default", "owner-1")

    assert report.completed == ()
    assert report.conflicted_intent_ids == ()
    assert tuple(root.rglob("*.blob")) == ()
    assert source.read_state("default", "owner-1", relative) == ABSENT


@pytest.mark.parametrize(
    "fault_stage",
    ["temp_file_fsynced", "atomic_installed", "parent_fsynced"],
)
def test_create_store_crash_windows_cleanup_temp_and_roll_forward(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    root = (tmp_path / fault_stage).resolve()
    root.mkdir()
    _, controls, _, _, crashing = _committer(root, hook=_CrashAt(fault_stage))
    document_id = new_document_id()
    relative = "knowledge/topics/store-crash.md"
    raw = render_new_document(document_id, f"store crash {fault_stage}")

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            _plan(f"store-{fault_stage}", document_id, DocumentEditKind.CREATE, ABSENT, relative, raw),
            actor_binding="trusted-runtime:owner-1",
            evidence_reference=f"explicit-command:{fault_stage}",
        )
    assert len(controls.incomplete_intents("default", "owner-1")) == 1

    recovered_source, _, _, _, recovered = _committer(root)
    report = recovered.recover_all("default", "owner-1")

    assert len(report.completed) == 1
    assert report.conflicted_intent_ids == ()
    assert recovered_source.read_raw("default", "owner-1", relative_path=relative) == raw
    assert tuple(user_memory_root(root, "default", "owner-1").rglob("*.tmp")) == ()


def test_crash_after_control_completion_retries_as_completed(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _, controls, _, _, crashing = _committer(root, hook=_CrashAt("completed"))
    document_id = new_document_id()
    relative = "knowledge/topics/completed-crash.md"
    raw = render_new_document(document_id, "completed crash")
    plan = _plan("completed-crash", document_id, DocumentEditKind.CREATE, ABSENT, relative, raw)

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            plan,
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:completed-crash",
        )
    assert controls.incomplete_intents("default", "owner-1") == ()

    source, _, _, _, recovered = _committer(root)
    result = recovered.commit(
        plan,
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:completed-crash",
    )

    assert result.status is DocumentIntentStatus.COMPLETED
    assert result.recovered is True
    assert source.read_raw("default", "owner-1", relative_path=relative) == raw


@pytest.mark.parametrize(
    ("operation", "fault_stage"),
    [
        ("update", "temp_file_fsynced"),
        ("update", "atomic_installed"),
        ("update", "parent_fsynced"),
        ("delete", "atomic_installed"),
        ("delete", "parent_fsynced"),
        ("rename", "atomic_installed"),
        ("rename", "parent_fsynced"),
        ("rename_edit", "temp_file_fsynced"),
        ("rename_edit", "atomic_installed"),
        ("rename_edit", "parent_fsynced"),
    ],
)
def test_existing_document_store_crash_windows_roll_forward(
    tmp_path: Path,
    operation: str,
    fault_stage: str,
) -> None:
    root = (tmp_path / f"{operation}-{fault_stage}").resolve()
    root.mkdir()
    source, controls, revisions, queue, initial = _committer(root)
    document_id = new_document_id()
    old_path = "knowledge/topics/existing.md"
    new_path = "knowledge/topics/renamed.md"
    before = render_new_document(document_id, "before")
    after = render_new_document(document_id, "after")
    initial.commit(
        _plan("initial", document_id, DocumentEditKind.CREATE, ABSENT, old_path, before),
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:initial",
    )
    expected = source.read_state("default", "owner-1", old_path)
    if operation == "update":
        plan = _plan("crash-update", document_id, DocumentEditKind.UPDATE, expected, old_path, after)
    elif operation == "delete":
        plan = _plan("crash-delete", document_id, DocumentEditKind.DELETE, expected, old_path, None)
    else:
        plan = DocumentEditPlan(
            idempotency_key=f"crash-{operation}",
            tenant_id="default",
            owner_user_id="owner-1",
            edit_kind=DocumentEditKind.RENAME,
            expected_state=expected,
            evidence_digest="f" * 64,
            edit_summary="rename crash matrix",
            document_id=document_id,
            relative_path=old_path,
            new_relative_path=new_path,
            after_bytes=after if operation == "rename_edit" else None,
            expected_registration_document_id=document_id,
        )
    crashing = MemoryDocumentCommitter(
        source,
        controls,
        revisions,
        cast(QueueStore, queue),
        test_hook=_CrashAt(fault_stage),
    )

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            plan,
            actor_binding="trusted-runtime:owner-1",
            evidence_reference=f"explicit-command:{operation}",
        )

    recovered_source, _, _, _, recovered = _committer(root)
    report = recovered.recover_all("default", "owner-1")

    assert len(report.completed) == 1
    assert report.conflicted_intent_ids == ()
    if operation == "update":
        assert recovered_source.read_raw("default", "owner-1", relative_path=old_path) == after
    elif operation == "delete":
        assert recovered_source.read_state("default", "owner-1", old_path) == ABSENT
    else:
        assert recovered_source.read_state("default", "owner-1", old_path) == ABSENT
        expected_after = after if operation == "rename_edit" else before
        assert recovered_source.read_raw("default", "owner-1", relative_path=new_path) == expected_after
    assert tuple(user_memory_root(root, "default", "owner-1").rglob("*.tmp")) == ()


def test_rename_edit_partial_target_is_third_state_and_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, revisions, queue, initial = _committer(root)
    document_id = new_document_id()
    old_path = "knowledge/topics/rename-edit-old.md"
    new_path = "knowledge/entities/rename-edit-new.md"
    before = render_new_document(document_id, "before rename edit")
    after = render_new_document(document_id, "after rename edit")
    initial.commit(
        _plan("rename-edit-initial", document_id, DocumentEditKind.CREATE, ABSENT, old_path, before),
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:rename-edit-initial",
    )
    plan = DocumentEditPlan(
        idempotency_key="rename-edit-partial",
        tenant_id="default",
        owner_user_id="owner-1",
        edit_kind=DocumentEditKind.RENAME,
        expected_state=source.read_state("default", "owner-1", old_path),
        evidence_digest="f" * 64,
        edit_summary="rename and edit partial crash",
        document_id=document_id,
        relative_path=old_path,
        new_relative_path=new_path,
        after_bytes=after,
        expected_registration_document_id=document_id,
    )
    crashing = MemoryDocumentCommitter(
        source,
        controls,
        revisions,
        cast(QueueStore, queue),
        test_hook=_CrashAt("rename_target_installed"),
    )

    with pytest.raises(_InjectedCrash):
        crashing.commit(
            plan,
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:rename-edit-partial",
        )
    assert source.read_raw("default", "owner-1", relative_path=old_path) == before
    assert source.read_raw("default", "owner-1", relative_path=new_path) == after

    report = _committer(root)[-1].recover_all("default", "owner-1")

    assert report.completed == ()
    assert len(report.conflicted_intent_ids) == 1
    recovered_source = _committer(root)[0]
    assert recovered_source.read_raw("default", "owner-1", relative_path=old_path) == before
    assert recovered_source.read_raw("default", "owner-1", relative_path=new_path) == after


def test_rename_edit_target_conflict_creates_no_intent_or_recovery_job(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source, controls, _, queue, committer = _committer(root)
    document_id = new_document_id()
    occupied_id = new_document_id()
    old_path = "knowledge/topics/rename-conflict-old.md"
    new_path = "knowledge/entities/rename-conflict-new.md"
    before = render_new_document(document_id, "before conflict")
    occupied = render_new_document(occupied_id, "occupied target")
    after = render_new_document(document_id, "must not overwrite")
    committer.commit(
        _plan("rename-conflict-initial", document_id, DocumentEditKind.CREATE, ABSENT, old_path, before),
        actor_binding="trusted-runtime:owner-1",
        evidence_reference="explicit-command:rename-conflict-initial",
    )
    source.create("default", "owner-1", new_path, occupied, expected=ABSENT)
    intent_count = len(controls.intents("default", "owner-1"))
    plan = DocumentEditPlan(
        idempotency_key="rename-edit-target-conflict",
        tenant_id="default",
        owner_user_id="owner-1",
        edit_kind=DocumentEditKind.RENAME,
        expected_state=source.read_state("default", "owner-1", old_path),
        evidence_digest="f" * 64,
        edit_summary="rename target conflict",
        document_id=document_id,
        relative_path=old_path,
        new_relative_path=new_path,
        after_bytes=after,
        expected_registration_document_id=document_id,
    )

    with pytest.raises(DocumentConflictError):
        committer.commit(
            plan,
            actor_binding="trusted-runtime:owner-1",
            evidence_reference="explicit-command:rename-target-conflict",
        )

    assert len(controls.intents("default", "owner-1")) == intent_count
    assert source.read_raw("default", "owner-1", relative_path=old_path) == before
    assert source.read_raw("default", "owner-1", relative_path=new_path) == occupied
    assert not [
        payload
        for payload in queue.payloads()
        if payload["queue_name"] == "memory_document_edit"
    ]
