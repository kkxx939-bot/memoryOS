from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.adapters.persistence.in_memory.queue_store import InMemoryQueueStore
from memoryos.memory.documents import (
    ABSENT,
    DocumentControlIntegrityError,
    DocumentControlRecord,
    DocumentUnsafeError,
    ExternalChangeKind,
    ExternalDocumentChange,
    MemoryCandidateKind,
    MemoryDocumentCommitter,
    MemoryDocumentContextOverlay,
    MemoryDocumentControlStore,
    MemoryDocumentPlanner,
    MemoryDocumentProjector,
    MemoryDocumentRevisionStore,
    MemoryDocumentScanner,
    MemoryEditProposal,
    QuarantinedDocument,
    explicit_evidence_digest,
    new_document_id,
    render_new_document,
)


def _proposal(kind: MemoryCandidateKind = MemoryCandidateKind.TOPIC_NOTE) -> MemoryEditProposal:
    return MemoryEditProposal(
        candidate_kind=kind,
        title="File memory",
        subject="File memory",
        body="Markdown is the live source of truth.",
        evidence_refs=("memoryos://user/u1/sessions/s1/events/e1",),
        occurred_at=datetime(2026, 7, 17, tzinfo=timezone.utc).isoformat(),
    )


def test_control_snapshot_enumeration_rejects_unexpected_artifacts(tmp_path) -> None:
    controls = MemoryDocumentControlStore(tmp_path)
    directory = tmp_path / "system" / "memory-documents" / "u1" / "documents"
    directory.mkdir(parents=True)
    (directory / "unexpected.tmp").write_bytes(b"must not be ignored")

    with pytest.raises(DocumentControlIntegrityError, match="unexpected artifact"):
        controls.controls("default", "u1")


def test_scanner_blocks_missing_identity_even_when_only_deleted_controls_remain(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    document_id = new_document_id()
    controls.write_control(
        DocumentControlRecord(
            tenant_id="default",
            owner_user_id="u1",
            document_id=document_id,
            relative_path="knowledge/topics/deleted.md",
            raw_sha256="",
            size=0,
            logical_revision=1,
            projection_generation=1,
            status="deleted",
            last_event_id=f"memchg_{'a' * 64}",
            updated_at=datetime(2026, 7, 18, tzinfo=timezone.utc).isoformat(),
        )
    )
    published: list[ExternalDocumentChange] = []
    scanner = MemoryDocumentScanner(
        store,
        control_store=controls,
        stability_seconds=0,
        change_publisher=published.append,
    )

    first = scanner.scan("default", "u1", force_stable=True)
    second = scanner.scan("default", "u1", force_stable=True)

    assert first.deletions_paused is True
    assert "root identity is missing" in first.pause_reason
    assert second.deletions_paused is True
    assert first.confirmed_changes == second.confirmed_changes == ()
    assert published == []
    assert controls.load_root_identity("default", "u1") is None


@pytest.mark.parametrize("authority", ["prepared_bootstrap", "completed_bootstrap", "receipt"])
def test_scanner_never_backfills_noncontrol_authority_without_root_identity(
    tmp_path,
    authority: str,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    store.probe_write_capabilities("default", "u1")
    if authority.endswith("bootstrap"):
        marker = tmp_path / "system" / "memory-documents" / "u1" / "bootstrap.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "schema": "memory_document_bootstrap_v1",
                    "status": "PREPARED" if authority.startswith("prepared") else "COMPLETED",
                    "tenant_id": "default",
                    "owner_user_id": "u1",
                }
            ),
            encoding="utf-8",
        )
    else:
        scan = store.full_scan("default", "u1")
        controls.ensure_root_identity("default", "u1", scan.root_identity)
        controls.prepare_adoption_receipt(
            "default",
            "u1",
            "knowledge/topics/receipt.md",
            "a" * 64,
            actor_binding="trusted:user:u1:u1",
        )
        (tmp_path / "system" / "memory-documents" / "u1" / "scan-root.json").unlink()

    scanner = MemoryDocumentScanner(store, control_store=controls, stability_seconds=0)
    result = scanner.scan("default", "u1", force_stable=True)

    assert result.deletions_paused is True
    assert result.confirmed_changes == ()
    assert "root identity is missing" in result.pause_reason
    assert controls.load_root_identity("default", "u1") is None


def test_router_and_planner_create_then_deterministic_noop(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    planner = MemoryDocumentPlanner(store)
    proposal = _proposal()
    digest = explicit_evidence_digest("evidence")

    create = planner.plan(
        proposal,
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="plan-1",
        evidence_digest=digest,
    )
    assert create.relative_path == "knowledge/topics/file-memory.md"
    assert create.expected_state == ABSENT
    document = store.create(
        "default",
        "u1",
        create.relative_path,
        create.after_bytes or b"",
        expected=create.expected_state,
    )

    update = planner.replan(
        proposal,
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="plan-1",
        evidence_digest=digest,
    )
    assert update.document_id == document.document_id
    assert update.after_bytes == document.raw_bytes


def test_projection_is_bounded_and_block_identity_uses_source_digest(tmp_path) -> None:
    document_id = new_document_id()
    raw = render_new_document(document_id, "# Topic\n\nIntro\n\n## Detail\n\nBody\n")
    first_digest = hashlib.sha256(raw).hexdigest()
    projector = MemoryDocumentProjector(max_blocks=2)
    first = projector.project(
        tenant_id="default",
        owner_user_id="u1",
        relative_path="knowledge/topics/topic.md",
        raw_bytes=raw,
        source_digest=first_digest,
        document_revision=1,
        projection_generation=1,
    )
    second = projector.project(
        tenant_id="default",
        owner_user_id="u1",
        relative_path="knowledge/topics/topic.md",
        raw_bytes=raw,
        source_digest="f" * 64,
        document_revision=2,
        projection_generation=2,
    )
    assert first.title == "Topic"
    assert len(first.blocks) == 2
    assert first.blocks[0].block_id != second.blocks[0].block_id


def test_scanner_requires_stability_and_temp_missing_does_not_delete(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    raw = render_new_document(document_id, "# Topic\n")
    store.create("default", "u1", "knowledge/topics/topic.md", raw, expected=ABSENT)
    now = [0.0]
    scanner = MemoryDocumentScanner(store, stability_seconds=5, clock=lambda: now[0])
    initial = scanner.scan("default", "u1", force_stable=True)
    assert initial.confirmed_changes[0].change_kind is ExternalChangeKind.CREATE

    path = tmp_path / "tenants" / "default" / "users" / "u1" / "memory" / "knowledge" / "topics" / "topic.md"
    original = path.read_bytes()
    path.unlink()
    missing = scanner.scan("default", "u1")
    assert missing.confirmed_changes == ()
    assert missing.pending_change_count == 1
    path.write_bytes(original)
    restored = scanner.scan("default", "u1")
    assert restored.confirmed_changes == ()
    assert restored.pending_change_count == 0


def test_scanner_external_update_and_overlay_reject_stale_digest(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    raw = render_new_document(document_id, "# Topic\n\nBefore\n")
    document = store.create("default", "u1", "knowledge/topics/topic.md", raw, expected=ABSENT)
    scanner = MemoryDocumentScanner(store, stability_seconds=0)
    scanner.scan("default", "u1")
    path = tmp_path / "tenants" / "default" / "users" / "u1" / "memory" / "knowledge" / "topics" / "topic.md"
    path.write_bytes(render_new_document(document_id, "# Topic\n\nAfter\n"))
    changed = scanner.scan("default", "u1")
    assert changed.confirmed_changes[0].change_kind is ExternalChangeKind.UPDATE

    overlay = MemoryDocumentContextOverlay(store)
    try:
        overlay.read(
            tenant_id="default",
            owner_user_id="u1",
            document_uri=document.uri,
            relative_path=document.relative_path,
            expected_source_digest=document.raw_sha256,
        )
    except Exception as exc:  # exact class is part of the storage boundary.
        assert type(exc).__name__ == "DocumentConflictError"
    else:  # pragma: no cover
        raise AssertionError("stale catalog projection must not hydrate live Markdown")


def test_scanner_journals_external_create_and_update_without_overwriting(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    control = MemoryDocumentControlStore(tmp_path)
    revisions = MemoryDocumentRevisionStore(tmp_path)
    queue = InMemoryQueueStore()
    committer = MemoryDocumentCommitter(store, control, revisions, queue)
    document_id = new_document_id()
    relative_path = "knowledge/topics/external.md"
    original = render_new_document(document_id, "# External\n\nBefore\n")
    store.create("default", "u1", relative_path, original, expected=ABSENT)

    committed = []

    def publish_external_change(change: ExternalDocumentChange) -> None:
        committed.append(committer.record_external_change(change))

    scanner = MemoryDocumentScanner(
        store,
        stability_seconds=0,
        change_publisher=publish_external_change,
    )

    created = scanner.scan("default", "u1", force_stable=True)
    assert created.confirmed_changes[0].change_kind is ExternalChangeKind.CREATE
    first = control.load_control("default", "u1", document_id)
    assert first is not None and first.logical_revision == 1
    assert committed[0] is not None
    assert committed[0].event is not None
    assert committed[0].event.actor_binding == "external-editor:stable-full-scan"
    assert committed[0].event.evidence_reference.startswith("scan-generation:")

    path = tmp_path / "tenants" / "default" / "users" / "u1" / "memory" / relative_path
    updated = render_new_document(document_id, "# External\n\nAfter\n")
    path.write_bytes(updated)
    changed = scanner.scan("default", "u1", force_stable=True)

    assert changed.confirmed_changes[0].change_kind is ExternalChangeKind.UPDATE
    second = control.load_control("default", "u1", document_id)
    assert second is not None and second.logical_revision == 2
    assert second.raw_sha256 == hashlib.sha256(updated).hexdigest()
    assert revisions.read_revision_blob("default", "u1", document_id, 1) == original
    assert revisions.read_revision_blob("default", "u1", document_id, 2) == updated
    assert len(queue.jobs) == 2


def test_scanner_journals_external_rename_then_stable_delete(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    control = MemoryDocumentControlStore(tmp_path)
    revisions = MemoryDocumentRevisionStore(tmp_path)
    queue = InMemoryQueueStore()
    committer = MemoryDocumentCommitter(store, control, revisions, queue)
    document_id = new_document_id()
    old_relative = "knowledge/topics/external-rename.md"
    new_relative = "knowledge/entities/external-rename.md"
    raw = render_new_document(document_id, "# External rename\n\nExact body\n")
    store.create("default", "u1", old_relative, raw, expected=ABSENT)

    committed = []

    def publish_external_change(change: ExternalDocumentChange) -> None:
        committed.append(committer.record_external_change(change))

    scanner = MemoryDocumentScanner(
        store,
        stability_seconds=0,
        change_publisher=publish_external_change,
    )
    scanner.scan("default", "u1", force_stable=True)
    memory_root = tmp_path / "tenants" / "default" / "users" / "u1" / "memory"
    old_path = memory_root / old_relative
    new_path = memory_root / new_relative
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.rename(new_path)

    renamed = scanner.scan("default", "u1", force_stable=True)

    assert len(renamed.confirmed_changes) == 1
    assert renamed.confirmed_changes[0].change_kind is ExternalChangeKind.RENAME
    rename_control = control.load_control("default", "u1", document_id)
    assert rename_control is not None
    assert rename_control.status == "present"
    assert rename_control.relative_path == new_relative
    assert rename_control.logical_revision == 2
    assert revisions.read_revision_blob("default", "u1", document_id, 2) == raw

    new_path.unlink()
    first_missing = scanner.scan("default", "u1", force_stable=True)
    assert first_missing.confirmed_changes == ()
    assert first_missing.pending_change_count == 1

    deleted = scanner.scan("default", "u1", force_stable=True)

    assert len(deleted.confirmed_changes) == 1
    assert deleted.confirmed_changes[0].change_kind is ExternalChangeKind.DELETE
    delete_control = control.load_control("default", "u1", document_id)
    assert delete_control is not None
    assert delete_control.status == "deleted"
    assert delete_control.logical_revision == 3
    assert revisions.read_revision_blob("default", "u1", document_id, 3) == raw
    assert len(committed) == 3
    assert len(queue.jobs) == 3


def test_scanner_copy_with_duplicate_id_never_emits_rename_or_delete(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    original_relative = "knowledge/topics/original.md"
    copied_relative = "knowledge/entities/copied.md"
    raw = render_new_document(document_id, "# Duplicate copy\n")
    store.create("default", "u1", original_relative, raw, expected=ABSENT)
    published: list[ExternalDocumentChange] = []
    scanner = MemoryDocumentScanner(
        store,
        stability_seconds=0,
        change_publisher=published.append,
    )
    scanner.scan("default", "u1", force_stable=True)
    published.clear()

    memory_root = tmp_path / "tenants" / "default" / "users" / "u1" / "memory"
    copied_path = memory_root / copied_relative
    copied_path.parent.mkdir(parents=True, exist_ok=True)
    copied_path.write_bytes(raw)

    first = scanner.scan("default", "u1", force_stable=True)
    second = scanner.scan("default", "u1", force_stable=True)

    assert first.deletions_paused is True
    assert second.deletions_paused is True
    assert first.confirmed_changes == second.confirmed_changes == ()
    assert published == []
    quarantined = tuple(
        item for item in first.generation.registrations if isinstance(item, QuarantinedDocument)
    )
    assert len(quarantined) == 2
    assert all("duplicate document_id" in item.reason for item in quarantined)
    assert (memory_root / original_relative).read_bytes() == raw
    assert copied_path.read_bytes() == raw


def test_force_stable_never_turns_one_missing_observation_into_delete(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/topics/delete-window.md"
    raw = render_new_document(document_id, "delete window")
    store.create("default", "u1", relative, raw, expected=ABSENT)
    now = [0.0]
    scanner = MemoryDocumentScanner(store, stability_seconds=5, clock=lambda: now[0])
    scanner.scan("default", "u1", force_stable=True)
    path = tmp_path / "tenants" / "default" / "users" / "u1" / "memory" / relative
    path.unlink()

    first = scanner.scan("default", "u1", force_stable=True)
    assert first.confirmed_changes == ()
    assert first.pending_change_count == 1
    now[0] = 4.9
    second = scanner.scan("default", "u1", force_stable=True)
    assert second.confirmed_changes == ()
    now[0] = 5.0
    third = scanner.scan("default", "u1")

    assert len(third.confirmed_changes) == 1
    assert third.confirmed_changes[0].change_kind is ExternalChangeKind.DELETE


def test_scanner_does_not_publish_identity_for_absent_owner_root(tmp_path) -> None:
    controls = MemoryDocumentControlStore(tmp_path)
    scanner = MemoryDocumentScanner(
        FileSystemMemoryDocumentStore(tmp_path),
        control_store=controls,
        stability_seconds=0,
    )

    result = scanner.scan("default", "u1", force_stable=True)

    assert result.generation.complete is True
    assert result.generation.root_identity == ""
    assert controls.load_root_identity("default", "u1") is None


def test_restart_seeds_durable_path_id_and_quarantines_same_path_changed_id(tmp_path) -> None:
    first_store = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    original_id = new_document_id()
    replacement_id = new_document_id()
    relative = "knowledge/topics/changed-id.md"
    original = render_new_document(original_id, "original")
    created = first_store.create("default", "u1", relative, original, expected=ABSENT)
    initial_scan = first_store.full_scan("default", "u1")
    controls.ensure_root_identity("default", "u1", initial_scan.root_identity)
    controls.write_control(
        DocumentControlRecord(
            tenant_id="default",
            owner_user_id="u1",
            document_id=original_id,
            relative_path=relative,
            raw_sha256=created.raw_sha256,
            size=created.size,
            logical_revision=1,
            projection_generation=1,
            status="present",
            last_event_id=f"memchg_{'b' * 64}",
            updated_at=datetime(2026, 7, 18, tzinfo=timezone.utc).isoformat(),
        )
    )
    path = tmp_path / "tenants" / "default" / "users" / "u1" / "memory" / relative
    path.write_bytes(render_new_document(replacement_id, "replacement"))

    restarted = MemoryDocumentScanner(
        FileSystemMemoryDocumentStore(tmp_path),
        control_store=controls,
        stability_seconds=0,
    ).scan("default", "u1", force_stable=True)

    assert restarted.deletions_paused is True
    assert restarted.confirmed_changes == ()
    registration = restarted.generation.registrations[0]
    assert isinstance(registration, QuarantinedDocument)
    assert "document_id changed" in registration.reason


def test_scanner_pauses_mass_delete_before_any_delete_is_confirmed(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    paths: list[str] = []
    for index in range(2):
        relative = f"knowledge/topics/mass-{index}.md"
        store.create(
            "default",
            "u1",
            relative,
            render_new_document(new_document_id(), f"mass {index}"),
            expected=ABSENT,
        )
        paths.append(relative)
    scanner = MemoryDocumentScanner(store, stability_seconds=0, mass_delete_threshold=2)
    scanner.scan("default", "u1", force_stable=True)
    root = tmp_path / "tenants" / "default" / "users" / "u1" / "memory"
    for relative in paths:
        (root / relative).unlink()

    result = scanner.scan("default", "u1", force_stable=True)

    assert result.deletions_paused is True
    assert "mass-delete" in result.pause_reason
    assert result.confirmed_changes == ()


def test_scanner_overflow_hint_is_reconciled_by_a_full_scan(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/topics/overflow.md"
    original = render_new_document(document_id, "before overflow")
    store.create("default", "u1", relative, original, expected=ABSENT)
    scanner = MemoryDocumentScanner(store, stability_seconds=0)
    scanner.scan("default", "u1", force_stable=True)
    path = tmp_path / "tenants" / "default" / "users" / "u1" / "memory" / relative
    path.write_bytes(render_new_document(document_id, "after overflow"))

    scanner.notify("default", "u1", overflow=True)
    result = scanner.scan("default", "u1")

    assert len(result.confirmed_changes) == 1
    assert result.confirmed_changes[0].change_kind is ExternalChangeKind.UPDATE


def test_scanner_pauses_when_bound_root_is_temporarily_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/topics/unavailable.md"
    store.create(
        "default",
        "u1",
        relative,
        render_new_document(document_id, "unavailable"),
        expected=ABSENT,
    )
    scanner = MemoryDocumentScanner(store, stability_seconds=0)
    scanner.scan("default", "u1", force_stable=True)

    def unavailable(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise DocumentUnsafeError("root temporarily unavailable")

    monkeypatch.setattr(store, "_open_user_root", unavailable)
    result = scanner.scan("default", "u1", force_stable=True)

    assert result.deletions_paused is True
    assert result.confirmed_changes == ()
    assert result.generation.complete is False
    assert result.generation.root_identity == ""
