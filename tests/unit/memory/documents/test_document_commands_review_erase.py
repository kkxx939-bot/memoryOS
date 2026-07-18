from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import cast

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.adapters.persistence.in_memory.queue_store import InMemoryQueueStore
from memoryos.application.memory.command_service import MemoryCommandService
from memoryos.application.memory.pending_review_service import MemoryEditReviewService
from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.core.readiness import RuntimeNotReadyError, RuntimeReadiness, RuntimeReadinessState
from memoryos.memory.documents import (
    ABSENT,
    DocumentConflictError,
    DocumentControlIntegrityError,
    DocumentDeletionStatus,
    DocumentEditKind,
    DocumentEditPlan,
    DocumentErasedError,
    DocumentEraseStatus,
    MemoryDocumentBootstrapper,
    MemoryDocumentCommitter,
    MemoryDocumentControlStore,
    MemoryDocumentEraser,
    MemoryDocumentPlanner,
    MemoryDocumentRevisionStore,
    MemoryDocumentScanner,
    MemoryEditReviewStatus,
    MemoryEditReviewStore,
    PresentPath,
    adopt_raw_document,
    new_document_id,
    parse_front_matter,
    render_new_document,
)
from memoryos.memory.documents.layout import user_memory_root
from memoryos.runtime.container import _publish_external_change
from memoryos.security.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    DEFAULT_AGENT_CAPABILITIES,
    HARD_ERASE_MEMORY,
    KNOWN_CAPABILITIES,
    READ_CONTEXT,
    TrustedRequestContext,
)
from memoryos.workers.memory_document_edit_worker import MemoryDocumentEditWorker


class _Queue:
    def __init__(self) -> None:
        self.jobs: dict[str, QueueJob] = {}

    def enqueue(self, job: QueueJob) -> QueueJob:
        self.jobs.setdefault(job.job_id, job)
        return self.jobs[job.job_id]


class _CleanupBackend:
    name = "vector"

    def __init__(self, acknowledgements: list[bool]) -> None:
        self.acknowledgements = acknowledgements
        self.requests = []

    def erase_document(self, request):  # noqa: ANN001, ANN201 - compact protocol test double.
        self.requests.append(request)
        return self.acknowledgements.pop(0)


def _caller(*, hard_erase: bool = True) -> TrustedRequestContext:
    capabilities = {READ_CONTEXT, AUTHORITATIVE_REMEMBER, AUTHORITATIVE_FORGET}
    if hard_erase:
        capabilities.add(HARD_ERASE_MEMORY)
    return TrustedRequestContext(
        tenant_id="default",
        user_id="user-a",
        actor_kind="user",
        actor_id="user-a",
        capabilities=frozenset(capabilities),
    )


def _components(root: Path, *, backend: _CleanupBackend | None = None):  # noqa: ANN202
    source = FileSystemMemoryDocumentStore(root)
    controls = MemoryDocumentControlStore(root)
    revisions = MemoryDocumentRevisionStore(root)
    reviews = MemoryEditReviewStore(root)
    committer = MemoryDocumentCommitter(source, controls, revisions, cast(QueueStore, _Queue()))
    eraser = MemoryDocumentEraser(
        source,
        controls,
        revisions,
        review_store=reviews,
        cleanup_backends=(backend,) if backend is not None else (),
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(source),
        committer,
        eraser,
        independent_evidence_locator=lambda *_args: ("memoryos://user/user-a/sessions/history/session-independent",),
    )
    review_service = MemoryEditReviewService(reviews, committer, erasure_store=eraser.erase_store)
    return source, controls, revisions, reviews, commands, review_service, eraser


def test_document_commands_reject_mutation_until_runtime_is_ready(tmp_path: Path) -> None:
    source, _, _, _, commands, _, _ = _components(tmp_path)
    readiness = RuntimeReadiness()
    commands.readiness = readiness

    with pytest.raises(RuntimeNotReadyError, match="STARTING"):
        commands.remember("must not be written during recovery", caller=_caller())
    assert source.full_scan("default", "user-a").registrations == ()

    readiness.transition(RuntimeReadinessState.READY)
    assert commands.remember("safe after recovery", caller=_caller()).changed is True


def test_explicit_rename_preserves_document_identity_and_requires_absent_target(tmp_path: Path) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    remembered = commands.remember("Rename source body", target_hint="topic:rename-source", caller=caller)
    target = "knowledge/entities/renamed-source.md"

    renamed = commands.rename_memory_document(
        remembered.document_uri,
        target,
        remembered.source_digest,
        caller=caller,
    )

    assert renamed.changed is True
    assert renamed.document_id == remembered.document_id
    assert renamed.document_uri == remembered.document_uri
    assert renamed.relative_path == target
    assert renamed.source_digest == remembered.source_digest
    assert renamed.document_revision > remembered.document_revision
    assert source.read_state("default", "user-a", remembered.relative_path) == ABSENT
    assert b"Rename source body" in source.read_raw("default", "user-a", document_id=remembered.document_id)
    control = controls.load_control("default", "user-a", remembered.document_id)
    assert control is not None and control.relative_path == target

    occupied = commands.remember("Occupied target", target_hint="topic:occupied", caller=caller)
    with pytest.raises(DocumentConflictError, match="target must be ABSENT"):
        commands.rename_memory_document(
            remembered.document_uri,
            occupied.relative_path,
            renamed.source_digest,
            caller=caller,
        )


def test_rename_and_content_edit_is_one_replayable_document_effect(tmp_path: Path) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    remembered = commands.remember(
        "Old body before combined rename edit",
        target_hint="topic:rename-edit-source",
        caller=caller,
    )
    target = "knowledge/entities/rename-edit-target.md"

    renamed = commands.rename_memory_document(
        remembered.document_uri,
        target,
        remembered.source_digest,
        edit="New body installed with the rename",
        caller=caller,
    )

    assert renamed.document_id == remembered.document_id
    assert renamed.document_uri == remembered.document_uri
    assert renamed.relative_path == target
    assert renamed.source_digest != remembered.source_digest
    assert renamed.edit_summary == "rename and edit memory document"
    assert source.read_state("default", "user-a", remembered.relative_path) == ABSENT
    assert b"New body installed with the rename" in source.read_raw(
        "default",
        "user-a",
        document_id=remembered.document_id,
    )
    assert b"Old body before combined rename edit" not in source.read_raw(
        "default",
        "user-a",
        document_id=remembered.document_id,
    )
    rename_job = next(
        job
        for job in cast(_Queue, commands.committer.projection_queue).jobs.values()
        if job.payload["edit_kind"] == "rename"
    )
    intent = controls.load_intent("default", "user-a", str(rename_job.payload["intent_id"]))
    assert intent is not None
    event = controls.load_event(intent)
    assert event is not None
    assert event.edit_kind is DocumentEditKind.RENAME
    assert event.old_relative_path == remembered.relative_path
    assert event.new_relative_path == target
    assert event.before_raw_digest == remembered.source_digest
    assert event.after_raw_digest == renamed.source_digest

    retried = commands.rename_memory_document(
        remembered.document_uri,
        target,
        remembered.source_digest,
        edit="New body installed with the rename",
        caller=caller,
    )
    assert retried == renamed


def test_adopt_unmanaged_document_binds_caller_and_enqueues_content_free_create(tmp_path: Path) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    relative_path = "knowledge/topics/user-note.md"
    original = b"# User note\r\n\r\nKeep these exact body bytes.\r\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()

    readiness = RuntimeReadiness()
    commands.readiness = readiness
    with pytest.raises(RuntimeNotReadyError):
        commands.adopt_memory_document(relative_path, expected, caller=caller)
    assert path.read_bytes() == original
    readiness.transition(RuntimeReadinessState.READY)

    without_capability = TrustedRequestContext(
        tenant_id=caller.tenant_id,
        user_id=caller.user_id,
        actor_kind="user",
        actor_id=caller.user_id,
        capabilities=frozenset({READ_CONTEXT}),
    )
    with pytest.raises(PermissionError, match="memory.authoritative.remember"):
        commands.adopt_memory_document(relative_path, expected, caller=without_capability)
    with pytest.raises(DocumentConflictError, match="expected_raw_sha256"):
        commands.adopt_memory_document(relative_path, "f" * 64, caller=caller)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="relative"):
        commands.adopt_memory_document(str(path), expected, caller=caller)

    adopted = commands.adopt_memory_document(relative_path, expected, caller=caller)

    assert adopted.changed is True
    assert adopted.document_revision == 1
    assert adopted.relative_path == relative_path
    assert adopted.projection_status == "ENQUEUED"
    raw = source.read_raw(
        caller.tenant_id,
        caller.user_id,
        document_id=adopted.document_id,
    )
    assert raw.endswith(original)
    parsed = parse_front_matter(raw, max_header_bytes=32 * 1024)
    assert parsed.document_id == adopted.document_id
    assert hashlib.sha256(raw).hexdigest() == adopted.source_digest

    control = controls.load_control(caller.tenant_id, caller.user_id, adopted.document_id)
    assert control is not None
    queue = cast(_Queue, commands.committer.projection_queue)
    assert len(queue.jobs) == 1
    job = next(iter(queue.jobs.values()))
    assert job.action == "memory_committed"
    assert job.payload["edit_kind"] == "create"
    assert job.payload["before_raw_digest"] == ""
    assert job.payload["after_raw_digest"] == adopted.source_digest
    assert not ({"content", "body", "raw_bytes"} & set(job.payload))
    intent = controls.load_intent(caller.tenant_id, caller.user_id, str(job.payload["intent_id"]))
    assert intent is not None
    event = controls.load_event(intent)
    assert event is not None
    assert event.edit_kind is DocumentEditKind.CREATE
    assert event.actor_binding == "trusted:user:user-a:user-a"
    assert event.evidence_reference.startswith("adoption-receipt:mdadopt_")
    assert event.evidence_digest == expected
    assert not ({"content", "body", "raw_bytes"} & set(event.to_dict()))

    retried = commands.adopt_memory_document(relative_path, expected, caller=caller)
    assert retried == adopted
    assert len(queue.jobs) == 1


def test_adoption_receipt_requires_prebound_identity_and_cannot_backfill_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    relative_path = "knowledge/topics/receipt-prebind.md"
    original = b"# Receipt prebind\n\nexact unmanaged source\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    durable_prepare = controls.prepare_adoption_receipt

    def stop_after_receipt(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - crash boundary.
        durable_prepare(*args, **kwargs)
        raise RuntimeError("process stopped after adoption receipt")

    monkeypatch.setattr(controls, "prepare_adoption_receipt", stop_after_receipt)
    with pytest.raises(RuntimeError, match="after adoption receipt"):
        commands.adopt_memory_document(relative_path, expected, caller=caller)

    identity = controls.load_root_identity(caller.tenant_id, caller.user_id)
    assert identity is not None
    receipts = controls.adoption_receipts(caller.tenant_id, caller.user_id)
    assert len(receipts) == 1
    assert path.read_bytes() == original
    assert controls.incomplete_intents(caller.tenant_id, caller.user_id) == ()
    assert controls.controls(caller.tenant_id, caller.user_id) == ()

    monkeypatch.setattr(controls, "prepare_adoption_receipt", durable_prepare)
    identity_path = tmp_path / "system" / "memory-documents" / caller.user_id / "scan-root.json"
    identity_path.unlink()
    memory_root = user_memory_root(tmp_path, caller.tenant_id, caller.user_id)
    detached_root = memory_root.with_name("memory-detached-receipt")
    memory_root.rename(detached_root)
    shutil.copytree(detached_root, memory_root)

    with pytest.raises(DocumentControlIntegrityError, match="missing its source root identity"):
        commands.adopt_memory_document(relative_path, expected, caller=caller)

    assert not identity_path.exists()
    assert (memory_root / relative_path).read_bytes() == original
    assert (detached_root / relative_path).read_bytes() == original
    assert controls.adoption_receipts(caller.tenant_id, caller.user_id) == receipts

    second_id = new_document_id()
    second_path = "knowledge/topics/receipt-must-not-backfill.md"
    second_raw = render_new_document(second_id, "unrelated direct create")
    with pytest.raises(DocumentControlIntegrityError, match="adoption receipts"):
        commands.committer.commit(
            DocumentEditPlan(
                idempotency_key="receipt-must-not-backfill",
                tenant_id=caller.tenant_id,
                owner_user_id=caller.user_id,
                edit_kind=DocumentEditKind.CREATE,
                expected_state=ABSENT,
                evidence_digest="a" * 64,
                edit_summary="receipt must not backfill identity",
                document_id=second_id,
                relative_path=second_path,
                after_bytes=second_raw,
                expected_registration_document_id=second_id,
            ),
            actor_binding="trusted:user:user-a:user-a",
            evidence_reference="explicit-command:receipt-backfill",
        )
    assert source.read_state(caller.tenant_id, caller.user_id, second_path) == ABSENT
    assert controls.incomplete_intents(caller.tenant_id, caller.user_id) == ()
    assert controls.controls(caller.tenant_id, caller.user_id) == ()
    assert not tuple(tmp_path.rglob("*.blob"))


def test_adopt_retry_recovers_after_frontmatter_rewrite_before_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    relative_path = "knowledge/topics/crash-after-rewrite.md"
    original = b"# Retry\n\nbody that must not enter the receipt\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    durable_adopt = source.adopt

    def crash_after_rewrite(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - fault-injection boundary.
        durable_adopt(*args, **kwargs)
        raise RuntimeError("simulated response loss after source CAS")

    monkeypatch.setattr(source, "adopt", crash_after_rewrite)
    with pytest.raises(RuntimeError, match="simulated response loss"):
        commands.adopt_memory_document(relative_path, expected, caller=caller)

    managed = source.full_scan("default", "user-a").managed
    assert len(managed) == 1
    assigned_id = managed[0].document_id
    retried = commands.adopt_memory_document(relative_path, expected, caller=caller)

    assert retried.document_id == assigned_id
    assert retried.document_revision == 1
    assert len(cast(_Queue, commands.committer.projection_queue).jobs) == 1
    receipt_files = tuple(tmp_path.rglob("mdadopt_*.json"))
    assert len(receipt_files) == 1
    assert original not in receipt_files[0].read_bytes()
    assert b"body that must not enter the receipt" not in receipt_files[0].read_bytes()
    assert controls.load_control("default", "user-a", assigned_id) is not None


@pytest.mark.parametrize("crash_stage", ["temp_file_fsynced", "atomic_installed"])
def test_adopt_receipt_reuses_deterministic_store_temp_after_process_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_stage: str,
) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    relative_path = f"knowledge/topics/adopt-{crash_stage}.md"
    original = f"# Crash {crash_stage}\n\nexact adoption body\n".encode()
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    durable_adopt = source.adopt
    operation_ids: list[str] = []

    def crash_in_store(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - exact store boundary.
        operation_id = str(kwargs.get("operation_id") or "")
        operation_ids.append(operation_id)

        def fault(stage: str) -> None:
            if stage == crash_stage:
                raise RuntimeError(f"process stopped at {crash_stage}")

        kwargs["fault_hook"] = fault
        return durable_adopt(*args, **kwargs)

    monkeypatch.setattr(source, "adopt", crash_in_store)
    with pytest.raises(RuntimeError, match=crash_stage):
        commands.adopt_memory_document(relative_path, expected, caller=caller)

    receipts = controls.adoption_receipts(caller.tenant_id, caller.user_id)
    assert len(receipts) == 1
    assert operation_ids == [receipts[0].receipt_id]
    deterministic_temp = source._temporary_name(path.name, receipts[0].receipt_id)
    if crash_stage == "temp_file_fsynced":
        assert (path.parent / deterministic_temp).exists()
        assert path.read_bytes() == original
    else:
        assert not (path.parent / deterministic_temp).exists()

    monkeypatch.setattr(source, "adopt", durable_adopt)
    recovered = commands.adopt_memory_document(relative_path, expected, caller=caller)

    assert recovered.document_id == receipts[0].document_id
    assert controls.load_control("default", "user-a", recovered.document_id) is not None
    assert not (path.parent / deterministic_temp).exists()


def test_restart_scanner_uses_durable_adoption_identity_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    relative_path = "knowledge/topics/restart-receipt.md"
    original = b"# Restart\n\ntrusted attribution survives process loss\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    durable_adopt = source.adopt
    commands.bootstrapper = MemoryDocumentBootstrapper(
        tmp_path,
        source,
        control_store=controls,
    )

    def stop_process_after_cas(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - crash boundary.
        durable_adopt(*args, **kwargs)
        raise RuntimeError("process stopped after adoption CAS")

    monkeypatch.setattr(source, "adopt", stop_process_after_cas)
    with pytest.raises(RuntimeError, match="process stopped"):
        commands.adopt_memory_document(relative_path, expected, caller=caller)
    assigned_id = source.full_scan("default", "user-a").managed[0].document_id
    assert controls.load_control("default", "user-a", assigned_id) is None
    assert controls.incomplete_intents("default", "user-a") == ()
    root_identity = controls.load_root_identity("default", "user-a")
    assert root_identity is not None
    assert root_identity.root_identity == source.full_scan("default", "user-a").root_identity

    # Recreate every process-local component. Startup's forced full scan must
    # recognize the indexed receipt and finish the trusted adoption intent.
    restarted_source = FileSystemMemoryDocumentStore(tmp_path)
    restarted_controls = MemoryDocumentControlStore(tmp_path)
    restarted_revisions = MemoryDocumentRevisionStore(tmp_path)
    restarted_queue = _Queue()
    restarted_committer = MemoryDocumentCommitter(
        restarted_source,
        restarted_controls,
        restarted_revisions,
        cast(QueueStore, restarted_queue),
    )
    restarted_bootstrapper = MemoryDocumentBootstrapper(
        tmp_path,
        restarted_source,
        control_store=restarted_controls,
    )
    scan_results = []
    scanner = MemoryDocumentScanner(
        restarted_source,
        stability_seconds=60,
        change_publisher=lambda change: scan_results.append(
            _publish_external_change(
                change,
                committer=restarted_committer,
                control_store=restarted_controls,
                document_store=restarted_source,
                bootstrapper=restarted_bootstrapper,
            )
        ),
    )
    reconciliation = scanner.scan("default", "user-a", force_stable=True)

    assert len(reconciliation.confirmed_changes) == 1
    assert len(scan_results) == 1 and scan_results[0] is not None
    event = scan_results[0].event
    assert event is not None
    assert event.actor_binding == "trusted:user:user-a:user-a"
    assert event.evidence_reference.startswith("adoption-receipt:mdadopt_")
    assert event.evidence_digest == expected
    assert len(restarted_queue.jobs) == 1

    restarted_eraser = MemoryDocumentEraser(
        restarted_source,
        restarted_controls,
        restarted_revisions,
    )
    restarted_commands = MemoryCommandService(
        MemoryDocumentPlanner(restarted_source),
        restarted_committer,
        restarted_eraser,
        bootstrapper=restarted_bootstrapper,
    )
    remembered = restarted_commands.remember(
        "direct remember after restart recovery",
        target_hint="topic:restart-direct",
        caller=caller,
    )
    assert remembered.changed is True
    retried = restarted_commands.adopt_memory_document(
        relative_path,
        expected,
        caller=caller,
    )
    assert retried.document_id == assigned_id
    assert retried.document_revision == 1
    assert len(restarted_queue.jobs) == 2


def test_adopt_first_bootstrap_preserves_template_and_survives_restart(tmp_path: Path) -> None:
    caller = _caller()
    source = FileSystemMemoryDocumentStore(tmp_path)
    controls = MemoryDocumentControlStore(tmp_path)
    revisions = MemoryDocumentRevisionStore(tmp_path)
    committer = MemoryDocumentCommitter(source, controls, revisions, cast(QueueStore, _Queue()))
    eraser = MemoryDocumentEraser(source, controls, revisions)
    bootstrapper = MemoryDocumentBootstrapper(
        tmp_path,
        source,
        control_store=controls,
    )
    commands = MemoryCommandService(
        MemoryDocumentPlanner(source),
        committer,
        eraser,
        bootstrapper=bootstrapper,
    )
    relative_path = "profile.md"
    original = b"# User-owned profile\n\nDo not replace this text.\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    unrelated_path = path.parent / "notes" / "draft.md"
    unrelated_path.parent.mkdir(parents=True, exist_ok=True)
    unrelated = b"unmanaged draft remains untouched\n"
    unrelated_path.write_bytes(unrelated)
    expected = hashlib.sha256(original).hexdigest()

    adopted = commands.adopt_memory_document(relative_path, expected, caller=caller)

    exact_adopted = adopt_raw_document(
        original,
        adopted.document_id,
        max_header_bytes=32 * 1024,
    )
    assert path.read_bytes() == exact_adopted
    assert unrelated_path.read_bytes() == unrelated
    first_scan = source.full_scan("default", "user-a")
    template_paths = {
        "MEMORY.md",
        "profile.md",
        "preferences.md",
        "knowledge/MEMORY.md",
        "knowledge/open-loops.md",
    }
    first_templates = {
        item.relative_path: item.document_id
        for item in first_scan.managed
        if item.relative_path in template_paths
    }
    assert set(first_templates) == template_paths
    assert first_templates[relative_path] == adopted.document_id
    marker = tmp_path / "system" / "memory-documents" / "user-a" / "bootstrap.json"
    assert '"status":"COMPLETED"' in marker.read_text(encoding="utf-8")
    root_identity = controls.load_root_identity("default", "user-a")
    assert root_identity is not None
    assert commands.remember("after adopt", target_hint="topic:adopt-first", caller=caller).changed

    restarted_source = FileSystemMemoryDocumentStore(tmp_path)
    restarted_controls = MemoryDocumentControlStore(tmp_path)
    restarted_revisions = MemoryDocumentRevisionStore(tmp_path)
    restarted_committer = MemoryDocumentCommitter(
        restarted_source,
        restarted_controls,
        restarted_revisions,
        cast(QueueStore, _Queue()),
    )
    restarted_bootstrapper = MemoryDocumentBootstrapper(
        tmp_path,
        restarted_source,
        control_store=restarted_controls,
    )
    restarted_commands = MemoryCommandService(
        MemoryDocumentPlanner(restarted_source),
        restarted_committer,
        MemoryDocumentEraser(restarted_source, restarted_controls, restarted_revisions),
        bootstrapper=restarted_bootstrapper,
    )

    restarted_bootstrapper.ensure_user("default", "user-a")
    assert restarted_controls.load_root_identity("default", "user-a") == root_identity
    assert restarted_commands.remember(
        "after restart",
        target_hint="topic:adopt-restart",
        caller=caller,
    ).changed
    restarted_scan = restarted_source.full_scan("default", "user-a")
    restarted_templates = {
        item.relative_path: item.document_id
        for item in restarted_scan.managed
        if item.relative_path in template_paths
    }
    assert restarted_templates == first_templates
    assert path.read_bytes() == exact_adopted
    assert unrelated_path.read_bytes() == unrelated


def test_adopt_retry_recovers_control_to_projection_tail(tmp_path: Path) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    queue = InMemoryQueueStore()
    commands.committer.projection_queue = queue
    caller = _caller()
    relative_path = "knowledge/topics/crash-before-projection.md"
    original = b"# Retry projection\n\nbody\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    crashed = False

    def fail_once(stage, _intent):  # noqa: ANN001, ANN202 - fault-injection boundary.
        nonlocal crashed
        if stage == "control_recorded" and not crashed:
            crashed = True
            raise RuntimeError("simulated crash before projection enqueue")

    commands.committer.test_hook = fail_once
    with pytest.raises(RuntimeError, match="before projection enqueue"):
        commands.adopt_memory_document(relative_path, expected, caller=caller)
    assigned_id = source.full_scan("default", "user-a").managed[0].document_id
    assert controls.load_control("default", "user-a", assigned_id) is not None
    recovery_jobs = [job for job in queue.jobs.values() if job.queue_name == "memory_document_edit"]
    assert len(recovery_jobs) == 1
    assert not [job for job in queue.jobs.values() if job.queue_name == "memory_projection"]

    commands.committer.test_hook = None
    worker_result = MemoryDocumentEditWorker(
        commands.committer,
        queue,
        tenant_id="default",
        worker_id="adopt-recovery-worker",
    ).process_pending()
    assert worker_result == {"claimed": 1, "committed": 1, "failed": 0, "dead_letter": 0}
    settled = queue.get(recovery_jobs[0].job_id)
    assert settled is not None and settled.status == "done"

    retried = commands.adopt_memory_document(relative_path, expected, caller=caller)

    assert retried.document_id == assigned_id
    assert retried.document_revision == 1
    projection_jobs = [job for job in queue.jobs.values() if job.queue_name == "memory_projection"]
    assert len(projection_jobs) == 1
    intents = [
        intent
        for intent in controls.incomplete_intents("default", "user-a")
        if intent.document_id == assigned_id
    ]
    assert intents == []


def test_hard_erase_retains_only_content_free_adoption_identity(tmp_path: Path) -> None:
    source, controls, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    relative_path = "knowledge/topics/adopt-then-erase.md"
    secret = b"adoption-secret-plaintext"
    original = b"# Secret\n\n" + secret + b"\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    adopted = commands.adopt_memory_document(relative_path, expected, caller=caller)
    receipt_files = tuple(tmp_path.rglob("mdadopt_*.json"))
    assert len(receipt_files) == 1

    erased = commands.forget(
        adopted.document_uri,
        mode="HARD_ERASE",
        expected_digest=adopted.source_digest,
        caller=caller,
    )

    assert erased.erasure_status == DocumentEraseStatus.ERASED.value
    assert receipt_files[0].exists()
    assert controls.load_publication_barrier("default", "user-a", adopted.document_id) is not None
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert secret not in artifact.read_bytes()
    with pytest.raises(DocumentErasedError):
        commands.adopt_memory_document(relative_path, expected, caller=caller)


def test_document_commands_soft_forget_history_and_restore(tmp_path: Path) -> None:
    source, _, _, _, commands, _, _ = _components(tmp_path)
    caller = _caller()

    remembered = commands.remember("Important\n\nInitial body", target_hint="topic:alpha", caller=caller)
    assert remembered.document_kind == "topic"
    assert remembered.changed is True
    assert remembered.projection_status == "ENQUEUED"

    edited = commands.edit_memory_document(
        remembered.document_uri,
        "# Main\n\nKeep\n\n## Secret\n\nremove me\n\n## Next\n\nstay\n",
        remembered.source_digest,
        caller=caller,
    )
    redacted = commands.forget(
        remembered.document_uri,
        section_anchor="Secret",
        expected_digest=edited.source_digest,
        caller=caller,
    )
    assert redacted.recoverable is True
    raw = source.read_raw("default", "user-a", document_id=remembered.document_id)
    assert b"Secret" not in raw and b"remove me" not in raw
    assert b"Next" in raw and b"stay" in raw

    history = commands.list_memory_history(remembered.document_uri, caller=caller)
    assert [item.document_revision for item in history.revisions] == [1, 2, 3]
    deleted = commands.forget(
        remembered.document_uri,
        expected_digest=redacted.source_digest,
        caller=caller,
    )
    assert deleted.recoverable is True
    assert source.read_state("default", "user-a", remembered.relative_path) == ABSENT
    assert len(commands.list_memory_history(remembered.document_uri, caller=caller).revisions) == 4

    restored = commands.restore_memory_revision(
        remembered.document_uri,
        revision=3,
        expected_digest="",
        caller=caller,
    )
    assert restored.changed is True
    assert b"stay" in source.read_raw("default", "user-a", document_id=remembered.document_id)


def test_review_approve_uses_exact_document_cas(tmp_path: Path) -> None:
    source, _, _, reviews, commands, review_service, _ = _components(tmp_path)
    caller = _caller()
    remembered = commands.remember("Base body", target_hint="topic:review", caller=caller)
    state = source.read_state("default", "user-a", remembered.relative_path)
    assert isinstance(state, PresentPath)
    raw = source.read_raw("default", "user-a", document_id=remembered.document_id)
    parsed = parse_front_matter(raw, max_header_bytes=32 * 1024)
    after = parsed.header_bytes + b"\nReviewed body\n"
    plan = DocumentEditPlan(
        idempotency_key="candidate-1",
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=DocumentEditKind.UPDATE,
        expected_state=state,
        evidence_digest=hashlib.sha256(b"candidate evidence").hexdigest(),
        edit_summary="reviewed candidate edit",
        document_id=remembered.document_id,
        relative_path=remembered.relative_path,
        after_bytes=after,
        expected_registration_document_id=remembered.document_id,
    )

    pending = review_service.seal_edit_proposal(plan, proposed_diff="-Base body\n+Reviewed body")
    assert pending.status == "PENDING"
    assert b"Reviewed body" in review_service.read_proposed_diff(pending.proposal_id, caller=caller)
    preview = review_service.preview_edit(pending.proposal_id, caller=caller)
    assert preview.proposal_id == pending.proposal_id
    assert preview.status == "PENDING"
    assert preview.proposed_diff == "-Base body\n+Reviewed body"
    assert preview.proposed_diff_digest == pending.proposed_diff_digest
    approved = review_service.review_edit(pending.proposal_id, "APPROVE", caller=caller)

    assert approved.status == "APPROVED"
    assert approved.changed is True
    assert source.read_raw("default", "user-a", document_id=remembered.document_id) == after
    record = reviews.load("default", "user-a", pending.proposal_id)
    assert record is not None and record.status == MemoryEditReviewStatus.APPROVED


def test_review_correct_then_reject_never_mutates_live_document(tmp_path: Path) -> None:
    source, _, _, reviews, commands, review_service, _ = _components(tmp_path)
    caller = _caller()
    remembered = commands.remember("Original", target_hint="topic:correction", caller=caller)
    state = source.read_state("default", "user-a", remembered.relative_path)
    assert isinstance(state, PresentPath)
    original = source.read_raw("default", "user-a", document_id=remembered.document_id)
    parsed = parse_front_matter(original, max_header_bytes=32 * 1024)

    def plan(key: str, body: bytes) -> DocumentEditPlan:
        return DocumentEditPlan(
            idempotency_key=key,
            tenant_id="default",
            owner_user_id="user-a",
            edit_kind=DocumentEditKind.UPDATE,
            expected_state=state,
            evidence_digest=hashlib.sha256(key.encode()).hexdigest(),
            edit_summary=f"candidate {key}",
            document_id=remembered.document_id,
            relative_path=remembered.relative_path,
            after_bytes=parsed.header_bytes + body,
            expected_registration_document_id=remembered.document_id,
        )

    pending = review_service.seal_edit_proposal(plan("first", b"\nFirst proposal\n"), proposed_diff="+first")
    replacement = review_service.review_edit(
        pending.proposal_id,
        "CORRECT",
        caller=caller,
        corrected_edit="Corrected proposal",
    )
    old = reviews.load("default", "user-a", pending.proposal_id)
    assert old is not None and old.status == MemoryEditReviewStatus.CORRECTED
    rejected = review_service.review_edit(replacement.proposal_id, "REJECT", caller=caller)
    assert rejected.status == "REJECTED"
    assert source.read_raw("default", "user-a", document_id=remembered.document_id) == original


def test_hard_erase_upgrades_soft_forget_and_purges_its_retained_revision(
    tmp_path: Path,
) -> None:
    source, controls, revisions, _, commands, _, _ = _components(tmp_path)
    caller = _caller()
    secret = b"soft-forgotten-secret-must-be-hard-erased"
    remembered = commands.remember(
        secret.decode(),
        target_hint="topic:soft-then-hard",
        caller=caller,
    )

    soft = commands.forget(
        remembered.document_uri,
        mode="SOFT_FORGET",
        expected_digest=remembered.source_digest,
        caller=caller,
    )

    assert soft.recoverable is True
    assert soft.source_digest == ""
    control = controls.load_control("default", "user-a", remembered.document_id)
    assert control is not None and control.status == "deleted"
    assert source.read_state("default", "user-a", remembered.relative_path) == ABSENT
    assert revisions.list_revisions("default", "user-a", remembered.document_id)

    hard = commands.forget(
        remembered.document_uri,
        mode="HARD_ERASE",
        caller=caller,
    )

    assert hard.erasure_status == DocumentEraseStatus.ERASED.value
    assert hard.recoverable is False
    assert revisions.list_revisions("default", "user-a", remembered.document_id) == ()
    assert controls.load_control("default", "user-a", remembered.document_id) is None
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert secret not in artifact.read_bytes()


def test_hard_erase_is_replayable_purges_body_blobs_and_blocks_resurrection(tmp_path: Path) -> None:
    backend = _CleanupBackend([False, True])
    source, controls, revisions, reviews, commands, review_service, eraser = _components(
        tmp_path,
        backend=backend,
    )
    caller = _caller()
    secret = "highly-sensitive-memory-body"
    remembered = commands.remember(secret, target_hint="topic:erase", caller=caller)
    state = source.read_state("default", "user-a", remembered.relative_path)
    assert isinstance(state, PresentPath)
    raw = source.read_raw("default", "user-a", document_id=remembered.document_id)
    parsed = parse_front_matter(raw, max_header_bytes=32 * 1024)
    proposal = DocumentEditPlan(
        idempotency_key="pending-secret-edit",
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=DocumentEditKind.UPDATE,
        expected_state=state,
        evidence_digest="c" * 64,
        edit_summary="pending body-bearing proposal",
        document_id=remembered.document_id,
        relative_path=remembered.relative_path,
        after_bytes=parsed.header_bytes + f"\n{secret}-review-copy\n".encode(),
        expected_registration_document_id=remembered.document_id,
    )
    pending = review_service.seal_edit_proposal(proposal, proposed_diff=f"+{secret}-diff-copy")

    first = commands.forget(
        remembered.document_uri,
        mode="HARD_ERASE",
        expected_digest=remembered.source_digest,
        caller=caller,
    )
    assert first.erasure_status == DocumentEraseStatus.ERASE_PENDING.value
    assert first.pending_backends == ("vector",)
    assert first.independent_evidence_retained == ("memoryos://user/user-a/sessions/history/session-independent",)
    assert source.read_state("default", "user-a", remembered.relative_path) == ABSENT
    assert revisions.list_revisions("default", "user-a", remembered.document_id) == ()
    assert controls.load_control("default", "user-a", remembered.document_id) is None
    hard_barrier = controls.load_publication_barrier(
        "default",
        "user-a",
        remembered.document_id,
    )
    assert hard_barrier is not None
    assert hard_barrier.status is DocumentDeletionStatus.HARD_ERASED
    assert hard_barrier.deletion_generation == remembered.document_revision + 1
    assert reviews.load("default", "user-a", pending.proposal_id) is None

    erasure_record = eraser.erase_store.load("default", "user-a", remembered.document_id)
    assert erasure_record is not None
    assert erasure_record.relative_path == remembered.relative_path
    resurrection = DocumentEditPlan(
        idempotency_key="forbidden-resurrection",
        tenant_id="default",
        owner_user_id="user-a",
        edit_kind=DocumentEditKind.CREATE,
        expected_state=ABSENT,
        evidence_digest="d" * 64,
        edit_summary="must be rejected by erasure epoch",
        document_id=remembered.document_id,
        relative_path=remembered.relative_path,
        after_bytes=raw,
    )
    with pytest.raises(DocumentErasedError):
        review_service.committer.commit(
            resurrection,
            actor_binding="trusted:user:user-a:user-a",
            evidence_reference="resurrection-attempt",
        )
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert secret.encode() not in artifact.read_bytes()
    with pytest.raises(DocumentErasedError):
        commands.restore_memory_revision(
            remembered.document_uri,
            revision=1,
            expected_digest="",
            caller=caller,
        )
    with pytest.raises(DocumentErasedError):
        eraser.erase_store.assert_projection_allowed(
            "default",
            "user-a",
            remembered.document_id,
            projection_generation=1,
        )

    recovery = eraser.recover_owner("default", "user-a")
    assert recovery.completed_document_ids == (remembered.document_id,)
    assert recovery.pending_document_ids == ()

    completed = commands.forget(
        remembered.document_uri,
        mode="HARD_ERASE",
        expected_digest=remembered.source_digest,
        caller=caller,
    )
    assert completed.erasure_status == DocumentEraseStatus.ERASED.value
    assert completed.pending_backends == ()
    assert len(backend.requests) == 2
    completed_barrier = controls.load_publication_barrier(
        "default", "user-a", remembered.document_id
    )
    assert completed_barrier is not None
    assert completed_barrier.status is DocumentDeletionStatus.HARD_ERASED
    assert completed_barrier.relative_path == ""
    assert completed_barrier.relative_path_digest == hard_barrier.relative_path_digest
    completed_record = eraser.erase_store.load("default", "user-a", remembered.document_id)
    assert completed_record is not None
    assert completed_record.relative_path == ""
    assert completed_record.relative_path_digest == hard_barrier.relative_path_digest
    assert completed_record.independent_evidence_retained == first.independent_evidence_retained


def test_hard_erase_requires_distinct_capability_and_whole_document(tmp_path: Path) -> None:
    assert HARD_ERASE_MEMORY in KNOWN_CAPABILITIES
    assert HARD_ERASE_MEMORY not in DEFAULT_AGENT_CAPABILITIES
    _, _, _, _, commands, _, _ = _components(tmp_path)
    privileged = _caller()
    remembered = commands.remember("body", target_hint="topic:capability", caller=privileged)

    with pytest.raises(PermissionError, match="memory.hard_erase"):
        commands.forget(
            remembered.document_uri,
            mode="HARD_ERASE",
            expected_digest=remembered.source_digest,
            caller=_caller(hard_erase=False),
        )
    with pytest.raises(ValueError, match="whole-document"):
        commands.forget(
            remembered.document_uri,
            section_anchor="body",
            mode="HARD_ERASE",
            expected_digest=remembered.source_digest,
            caller=privileged,
        )
