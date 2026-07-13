from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    head_set_path,
    load_current_head,
)
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.projection import (
    ProjectionIntegrityError,
    ProjectionOutboxIntegrityError,
)
from memoryos.memory.canonical.projection_proof import (
    AuthoritativeProjectionIntegrityError,
)
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.schema import MemoryTypeSchema
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.runtime.readiness import RuntimeNotReadyError, RuntimeReadinessState


class _EmptyExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemorySemanticProposal]:
        del archive, schemas
        return []


def _remember(
    client: MemoryOSClient,
    value: str,
    *,
    topic: str = "primary storage backend",
) -> dict:
    return client.remember(
        user_id="u1",
        content=value,
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": topic},
    )


def _require_queue_job(client: MemoryOSClient, job_id: str) -> QueueJob:
    job = client.queue_store.get(job_id)
    assert job is not None
    return job


def _resolve_replacement(client: MemoryOSClient) -> None:
    pending = _remember(client, "MySQL")
    assert pending["status"] == "PENDING"
    record = client.list_pending(user_id="u1", lifecycle_states=["PENDING"])[0]
    client.review_pending(
        user_id="u1",
        pending_uri=record["uri"],
        decision="CONFIRM_AND_APPLY",
        expected_lifecycle_revision=record["lifecycle_revision"],
        expected_proposal_fingerprint=record["proposal_fingerprint"],
        command_id="tamper-matrix-review",
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "source_metadata",
        "source_content",
        "source_relations",
        "source_pointer_symlink",
        "current_head",
        "current_head_alias",
        "current_head_parent_symlink",
        "current_head_receipt_alias",
        "historical_receipt",
        "outbox",
        "outbox_alias",
        "outbox_symlink",
        "diff",
        "diff_symlink",
        "planning_envelope",
        "planning_envelope_symlink",
        "planning_anchor_path",
        "pending_history",
    ],
)
def test_authoritative_artifact_tamper_forces_startup_not_ready(
    tmp_path: Path,
    artifact: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    claim_uri = committed["uri"]
    slot_uri = claim_uri.rsplit("/claims/", 1)[0]
    head, receipt, _snapshot = load_current_head(tmp_path, claim_uri)

    if artifact.startswith("source_"):
        assert isinstance(client.source_store, FileSystemSourceStore)
        directory = client.source_store._object_dir(claim_uri)
        if artifact == "source_pointer_symlink":
            path = directory / ".bundle-current.json"
            target = tmp_path / "source-pointer-symlink-target.bin"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        else:
            pointer = json.loads((directory / ".bundle-current.json").read_text(encoding="utf-8"))
            generation = directory / ".bundle-generations" / pointer["generation_id"]
            filename = {
                "source_metadata": ".meta.json",
                "source_content": "content.md",
                "source_relations": ".relations.json",
            }[artifact]
            path = generation / filename
            if artifact == "source_content":
                path.write_text("tampered canonical content", encoding="utf-8")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["tampered"] = True
                atomic_write_json(path, payload, artifact_root=tmp_path)
    elif artifact == "current_head":
        path = head_set_path(tmp_path, slot_uri)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["heads"][claim_uri]["object_digest"] = "0" * 64
        atomic_write_json(path, payload, artifact_root=tmp_path)
    elif artifact == "current_head_alias":
        path = head_set_path(tmp_path, slot_uri)
        alias = path.with_name("0" * 64 + ".json")
        assert alias != path
        alias.write_bytes(path.read_bytes())
    elif artifact == "current_head_parent_symlink":
        directory = tmp_path / "system" / "current-heads"
        redirected = tmp_path / "system" / "redirected-current-heads"
        directory.rename(redirected)
        directory.symlink_to(redirected, target_is_directory=True)
    elif artifact == "current_head_receipt_alias":
        receipt_path = tmp_path / str(head["receipt_path"])
        alias = tmp_path / "system" / "receipt-alias.json"
        alias.write_bytes(receipt_path.read_bytes())
        path = head_set_path(tmp_path, slot_uri)
        payload = json.loads(path.read_text(encoding="utf-8"))
        member = payload["heads"][claim_uri]
        member["receipt_path"] = str(alias.relative_to(tmp_path))
        member_core = {key: value for key, value in member.items() if key != "head_digest"}
        member["head_digest"] = canonical_digest(member_core)
        set_core = {key: value for key, value in payload.items() if key != "head_set_digest"}
        payload["head_set_digest"] = canonical_digest(set_core)
        atomic_write_json(path, payload, artifact_root=tmp_path)
    elif artifact == "historical_receipt":
        _resolve_replacement(client)
        referenced = {
            str(current[0]["receipt_path"])
            for uri in (slot_uri, claim_uri)
            if (current := load_current_head(tmp_path, uri))
        }
        historical = next(
            path
            for path in (tmp_path / "system" / "transactions").glob("*.json")
            if str(path.relative_to(tmp_path)) not in referenced
        )
        payload = json.loads(historical.read_text(encoding="utf-8"))
        payload["created_at"] = "tampered"
        atomic_write_json(historical, payload, artifact_root=tmp_path)
    elif artifact in {"outbox", "outbox_alias", "outbox_symlink"}:
        path = tmp_path / "system" / "outbox" / f"{head['current_transaction_id']}.json"
        if artifact == "outbox_alias":
            alias = path.with_name("outbox-alias.json")
            alias.write_bytes(path.read_bytes())
        elif artifact == "outbox_symlink":
            target = tmp_path / "outbox-symlink-target.bin"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["receipt_digest"] = "0" * 64
            atomic_write_json(path, payload, artifact_root=tmp_path)
    elif artifact in {"diff", "diff_symlink"}:
        path = tmp_path / "system" / "diffs" / f"{receipt['diff']['diff_id']}.json"
        if artifact == "diff_symlink":
            target = tmp_path / "diff-symlink-target.bin"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["created_at"] = "tampered"
            atomic_write_json(path, payload, artifact_root=tmp_path)
    elif artifact in {
        "planning_envelope",
        "planning_envelope_symlink",
        "planning_anchor_path",
    }:
        archive = SessionArchive(
            user_id="u1",
            session_id="tamper-planning",
            archive_uri="memoryos://user/u1/sessions/history/tamper-planning",
            messages=[{"id": "m1", "role": "user", "content": "Remember this durable rule."}],
            metadata={"tenant_id": "default", "project_id": "memoryos"},
            task_id="tamper-planning-task",
        )
        # Durable planning is allowed to consume only a published immutable
        # session archive.  Persist the evidence first so this branch tests
        # planning-envelope tamper detection rather than the earlier archive
        # visibility boundary.
        client.session_archive_store.write_sync_archive(archive)
        planner = MemoryCommitPlanner(
            extractor=_EmptyExtractor(),
            source_store=client.source_store,
            index_store=client.index_store,
            relation_store=client.relation_store,
        )
        planner.plan(archive)
        assert planner.planning_store is not None
        path = planner.planning_store.path(archive.task_id)
        if artifact == "planning_envelope_symlink":
            target = tmp_path / "planning-envelope-symlink-target.bin"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        elif artifact == "planning_anchor_path":
            path = planner.planning_store.anchor_path(archive.task_id)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["envelope_path"] = "system/planning-envelopes/not-the-task.json"
            core = {key: value for key, value in payload.items() if key != "anchor_digest"}
            payload["anchor_digest"] = canonical_digest(core)
            atomic_write_json(path, payload, artifact_root=tmp_path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["proposal_set_digest"] = "0" * 64
            atomic_write_json(path, payload, artifact_root=tmp_path)
    else:
        pending = _remember(client, "MySQL")
        obj = client.source_store.read_object(pending["uri"])
        obj.metadata["lifecycle_history"] = [{"from": "pending", "to": "confirmed", "reason": "forged"}]
        client.source_store.write_object(obj, content=client.source_store.read_content(obj.uri))

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert restarted.readiness.reasons


def test_projection_outbox_receipt_path_is_bound_to_its_unique_receipt(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    head, _receipt, _snapshot = load_current_head(tmp_path, committed["uri"])
    transaction_id = str(head["current_transaction_id"])
    path = tmp_path / "system" / "outbox" / f"{transaction_id}.json"
    outbox = json.loads(path.read_text(encoding="utf-8"))
    receipt_path = tmp_path / str(outbox["receipt_path"])
    alias = tmp_path / "system" / "projection-receipt-alias.json"
    alias.write_bytes(receipt_path.read_bytes())
    outbox["receipt_path"] = str(alias.relative_to(tmp_path))
    core = {key: value for key, value in outbox.items() if key != "outbox_digest"}
    outbox["outbox_digest"] = canonical_digest(core)

    with pytest.raises(ProjectionIntegrityError, match="unique immutable receipt"):
        client.memory_projection_worker._load_bound_receipt(
            outbox,
            transaction_id,
            str(outbox["commit_group_id"]),
        )


def test_committed_read_rejects_a_current_head_symlink_without_restart(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    slot_uri = committed["uri"].rsplit("/claims/", 1)[0]
    path = head_set_path(tmp_path, slot_uri)
    target = tmp_path / "current-head-symlink-target.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(CurrentHeadIntegrityError, match="symbolic link"):
        load_current_head(tmp_path, committed["uri"])


def test_committed_read_rejects_current_head_parent_symlink_without_restart(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    directory = tmp_path / "system" / "current-heads"
    redirected = tmp_path / "system" / "redirected-current-heads"
    directory.rename(redirected)
    directory.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(CurrentHeadIntegrityError, match="symbolic link"):
        load_current_head(tmp_path, committed["uri"])


@pytest.mark.parametrize(
    "artifact",
    [
        "projection_pointer",
        "projection_attempt_record",
        "projection_attempt_alias",
        "index_metadata",
    ],
)
def test_disposable_projection_tamper_is_rebuilt_before_ready(
    tmp_path: Path,
    artifact: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    claim_uri = committed["uri"]
    record_store = client.memory_projection_worker.projector.record_store
    current = record_store.load_current(claim_uri, source_revision=1)
    assert current is not None
    alias_path: Path | None = None
    if artifact == "projection_pointer":
        path = record_store.current_path(claim_uri)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["publish_token"] = "tampered"
        atomic_write_json(path, payload, artifact_root=tmp_path)
    elif artifact in {"projection_attempt_record", "projection_attempt_alias"}:
        path = record_store.attempt_path_for(current)
        if artifact == "projection_attempt_alias":
            alias_path = path.with_name("attempt-alias.json")
            alias_path.write_bytes(path.read_bytes())
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["projected_content_digest"] = "0" * 64
            atomic_write_json(path, payload, artifact_root=tmp_path)
    else:
        assert isinstance(client.index_store, SQLiteIndexStore)
        index_path = client.index_store.path
        with sqlite3.connect(index_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM contexts WHERE uri = ?",
                (claim_uri,),
            ).fetchone()
            assert row is not None
            metadata = json.loads(row[0])
            metadata["projection_publish_token"] = "tampered"
            connection.execute(
                "UPDATE contexts SET metadata_json = ? WHERE uri = ?",
                (json.dumps(metadata, sort_keys=True), claim_uri),
            )

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.READY, (
        list(restarted.readiness.details),
        restarted.readiness.details.get("projection_repairs"),
        restarted.readiness.details.get("projection_queue"),
        restarted.readiness.reasons,
    )
    if alias_path is not None:
        assert not alias_path.exists()
    restarted.memory_projection_worker.verify_current_projections()
    repaired = restarted.memory_projection_worker.projector.record_store.load_current(
        claim_uri,
        source_revision=1,
    )
    assert repaired is not None and repaired.publish_token != "tampered"
    assert repaired.to_dict()["record_digest"]


@pytest.mark.parametrize(
    "artifact",
    ["publication", "completion", "publication_symlink", "completion_symlink"],
)
def test_immutable_projection_proof_tamper_forces_startup_not_ready(
    tmp_path: Path,
    artifact: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    head, _receipt, _snapshot = load_current_head(tmp_path, committed["uri"])
    transaction_id = str(head["current_transaction_id"])
    if artifact.startswith("completion"):
        client = MemoryOSClient(str(tmp_path))
        assert client.readiness.state == RuntimeReadinessState.READY
        path = client.memory_projection_worker.proof_store.completion_path(transaction_id)
    else:
        path = client.memory_projection_worker.proof_store.publication_path(transaction_id)
    if artifact.endswith("_symlink"):
        target = tmp_path / f"{artifact}-target.bin"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["receipt_digest"] = "0" * 64
        atomic_write_json(path, payload, artifact_root=tmp_path)

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "ProjectionIntegrityError" in " ".join(restarted.readiness.reasons)


def test_live_corrupt_committed_outbox_aborts_batch_and_marks_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    worker = client.memory_projection_worker
    with monkeypatch.context() as scoped:
        scoped.setattr(
            worker,
            "process_pending",
            lambda *args, **kwargs: {
                "processed": [],
                "stale": [],
                "failed": [],
                "dead_letter": [],
                "quarantine": [],
            },
        )
        first = _remember(client, "PostgreSQL", topic="storage-a")
        second = _remember(client, "SQLite", topic="storage-b")

    outboxes = sorted((tmp_path / "system" / "outbox").glob("*.json"))
    assert len(outboxes) == 2
    corrupt = outboxes[0]
    payload = json.loads(corrupt.read_text(encoding="utf-8"))
    payload["outbox_digest"] = "0" * 64
    atomic_write_json(corrupt, payload, artifact_root=tmp_path)
    second_head, _receipt, _snapshot = load_current_head(tmp_path, second["uri"])
    second_job_id = f"outbox_{second_head['current_transaction_id']}"
    assert _require_queue_job(client, second_job_id).status == "pending"
    assert client.index_store.get_index_metadata(first["uri"]) is None
    assert client.index_store.get_index_metadata(second["uri"]) is None

    with pytest.raises(ProjectionOutboxIntegrityError, match="failed before projection"):
        worker.process_pending(limit=100)

    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    assert client.health()["status"] == "not_ready"
    assert not corrupt.exists()
    assert list((tmp_path / "system" / "quarantine" / "outbox").glob("*.original"))
    assert _require_queue_job(client, second_job_id).status == "pending"
    assert client.index_store.get_index_metadata(first["uri"]) is None
    assert client.index_store.get_index_metadata(second["uri"]) is None
    with pytest.raises(RuntimeNotReadyError):
        _remember(client, "DuckDB", topic="storage-c")


def test_startup_corrupt_outbox_stops_before_valid_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    with monkeypatch.context() as scoped:
        scoped.setattr(
            client.memory_projection_worker,
            "process_pending",
            lambda *args, **kwargs: {
                "processed": [],
                "stale": [],
                "failed": [],
                "dead_letter": [],
                "quarantine": [],
            },
        )
        first = _remember(client, "PostgreSQL", topic="startup-storage-a")
        second = _remember(client, "SQLite", topic="startup-storage-b")
    outboxes = sorted((tmp_path / "system" / "outbox").glob("*.json"))
    assert len(outboxes) == 2
    payload = json.loads(outboxes[0].read_text(encoding="utf-8"))
    payload["outbox_digest"] = "0" * 64
    atomic_write_json(outboxes[0], payload, artifact_root=tmp_path)
    second_head, _receipt, _snapshot = load_current_head(tmp_path, second["uri"])
    second_job_id = f"outbox_{second_head['current_transaction_id']}"

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert restarted.health()["status"] == "not_ready"
    assert _require_queue_job(restarted, second_job_id).status == "pending"
    assert restarted.index_store.get_index_metadata(first["uri"]) is None
    assert restarted.index_store.get_index_metadata(second["uri"]) is None


def test_live_self_consistent_projection_proof_tamper_marks_not_ready(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = _remember(client, "PostgreSQL")
    head, _receipt, _snapshot = load_current_head(tmp_path, committed["uri"])
    transaction_id = str(head["current_transaction_id"])
    proof = client.memory_projection_worker.proof_store.publication_path(transaction_id)
    payload = json.loads(proof.read_text(encoding="utf-8"))
    payload["receipt_digest"] = "0" * 64
    core = {key: value for key, value in payload.items() if key != "publication_digest"}
    payload["publication_digest"] = canonical_digest(core)
    atomic_write_json(proof, payload, artifact_root=tmp_path)

    with pytest.raises(
        AuthoritativeProjectionIntegrityError,
        match="transaction boundary",
    ):
        client.memory_projection_worker.process_pending()

    assert proof.exists()
    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    assert client.readiness.details["artifact"] == "projection_proof"
    assert client.health()["status"] == "not_ready"


def test_projection_retry_stays_ready_but_terminal_dead_letter_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    worker = client.memory_projection_worker
    with monkeypatch.context() as scoped:
        scoped.setattr(
            worker,
            "process_pending",
            lambda *args, **kwargs: {
                "processed": [],
                "stale": [],
                "failed": [],
                "dead_letter": [],
                "quarantine": [],
            },
        )
        committed = _remember(client, "PostgreSQL")
    head, _receipt, _snapshot = load_current_head(tmp_path, committed["uri"])
    job_id = f"outbox_{head['current_transaction_id']}"

    with monkeypatch.context() as scoped:
        scoped.setattr(
            worker.projector,
            "project",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("temporary index outage")),
        )
        retry = worker.process_pending(max_retries=3)
    assert retry["failed"] == [job_id]
    assert retry["dead_letter"] == []
    assert _require_queue_job(client, job_id).status == "pending"
    assert client.readiness.state == RuntimeReadinessState.READY

    with monkeypatch.context() as scoped:
        scoped.setattr(
            worker.projector,
            "project",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("permanent index outage")),
        )
        terminal = worker.process_pending(max_retries=2)
    assert terminal["dead_letter"] == [job_id]
    assert _require_queue_job(client, job_id).status == "dead_letter"
    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    assert client.readiness.details["artifact"] == "projection_queue_dead_letter"
    with pytest.raises(RuntimeNotReadyError):
        worker.process_pending()


@pytest.mark.parametrize("during_startup", [False, True])
@pytest.mark.parametrize("fault_kind", ["outbox", "proof", "queue"])
def test_authoritative_race_aborts_already_leased_projection_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    during_startup: bool,
    fault_kind: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    worker = client.memory_projection_worker
    with monkeypatch.context() as scoped:
        scoped.setattr(
            worker,
            "process_pending",
            lambda *args, **kwargs: {
                "processed": [],
                "stale": [],
                "failed": [],
                "dead_letter": [],
                "quarantine": [],
                "released": [],
            },
        )
        first = _remember(client, "PostgreSQL", topic="leased-race-a")
        second = _remember(client, "SQLite", topic="leased-race-b")
    heads = [load_current_head(tmp_path, item["uri"])[0] for item in (first, second)]
    transaction_ids = [str(head["current_transaction_id"]) for head in heads]
    job_ids = {f"outbox_{transaction_id}" for transaction_id in transaction_ids}
    assert all(_require_queue_job(client, job_id).status == "pending" for job_id in job_ids)

    original_load = worker._load_projection_job_outbox
    original_ensure = worker._ensure_projection_publication
    tampered_job_ids: list[str] = []

    def tamper_after_batch_lease(job, **kwargs):  # noqa: ANN001, ANN202
        if not tampered_job_ids:
            path = Path(str(job.payload["outbox_path"]))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["outbox_digest"] = "0" * 64
            atomic_write_json(path, payload, artifact_root=tmp_path)
            tampered_job_ids.append(job.job_id)
        return original_load(job, **kwargs)

    def tamper_proof_after_projection(outbox, job):  # noqa: ANN001, ANN202
        if not tampered_job_ids:
            transaction_id = str(outbox["transaction_id"])
            atomic_write_json(
                worker.proof_store.publication_path(transaction_id),
                {
                    "schema_version": "projection_publication_receipt_v1",
                    "transaction_id": transaction_id,
                    "publication_digest": "0" * 64,
                },
                artifact_root=tmp_path,
            )
            tampered_job_ids.append(job.job_id)
        return original_ensure(outbox, job)

    def tamper_queue_after_batch_lease(job, **kwargs):  # noqa: ANN001, ANN202
        if not tampered_job_ids:
            assert isinstance(client.queue_store, SQLiteQueueStore)
            payload = dict(job.payload)
            payload["operation_ids"] = [*payload.get("operation_ids", []), "forged-operation"]
            with sqlite3.connect(client.queue_store.path) as connection:
                connection.execute(
                    "UPDATE queue_jobs SET payload_json = ? WHERE job_id = ?",
                    (
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        job.job_id,
                    ),
                )
            tampered_job_ids.append(job.job_id)
        return original_load(job, **kwargs)

    if fault_kind == "outbox":
        monkeypatch.setattr(worker, "_load_projection_job_outbox", tamper_after_batch_lease)
    elif fault_kind == "proof":
        monkeypatch.setattr(worker, "_ensure_projection_publication", tamper_proof_after_projection)
    else:
        monkeypatch.setattr(worker, "_load_projection_job_outbox", tamper_queue_after_batch_lease)
    if during_startup:
        client.readiness.transition(RuntimeReadinessState.RECOVERING)
        result = worker._process_pending_during_startup(limit=2)
    else:
        result = worker.process_pending(limit=2)

    assert len(tampered_job_ids) == 1
    corrupt_job_id = tampered_job_ids[0]
    remaining_job_id = next(iter(job_ids - {corrupt_job_id}))
    assert result["processed"] == []
    assert result["quarantine"] == [corrupt_job_id]
    assert result["released"] == [remaining_job_id]
    corrupt_job = client.queue_store.get(corrupt_job_id)
    remaining_job = client.queue_store.get(remaining_job_id)
    assert corrupt_job is not None
    assert corrupt_job.status == "quarantine"
    assert remaining_job is not None and remaining_job.status == "pending"
    assert remaining_job.retry_count == 0
    assert remaining_job.lease_token == remaining_job.lease_owner == ""
    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    remaining_transaction_id = remaining_job_id.removeprefix("outbox_")
    remaining_item = (first, second)[transaction_ids.index(remaining_transaction_id)]
    assert client.index_store.get_index_metadata(remaining_item["uri"]) is None
    assert not worker.proof_store.publication_path(remaining_transaction_id).exists()
    if fault_kind in {"outbox", "queue"}:
        for item, transaction_id in zip((first, second), transaction_ids, strict=True):
            assert client.index_store.get_index_metadata(item["uri"]) is None
            assert not worker.proof_store.publication_path(transaction_id).exists()

    restarted = MemoryOSClient(str(tmp_path))
    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    restarted_remaining = restarted.queue_store.get(remaining_job_id)
    assert restarted_remaining is not None and restarted_remaining.status == "pending"
    assert restarted.index_store.get_index_metadata(remaining_item["uri"]) is None
