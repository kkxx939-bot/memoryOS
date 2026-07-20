from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from infrastructure.store.filesystem import SessionArchiveStore
from memory.commit.evidence.errors import AsyncOutputIntegrityError
from pre.session import SessionArchive
from tests.support.session_archive import build_session_archive_store

ARCHIVE_URI = "memoryos://user/u1/sessions/history/generation-race"


class InjectedAsyncCrash(RuntimeError):
    pass


def _archive(task_id: str) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="generation-race",
        archive_uri=ARCHIVE_URI,
        metadata={"tenant_id": "default"},
        task_id=task_id,
    )


def _publish(
    store: SessionArchiveStore,
    archive: SessionArchive,
    *,
    created_at: str,
) -> None:
    identity = {
        "task_id": archive.task_id,
        "archive_uri": archive.archive_uri,
        "tenant_id": "default",
        "created_at": created_at,
        "complete": True,
    }
    store.write_async_outputs(
        archive.archive_uri,
        abstract=f"abstract-{archive.task_id}",
        overview=f"overview-{archive.task_id}",
        memory_diff={"task_id": archive.task_id, "status": "committed", "kind": "memory"},
        behavior_diff={"task_id": archive.task_id, "status": "committed", "kind": "behavior"},
        action_policy_diff={
            "task_id": archive.task_id,
            "status": "committed",
            "kind": "action_policy",
        },
        context_diff={"task_id": archive.task_id, "status": "committed", "kind": "context"},
        tenant_id="default",
        commit_group_status=identity,
        complete=True,
        task_id=archive.task_id,
        created_at=created_at,
    )


def _publish_process(
    root: str,
    task_id: str,
    created_at: str,
    barrier: Any,
    results: Any,
) -> None:
    try:
        barrier.wait(timeout=15)
        _publish(build_session_archive_store(root), _archive(task_id), created_at=created_at)
        results.put((task_id, "ok"))
    except BaseException as exc:  # pragma: no cover - surfaced in the parent.
        results.put((task_id, type(exc).__name__))


@pytest.mark.parametrize(
    ("crash_stage", "b_is_current"),
    [
        ("after_abstract.md", False),
        ("after_overview.md", False),
        ("after_memory_diff.json", False),
        ("after_behavior_diff.json", False),
        ("after_action_policy_diff.json", False),
        ("after_context_diff.json", False),
        ("after_commit_group_status.json", False),
        ("after_files", False),
        ("after_manifest", False),
        ("before_current", False),
        ("after_current", True),
    ],
)
def test_async_generation_crash_never_mixes_tasks(
    tmp_path: Path,
    crash_stage: str,
    b_is_current: bool,
) -> None:
    archive_a = _archive("task-a")
    archive_b = _archive("task-b")
    store = build_session_archive_store(tmp_path)
    _publish(store, archive_a, created_at="2026-07-12T00:00:00+00:00")

    def crash(stage: str, task_id: str) -> None:
        if task_id == archive_b.task_id and stage == crash_stage:
            raise InjectedAsyncCrash(stage)

    store.test_hook = crash
    with pytest.raises(InjectedAsyncCrash, match=crash_stage.replace(".", r"\.")):
        _publish(store, archive_b, created_at="2026-07-12T00:01:00+00:00")

    restarted = build_session_archive_store(tmp_path)
    assert restarted.async_outputs_done_for_task(archive_b) is b_is_current
    assert restarted.async_outputs_done_for_task(archive_a) is (not b_is_current)
    selected = restarted.read_async_outputs(archive_b if b_is_current else archive_a)
    selected_task = archive_b.task_id if b_is_current else archive_a.task_id
    assert selected["head"]["task_id"] == selected_task
    assert selected["manifest"]["task_id"] == selected_task
    for name in ("memory_diff", "behavior_diff", "action_policy_diff", "context_diff"):
        assert selected[name]["task_id"] == selected_task

    archive_dir = store._dir(ARCHIVE_URI)
    assert not (archive_dir / ".done").exists()
    assert not (archive_dir / "memory_diff.json").exists()


def test_async_manifest_tamper_is_quarantined_and_not_reprocessed(tmp_path: Path) -> None:
    archive = _archive("task-tamper")
    store = build_session_archive_store(tmp_path)
    _publish(store, archive, created_at="2026-07-12T00:00:00+00:00")
    generation = store._dir(ARCHIVE_URI) / "async_outputs" / archive.task_id
    (generation / "memory_diff.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AsyncOutputIntegrityError, match="digest"):
        store.read_async_outputs(archive)
    assert store.async_outputs_done_for_task(archive) is False
    assert store.last_async_output_error == "AsyncOutputIntegrityError"
    quarantine = tmp_path / "system" / "quarantine" / "async_output"
    first_records = sorted(quarantine.glob("*.json"))
    assert len(first_records) == 2
    assert store.async_outputs_done_for_task(archive) is False
    assert sorted(quarantine.glob("*.json")) == first_records


def test_concurrent_async_publish_uses_deterministic_head_cas(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_publish_process,
            args=(
                str(tmp_path),
                task_id,
                created_at,
                barrier,
                results,
            ),
        )
        for task_id, created_at in (
            ("task-old", "2026-07-12T00:00:00+00:00"),
            ("task-new", "2026-07-12T00:01:00+00:00"),
        )
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    assert {results.get(timeout=5) for _ in processes} == {
        ("task-old", "ok"),
        ("task-new", "ok"),
    }

    store = build_session_archive_store(tmp_path)
    assert store.async_outputs_done_for_task(_archive("task-new")) is True
    assert store.async_outputs_done_for_task(_archive("task-old")) is False
    selected = store.read_async_outputs(_archive("task-new"))
    assert selected["head"]["task_id"] == "task-new"
    assert selected["memory_diff"]["task_id"] == "task-new"
    generation = store._dir(ARCHIVE_URI) / "async_outputs" / "task-new"
    assert generation.stat().st_mode & 0o777 == 0o700
    assert (generation / "manifest.json").stat().st_mode & 0o777 == 0o600
