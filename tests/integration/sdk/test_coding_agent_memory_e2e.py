from __future__ import annotations

from pathlib import Path

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    HARD_ERASE_MEMORY,
    READ_CONTEXT,
    TrustedRequestContext,
)
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.documents.model import ABSENT


def _caller(*, tenant_id: str = "default") -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id=tenant_id,
        user_id="u1",
        actor_kind="user",
        actor_id="u1",
        capabilities=frozenset(
            {
                READ_CONTEXT,
                AUTHORITATIVE_REMEMBER,
                AUTHORITATIVE_FORGET,
                HARD_ERASE_MEMORY,
            }
        ),
    )


def test_explicit_document_commands_round_trip_through_sdk(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()

    remembered = client.remember(
        "I prefer concise engineering responses.",
        target_hint="preference:Response style",
        caller=caller,
    )
    assert remembered["changed"] is True
    assert remembered["projection_status"] == "ENQUEUED"
    assert remembered["relative_path"] == "preferences.md"

    edited = client.edit_memory_document(
        remembered["document_uri"],
        "I prefer concise responses with runnable examples.",
        remembered["source_digest"],
        caller=caller,
    )
    assert edited["document_id"] == remembered["document_id"]
    assert edited["document_revision"] > remembered["document_revision"]

    history = client.list_memory_history(remembered["document_uri"], caller=caller)
    assert history["document_id"] == remembered["document_id"]
    assert len(history["revisions"]) >= 2

    forgotten = client.forget(
        remembered["document_uri"],
        mode="SOFT_FORGET",
        expected_digest=edited["source_digest"],
        caller=caller,
    )
    assert forgotten["mode"] == "SOFT_FORGET"
    assert forgotten["recoverable"] is True

    restored = client.restore_memory_revision(
        remembered["document_uri"],
        revision=1,
        expected_digest="",
        caller=caller,
    )
    assert restored["changed"] is True
    assert restored["document_id"] == remembered["document_id"]


def test_rename_and_roll_forward_merge_are_public_sdk_operations(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    target = client.remember("Merge target before", target_hint="topic:merge-target", caller=caller)
    source = client.remember("Merge source body", target_hint="topic:merge-source", caller=caller)

    renamed = client.rename_memory_document(
        target["document_uri"],
        "knowledge/entities/merge-target.md",
        target["source_digest"],
        caller=caller,
    )

    assert renamed["document_id"] == target["document_id"]
    assert renamed["document_uri"] == target["document_uri"]
    assert renamed["relative_path"] == "knowledge/entities/merge-target.md"
    waiting = client.merge_memory_documents(
        renamed["document_uri"],
        "Merge target before\n\nMerge source body",
        renamed["source_digest"],
        [
            {
                "document_uri": source["document_uri"],
                "expected_digest": source["source_digest"],
            }
        ],
        caller=caller,
    )
    assert waiting["status"] == "AWAITING_TARGET_PROJECTION"
    assert waiting["pending_document_ids"] == [source["document_id"]]

    client._process_memory_projections_or_raise()
    completed = client.resume_memory_consolidation(waiting["saga_id"], caller=caller)

    assert completed["status"] == "COMPLETED"
    assert completed["soft_forgotten_document_ids"] == [source["document_id"]]
    assert b"Merge source body" in client.memory_document_store.read_raw(
        "default",
        "u1",
        document_id=target["document_id"],
    )
    assert client.memory_document_store.read_state(
        "default",
        "u1",
        source["relative_path"],
    ) == ABSENT


def test_hard_erase_reports_derived_cleanup_and_retained_evidence(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    remembered = client.remember("Erase this exact memory document.", caller=caller)

    result = client.forget(
        remembered["document_uri"],
        mode="HARD_ERASE",
        expected_digest=remembered["source_digest"],
        caller=caller,
    )

    assert result["mode"] == "HARD_ERASE"
    assert result["recoverable"] is False
    assert result["erasure_status"] in {"ERASE_PENDING", "ERASED"}
    assert isinstance(result["pending_backends"], (list, tuple))
    assert isinstance(result["independent_evidence_retained"], (list, tuple))
    assert result["media_disclaimer"]


def test_real_session_uncertain_memory_waits_for_review_then_uses_document_committer(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    archive = SessionArchive(
        user_id="u1",
        session_id="session-auto-review",
        archive_uri="memoryos://user/u1/sessions/history/session-auto-review",
        task_id="task-auto-review",
        created_at="2026-07-17T10:00:00+08:00",
        messages=[
            {
                "id": "event-auto-review",
                "role": "user",
                "content": "请记住 auto-review-proof-token 的部署说明",
                "occurred_at": "2026-07-17T10:00:00+08:00",
            }
        ],
        metadata={"tenant_id": "default"},
    )

    committed = client.context_db.commit_session(archive, async_commit=True)

    assert committed.done is True
    assert committed.memory_document_change_count == 0
    assert committed.edit_proposal_count == 1
    assert len(committed.edit_proposal_ids) == 1
    proposal_id = committed.edit_proposal_ids[0]
    record = client.memory_review_service.review_store.load("default", "u1", proposal_id)
    assert record is not None
    assert record.independent_evidence_references == (archive.archive_uri,)
    assert client.memory_document_store.read_state("default", "u1", record.relative_path) == ABSENT
    preview = client.preview_memory_edit(proposal_id, caller=caller)
    assert preview["proposal_id"] == proposal_id
    assert "auto-review-proof-token" in preview["proposed_diff"]

    approved = client.review_memory_edit(proposal_id, "APPROVE", caller=caller)
    client._process_memory_projections_or_raise()

    assert approved["changed"] is True
    assert approved["projection_status"] == "ENQUEUED"
    assert b"auto-review-proof-token" in client.memory_document_store.read_raw(
        "default",
        "u1",
        document_id=approved["document_id"],
    )

    erased = client.forget(
        approved["document_uri"],
        mode="HARD_ERASE",
        expected_digest=approved["source_digest"],
        caller=caller,
    )
    assert erased["erasure_status"] == "ERASED"
    assert erased["independent_evidence_retained"] == [archive.archive_uri]

    restarted = MemoryOSClient(str(tmp_path))
    replayed = restarted.forget(
        approved["document_uri"],
        mode="HARD_ERASE",
        expected_digest=approved["source_digest"],
        caller=caller,
    )
    assert replayed["erasure_status"] == "ERASED"
    assert replayed["independent_evidence_retained"] == [archive.archive_uri]
