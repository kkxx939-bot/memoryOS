"""Builders shared by pending-memory lifecycle tests."""

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType


def _archive(*, task_id: str = "pending-task") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="pending-session",
        archive_uri="memoryos://user/u1/sessions/history/pending-session",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "Project rule: Redis must perhaps be used after review.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id=task_id,
        created_at="2026-07-11T01:00:00Z",
    )


def _pending_draft(*, missing_source: bool = False) -> MemoryCandidateDraft:
    return MemoryCandidateDraft(
        memory_type=MemoryType.PROJECT_RULE,
        title="Redis review",
        content="Project rule: Redis must perhaps be used after review.",
        fields={"rule_topic": "redis_usage", "rule": "Redis", "project_id": "memoryos"},
        confidence=0.68,
        source_role="user",
        source_adapter_id="codex",
        source_session_id="pending-session",
        source_message_ids=["missing" if missing_source else "m1"],
        merge_key="project_rule:redis_usage",
        reason="needs review",
    )


__all__ = ["_archive", "_pending_draft"]
