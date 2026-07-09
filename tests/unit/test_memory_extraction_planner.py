from __future__ import annotations

from collections.abc import Sequence

from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.extraction import RuleFallbackExtractor
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema


class FakeExtractor:
    def __init__(self, candidates: list[MemoryCandidateDraft]) -> None:
        self.candidates = candidates

    def extract(self, archive: SessionArchive, schemas: Sequence[MemoryTypeSchema]) -> list[MemoryCandidateDraft]:  # noqa: ARG002
        return self.candidates


def _draft(memory_type: MemoryType, content: str, *, fields: dict, confidence: float = 0.9, role: str = "user") -> MemoryCandidateDraft:
    return MemoryCandidateDraft(
        memory_type=memory_type,
        title=content[:40],
        content=content,
        fields=fields,
        confidence=confidence,
        source_role=role,
        source_adapter_id="codex",
        source_session_id="s1",
        source_message_ids=["m1"],
        evidence=[{"source": "m1"}],
        merge_key=f"{memory_type.value}:m1",
        reason="test",
    )


def test_memory_extractor_outputs_structured_candidates() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {"id": "m1", "role": "user", "content": "I prefer concise answers during code reviews."},
            {"id": "m2", "role": "user", "content": "Project rule: MemoryOS must not change the L0/L1/L2 URI tree."},
            {
                "id": "m3",
                "role": "assistant",
                "content": "Implemented schema-driven planning. Approach: emit candidates before operations. Outcome: verified with pytest. This is reusable experience.",
            },
        ],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    candidates = RuleFallbackExtractor().extract(archive, [])

    assert candidates
    assert all(isinstance(candidate, MemoryCandidateDraft) for candidate in candidates)
    assert {candidate.memory_type for candidate in candidates} >= {
        MemoryType.PREFERENCE,
        MemoryType.PROJECT_RULE,
        MemoryType.AGENT_EXPERIENCE,
    }


def test_memory_commit_planner_returns_operations_for_accepted_and_pending_only() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )
    planner = MemoryCommitPlanner(
        extractor=FakeExtractor(
            [
                _draft(MemoryType.PROJECT_RULE, "MemoryOS must keep URI trees stable.", fields={"rule": "keep URI trees stable", "project_id": "memoryos"}),
                _draft(MemoryType.PROJECT_RULE, "MemoryOS must maybe do something.", fields={"rule": "maybe do something"}, confidence=0.68),
                _draft(MemoryType.EVENT, "shell output\nExit code: 1", fields={"event": "tool"}, role="tool"),
                _draft(MemoryType.PREFERENCE, "api_key=sk-test", fields={"preference": "api_key=sk-test"}),
                _draft(MemoryType.EVENT, "This chat discussed memory.", fields={"event": "chat discussed memory"}, confidence=0.5),
            ]
        )
    )

    operations = planner.plan(archive)

    assert len(operations) == 2
    assert planner.last_group.summary() == {
        "accepted": 1,
        "pending": 1,
        "rejected": 1,
        "archive_only": 1,
        "private_only": 0,
        "restricted": 1,
    }
    assert {operation.payload["memory_type"] for operation in operations} == {"project_rule"}
    assert {operation.payload["admission"]["decision"] for operation in operations} == {"accept", "pending"}


def test_memory_operation_payload_contains_schema_metadata() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "Project rule: MemoryOS must keep OperationCommitter in the write path."}],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner().plan(archive)[0]
    metadata = operation.payload["context_object"]["metadata"]

    assert operation.payload["memory_type"] == "project_rule"
    assert operation.payload["admission"]["decision"] == "accept"
    assert "project:memoryos:rules" in operation.payload["retrieval_views"]
    assert operation.payload["source_adapter_id"] == "codex"
    assert operation.payload["source_session_id"] == "s1"
    assert operation.payload["merge_key"]
    assert operation.payload["schema_version"] == "memory_schema_v1"
    assert metadata["memory_type"] == "project_rule"
    assert metadata["retrieval_views"] == operation.payload["retrieval_views"]
