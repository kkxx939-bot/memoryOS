from __future__ import annotations

from collections.abc import Sequence

from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.extraction import LLMMemoryExtractor, RuleFallbackExtractor, RuleMemoryExtractor
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.memory.view import MemoryViewRouter


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


def test_legacy_extractors_not_used_by_memory_commit_planner() -> None:
    planner = MemoryCommitPlanner()

    assert isinstance(planner.extractor, RuleFallbackExtractor)
    assert not isinstance(planner.extractor, RuleMemoryExtractor)
    assert not isinstance(planner.extractor, LLMMemoryExtractor)


def test_memory_commit_planner_uses_schema_pipeline() -> None:
    planner = MemoryCommitPlanner()

    assert isinstance(planner.extractor, RuleFallbackExtractor)
    assert hasattr(planner, "admission_gate")
    assert hasattr(planner, "last_group")


def test_legacy_extractors_not_default() -> None:
    test_legacy_extractors_not_used_by_memory_commit_planner()


def test_memory_commit_planner_does_not_use_legacy_rule_extractor() -> None:
    assert not isinstance(MemoryCommitPlanner().extractor, RuleMemoryExtractor)


def test_legacy_llm_extractor_not_default_commit_backend() -> None:
    assert not isinstance(MemoryCommitPlanner().extractor, LLMMemoryExtractor)


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
    assert metadata["source_adapter_id"] == "codex"
    assert metadata["source_session_id"] == "s1"
    assert metadata["source_roles"] == ["user"]


def test_project_rule_accepted_with_retrieval_views() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "Project rule: MemoryOS must keep source-only audits."}],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner().plan(archive)[0]

    assert operation.payload["admission"]["decision"] == "accept"
    assert "project:memoryOS:rules" in operation.payload["retrieval_views"]


def test_user_preference_accepted_with_user_views() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer concise code review findings first."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner().plan(archive)[0]

    assert operation.payload["memory_type"] == "preference"
    assert operation.payload["admission"]["decision"] in {"accept", "pending"}
    assert "user:u1:preferences" in operation.payload["retrieval_views"]


def test_agent_experience_from_assistant_result() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "role": "assistant",
                "content": "Reusable lesson implemented. Approach: run focused checks before broad verification. Outcome: completed successfully.",
            }
        ],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner().plan(archive)[0]

    assert operation.payload["memory_type"] == "agent_experience"
    assert operation.payload["admission"]["decision"] == "accept"


def test_raw_tool_output_archive_only() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        tool_results=[{"tool_name": "shell", "tool_output": "pytest failed\nTraceback (most recent call last):\nAssertionError"}],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )
    planner = MemoryCommitPlanner()

    assert planner.plan(archive) == []
    assert planner.last_group.archive_only


def test_secret_restricted() -> None:
    planner = MemoryCommitPlanner(
        extractor=FakeExtractor(
            [
                _draft(
                    MemoryType.PREFERENCE,
                    "api key OPENAI_API_KEY=sk-test",
                    fields={"preference": "api key OPENAI_API_KEY=sk-test"},
                )
            ]
        )
    )

    assert planner.plan(
        SessionArchive(user_id="u1", session_id="s1", archive_uri="memoryos://user/u1/sessions/history/s1")
    ) == []
    assert planner.last_group.restricted


def test_memory_commit_planner_accepts_view_router_injection() -> None:
    class FixedViewRouter(MemoryViewRouter):
        def route(self, candidate, schema, *, user_id: str, project_id: str = "", adapter_id: str = "") -> list[str]:  # noqa: ANN001, ARG002
            return [f"custom:{user_id}:{project_id}:{adapter_id}"]

    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "Project rule: MemoryOS must keep OperationCommitter in the write path."}],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner(view_router=FixedViewRouter()).plan(archive)[0]

    assert operation.payload["retrieval_views"] == ["custom:u1:memoryos:codex"]
    assert operation.payload["context_object"]["metadata"]["retrieval_views"] == ["custom:u1:memoryos:codex"]
