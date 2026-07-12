from __future__ import annotations

from collections.abc import Sequence

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.admission import MemoryAdmissionGate
from memoryos.memory.canonical import CandidateProposalAdapter, MemorySemanticProposal, SessionArchiveEpisodeAdapter
from memoryos.memory.extraction import RuleFallbackExtractor
from memoryos.memory.schema import (
    AdmissionDecision,
    MemoryCandidateDraft,
    MemoryType,
    MemoryTypeRegistry,
    MemoryTypeSchema,
)
from memoryos.memory.view import MemoryViewRouter


class FakeExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self, candidates: list[MemoryCandidateDraft]) -> None:
        self.candidates = candidates

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemorySemanticProposal]:
        return self.extract_with_context(
            archive,
            schemas,
            existing_memories=(),
            episode=SessionArchiveEpisodeAdapter().adapt(archive),
        )

    def extract_with_context(self, archive, schemas, *, existing_memories, episode):  # noqa: ANN001, ANN201, ARG002
        adapter = CandidateProposalAdapter()
        gate = MemoryAdmissionGate()
        return [
            adapter.adapt(candidate, episode, archive)
            for candidate in self.candidates
            if gate.evaluate(
                candidate,
                user_id=archive.user_id,
                project_id=str(archive.metadata.get("project_id") or ""),
                adapter_id="codex",
            ).decision
            in {AdmissionDecision.ACCEPT, AdmissionDecision.PENDING}
        ]


def _draft(
    memory_type: MemoryType, content: str, *, fields: dict, confidence: float = 0.9, role: str = "user"
) -> MemoryCandidateDraft:
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


class LegacyDraftOnlyExtractor:
    def extract(self, archive, schemas):  # noqa: ANN001, ANN201, ARG002
        return [_draft(MemoryType.PREFERENCE, "concise", fields={"preference": "concise"})]


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

    candidates = RuleFallbackExtractor().extract_drafts(archive, [])

    assert candidates
    assert all(isinstance(candidate, MemoryCandidateDraft) for candidate in candidates)
    assert {candidate.memory_type for candidate in candidates} >= {
        MemoryType.PREFERENCE,
        MemoryType.PROJECT_RULE,
        MemoryType.AGENT_EXPERIENCE,
    }


def test_runtime_and_planner_reject_legacy_draft_only_extractor_before_writes(tmp_path) -> None:  # noqa: ANN001
    backend = LegacyDraftOnlyExtractor()
    archive = SessionArchive(
        user_id="u1",
        session_id="legacy-draft",
        archive_uri="memoryos://user/u1/sessions/history/legacy-draft",
        messages=[{"id": "m1", "role": "user", "content": "I prefer concise answers."}],
    )

    with pytest.raises(TypeError, match="LLM MemorySemanticProposal"):
        MemoryOSClient(str(tmp_path), memory_extractor=backend)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MemorySemanticProposal"):
        MemoryCommitPlanner(extractor=backend).plan(archive)  # type: ignore[arg-type]


def test_runtime_rejects_rule_fallback_as_natural_language_semantic_backend(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(TypeError, match="LLM MemorySemanticProposal"):
        MemoryOSClient(str(tmp_path), memory_extractor=RuleFallbackExtractor())


def test_fallback_identity_is_semantic_and_unknown_constraint_is_archive_only() -> None:
    extractor = RuleFallbackExtractor()
    schemas = MemoryTypeRegistry().list()
    first = extractor.extract_drafts(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "I prefer findings first during code reviews."}],
        ),
        schemas,
    )[0]
    second = extractor.extract_drafts(
        SessionArchive(
            user_id="u1",
            session_id="s2",
            archive_uri="memoryos://user/u1/sessions/history/s2",
            messages=[{"id": "m2", "role": "user", "content": "I prefer concise code review results."}],
        ),
        schemas,
    )[0]
    unknown = extractor.extract_drafts(
        SessionArchive(
            user_id="u1",
            session_id="s3",
            archive_uri="memoryos://user/u1/sessions/history/s3",
            messages=[{"id": "m3", "role": "user", "content": "The project must preserve the frobnicator convention."}],
            metadata={"project_id": "memoryos"},
        ),
        schemas,
    )

    assert first.fields["dimension"] == second.fields["dimension"] == "code_review_style"
    assert first.merge_key == second.merge_key
    assert not any(candidate.memory_type == MemoryType.PROJECT_RULE for candidate in unknown)


def test_memory_commit_planner_uses_schema_pipeline() -> None:
    planner = MemoryCommitPlanner(extractor=RuleFallbackExtractor())

    assert isinstance(planner.extractor, RuleFallbackExtractor)
    assert hasattr(planner, "admission_gate")
    assert not hasattr(planner, "last_group")
    assert not hasattr(planner, "last_canonical_inputs")


def test_memory_commit_planner_returns_operations_for_accepted_and_pending_only() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"id": "m1", "role": "user", "content": "MemoryOS must keep URI trees stable."}],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )
    planner = MemoryCommitPlanner(
        extractor=FakeExtractor(
            [
                _draft(
                    MemoryType.PROJECT_RULE,
                    "MemoryOS must keep URI trees stable.",
                    fields={
                        "rule_topic": "uri_tree_stability",
                        "canonical_value": "keep URI trees stable",
                        "rule": "keep URI trees stable",
                        "project_id": "memoryos",
                    },
                ),
                _draft(
                    MemoryType.PROJECT_RULE,
                    "MemoryOS must maybe do something.",
                    fields={"rule": "maybe do something"},
                    confidence=0.68,
                ),
                _draft(MemoryType.EVENT, "shell output\nExit code: 1", fields={"event": "tool"}, role="tool"),
                _draft(MemoryType.PREFERENCE, "api_key=sk-test", fields={"preference": "api_key=sk-test"}),
                _draft(
                    MemoryType.EVENT,
                    "This chat discussed memory.",
                    fields={"event": "chat discussed memory"},
                    confidence=0.5,
                ),
            ]
        )
    )

    operations = planner.plan(archive)

    assert len(operations.operations) == 2
    assert {item.decision for item in operations.context.proposal_outcomes} == {"PENDING"}
    assert all(operation.payload.get("canonical_pending_proposal") is True for operation in operations.operations)
    assert {operation.payload["memory_type"] for operation in operations.operations} == {"project_rule"}
    assert {operation.payload["admission"]["decision"] for operation in operations.operations} == {"pending"}
    assert (
        len([operation for operation in operations.operations if operation.payload.get("canonical_memory") is True])
        == 0
    )
    assert all(operation.payload.get("canonical_pending_proposal") is True for operation in operations.operations)


def test_memory_operation_payload_contains_schema_metadata() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {"role": "user", "content": "Project rule: MemoryOS must keep OperationCommitter in the write path."}
        ],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive).operations[0]
    metadata = operation.payload["context_object"]["metadata"]

    assert operation.payload["memory_type"] == "project_rule"
    assert operation.payload["admission"]["decision"] == "pending"
    assert operation.payload["canonical_pending_proposal"] is True
    assert "project:memoryos:rules" in operation.payload["retrieval_views"]
    assert operation.payload["schema_version"] == "canonical_pending_proposal_v1"
    assert metadata["canonical_kind"] == "pending_proposal"
    assert metadata["proposal"]["memory_type"] == "project_rule"
    assert metadata["retrieval_views"] == operation.payload["retrieval_views"]
    assert metadata["proposal"]["metadata"]["source_adapter_id"] == "codex"
    assert metadata["proposal"]["metadata"]["source_session_id"] == "s1"
    assert metadata["source_role"] == "user"


def test_project_rule_fallback_is_pending_with_retrieval_views() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "Project rule: MemoryOS must keep source-only audits."}],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive).operations[0]

    assert operation.payload["admission"]["decision"] == "pending"
    assert operation.payload["canonical_pending_proposal"] is True
    assert "project:memoryOS:rules" in operation.payload["retrieval_views"]


def test_user_preference_accepted_with_user_views() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"role": "user", "content": "I prefer concise code review findings first."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive).operations[0]

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

    operation = MemoryCommitPlanner(extractor=RuleFallbackExtractor()).plan(archive).operations[0]

    assert operation.payload["memory_type"] == "agent_experience"
    assert operation.payload["admission"]["decision"] == "pending"
    assert operation.payload["canonical_pending_proposal"] is True


def test_raw_tool_output_archive_only() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        tool_results=[
            {"tool_name": "shell", "tool_output": "pytest failed\nTraceback (most recent call last):\nAssertionError"}
        ],
        metadata={"project_id": "memoryOS", "connect": {"adapter_id": "codex"}},
    )
    planner = MemoryCommitPlanner(extractor=RuleFallbackExtractor())

    result = planner.plan(archive)
    assert result.operations == ()
    assert "ordinary_tool_result" in result.context.salience_reasons
    assert "unconfirmed_tool_output" in result.context.salience_reasons


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

    result = planner.plan(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "api key OPENAI_API_KEY=sk-test"}],
        )
    )
    assert result.operations == ()
    assert "privacy_or_sensitivity_risk" in result.context.salience_reasons


def test_memory_commit_planner_accepts_view_router_injection() -> None:
    class FixedViewRouter(MemoryViewRouter):
        def route(self, candidate, schema, *, user_id: str, project_id: str = "", adapter_id: str = "") -> list[str]:  # noqa: ANN001, ARG002
            return [f"custom:{user_id}:{project_id}:{adapter_id}"]

    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {"role": "user", "content": "Project rule: MemoryOS must keep OperationCommitter in the write path."}
        ],
        metadata={"project_id": "memoryos", "connect": {"adapter_id": "codex"}},
    )

    operation = MemoryCommitPlanner(
        extractor=RuleFallbackExtractor(),
        view_router=FixedViewRouter(),
    ).plan(archive).operations[0]

    assert operation.payload["retrieval_views"] == ["custom:u1:memoryos:codex"]
    assert operation.payload["context_object"]["metadata"]["retrieval_views"] == ["custom:u1:memoryos:codex"]
