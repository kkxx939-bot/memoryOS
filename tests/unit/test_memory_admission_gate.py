from __future__ import annotations

from memoryos.memory.admission import MemoryAdmissionGate
from memoryos.memory.schema import AdmissionDecision, MemoryCandidateDraft, MemoryType, MemoryTypeRegistry


def _candidate(memory_type: MemoryType, content: str, **kwargs) -> MemoryCandidateDraft:  # noqa: ANN003
    fields = kwargs.pop("fields", {})
    return MemoryCandidateDraft(
        memory_type=memory_type,
        title=content[:32],
        content=content,
        fields=fields,
        confidence=kwargs.pop("confidence", 0.9),
        source_role=kwargs.pop("source_role", "user"),
        source_adapter_id=kwargs.pop("source_adapter_id", "codex"),
        source_session_id="s1",
        evidence=[{"source": "test"}],
        **kwargs,
    )


def test_memory_schema_registry_builtin_types() -> None:
    registry = MemoryTypeRegistry()
    schemas = registry.by_value()

    assert set(schemas) == {
        "profile",
        "preference",
        "entity",
        "event",
        "project_rule",
        "project_decision",
        "agent_experience",
    }
    for schema in schemas.values():
        assert schema.required_fields
        assert schema.operation_mode
        assert schema.default_retrieval_views


def test_admission_gate_rejects_raw_tool_output() -> None:
    gate = MemoryAdmissionGate()
    raw_items = [
        _candidate(MemoryType.EVENT, "shell output\nExit code: 1", fields={"event": "shell"}, source_role="tool"),
        _candidate(MemoryType.EVENT, "pytest failed with AssertionError", fields={"event": "pytest"}),
        _candidate(MemoryType.EVENT, "Traceback (most recent call last):\nValueError", fields={"event": "traceback"}),
        _candidate(MemoryType.PROJECT_RULE, "diff --git a/file b/file\n@@ patch", fields={"rule": "patch", "project_id": "p1"}),
    ]

    decisions = [gate.evaluate(item, user_id="u1", project_id="p1").decision for item in raw_items]

    assert set(decisions) <= {AdmissionDecision.ARCHIVE_ONLY, AdmissionDecision.PRIVATE_ONLY}
    assert AdmissionDecision.ACCEPT not in decisions


def test_admission_gate_restricts_secret_like_content() -> None:
    gate = MemoryAdmissionGate()
    secrets = [
        _candidate(MemoryType.PREFERENCE, "OPENAI_API_KEY=sk-test", fields={"preference": "OPENAI_API_KEY=sk-test"}),
        _candidate(MemoryType.PREFERENCE, "password: hunter2", fields={"preference": "password: hunter2"}),
        _candidate(MemoryType.PREFERENCE, "token=abc123", fields={"preference": "token=abc123"}),
        _candidate(MemoryType.PREFERENCE, "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----", fields={"preference": "key"}),
    ]

    results = [gate.evaluate(item, user_id="u1", project_id="p1") for item in secrets]

    assert {result.decision for result in results} == {AdmissionDecision.RESTRICTED}
    assert all(result.restricted for result in results)


def test_admission_gate_accepts_project_rule() -> None:
    gate = MemoryAdmissionGate()
    candidate = _candidate(
        MemoryType.PROJECT_RULE,
        "MemoryOS must keep the L0/L1/L2 URI tree unchanged.",
        fields={"rule": "keep L0/L1/L2 URI tree unchanged", "project_id": "memoryos"},
    )

    result = gate.evaluate(candidate, user_id="u1", project_id="memoryos")

    assert result.decision == AdmissionDecision.ACCEPT
    assert "project:memoryos:rules" in result.retrieval_views


def test_admission_gate_accepts_user_preference() -> None:
    gate = MemoryAdmissionGate()
    candidate = _candidate(
        MemoryType.PREFERENCE,
        "I prefer concise code review findings first.",
        fields={"preference": "concise code review findings first"},
    )

    result = gate.evaluate(candidate, user_id="u1")

    assert result.decision in {AdmissionDecision.ACCEPT, AdmissionDecision.PENDING}
    assert "user:u1:preferences" in result.retrieval_views
