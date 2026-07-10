from __future__ import annotations

import json

import pytest

from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.memory.schema import AdmissionDecision, MemoryTypeRegistry


def _archive() -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        metadata={"connect": {"adapter_id": "codex"}},
    )


def test_fake_llm_extractor_backend_outputs_candidates() -> None:
    response = json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "preference",
                    "title": "Review style",
                    "content": "I prefer findings first during code reviews.",
                    "fields": {"preference": "findings first during code reviews"},
                    "confidence": 0.9,
                    "source_role": "user",
                    "merge_key": "pref:review",
                }
            ]
        }
    )
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(response))
    archive = _archive()

    candidates = backend.extract(archive, MemoryTypeRegistry().list())
    operations = MemoryCommitPlanner(extractor=backend).plan(archive)

    assert candidates[0].memory_type.value == "preference"
    assert operations[0].payload["admission"]["decision"] == AdmissionDecision.ACCEPT.value


def test_llm_backend_cannot_bypass_admission() -> None:
    response = json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "preference",
                    "title": "Raw output",
                    "content": "pytest failed\nTraceback (most recent call last):\nAssertionError",
                    "fields": {"preference": "pytest failed"},
                    "confidence": 0.99,
                    "source_role": "user",
                    "retrieval_views": ["project:memoryOS:rules"],
                },
                {
                    "memory_type": "preference",
                    "title": "Secret",
                    "content": "api_key=sk-test",
                    "fields": {"preference": "api_key=sk-test"},
                    "confidence": 0.99,
                    "source_role": "user",
                },
            ]
        }
    )
    planner = MemoryCommitPlanner(extractor=LLMMemoryExtractorBackend(FakeMemoryModelProvider(response)))

    operations = planner.plan(_archive())

    assert operations == []
    assert planner.last_group.archive_only
    assert planner.last_group.restricted


def test_llm_backend_rejects_illegal_memory_type() -> None:
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider('{"candidates":[{"memory_type":"tool_log","content":"x","fields":{},"source_role":"user"}]}'))

    with pytest.raises(ValueError):
        backend.extract(_archive(), MemoryTypeRegistry().list())
