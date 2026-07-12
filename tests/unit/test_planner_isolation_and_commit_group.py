from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import CandidateProposalAdapter, MemorySemanticProposal, SessionArchiveEpisodeAdapter
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.workers.session_commit_worker import SessionCommitWorker


class BarrierExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier

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

    def _drafts(self, archive: SessionArchive) -> list[MemoryCandidateDraft]:
        self.barrier.wait(timeout=5)
        message = archive.messages[0]
        project_id = str(archive.metadata["project_id"])
        return [
            MemoryCandidateDraft(
                memory_type=MemoryType.PROJECT_RULE,
                title=f"Rule {archive.session_id}",
                content=str(message["content"]),
                fields={
                    "rule_topic": f"rule_{archive.session_id}",
                    "rule": str(message["content"]),
                    "project_id": project_id,
                },
                confidence=0.99,
                source_role="user",
                source_adapter_id="codex",
                source_session_id=archive.session_id,
                source_message_ids=[str(message["id"])],
                merge_key=f"rule:{archive.session_id}",
            )
        ]

    def extract_with_context(self, archive, schemas, *, existing_memories, episode):  # noqa: ANN001, ANN201, ARG002
        adapter = CandidateProposalAdapter()
        return [adapter.adapt(candidate, episode, archive) for candidate in self._drafts(archive)]


def _archive(name: str) -> SessionArchive:
    return SessionArchive(
        user_id=f"user-{name}",
        session_id=name,
        archive_uri=f"memoryos://user/user-{name}/sessions/history/{name}",
        messages=[
            {
                "id": f"{name}:m1",
                "role": "user",
                "actor_id": f"user-{name}",
                "content": f"Project rule: must preserve request {name} only.",
                "occurred_at": "2026-07-11T01:00:00Z",
                "ingested_at": "2026-07-11T01:00:01Z",
                "sequence": 1,
            }
        ],
        metadata={"tenant_id": "default", "project_id": f"workspace-{name}"},
        task_id=f"task-{name}",
        created_at="2026-07-11T01:00:01Z",
    )


def _assert_isolated(result_a, result_b) -> None:  # noqa: ANN001
    assert result_a.context.session_id == "a"
    assert result_b.context.session_id == "b"
    assert result_a.context.operation_group_identity == "commit_group_task-a"
    assert result_b.context.operation_group_identity == "commit_group_task-b"
    assert {ref.event_id for ref in result_a.context.evidence_references} == {"a:m1"}
    assert {ref.event_id for ref in result_b.context.evidence_references} == {"b:m1"}
    assert all("workspace-a" in snapshot.payload_json for snapshot in result_a.context.staged_objects)
    assert all("workspace-b" in snapshot.payload_json for snapshot in result_b.context.staged_objects)
    assert not any("workspace-b" in snapshot.payload_json for snapshot in result_a.context.staged_objects)
    assert not any("workspace-a" in snapshot.payload_json for snapshot in result_b.context.staged_objects)


def test_one_planner_is_request_isolated_across_threads_and_replan() -> None:
    planner = MemoryCommitPlanner(extractor=BarrierExtractor(threading.Barrier(2)))
    archive_a, archive_b = _archive("a"), _archive("b")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(planner.plan, archive_a)
        future_b = pool.submit(planner.plan, archive_b)
        result_a, result_b = future_a.result(timeout=10), future_b.result(timeout=10)

    _assert_isolated(result_a, result_b)
    replanned_a = planner.replan(result_a.context, archive_a)
    assert {ref.event_id for ref in replanned_a.context.evidence_references} == {"a:m1"}
    assert {item.proposal.metadata["source_session_id"] for item in replanned_a.context.proposal_inputs} == {"a"}
    assert not hasattr(planner, "last_prefetch")
    assert not hasattr(planner.formation, "_planning_objects")


def test_one_planner_is_request_isolated_across_asyncio_tasks() -> None:
    planner = MemoryCommitPlanner(extractor=BarrierExtractor(threading.Barrier(2)))

    async def run_concurrently():  # noqa: ANN202
        return await asyncio.gather(
            asyncio.to_thread(planner.plan, _archive("a")),
            asyncio.to_thread(planner.plan, _archive("b")),
        )

    result_a, result_b = asyncio.run(run_concurrently())
    _assert_isolated(result_a, result_b)


def test_commit_group_restart_retries_only_failed_projection_consumer(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    text = "I confirm the primary storage backend continues to use SQLite."
    atomic = {"event_id": "m1", "span_start": 0, "span_end": len(text)}
    semantic = {
        "speech_act": "confirmation",
        "commitment": "confirmed",
        "temporal_scope": "current",
        "relation_to_existing": "unrelated",
        "utterance_mode": "assertion",
        "attribution": "source_actor",
        "durability": "durable",
        "modal_force": "none",
        "atomicity": "atomic",
    }
    identity = {"decision_topic": "primary storage backend"}
    values = {"canonical_value": "SQLite"}
    response = json.dumps(
        {
            "candidates": [
                {
                    "proposal_id": "p-consumer-recovery",
                    "memory_type": "project_decision",
                    "identity_fields": identity,
                    "value_fields": values,
                    "semantic": semantic,
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [
                        {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
                    ],
                    "evidence_refs": [atomic],
                    "atomic_evidence_ref": atomic,
                    "field_evidence_refs": {
                        **{f"identity.{key}": [atomic] for key in identity},
                        **{f"value.{key}": [atomic] for key in values},
                        **{f"semantic.{key}": [atomic] for key in semantic},
                        "transition": [atomic],
                    },
                    "confidence": 0.99,
                    "source_role": "user",
                }
            ]
        }
    )
    client = MemoryOSClient(
        str(tmp_path),
        memory_extractor=LLMMemoryExtractorBackend(FakeMemoryModelProvider(response)),
    )
    projection_calls = 0

    def fail_projection(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal projection_calls
        projection_calls += 1
        raise OSError("projection unavailable")

    monkeypatch.setattr(client.memory_projection_worker, "process_commit_group", fail_projection)
    result = client.commit_agent_session(
        user_id="u1",
        session_id="consumer-recovery",
        project_id="memoryos",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "actor_id": "u1",
                "content": text,
            }
        ],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )

    assert result.canonical_committed
    assert not result.done
    assert result.status == "derived_failed"
    assert projection_calls == 1
    before = result.commit_group_status
    assert before["canonical_status"] == "completed"
    assert before["consumers"]["projection"]["status"] == "failed"
    assert before["consumers"]["context"]["status"] == "completed"
    assert client.search_context(
        "SQLite",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )
    archived = client.session_archive_store.read_archive(result.archive_uri)
    assert not client.session_archive_store.async_outputs_done_for_task(archived)

    restarted = MemoryOSClient(str(tmp_path))
    recovery = SessionCommitWorker(restarted.session_commit_service).process_pending()
    assert recovery["recovered"] == 1
    after = restarted.session_commit_service.commit_group_store.load(result.commit_group_id)
    assert after is not None and after.complete
    assert after.canonical_attempt_count == before["canonical_attempt_count"]
    assert after.consumers["projection"].attempt_count == 2
    for consumer in ("behavior", "action_policy", "context"):
        assert after.consumers[consumer].attempt_count == before["consumers"][consumer]["attempt_count"]
    assert restarted.session_archive_store.async_outputs_done_for_task(archived)
