from __future__ import annotations

import json
from typing import Any, cast

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.workers.memory_proposal_worker import MemoryProposalWorker


def _response(candidates: list[dict]) -> str:
    return json.dumps({"candidates": candidates}, ensure_ascii=False)


def _proposal(
    proposal_id: str,
    memory_type: str,
    identity_fields: dict,
    value: str,
    *,
    speech_act: str,
    commitment: str,
    temporal_scope: str = "current",
    relation: str = "unrelated",
    scopes: list[dict] | None = None,
) -> dict:
    evidence_refs = [{"event_id": "message:0"}]
    value_fields = {"canonical_value": value}
    return {
        "proposal_id": proposal_id,
        "memory_type": memory_type,
        "identity_fields": identity_fields,
        "value_fields": value_fields,
        "semantic": {
            "speech_act": speech_act,
            "commitment": commitment,
            "temporal_scope": temporal_scope,
            "relation_to_existing": relation,
        },
        "epistemic_status": "EXPLICIT",
        "suggested_scope_refs": scopes or [],
        "evidence_refs": evidence_refs,
        "field_evidence_refs": {
            **{f"identity.{key}": evidence_refs for key in identity_fields},
            **{f"value.{key}": evidence_refs for key in value_fields},
            "semantic.speech_act": evidence_refs,
            "semantic.commitment": evidence_refs,
            "semantic.temporal_scope": evidence_refs,
            "semantic.relation_to_existing": evidence_refs,
            "transition": evidence_refs,
        },
        "confidence": 0.98,
        "source_role": "user",
    }


def test_coding_agent_event_to_projection_retrieval_and_safe_transition(tmp_path) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    provider = FakeMemoryModelProvider(
        _response(
            [
                _proposal(
                    "p-sqlite",
                    "project_decision",
                    {"decision_topic": "primary storage backend"},
                    "SQLite",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                ),
                _proposal(
                    "p-postgres-option",
                    "project_decision",
                    {"decision_topic": "primary storage backend"},
                    "PostgreSQL",
                    speech_act="future_option",
                    commitment="exploratory",
                    temporal_scope="future",
                    relation="alternative",
                    scopes=[workspace],
                ),
                _proposal(
                    "p-redis-rule",
                    "project_rule",
                    {"rule_topic": "Redis"},
                    "forbidden",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                ),
            ]
        )
    )
    prompts: list[str] = []
    provider.prompts = prompts
    client = MemoryOSClient(
        str(tmp_path),
        memory_extractor=LLMMemoryExtractorBackend(provider, model_id="fake-memory-model"),
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="memoryos",
        messages=[
            {
                "role": "user",
                "content": "The primary storage backend is confirmed as SQLite. PostgreSQL is a future option. Redis is forbidden.",
            }
        ],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    current = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="CURRENT",
    )
    options = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="OPTIONS",
    )
    assert {item["metadata"]["canonical_value"] for item in current} == {"sqlite", "forbidden"}
    assert {item["metadata"]["canonical_value"] for item in options} == {"postgresql"}
    assert "EXISTING_MEMORIES=" in prompts[0]
    assert all(item["metadata"]["projection_revision"] == item["metadata"]["revision"] for item in options)

    provider.response = _response(
        [
            _proposal(
                "p-postgres-confirm",
                "project_decision",
                {"decision_topic": "primary storage backend"},
                "PostgreSQL",
                speech_act="confirmation",
                commitment="confirmed",
                scopes=[workspace],
            )
        ]
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="s2",
        project_id="memoryos",
        messages=[
            {
                "role": "user",
                "content": "I confirm: formally change the primary storage backend to PostgreSQL.",
            }
        ],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    active_backend = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    backend_values = {
        item["metadata"]["canonical_value"]
        for item in active_backend
        if item["metadata"]["memory_type"] == "project_decision"
    }
    assert backend_values == {"postgresql"}
    history = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["SUPERSEDED"],
    )
    assert {item["metadata"]["canonical_value"] for item in history} == {"sqlite"}

    provider.response = _response(
        [
            _proposal(
                "p-model-mistake",
                "project_decision",
                {"decision_topic": "primary storage backend"},
                "SQLite",
                speech_act="confirmation",
                commitment="confirmed",
                scopes=[workspace],
            )
        ]
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="s3",
        project_id="memoryos",
        messages=[
            {
                "role": "user",
                "content": "SQLite is only a future option; no confirmation has been made for the primary storage backend.",
            }
        ],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    still_active = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    assert {
        item["metadata"]["canonical_value"]
        for item in still_active
        if item["metadata"]["memory_type"] == "project_decision"
    } == {"postgresql"}


def test_default_fallback_mixed_database_statement_and_user_confirmation(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    connect = ConnectMetadata.default_agent("codex").to_dict()
    client.commit_agent_session(
        user_id="u1",
        session_id="fallback-s1",
        project_id="memoryos",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": "这个项目继续使用 SQLite，不要引入 Redis。PostgreSQL 以后可以评估。",
            }
        ],
        connect_metadata=connect,
    )
    current = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="CURRENT",
    )
    options = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="OPTIONS",
    )
    assert {
        (item["metadata"]["memory_type"], item["metadata"]["canonical_value"], item["memory_state"]) for item in current
    } == {
        ("project_decision", "sqlite", "ACTIVE"),
        ("project_rule", "forbidden", "ACTIVE"),
    }
    postgres = next(item for item in options if item["metadata"]["canonical_value"] == "postgresql")
    assert postgres["memory_state"] == "PROPOSED"
    assert postgres["metadata"]["revisions"][-1]["qualifiers"]["phase"] == "evaluation_candidate"

    client.commit_agent_session(
        user_id="u1",
        session_id="fallback-s2",
        project_id="memoryos",
        messages=[{"id": "m2", "role": "assistant", "content": "PostgreSQL 更适合并发，建议评估。"}],
        connect_metadata=connect,
    )
    assert {
        item["metadata"]["canonical_value"]
        for item in client.search_context(
            "",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
            query_intent="CURRENT",
        )
        if item["metadata"]["memory_type"] == "project_decision"
    } == {"sqlite"}

    client.commit_agent_session(
        user_id="u1",
        session_id="fallback-s3",
        project_id="memoryos",
        messages=[{"id": "m3", "role": "user", "content": "评估通过，主存储正式改成 PostgreSQL。"}],
        connect_metadata=connect,
    )
    active = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    history = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["SUPERSEDED"],
    )
    assert {
        item["metadata"]["canonical_value"] for item in active if item["metadata"]["memory_type"] == "project_decision"
    } == {"postgresql"}
    assert {item["metadata"]["canonical_value"] for item in history} == {"sqlite"}


def test_canonical_memory_shares_workspace_subject_but_isolates_tenant_and_workspace(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    connect = ConnectMetadata.default_agent("codex").to_dict()

    def commit(user_id: str, session_id: str, project_id: str, text: str, *, tenant_id: str = "default") -> None:
        client.commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            messages=[{"id": f"{session_id}:m1", "role": "user", "content": text}],
            connect_metadata=connect,
            scope={"tenant_id": tenant_id},
        )

    commit("u1", "u1-a-sqlite", "workspace-a", "这个项目继续使用 SQLite。")
    commit("u2", "u2-a-sqlite", "workspace-a", "这个项目继续使用 SQLite。")
    commit("u1", "u1-b-sqlite", "workspace-b", "这个项目继续使用 SQLite。")
    commit("u1", "u1-other-sqlite", "workspace-a", "这个项目继续使用 SQLite。", tenant_id="other")
    commit("u1", "u1-a-postgres", "workspace-a", "主存储正式改成 PostgreSQL。")

    def active(user_id: str, project_id: str, tenant_id: str = "default") -> set[str]:
        return {
            item["metadata"]["canonical_value"]
            for item in client.search_context(
                "",
                user_id=user_id,
                project_id=project_id,
                tenant_id=tenant_id,
                context_type="memory",
                memory_states=["ACTIVE"],
            )
            if item["metadata"]["memory_type"] == "project_decision"
        }

    assert active("u1", "workspace-a") == {"postgresql"}
    assert active("u2", "workspace-a") == {"postgresql"}
    assert active("u1", "workspace-b") == {"sqlite"}
    assert active("u1", "workspace-a", "other") == {"sqlite"}


def test_reachy_compatible_origin_forms_person_environment_preference(tmp_path) -> None:  # noqa: ANN001
    principal = {"namespace": "memoryos", "kind": "principal", "id": "user_1"}
    environment = {"namespace": "memoryos", "kind": "environment", "id": "home_01"}
    provider = FakeMemoryModelProvider(
        _response(
            [
                _proposal(
                    "p-quiet-hours",
                    "preference",
                    {"subject": "music", "dimension": "22:00"},
                    "do not play music after 22:00",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[principal, environment],
                )
            ]
        )
    )
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    client.commit_agent_session(
        user_id="user_1",
        session_id="interaction_1",
        messages=[
            {
                "role": "user",
                "content": "My music preference is confirmed: do not play music after 22:00.",
            }
        ],
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini", "reachy_01").to_dict(),
        scope={
            "tenant_id": "home",
            "subjects": [{"kind": "person", "id": "user_1"}],
            "origin": {
                "world_domain": "physical",
                "connect_type": "robot",
                "adapter_id": "reachy_mini",
                "instance_id": "reachy_01",
                "primary_scope": environment,
                "qualifiers": [
                    {"kind": "location", "id": "home_01:kitchen", "parent_id": "home_01"},
                    {"kind": "asset", "id": "reachy_01", "parent_id": "home_01"},
                ],
            },
        },
    )
    results = client.search_context(
        "music",
        user_id="user_1",
        tenant_id="home",
        applicability_scopes=[principal, environment, {"kind": "asset", "id": "reachy_01"}],
        context_type="memory",
    )
    assert len(results) == 1
    applicability = results[0]["metadata"]["scope"]["applicability"]["all_of"]
    assert {(item["kind"], item["id"]) for item in applicability} == {
        ("principal", "user_1"),
        ("environment", "home_01"),
    }
    origin = results[0]["metadata"]["scope"]["origin_refs"]
    assert {(item["kind"], item["id"]) for item in origin} >= {
        ("environment", "home_01"),
        ("location", "home_01:kitchen"),
        ("asset", "reachy_01"),
    }
    visibility = results[0]["metadata"]["scope"]["visibility"]
    assert visibility["tenant_id"] == "home"


def test_runtime_aliases_converge_llm_wording_to_one_slot(tmp_path) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    provider = FakeMemoryModelProvider(
        _response(
            [
                _proposal(
                    "p1",
                    "project_decision",
                    {"decision_topic": "primary storage backend"},
                    "SQLite",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                )
            ]
        )
    )
    client = MemoryOSClient(
        str(tmp_path),
        memory_extractor=LLMMemoryExtractorBackend(provider),
        memory_aliases={
            "project_decision:decision_topic": {
                "primary storage backend": "storage_backend",
                "database backend": "storage_backend",
            }
        },
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="memoryos",
        messages=[{"role": "user", "content": "I confirm the primary storage backend is SQLite."}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    provider.response = _response(
        [
            _proposal(
                "p2",
                "project_decision",
                {"decision_topic": "database backend"},
                "PostgreSQL",
                speech_act="future_option",
                commitment="exploratory",
                scopes=[workspace],
            )
        ]
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="s2",
        project_id="memoryos",
        messages=[{"role": "user", "content": "PostgreSQL is a future database backend option."}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )

    options = client.search_context(
        "backend",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="OPTIONS",
    )
    decisions = [item for item in options if item["metadata"]["memory_type"] == "project_decision"]
    assert {item["metadata"]["canonical_value"] for item in decisions} == {"postgresql"}
    assert len({item["metadata"]["slot_id"] for item in decisions}) == 1


def test_llm_outage_archives_evidence_and_deferred_proposal_replays(tmp_path) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}

    class FailOnceProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> str:  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model unavailable")
            return _response(
                [
                    _proposal(
                        "deferred-p1",
                        "project_decision",
                        {"decision_topic": "storage backend"},
                        "SQLite",
                        speech_act="confirmation",
                        commitment="confirmed",
                        scopes=[workspace],
                    )
                ]
            )

    provider = FailOnceProvider()
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="memoryos",
        messages=[{"role": "user", "content": "I confirm the storage backend is SQLite."}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    assert not result.done
    assert result.status == "canonical_pending"
    assert not result.canonical_committed
    assert cast(Any, client.queue_store).stats().get("pending", 0) >= 1

    replay = MemoryProposalWorker(client.session_commit_service).process_pending()
    assert replay["committed"] == 1
    current = client.search_context(
        "storage",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="CURRENT",
    )
    assert {item["metadata"]["canonical_value"] for item in current} == {"sqlite"}


def test_explicit_forget_creates_retracted_revision_without_physical_delete(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    remembered = client.remember(
        user_id="u1",
        title="storage backend",
        content="SQLite",
        memory_type="project_decision",
        project_id="memoryos",
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    forgotten = client.forget(user_id="u1", uri=remembered["uri"])
    assert forgotten["memory_state"] == "RETRACTED"
    obj = client.source_store.read_object(remembered["uri"])
    assert obj.lifecycle_state.value == "active"
    assert obj.metadata["state"] == "RETRACTED"
    evidence = obj.metadata["revisions"][-1]["evidence_refs"][0]
    assert evidence["source_uri"].endswith("/sessions/history/" + evidence["event_id"])
    archived = client.session_archive_store.read_archive(evidence["source_uri"])
    assert archived.messages[0]["id"] == evidence["event_id"]
    assert archived.messages[0]["event_type"] == "RETRACTION"
    slot = client.source_store.read_object(remembered["uri"].rsplit("/claims/", 1)[0])
    assert slot.metadata["active_claim_id"] is None
    assert (
        client.search_context(
            "SQLite",
            user_id="u1",
            project_id="memoryos",
            context_type="memory",
            query_intent="CURRENT",
        )
        == []
    )
    history = client.search_context(
        "SQLite",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["RETRACTED"],
    )
    assert [item["uri"] for item in history] == [remembered["uri"]]
