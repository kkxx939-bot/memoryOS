from __future__ import annotations

import json
from typing import Any, cast

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_model import SessionArchive
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
    related_candidate_refs: list[str] | None = None,
    source_text: str,
    atomic_text: str | None = None,
    utterance_mode: str = "assertion",
    attribution: str = "source_actor",
    durability: str = "durable",
    modal_force: str | None = None,
    atomicity: str = "atomic",
    event_id: str = "message:0",
) -> dict:
    proposition = atomic_text or source_text
    span_start = source_text.index(proposition)
    atomic_ref = {
        "event_id": event_id,
        "span_start": span_start,
        "span_end": span_start + len(proposition),
    }
    evidence_refs = [atomic_ref]
    value_fields = {"canonical_value": value}
    if modal_force is None:
        if memory_type == "project_rule":
            modal_force = {
                "required": "require",
                "forbidden": "forbid",
                "allowed": "allow",
                "preferred": "prefer",
                "discouraged": "discourage",
            }.get(value.casefold(), "unknown")
        elif memory_type == "preference":
            modal_force = "prefer"
        else:
            modal_force = "none"
    semantic = {
        "speech_act": speech_act,
        "commitment": commitment,
        "temporal_scope": temporal_scope,
        "relation_to_existing": relation,
        "utterance_mode": utterance_mode,
        "attribution": attribution,
        "durability": durability,
        "modal_force": modal_force,
        "atomicity": atomicity,
    }
    return {
        "proposal_id": proposal_id,
        "memory_type": memory_type,
        "identity_fields": identity_fields,
        "value_fields": value_fields,
        "semantic": semantic,
        "epistemic_status": "EXPLICIT",
        "suggested_scope_refs": scopes or [],
        "related_candidate_refs": related_candidate_refs or [],
        "evidence_refs": evidence_refs,
        "atomic_evidence_ref": atomic_ref,
        "field_evidence_refs": {
            **{f"identity.{key}": evidence_refs for key in identity_fields},
            **{f"value.{key}": evidence_refs for key in value_fields},
            **{f"semantic.{key}": evidence_refs for key in semantic},
            "transition": evidence_refs,
        },
        "confidence": 0.98,
        "source_role": "user",
    }


def test_coding_agent_event_to_projection_retrieval_and_safe_transition(tmp_path) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    initial_text = (
        "The primary storage backend is confirmed as SQLite. "
        "PostgreSQL is a future option. Redis is forbidden."
    )
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
                    source_text=initial_text,
                    atomic_text="The primary storage backend is confirmed as SQLite.",
                ),
                _proposal(
                    "p-postgres-option",
                    "project_decision",
                    {"decision_topic": "primary storage backend"},
                    "PostgreSQL",
                    speech_act="proposal",
                    commitment="exploratory",
                    temporal_scope="future",
                    relation="alternative",
                    scopes=[workspace],
                    source_text=initial_text,
                    atomic_text="PostgreSQL is a future option.",
                ),
                _proposal(
                    "p-redis-rule",
                    "project_rule",
                    {"rule_topic": "Redis"},
                    "forbidden",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                    source_text=initial_text,
                    atomic_text="Redis is forbidden.",
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
    initial_result = client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="memoryos",
        messages=[
            {
                "role": "user",
                "content": initial_text,
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
    assert options == []
    assert initial_result.pending_count == 1
    assert initial_result.pending_persisted is True
    pending = [
        obj
        for obj in client.source_store.list_objects()
        if obj.metadata.get("canonical_kind") == "pending_proposal"
    ]
    assert {obj.metadata["proposal"]["value_fields"]["canonical_value"].casefold() for obj in pending} == {
        "postgresql"
    }
    assert "EXISTING_MEMORIES=" in prompts[0]
    replacement_text = "I confirm: formally change the primary storage backend to PostgreSQL."
    provider.response = _response(
        [
            _proposal(
                "p-postgres-confirm",
                "project_decision",
                {"decision_topic": "primary storage backend"},
                "PostgreSQL",
                speech_act="correction",
                commitment="confirmed",
                relation="supersedes",
                scopes=[workspace],
                related_candidate_refs=["existing_0"],
                source_text=replacement_text,
            )
        ]
    )
    replacement_result = client.commit_agent_session(
        user_id="u1",
        session_id="s2",
        project_id="memoryos",
        messages=[
            {
                "role": "user",
                "content": replacement_text,
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
    assert backend_values == {"sqlite"}
    assert replacement_result.pending_count == 1
    replacement_pending = next(
        obj
        for obj in client.source_store.list_objects()
        if obj.metadata.get("canonical_kind") == "pending_proposal"
        and obj.metadata.get("proposal", {}).get("proposal_id") == "p-postgres-confirm"
    )
    assert replacement_pending.metadata["pending_reason_code"].endswith(
        "destructive_effect_requires_structured_review"
    )
    history = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["SUPERSEDED"],
    )
    assert history == []

    reviewable = next(
        item
        for item in client.list_pending(user_id="u1", lifecycle_states=["PENDING"])
        if item["uri"] == replacement_pending.uri
    )
    review_result = client.review_pending(
        user_id="u1",
        pending_uri=replacement_pending.uri,
        decision="CONFIRM_AND_APPLY",
        expected_lifecycle_revision=reviewable["lifecycle_revision"],
        expected_proposal_fingerprint=reviewable["proposal_fingerprint"],
        command_id="review-postgres-replacement",
        reason="structured_human_review",
    )
    assert review_result["status"] == LifecycleState.RESOLVED.value

    active_after_review = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    assert {
        item["metadata"]["canonical_value"]
        for item in active_after_review
        if item["metadata"]["memory_type"] == "project_decision"
    } == {"postgresql"}
    history = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["SUPERSEDED"],
    )
    assert {item["metadata"]["canonical_value"] for item in history} == {"sqlite"}
    assert client.list_pending(
        user_id="u1",
        lifecycle_states=[LifecycleState.RESOLVED.value],
    )[0]["uri"] == replacement_pending.uri

    model_mistake_text = (
        "SQLite is only a future option; no confirmation has been made for the primary storage backend."
    )
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
                source_text=model_mistake_text,
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
                "content": model_mistake_text,
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


def test_llm_mixed_database_statement_is_split_into_atomic_safe_outcomes(tmp_path) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    source_text = "数据库继续使用 SQLite，不要引入 Redis。数据库的 PostgreSQL 方案以后可以评估。"
    provider = FakeMemoryModelProvider(
        _response(
            [
                _proposal(
                    "mixed-sqlite",
                    "project_decision",
                    {"decision_topic": "数据库"},
                    "SQLite",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                    source_text=source_text,
                    atomic_text="数据库继续使用 SQLite",
                ),
                _proposal(
                    "mixed-redis",
                    "project_rule",
                    {"rule_topic": "Redis"},
                    "forbidden",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                    source_text=source_text,
                    atomic_text="不要引入 Redis",
                    utterance_mode="directive",
                    modal_force="forbid",
                ),
                _proposal(
                    "mixed-postgres-option",
                    "project_decision",
                    {"decision_topic": "数据库"},
                    "PostgreSQL",
                    speech_act="proposal",
                    commitment="exploratory",
                    temporal_scope="future",
                    relation="alternative",
                    scopes=[workspace],
                    source_text=source_text,
                    atomic_text="数据库的 PostgreSQL 方案以后可以评估",
                    utterance_mode="hypothetical",
                ),
            ]
        )
    )
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    connect = ConnectMetadata.default_agent("codex").to_dict()
    first = client.commit_agent_session(
        user_id="u1",
        session_id="mixed-s1",
        project_id="memoryos",
        messages=[{"role": "user", "content": source_text}],
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
    assert {item["metadata"]["canonical_value"] for item in current} == {"sqlite", "forbidden"}
    assert options == []
    assert first.pending_count == 0
    assert first.pending_persisted is False

    provider.response = _response([])
    client.commit_agent_session(
        user_id="u1",
        session_id="mixed-s2",
        project_id="memoryos",
        messages=[{"role": "assistant", "content": "PostgreSQL 更适合并发，建议评估。"}],
        connect_metadata=connect,
    )
    still_current = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="CURRENT",
    )
    assert {item["metadata"]["canonical_value"] for item in still_current} == {"sqlite", "forbidden"}
    assert not any(
        obj.metadata.get("canonical_value") == "postgresql"
        for obj in client.source_store.list_objects()
        if obj.metadata.get("canonical_kind") == "claim"
    )


@pytest.mark.parametrize(
    "text",
    [
        "我们讨论过改成 MySQL，不过最后决定继续使用 PostgreSQL",
        "不是把 PostgreSQL 改为 MySQL，而是继续使用 PostgreSQL",
    ],
)
def test_discussed_or_negated_switch_cannot_self_supersede_active_claim(tmp_path, text: str) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    base_text = "数据库继续使用 PostgreSQL"
    provider = FakeMemoryModelProvider(
        _response(
            [
                _proposal(
                    "self-target-base",
                    "project_decision",
                    {"decision_topic": "数据库"},
                    "PostgreSQL",
                    speech_act="confirmation",
                    commitment="confirmed",
                    scopes=[workspace],
                    source_text=base_text,
                )
            ]
        )
    )
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    connect = ConnectMetadata.default_agent("codex").to_dict()
    client.commit_agent_session(
        user_id="u1",
        session_id="self-target-base",
        project_id="memoryos",
        messages=[{"role": "user", "content": base_text}],
        connect_metadata=connect,
    )

    followup_text = f"数据库：{text}"
    provider.response = _response(
        [
            _proposal(
                "self-target-followup",
                "project_decision",
                {"decision_topic": "数据库"},
                "PostgreSQL",
                speech_act="confirmation",
                commitment="confirmed",
                relation="unrelated",
                scopes=[workspace],
                source_text=followup_text,
            )
        ]
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="self-target-followup",
        project_id="memoryos",
        messages=[{"role": "user", "content": followup_text}],
        connect_metadata=connect,
    )
    active = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    superseded = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["SUPERSEDED"],
    )
    assert {
        item["metadata"]["canonical_value"]
        for item in active
        if item["metadata"].get("memory_type") == "project_decision"
    } == {"postgresql"}
    assert not superseded


@pytest.mark.parametrize(
    ("text", "exception"),
    [
        ("除非只是缓存，否则不得使用 Redis", "只是缓存"),
        (
            "Unless Redis is used only as a cache, it must not be enabled.",
            "Redis is used only as a cache",
        ),
    ],
)
def test_preposed_exception_is_preserved_through_llm_canonical_pipeline(tmp_path, text: str, exception: str) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    candidate = _proposal(
        "conditional-redis",
        "project_rule",
        {"rule_topic": "Redis"},
        "forbidden",
        speech_act="confirmation",
        commitment="confirmed",
        scopes=[workspace],
        source_text=text,
        utterance_mode="directive",
        modal_force="conditional_forbid",
    )
    candidate["value_fields"]["exception"] = exception
    candidate["field_evidence_refs"]["value.exception"] = candidate["evidence_refs"]
    provider = FakeMemoryModelProvider(_response([candidate]))
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="preposed-exception",
        project_id="memoryos",
        messages=[{"role": "user", "content": text}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    active = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    assert result.pending_count == 0
    assert len(active) == 1
    claim = next(
        obj
        for obj in client.source_store.list_objects()
        if obj.metadata.get("canonical_kind") == "claim"
    )
    current_revision = claim.metadata["revisions"][claim.metadata["current_revision"] - 1]
    values = current_revision["value_fields"]
    assert values["canonical_value"] == "forbidden"
    assert values["exception"] == exception
    assert current_revision["evidence_refs"]


@pytest.mark.parametrize(
    "text",
    [
        "经理说项目必须使用 Redis",
        "我听说项目禁止使用 Redis",
        "据说项目必须使用 Redis",
        "Redis 或 PostgreSQL 必须使用",
    ],
)
def test_attributed_or_competing_subject_rule_cannot_become_active(tmp_path, text: str) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    attributed = text != "Redis 或 PostgreSQL 必须使用"
    candidate = _proposal(
        "unsafe-rule",
        "project_rule",
        {"rule_topic": "Redis usage"},
        "required",
        speech_act="observation",
        commitment="unknown",
        temporal_scope="unspecified",
        scopes=[workspace],
        source_text=text,
        attribution="third_party" if attributed else "mixed",
        atomicity="atomic" if attributed else "compound",
        modal_force="require",
    )
    provider = FakeMemoryModelProvider(_response([candidate]))
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="unsafe-rule-fallback",
        project_id="memoryos",
        messages=[{"role": "user", "content": text}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    visible = client.search_context(
        "",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )

    assert result.archive_committed is True
    assert result.pending_count == 1
    assert result.pending_persisted is True
    assert not any(item["metadata"].get("memory_type") == "project_rule" for item in visible)


def test_canonical_memory_shares_workspace_subject_but_isolates_tenant_and_workspace(tmp_path) -> None:  # noqa: ANN001
    provider = FakeMemoryModelProvider("")
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    connect = ConnectMetadata.default_agent("codex").to_dict()

    def commit(user_id: str, session_id: str, project_id: str, text: str, *, tenant_id: str = "default") -> None:
        value = "PostgreSQL" if "PostgreSQL" in text else "SQLite"
        existing = client.search_context(
            "",
            user_id=user_id,
            project_id=project_id,
            tenant_id=tenant_id,
            context_type="memory",
            memory_states=["ACTIVE"],
        )
        target = next(
            (
                item["metadata"]["claim_id"]
                for item in existing
                if item["metadata"].get("memory_type") == "project_decision"
            ),
            None,
        )
        event_id = f"{session_id}:m1"
        provider.response = _response(
            [
                _proposal(
                    f"proposal:{session_id}",
                    "project_decision",
                    {"decision_topic": "primary storage backend"},
                    value,
                    speech_act="correction" if target else "confirmation",
                    commitment="confirmed",
                    relation="supersedes" if target else "unrelated",
                    scopes=[{"namespace": "memoryos", "kind": "workspace", "id": project_id}],
                    related_candidate_refs=["existing_0"] if target else [],
                    source_text=text,
                    event_id=event_id,
                )
            ]
        )
        client.commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            messages=[{"id": event_id, "role": "user", "content": text}],
            connect_metadata=connect,
            scope={"tenant_id": tenant_id},
        )

    commit("u1", "u1-a-postgres", "workspace-a", "I confirm the primary storage backend uses PostgreSQL.")
    commit("u2", "u2-a-postgres", "workspace-a", "I confirm the primary storage backend uses PostgreSQL.")
    commit("u1", "u1-b-sqlite", "workspace-b", "I confirm the primary storage backend uses SQLite.")
    commit(
        "u1",
        "u1-other-sqlite",
        "workspace-a",
        "I confirm the primary storage backend uses SQLite.",
        tenant_id="other",
    )

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
    source_text = "My music preference is confirmed: do not play music after 22:00."
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
                    source_text=source_text,
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
                "content": source_text,
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
    first_text = "I confirm the primary storage backend is SQLite."
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
                    source_text=first_text,
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
        messages=[{"role": "user", "content": first_text}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    second_text = "I confirm the database backend remains SQLite."
    provider.response = _response(
        [
            _proposal(
                "p2",
                "project_decision",
                {"decision_topic": "database backend"},
                "SQLite",
                speech_act="confirmation",
                commitment="confirmed",
                scopes=[workspace],
                source_text=second_text,
            )
        ]
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="s2",
        project_id="memoryos",
        messages=[{"role": "user", "content": second_text}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )

    current = client.search_context(
        "backend",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="CURRENT",
    )
    decisions = [item for item in current if item["metadata"]["memory_type"] == "project_decision"]
    assert {item["metadata"]["canonical_value"] for item in decisions} == {"sqlite"}
    assert len({item["metadata"]["slot_id"] for item in decisions}) == 1


def test_llm_outage_archives_evidence_and_deferred_proposal_replays(tmp_path) -> None:  # noqa: ANN001
    workspace = {"namespace": "memoryos", "kind": "workspace", "id": "memoryos"}
    source_text = "I confirm the storage backend is SQLite."

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
                        source_text=source_text,
                    )
                ]
            )

    provider = FailOnceProvider()
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="memoryos",
        messages=[{"role": "user", "content": source_text}],
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
    remembered = cast(Any, client.remember)(
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
    assert archived.messages[0]["event_type"] == "EXPLICIT_MEMORY_COMMAND"
    assert json.loads(archived.messages[0]["content"])["command"] == "RETRACT_CANONICAL_CLAIM"
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


@pytest.mark.parametrize(
    ("memory_type", "title", "content", "extra"),
    [
        ("preference", "response style", "concise", {}),
        (
            "project_rule",
            "redis usage",
            "Redis is forbidden except as a short-term cache",
            {"constraint_polarity": "FORBID", "exception": "short-term cache"},
        ),
    ],
)
def test_structured_remember_and_forget_support_type_specific_semantics(
    tmp_path,
    memory_type: str,
    title: str,
    content: str,
    extra: dict[str, str],
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    remembered = cast(Any, client.remember)(
        user_id="u1",
        title=title,
        content=content,
        memory_type=memory_type,
        project_id="memoryos" if memory_type == "project_rule" else "",
        **extra,
    )
    active = client.source_store.read_object(remembered["uri"])
    assert active.metadata["state"] == "ACTIVE"
    initial_evidence = active.metadata["revisions"][0]["evidence_refs"][0]
    initial_archive = client.session_archive_store.read_archive(initial_evidence["source_uri"])
    command = json.loads(initial_archive.messages[0]["content"])
    assert initial_archive.messages[0]["event_type"] == "EXPLICIT_MEMORY_COMMAND"
    assert command["command"] == "REMEMBER_CANONICAL_VALUE"
    if memory_type == "project_rule":
        assert command["value_fields"]["constraint_polarity"] == "CONDITIONAL_FORBID"
        assert command["value_fields"]["exception"] == "short-term cache"

    forgotten = client.forget(user_id="u1", uri=remembered["uri"])

    assert forgotten["memory_state"] == "RETRACTED"
    retracted = client.source_store.read_object(remembered["uri"])
    assert retracted.metadata["state"] == "RETRACTED"
    assert len(retracted.metadata["revisions"]) == 2
    assert (
        client.search_context(
            content,
            user_id="u1",
            project_id="memoryos" if memory_type == "project_rule" else "",
            context_type="memory",
            query_intent="CURRENT",
        )
        == []
    )


def test_structured_project_rule_requires_explicit_polarity(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))

    with pytest.raises(ValueError, match="project_rule requires constraint_polarity"):
        client.remember(
            user_id="u1",
            title="redis usage",
            content="Redis policy",
            memory_type="project_rule",
            project_id="memoryos",
        )


def test_structured_remember_persists_ambiguous_second_value_as_pending(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    first = client.remember(
        user_id="u1",
        title="storage backend",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
    )

    second = client.remember(
        user_id="u1",
        title="storage backend",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
    )
    repeated = client.remember(
        user_id="u1",
        title="storage backend",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
    )

    assert first["status"] == "COMMITTED"
    assert second["status"] == "PENDING"
    assert repeated["status"] == "PENDING"
    assert repeated["uri"] == second["uri"]
    assert second["pending_persisted"] is True
    assert client.source_store.read_object(second["uri"]).metadata["canonical_kind"] == "pending_proposal"
    pending = client.list_pending(user_id="u1", lifecycle_states=["PENDING"])
    assert [item["uri"] for item in pending] == [second["uri"]]
    reviewable = pending[0]
    evidence = reviewable["evidence_refs"][0]
    archived = client.session_archive_store.read_archive(evidence["source_uri"])
    assert archived.messages[0]["content"] == json.dumps(
        {
            "command": "REMEMBER_CANONICAL_VALUE",
            "identity_fields": {"decision_topic": "storage backend"},
            "memory_type": "project_decision",
            "value_fields": {"canonical_value": "MySQL"},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(PermissionError, match="owner"):
        client.review_pending(
            user_id="u2",
            pending_uri=second["uri"],
            decision="REJECT",
            expected_lifecycle_revision=reviewable["lifecycle_revision"],
            expected_proposal_fingerprint=reviewable["proposal_fingerprint"],
            command_id="cross-owner-review",
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        client.review_pending(
            user_id="u1",
            pending_uri=second["uri"],
            decision="REJECT",
            expected_lifecycle_revision=reviewable["lifecycle_revision"],
            expected_proposal_fingerprint="forged",
            command_id="forged-review",
        )
    current = client.search_context(
        "storage backend",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        query_intent="CURRENT",
    )
    assert {item["metadata"]["canonical_value"] for item in current} == {"postgresql"}


@pytest.mark.parametrize(
    ("source_text", "identity", "value"),
    [
        (
            "I confirm the primary storage backend is MySQL.",
            {"decision_topic": "primary storage backend"},
            "PostgreSQL",
        ),
        (
            "Remember: Should we use PostgreSQL?",
            {"decision_topic": "PostgreSQL"},
            "PostgreSQL",
        ),
    ],
)
def test_llm_semantic_hallucination_or_question_cannot_create_active_claim(
    tmp_path,
    source_text: str,
    identity: dict[str, str],
    value: str,
) -> None:  # noqa: ANN001
    provider = FakeMemoryModelProvider(
        _response(
            [
                _proposal(
                    "unsafe-semantic",
                    "project_decision",
                    identity,
                    value,
                    speech_act="confirmation",
                    commitment="confirmed",
                    source_text=source_text,
                )
            ]
        )
    )
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))

    result = client.commit_agent_session(
        user_id="u1",
        session_id="unsafe-semantic",
        messages=[{"id": "message:0", "role": "user", "content": source_text}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )

    assert result.canonical_active_operation_count == 0
    assert result.pending_count == 1
    assert client.search_context("PostgreSQL", user_id="u1", context_type="memory") == []


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_pending_count"),
    [
        ("RETRY", "RETRYABLE", 1),
        ("CONFIRM", "CONFIRMED", 1),
        ("REJECT", "REJECTED", 0),
        ("EXPIRE", "EXPIRED", 0),
    ],
)
def test_exact_remember_after_pending_review_returns_existing_lifecycle_without_rewrite(
    tmp_path,
    decision: str,
    expected_status: str,
    expected_pending_count: int,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        user_id="u1",
        title="storage backend",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
    )
    pending = client.remember(
        user_id="u1",
        title="storage backend",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
    )
    record = client.list_pending(user_id="u1")[0]
    reviewed = client.review_pending(
        user_id="u1",
        pending_uri=pending["uri"],
        decision=decision,
        expected_lifecycle_revision=record["lifecycle_revision"],
        expected_proposal_fingerprint=record["proposal_fingerprint"],
        command_id=f"review-{decision.casefold()}",
    )

    repeated = client.remember(
        user_id="u1",
        title="storage backend",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
    )

    assert repeated["uri"] == pending["uri"]
    assert repeated["status"] == expected_status
    assert repeated["lifecycle_revision"] == reviewed["lifecycle_revision"]
    assert repeated["pending_count"] == expected_pending_count
    assert repeated["pending_persisted"] is (expected_pending_count == 1)
    assert repeated["proposal_record_persisted"] is True
    assert repeated["diff_id"] == ""


def test_structured_command_retry_reuses_nondefault_tenant_evidence_archive(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    uri = "memoryos://user/u1/sessions/history/tenant-command"
    message = {
        "id": "tenant-command",
        "role": "user",
        "event_type": "EXPLICIT_MEMORY_COMMAND",
        "content": '{"command":"RETRACT_CANONICAL_CLAIM"}',
    }
    first = client._persist_structured_command_archive(
        SessionArchive(
            user_id="u1",
            session_id="tenant-command",
            archive_uri=uri,
            messages=[message],
            metadata={"tenant_id": "t1", "structured_memory_command": True},
        )
    )
    repeated = client._persist_structured_command_archive(
        SessionArchive(
            user_id="u1",
            session_id="tenant-command",
            archive_uri=uri,
            messages=[message],
            metadata={"tenant_id": "t1", "structured_memory_command": True},
        )
    )

    assert repeated.archive_digest == first.archive_digest
    assert repeated.manifest_digest == first.manifest_digest
    assert repeated.task_id == first.task_id
    assert repeated.created_at == first.created_at
    assert client.session_archive_store.archive_exists(uri, tenant_id="t1")
    assert not client.session_archive_store.archive_exists(uri, tenant_id="default")
