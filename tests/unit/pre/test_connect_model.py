"""业务前置连接模型的解析、校验和序列化测试。"""

from __future__ import annotations

import pytest

from pre.connect import CapabilityProfile, ConnectMetadata, ConnectType, PipelineMode


def test_capability_profile_roundtrip_defaults() -> None:
    profile = CapabilityProfile.from_dict(None)

    assert profile.can_write_memory is True
    assert profile.can_predict_behavior is False
    assert profile.can_execute_action is False
    assert CapabilityProfile.from_dict(profile.to_dict()) == profile


def test_connect_metadata_roundtrip_defaults_and_extra() -> None:
    metadata = ConnectMetadata.from_dict({"adapter_id": "codex", "extra": {"workspace": "/tmp/repo"}})

    assert metadata.connect_type == ConnectType.AGENT
    assert metadata.run_mode == PipelineMode.CONTEXT_REDUCTION
    assert metadata.capabilities.can_predict_behavior is False
    assert metadata.capabilities.can_execute_action is False
    assert metadata.extra == {"workspace": "/tmp/repo"}
    assert ConnectMetadata.from_dict(metadata.to_dict()) == metadata


def test_default_agent_and_action_capable_embodied_profiles() -> None:
    agent = ConnectMetadata.default_agent("claude_code")
    embodied = ConnectMetadata.action_capable_embodied(adapter_id="reachy_mini", agent_instance_id="r1")

    assert agent.connect_type == ConnectType.AGENT
    assert agent.run_mode == PipelineMode.CONTEXT_REDUCTION
    assert embodied.connect_type == ConnectType.EMBODIED
    assert embodied.run_mode == PipelineMode.ACTION_CAPABLE
    assert embodied.agent_instance_id == "r1"
    assert embodied.capabilities.can_predict_behavior is True
    assert embodied.capabilities.can_execute_action is True
    assert embodied.to_dict()["modality"] == ["text", "sensor", "action"]


def test_connect_metadata_rejects_invalid_type_and_mode() -> None:
    with pytest.raises(ValueError):
        ConnectMetadata.from_dict({"connect_type": "wearable"})

    with pytest.raises(ValueError):
        ConnectMetadata.from_dict({"run_mode": "plugin_runtime"})


def test_connect_metadata_rejects_malformed_field_types() -> None:
    for payload in (
        "not-object",
        {"modality": 1},
        {"modality": {"kind": "text"}},
        {"modality": ["text", 1]},
        {"extra": "not-object"},
        {"extra": ["workspace", "/tmp/repo"]},
        {"capabilities": "not-object"},
    ):
        with pytest.raises(ValueError):
            ConnectMetadata.from_dict(payload)  # type: ignore[arg-type]


def test_connect_metadata_rejects_empty_identity_fields() -> None:
    for field in ("adapter_id", "source_kind", "world_domain"):
        with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
            ConnectMetadata.from_dict({field: ""})
