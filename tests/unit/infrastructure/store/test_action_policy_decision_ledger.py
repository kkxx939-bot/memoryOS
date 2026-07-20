from __future__ import annotations

from behavior.core.model.observation import Observation
from infrastructure.store.action_policy import ActionPolicyDecisionLedger
from policy.action_policy.decision.action_context import ActionContext
from policy.action_policy.decision.result import PolicyDecision, PredictionResult
from policy.action_policy.model.action_policy import ActionCandidate


def _result() -> PredictionResult:
    return PredictionResult(
        request_id="request-1",
        episode_id="episode-1",
        observation=Observation(
            user_id="u1",
            raw_text="private observation text",
            location="private location",
            explicit_scene_key="hot-room",
        ),
        candidates=[
            ActionCandidate(
                action="turn_on_fan",
                score=0.92,
                policy_uri="memoryos://user/u1/action_policies/hot-room/turn_on_fan",
                reason="test",
                features={"q_value": 0.9, "debug_text": "do not persist"},
            )
        ],
        action_context=ActionContext(
            user_id="u1",
            candidate_actions=["turn_on_fan"],
            packed_context={"secret": "private packed context"},
            source_uris=["memoryos://user/u1/private/source"],
        ),
        decision=PolicyDecision(
            mode="execute",
            allowed=True,
            action="turn_on_fan",
            reason="Low-risk action is authorized.",
        ),
    )


def test_ledger_is_tenant_scoped_immutable_and_excludes_context_bodies(tmp_path) -> None:
    ledger = ActionPolicyDecisionLedger(tmp_path)

    first = ledger.record(_result(), tenant_id="tenant-a")
    second = ledger.record(_result(), tenant_id="tenant-a")
    payload = first.read_text(encoding="utf-8")

    assert first == second
    assert first.is_file()
    assert first.is_relative_to(tmp_path / "tenants" / "tenant-a")
    assert "private observation text" not in payload
    assert "private packed context" not in payload
    assert "private/source" not in payload
    assert "do not persist" not in payload
    assert '"schema_version":"action_policy_decision_v1"' in payload


def test_ledger_rejects_unsafe_tenant_and_user_segments(tmp_path) -> None:
    ledger = ActionPolicyDecisionLedger(tmp_path)

    try:
        ledger.record(_result(), tenant_id="../tenant-a")
    except ValueError:
        pass
    else:  # pragma: no cover - 防止路径校验被意外移除。
        raise AssertionError("unsafe tenant_id must be rejected")
