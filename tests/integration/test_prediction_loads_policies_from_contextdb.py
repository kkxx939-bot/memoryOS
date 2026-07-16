from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.pipeline.observation_normalizer import ObservationNormalizer


def _seed_policy(
    client: MemoryOSClient, policy: ActionPolicy, lifecycle: LifecycleState = LifecycleState.ACTIVE
) -> None:
    anchor = ContextObject(
        uri=policy.memory_anchor_uri, context_type=ContextType.MEMORY, title="anchor", owner_user_id=policy.user_id
    )
    obj = policy.to_context_object()
    obj.lifecycle_state = lifecycle
    client.context_db.seed_object(anchor, content="anchor")
    client.context_db.commit_operation(
        ContextOperation(
            user_id=policy.user_id,
            context_type=ContextType.ACTION_POLICY,
            action=OperationAction.ADD,
            target_uri=policy.uri,
            payload={
                "context_object": obj.to_dict(),
                "content": json.dumps(policy.to_dict()),
            },
        )
    )


def _request(actions: list[str]) -> PredictionRequest:
    return PredictionRequest(
        user_id="u1",
        episode_id="ep",
        observation={"scene_key": "hot", "raw_text": "hot room", "location": "home"},
        available_actions=actions,
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    )


def test_prediction_loads_policy_from_contextdb_without_manual_policies(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        q_value=0.95,
        confidence=0.95,
    )
    _seed_policy(client, policy)

    result = client.predict(_request(["turn_on_ac", "ask_user", "do_nothing"]))

    assert result.candidates[0].action == "turn_on_ac"
    assert result.memory_operations == []


def test_available_actions_and_deleted_obsolete_filtering(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    deleted_policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_fan",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
    )
    obsolete_policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="smoke",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
    )
    _seed_policy(
        client,
        ActionPolicy(
            user_id="u1",
            scene_key="hot",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        ),
    )
    _seed_policy(
        client,
        deleted_policy,
        lifecycle=LifecycleState.DELETED,
    )
    _seed_policy(
        client,
        obsolete_policy,
        lifecycle=LifecycleState.OBSOLETE,
    )

    # Lifecycle controls the derived RelationStore, not the durable ordinary
    # Source fact.  A rebuild can still inspect the Source relation, while
    # online ActionPolicy lookup cannot traverse either retired endpoint.
    for policy in (deleted_policy, obsolete_policy):
        source = client.source_store.read_object(policy.uri)
        assert any(
            relation.relation_type == "anchored_by"
            and relation.target_uri == policy.memory_anchor_uri
            for relation in source.relations
        )
        assert client.relation_store.relations_of(policy.uri, tenant_id="default") == []

    result = client.predict(_request(["turn_on_fan", "smoke", "ask_user", "do_nothing"]))

    assert result.candidates == []
    assert result.decision.mode in {"ask_user", "do_nothing"}


def test_disabled_auto_execute_policy_ranks_but_gate_asks_user(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        q_value=0.95,
        confidence=0.95,
        auto_execute_allowed=True,
        status=ActionPolicyStatus.DISABLED_AUTO_EXECUTE,
    )
    _seed_policy(client, policy)

    result = client.predict(_request(["turn_on_ac", "ask_user", "do_nothing"]))

    assert result.candidates[0].action == "turn_on_ac"
    assert result.decision.mode == "ask_user"


def test_no_policy_does_not_crash(tmp_path) -> None:
    result = MemoryOSClient(str(tmp_path)).predict(_request(["turn_on_ac", "ask_user", "do_nothing"]))

    assert result.candidates == []
    assert result.decision.mode in {"ask_user", "do_nothing"}


def test_dict_observation_explicit_scene_key_is_preserved() -> None:
    observation = ObservationNormalizer().normalize(
        "u1",
        {
            "raw_text": "room is hot",
            "location": "home",
            "scene_key": "hot_room",
        },
    )

    assert observation.scene_key == "hot_room"


def test_dict_observation_scene_key_none_is_not_string_none() -> None:
    observation = ObservationNormalizer().normalize(
        "u1",
        {
            "raw_text": "room is hot",
            "location": "home",
            "scene_key": None,
        },
    )

    assert observation.explicit_scene_key == ""
    assert observation.scene_key != "None"


def test_packed_fallback_behavior_hit_enters_source_uris_and_archive_used_contexts(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path))
    scene_key = "hot"
    behavior = BehaviorPattern(
        user_id="u1",
        scene_key=scene_key,
        trigger_conditions={"scene_key": scene_key},
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        case_refs=["case-1"],
        action_distribution=[{"action": "turn_on_ac", "count": 1}],
    )
    client.context_db.seed_object(behavior.to_context_object(), content="hot turn_on_ac behavior")
    policy = ActionPolicy(
        user_id="u1",
        scene_key=scene_key,
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        auto_execute_allowed=True,
        q_value=0.95,
        confidence=0.95,
    )

    result = client.process_observation(
        PredictionRequest(
            user_id="u1",
            episode_id="fallback-hit",
            observation={"scene_key": scene_key, "raw_text": "hot room", "location": "home"},
            available_actions=["turn_on_ac", "ask_user", "do_nothing"],
            token_budget=2000,
            connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
        ),
        policies=[policy],
        async_commit=False,
    )

    action_context = result.prediction_result.action_context
    fallback_uris = {item["uri"] for item in action_context.packed_context["slices"]["behavior_pattern"]["items"]}
    assert behavior.uri in fallback_uris
    assert behavior.uri in action_context.source_uris

    assert result.archive_uri is not None
    archived = client.session_archive_store.read_archive(result.archive_uri)
    assert behavior.uri in {item["uri"] for item in archived.used_contexts}
