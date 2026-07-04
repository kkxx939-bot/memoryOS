# MemoryOS - Predictive Context Database for AI Agents

MemoryOS is a production-oriented Predictive Context Database for AI agents.

It is not a memory-only SDK. The runtime is:

```text
ContextDB + Memory + Behavior + ActionPolicy + PredictionEngine + Operation Plane
```

The production entrypoint is `MemoryOSClient.process_observation()`. It predicts likely actions from similar behavior and durable action policies, retrieves only the execution context needed for the top candidates, gates every action through PolicyGate, archives the session, and commits durable updates through ContextOperation.

`MemoryOSClient.predict()` is a low-level prediction API. Use `process_observation()` for production flows because it connects PredictionEngine, SessionArchive, SessionCommit, OperationCommitter, Diff, Redo, SourceStore, IndexStore, and RelationStore.

## Architecture

### ContextDB

ContextDB is the shared substrate for all durable context:

- URI-addressed ContextObject records
- L0 abstract, L1 overview, and L2 source content
- Session archive and async commit
- Resource and Skill context
- hierarchical retrieval and token budget packing
- Relation graph
- SourceStore, IndexStore, QueueStore, LockStore
- Diff, Redo, Audit, and recovery

SourceStore is the source of truth. IndexStore is derived and can be rebuilt from SourceStore.

### Memory

Memory stores semantic user context:

- explicit user facts
- explicit preferences
- policy memories
- MemoryAnchor
- MemoryCandidate
- confirmed inferred memory

Behavior updates do not overwrite Memory. Memory can constrain Behavior and ActionPolicy through relations and policy memories.

### Behavior

Behavior stores evidence and patterns:

- TemporaryBehaviorCase
- BehaviorCluster
- BehaviorPattern
- OpportunityStats
- Opportunity-Aware Decay

BehaviorCluster and BehaviorPattern bind to MemoryAnchor. Behavior lifecycle planning can aggregate current session observations with historical BehaviorCase and BehaviorCluster objects from ContextDB.

### ActionPolicy

ActionPolicy is a durable ContextObject, not a temporary candidate list:

- action
- q_value
- reward_score
- penalty_score
- cooldown
- suppress
- disable
- auto_execute_allowed
- required_resource_uris
- required_skill_uris
- supported_behavior_pattern_uris
- constrained_by_memory_uris

ActionPolicy must bind to a MemoryAnchor. Reward and penalty updates are applied through ContextOperation and OperationCommitter, and are idempotent by operation_id.

### PredictionEngine

The prediction chain is:

```text
Observation
-> ObservationNormalizer
-> SimilarBehaviorRetriever
-> ActionPolicyRanker
-> ActionContextBuilder
-> PolicyGate
-> PredictionLedger
-> PredictionResult
```

PredictionResult never contains durable memory operations. Long-term updates happen through SessionCommit and OperationCommitter.

## Operation Plane

All durable writes use ContextOperation:

- add
- update
- delete
- supersede
- merge
- confirm
- reject
- reward
- penalize
- cooldown
- suppress
- disable
- archive
- compress
- refresh_layers
- reindex

OperationCommitter applies operations in this order:

```text
TargetResolver -> OperationCoalescer -> ConflictResolver -> PathLock -> apply source -> apply index -> audit -> diff -> redo commit
```

Pending and rejected operations are recorded in ContextDiff and are not applied. Redo phases are tracked so interrupted writes can resume without applying reward or penalty twice.

## Session Commit

Session commit is two-phase:

1. `sync_archive()` writes messages, observations, predictions, feedback, used contexts, used skills, tool results, and manifest.
2. `async_commit()` generates session L0/L1 and real diffs:
   - `memory_diff.json`
   - `behavior_diff.json`
   - `action_policy_diff.json`
   - `context_diff.json`

By default, `SessionCommitService` requires a real OperationCommitter. Plan-only behavior must be explicitly enabled with `allow_plan_only=True` for planner-only tests.

## Action Context

ActionContextBuilder is relation-first:

- `anchored_by` -> MemoryAnchor
- `constrained_by` -> PolicyMemory or rules
- `supported_by` -> BehaviorPattern
- `requires_resource` -> Resource
- `requires_skill` -> Skill
- `uses_session` -> recent Session

It prefers L1/L0 content and avoids loading L2 by default. Search is a fallback when relations are incomplete.

## PolicyGate

Automatic execution is always gated by:

- action risk level
- ActionPolicy status
- ActionPolicy auto_execute_allowed
- cooldown_until
- user policy memory
- recent negative feedback
- resource availability
- skill availability
- current session context

PolicyGate returns one of:

```text
execute
ask_user
suggest
do_nothing
suppress
blocked
```

## Quick Start

```python
from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.prediction.model.prediction_request import PredictionRequest

client = MemoryOSClient("./memory-root")

anchor_uri = "memoryos://user/u1/memories/anchors/home_comfort"
pattern_uri = "memoryos://user/u1/behavior/patterns/hot_room"
resource_uri = "memoryos://resources/devices/ac-living-room"
skill_uri = "memoryos://skills/smart_home/ac-control"

client.source_store.write_object(
    ContextObject(
        uri=anchor_uri,
        context_type=ContextType.MEMORY,
        title="Home comfort anchor",
        owner_user_id="u1",
        metadata={"memory_kind": "anchor_memory", "summary": "Hot room comfort behavior."},
    ),
    content="Hot room comfort behavior.",
)
client.source_store.write_object(
    ContextObject(
        uri=pattern_uri,
        context_type=ContextType.BEHAVIOR_PATTERN,
        title="Hot room pattern",
        owner_user_id="u1",
        metadata={"scene_key": "hot_room", "memory_anchor_uri": anchor_uri},
    ),
    content="The user often cools the room when temperature is high.",
)
client.source_store.write_object(ContextObject(uri=resource_uri, context_type=ContextType.RESOURCE, title="Living room AC"), content="available")
client.source_store.write_object(ContextObject(uri=skill_uri, context_type=ContextType.SKILL, title="AC control"), content="executable")

policy = ActionPolicy(
    user_id="u1",
    scene_key="hot_room",
    action="turn_on_ac",
    memory_anchor_uri=anchor_uri,
    q_value=0.85,
    confidence=0.9,
    auto_execute_allowed=True,
    required_resource_uris=[resource_uri],
    required_skill_uris=[skill_uri],
    supported_behavior_pattern_uris=[pattern_uri],
)

client.source_store.write_object(policy.to_context_object(), content="turn on ac policy")
client.index_store.upsert_index(policy.to_context_object(), content="hot room turn on ac")
client.relation_store.add_relation(ContextRelation(source_uri=policy.uri, relation_type="anchored_by", target_uri=anchor_uri, metadata={"owner_user_id": "u1"}))
client.relation_store.add_relation(ContextRelation(source_uri=policy.uri, relation_type="supported_by", target_uri=pattern_uri, metadata={"owner_user_id": "u1"}))
client.relation_store.add_relation(ContextRelation(source_uri=policy.uri, relation_type="requires_resource", target_uri=resource_uri, metadata={"owner_user_id": "u1"}))
client.relation_store.add_relation(ContextRelation(source_uri=policy.uri, relation_type="requires_skill", target_uri=skill_uri, metadata={"owner_user_id": "u1"}))

request = PredictionRequest(
    user_id="u1",
    episode_id="ep-001",
    observation={
        "scene_key": "hot_room",
        "raw_text": "Room temperature is 30C and the user is home.",
        "location": "home",
        "environment": {"temperature": 30},
    },
    available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
)

result = client.process_observation(request, [policy], archive_session=True, async_commit=True)
print(result.candidates[0].action)
print(result.decision.mode)
print(result.memory_operations)  # always []
```

After the call, the session archive contains committed `memory_diff.json`, `behavior_diff.json`, `action_policy_diff.json`, and `context_diff.json`.

## Invariants

1. Memory, Behavior, and ActionPolicy are semantically independent.
2. They share ContextDB, Operation, Relation, L0/L1/L2, Diff, and Redo.
3. BehaviorPattern and ActionPolicy require MemoryAnchor.
4. Behavior updates do not overwrite Memory.
5. Memory can constrain ActionPolicy.
6. PredictionResult does not write Memory.
7. Long-term updates happen through async commit.
8. IndexStore is derived; SourceStore is the source of truth.
9. Cooling is opportunity-aware, not time-only.
10. Automatic execution must pass PolicyGate.
