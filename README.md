# MemoryOS - Predictive Context Database for AI Agents

MemoryOS is a production-oriented Predictive Context Database for AI agents.

It is not a memory-only SDK. The core design is:

```text
ContextDB + Memory + Behavior + ActionPolicy + PredictionEngine
```

The system predicts likely actions from similar behavior and durable action policies, then retrieves the minimum execution context through ContextDB, and finally passes every executable decision through PolicyGate.

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

SourceStore is the source of truth. IndexStore is derived and can be rebuilt.

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

BehaviorPattern must bind to a MemoryAnchor.

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
- evidence refs
- required context types

ActionPolicy must bind to a MemoryAnchor. Reward and penalty updates are applied through ContextOperation and OperationCommitter.

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

PredictionResult never contains durable memory operations. Long-term updates happen through async SessionCommit and OperationCommitter.

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

OperationCommitter applies operations with:

- TargetResolver
- ConflictResolver
- OperationCoalescer
- PathLock
- RedoLog
- AuditWriter
- DiffWriter
- Index updates

Redo phases are tracked so interrupted writes can recover from source/index inconsistency.

## Session Commit

Session commit is two-phase:

1. `sync_archive()` writes messages, observations, predictions, feedback, used contexts, used skills, tool results, and manifest.
2. `async_commit()` generates session L0/L1 and real diffs:
   - `memory_diff.json`
   - `behavior_diff.json`
   - `action_policy_diff.json`
   - `context_diff.json`

The async planners emit ContextOperation records for Memory, Behavior, ActionPolicy, and context maintenance.

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
from memoryos.prediction.model.prediction_request import PredictionRequest

client = MemoryOSClient("./memory-root")

request = PredictionRequest(
    user_id="u1",
    episode_id="ep-001",
    observation={
        "raw_text": "Room temperature is 30C and the user is home.",
        "location": "home",
        "activity": "resting",
        "signals": ["user_present"],
        "environment": {"temperature": 30},
    },
    available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
)

policies = [
    ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/home_comfort",
        q_value=0.8,
        confidence=0.8,
    )
]

result = client.predict(request, policies)
print(result.decision.mode)
```

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
