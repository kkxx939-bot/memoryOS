# 008 Production Closure

MemoryOS is now closed around a single production path:

```text
MemoryOSClient.process_observation()
-> PredictionEngine
-> ActionContextBuilder
-> PolicyGate
-> SessionArchive
-> SessionCommitService
-> OperationCommitter
-> SourceStore / IndexStore / RelationStore / Diff / Redo
```

## No Legacy Compatibility

The old memory-centered service path was removed because it allowed a second durable-write route beside ContextOperation. A Predictive Context Database needs one operation plane so Memory, Behavior, and ActionPolicy remain semantically independent while sharing SourceStore, IndexStore, Relation, Diff, and Redo.

## Session Commit Must Commit

Session commit is not a planner-only step in production. `SessionCommitService.async_commit()` requires an OperationCommitter by default and raises if none is provided. Plan-only mode is only available through `allow_plan_only=True` for focused planner tests.

## Reward And Penalty Idempotency

Reward and penalty mutate ActionPolicy counters and q_value. They are not naturally idempotent, so ActionPolicy records recent `applied_operation_ids`. OperationCommitter passes operation_id into ActionPolicyUpdater, and Redo recovery resumes from source/index/audit/diff phases without reapplying source mutations after `source_written`.

## Relation-First Context

ActionContextBuilder first follows RelationStore:

- ActionPolicy `anchored_by` MemoryAnchor
- ActionPolicy `constrained_by` PolicyMemory
- ActionPolicy `supported_by` BehaviorPattern
- ActionPolicy `requires_resource` Resource
- ActionPolicy `requires_skill` Skill

Search is a fallback. This keeps execution context deterministic, lower token, and aligned with the audit graph.

## Cross-Session Behavior Lifecycle

BehaviorCommitPlanner creates TemporaryBehaviorCase from the current archive, then reads historical BehaviorCase and BehaviorCluster objects from ContextDB. Two similar cases can form a BehaviorCluster and MemoryAnchor; three can form a BehaviorPattern. Single short-lived behavior can be compressed or archived instead of becoming durable preference memory.

## Source And Index

SourceStore is the fact source. SQLiteIndexStore is a rebuildable derived index with FTS5 when available and explicit fallback otherwise. ConsistencyVerifier reports missing index rows, orphan index rows, deleted objects still returned by default search, and broken relations. ReindexWorker rebuilds IndexStore from SourceStore without modifying source content.

## Checklist

- [x] Single production entrypoint is `MemoryOSClient.process_observation()`.
- [x] PredictionResult does not write durable Memory operations.
- [x] SessionCommit generates real memory, behavior, action policy, and context diffs.
- [x] OperationCommitter resolves targets before coalescing and conflict resolution.
- [x] Reward and penalty are idempotent by operation_id.
- [x] Redo recovery resumes by phase.
- [x] ActionContextBuilder is relation-first and L0/L1-first.
- [x] PolicyGate reads policy memory, resource, skill, recent feedback, cooldown, and risk.
- [x] Behavior lifecycle aggregates across historical ContextDB state.
- [x] Resource and Skill are ContextObject types required by ActionPolicy.
- [x] SourceStore and IndexStore are separated and index can be rebuilt.
- [x] Old service/domain/usecase/docs paths are removed.

