# Memory, Behavior, ActionPolicy

Memory is the semantic anchor:

- explicit_memory
- anchor_memory
- memory_candidate
- confirmed_inferred_memory
- policy_memory

Behavior is the evidence chain:

- TemporaryBehaviorCase
- BehaviorCluster
- BehaviorPattern
- OpportunityStats

ActionPolicy is the prediction strategy:

- q_value
- reward_score
- penalty_score
- cooldown
- suppress
- disabled_auto_execute

Behavior updates do not overwrite Memory. Explicit rules create policy memory and constrain ActionPolicy.
