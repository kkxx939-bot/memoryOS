"""Action-policy commit semantics owned by the action-policy domain."""

from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class ActionPolicyCommitHandler:
    """Apply and materialize durable ActionPolicy mutations."""

    @staticmethod
    def _validate_action_policy_operation(committer, operation: ContextOperation) -> None:
        if operation.action == OperationAction.REWARD:
            RewardSignal.from_payload(operation.payload)
        elif operation.action == OperationAction.PENALIZE:
            PenaltySignal.from_payload(operation.payload)
        elif operation.action == OperationAction.COOLDOWN:
            cooldown_until = operation.payload.get("cooldown_until")
            if cooldown_until is not None and not isinstance(cooldown_until, str):
                raise ValueError("cooldown_until must be a string or null")

    @staticmethod
    def _apply_action_policy_mutation(committer, policy: ActionPolicy, operation: ContextOperation) -> ActionPolicy:
        if operation.action == OperationAction.REWARD:
            return committer.action_policy_updater.reward(
                policy, RewardSignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.PENALIZE:
            return committer.action_policy_updater.penalize(
                policy, PenaltySignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.COOLDOWN:
            return committer.action_policy_updater.cooldown(
                policy, operation.payload.get("cooldown_until"), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.SUPPRESS:
            return committer.action_policy_updater.suppress(policy, operation_id=operation.operation_id)
        if operation.action == OperationAction.DISABLE:
            return committer.action_policy_updater.disable_auto_execute(policy, operation_id=operation.operation_id)
        return policy

    @staticmethod
    def _read_action_policy(committer, uri: str) -> ActionPolicy:
        obj = committer.source_store.read_object(uri)
        data = dict(obj.metadata)
        if not data:
            content = committer._read_content_or_empty(uri)
            data = json.loads(content) if content else {}
        return ActionPolicy(**data)

    @staticmethod
    def _write_action_policy(committer, policy: ActionPolicy) -> None:
        obj = committer._materialize_action_policy_source_relations(policy.to_context_object())
        committer.source_store.write_object(
            obj,
            content=json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
        )
        committer._apply_relations(
            obj,
            ContextOperation(
                user_id=policy.user_id,
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.UPDATE,
                target_uri=policy.uri,
                payload={},
            ),
        )

    @staticmethod
    def _materialize_action_policy_source_relations(committer, obj: ContextObject) -> ContextObject:
        """Persist ActionPolicy relation facts even when they cannot be served.

        Public ``ContextDB.add_relation`` is an online-serving operation and
        therefore rejects deleted or obsolete endpoints.  ActionPolicy writes
        have a narrower requirement: their typed anchor/resource/skill fields
        are durable Source facts, while ``_apply_relations`` independently
        decides whether a rebuildable RelationStore row is currently eligible.
        Materializing those facts here keeps the two contracts separate.
        """

        if obj.context_type != ContextType.ACTION_POLICY:
            return obj
        by_identity = {
            (relation.source_uri, relation.relation_type, relation.target_uri): relation for relation in obj.relations
        }
        for spec in committer._relation_specs_for_object(obj):
            identity = (
                str(spec["source_uri"]),
                str(spec["relation_type"]),
                str(spec["target_uri"]),
            )
            by_identity.setdefault(
                identity,
                ContextRelation(
                    source_uri=identity[0],
                    relation_type=identity[1],
                    target_uri=identity[2],
                    weight=float(spec.get("weight", 1.0)),
                    metadata=dict(spec.get("metadata", {}) or {}),
                    created_at=str(obj.created_at or obj.updated_at or ""),
                ),
            )
        obj.relations = [by_identity[key] for key in sorted(by_identity)]
        return obj
