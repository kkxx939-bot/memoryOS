"""ActionPolicy 领域自有的持久化变更语义。"""

from __future__ import annotations

import json

from infrastructure.context.relations.ordinary import ordinary_relation_specs_for_object
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from policy.action_policy.model.action_policy import ActionPolicy
from policy.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from policy.action_policy.update.action_policy_updater import ActionPolicyUpdater
from transaction.commit.control import RedoIntegrityError
from transaction.commit.domain_protocols import OperationDomainHost, RelationEligibility
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction

_POLICY_ACTIONS = frozenset(
    {
        OperationAction.REWARD,
        OperationAction.PENALIZE,
        OperationAction.COOLDOWN,
        OperationAction.SUPPRESS,
        OperationAction.DISABLE,
    }
)


class ActionPolicyCommitHandler:
    """完全由 ActionPolicy 领域拥有的事务副作用处理器。"""

    def __init__(self, updater: ActionPolicyUpdater | None = None) -> None:
        self.updater = updater or ActionPolicyUpdater()

    def handles(self, operation: ContextOperation) -> bool:
        return operation.context_type == ContextType.ACTION_POLICY and operation.action in _POLICY_ACTIONS

    @staticmethod
    def owns_object(obj: ContextObject) -> bool:
        return obj.context_type == ContextType.ACTION_POLICY

    def validate(self, host: OperationDomainHost, operation: ContextOperation) -> None:
        if operation.action == OperationAction.REWARD:
            RewardSignal.from_payload(operation.payload)
        elif operation.action == OperationAction.PENALIZE:
            PenaltySignal.from_payload(operation.payload)
        elif operation.action == OperationAction.COOLDOWN:
            cooldown_until = operation.payload.get("cooldown_until")
            if cooldown_until is not None and not isinstance(cooldown_until, str):
                raise ValueError("cooldown_until must be a string or null")
        if not operation.target_uri:
            raise ValueError(f"{operation.action.value} operation requires a target URI")
        self._read_policy(host, operation.target_uri)

    def apply_source(self, host: OperationDomainHost, operation: ContextOperation) -> None:
        if not operation.target_uri:
            raise ValueError("ActionPolicy mutation requires a target URI")
        policy = self._read_policy(host, operation.target_uri)
        if operation.action == OperationAction.REWARD:
            policy = self.updater.reward(
                policy, RewardSignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        elif operation.action == OperationAction.PENALIZE:
            policy = self.updater.penalize(
                policy, PenaltySignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        elif operation.action == OperationAction.COOLDOWN:
            policy = self.updater.cooldown(
                policy, operation.payload.get("cooldown_until"), operation_id=operation.operation_id
            )
        elif operation.action == OperationAction.SUPPRESS:
            policy = self.updater.suppress(policy, operation_id=operation.operation_id)
        elif operation.action == OperationAction.DISABLE:
            policy = self.updater.disable_auto_execute(policy, operation_id=operation.operation_id)
        self._write_policy(host, policy, operation)

    @staticmethod
    def _read_policy(host: OperationDomainHost, uri: str) -> ActionPolicy:
        obj = host.source_store.read_object(uri)
        data = dict(obj.metadata)
        if not data:
            content = host._read_content_or_empty(uri)
            data = json.loads(content) if content else {}
        return ActionPolicy(**data)

    def _write_policy(
        self,
        host: OperationDomainHost,
        policy: ActionPolicy,
        operation: ContextOperation,
    ) -> None:
        obj = self.materialize_object(host, policy.to_context_object())
        host.source_store.write_object(
            obj,
            content=json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
        )
        host._apply_relations(obj, operation)

    def validate_postcondition(
        self,
        host: OperationDomainHost,
        operation: ContextOperation,
        effect: dict,
    ) -> None:
        del host
        if not operation.target_uri:
            raise RedoIntegrityError("ActionPolicy recovery effect has no target URI")
        snapshots = {
            str(item.get("uri") or ""): item for item in effect.get("snapshots", []) or [] if isinstance(item, dict)
        }
        snapshot = snapshots.get(operation.target_uri)
        if snapshot is None or not snapshot.get("exists") or not isinstance(snapshot.get("object"), dict):
            raise RedoIntegrityError("ActionPolicy recovery effect is missing its target object")
        target = ContextObject.from_dict(snapshot["object"])
        applied = {str(item) for item in target.metadata.get("applied_operation_ids", []) or []}
        if operation.operation_id not in applied:
            raise RedoIntegrityError("ActionPolicy recovery effect is missing its operation id")

    def materialize_object(self, host: OperationDomainHost, obj: ContextObject) -> ContextObject:
        """即使关系当前不能参与在线服务，也要保留 ActionPolicy 的关系事实。

        ``ContextDB.add_relation`` 面向在线服务，会拒绝 deleted 或 obsolete 端点；
        ActionPolicy 写入中的锚点、资源和技能关系属于 Source 层事实，是否投影到可重建的
        RelationStore 则由 ``_apply_relations`` 独立判断，两种契约不能混在一起。
        """

        del host
        if not self.owns_object(obj):
            return obj
        by_identity = {
            (relation.source_uri, relation.relation_type, relation.target_uri): relation for relation in obj.relations
        }
        for spec in ordinary_relation_specs_for_object(obj):
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

    def allows_source_only_relation(
        self,
        host: OperationDomainHost,
        obj: ContextObject,
        spec: dict,
        eligibility: RelationEligibility,
    ) -> bool:
        if (
            not self.owns_object(obj)
            or obj.lifecycle_state == LifecycleState.ACTIVE
            or str(spec.get("source_uri") or "") != obj.uri
            or eligibility.reason != "source endpoint is not serving"
        ):
            return False
        schema_authority = ContextObject.from_dict(obj.to_dict())
        schema_authority.relations = []
        schema_keys = {host._relation_spec_key(item) for item in ordinary_relation_specs_for_object(schema_authority)}
        return host._relation_spec_key(spec) in schema_keys
