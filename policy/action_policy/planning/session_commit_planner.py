"""把真实行为证据和用户反馈规划为 ActionPolicy 写操作。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from policy.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from policy.action_policy.risk import action_spec, canonical_action
from policy.action_policy.update.action_policy_factory import ActionPolicyEvidence, ActionPolicyFactory
from policy.action_policy.update.feedback_commit_planner import FeedbackCommitPlanner
from pre.session import SessionArchive
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


class ActionPolicyCommitPlanner:
    def __init__(self, index_store: IndexStore | None = None, source_store: SourceStore | None = None) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.factory = ActionPolicyFactory()
        self.feedback_planner = FeedbackCommitPlanner()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        operations.extend(self._auto_policy_operations(archive))
        for feedback in archive.feedback:
            reward = float(feedback.get("reward", feedback.get("reward_value", 0.0)) or 0.0)
            explicit_rule = str(feedback.get("explicit_rule", ""))
            if reward == 0 and not explicit_rule:
                continue
            policy_uri = (
                feedback.get("policy_uri")
                or feedback.get("action_policy_uri")
                or self._policy_uri_from_feedback(archive.user_id, feedback)
            )
            if not policy_uri:
                continue
            if reward > 0:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.REWARD,
                        target_uri=policy_uri,
                        payload={
                            "reward": min(1.0, reward),
                            "signal_type": feedback.get("feedback_type", "implicit_positive"),
                        },
                        evidence=[{"source": "session_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
            elif reward < 0:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.PENALIZE,
                        target_uri=policy_uri,
                        payload={
                            "penalty": min(1.0, abs(reward)),
                            "signal_type": feedback.get("feedback_type", "implicit_negative"),
                            "explicit_rule": explicit_rule,
                        },
                        evidence=[{"source": "session_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
            if explicit_rule:
                operations.extend(
                    self.feedback_planner.explicit_negative_rule_operations(
                        user_id=archive.user_id,
                        policy_uri=policy_uri,
                        explicit_rule=explicit_rule,
                        signal_type=str(feedback.get("feedback_type", "explicit_negative")),
                        evidence_uri=str(feedback.get("evidence_uri", "")),
                        source_session_id=archive.session_id,
                        disable_policy=reward >= 0,
                    )
                )
        return operations

    def _auto_policy_operations(self, archive: SessionArchive) -> list[ContextOperation]:
        evidence_by_key = self._collect_evidence(archive)
        operations: list[ContextOperation] = []
        for evidence in evidence_by_key.values():
            if not evidence.support_anchor_uri:
                continue
            if not self._stable_enough(evidence):
                continue
            if not self._anchor_exists(evidence.support_anchor_uri, archive.user_id):
                # ActionPolicy 只能消费 Behavior 已提交的支撑锚点，不能代替 Behavior 创建领域对象。
                continue
            existing = self._read_existing_policy(archive.user_id, evidence.scene_key, evidence.action)
            if existing is not None and existing.status in {
                ActionPolicyStatus.SUPPRESSED,
                ActionPolicyStatus.OBSOLETE,
                ActionPolicyStatus.DELETED,
            }:
                continue
            policy = self.factory.build(evidence, existing=existing)
            operation_action = OperationAction.UPDATE if existing is not None else OperationAction.ADD
            operations.append(
                ContextOperation(
                    user_id=archive.user_id,
                    context_type=ContextType.ACTION_POLICY,
                    action=operation_action,
                    target_uri=policy.uri,
                    payload={"context_object": policy.to_context_object().to_dict(), "content": json.dumps(policy.to_dict(), ensure_ascii=False, indent=2)},
                    evidence=[
                        {
                            "source": "action_policy_auto_generation",
                            "scene_key": evidence.scene_key,
                            "action": evidence.action,
                            "support_anchor_uri": evidence.support_anchor_uri,
                        }
                    ],
                    source_session_id=archive.session_id,
                    confidence=policy.confidence,
                )
            )
        return operations

    def _collect_evidence(self, archive: SessionArchive) -> dict[tuple[str, str], ActionPolicyEvidence]:
        evidence_by_key: dict[tuple[str, str], ActionPolicyEvidence] = {}
        scenes = self._scene_keys(archive)
        resource_uris = self._used_uris(archive, "resources")
        skill_uris = self._used_uris(archive, "skills")
        for scene_key in scenes:
            for metadata, uri in self._behavior_metadata(archive.user_id, scene_key):
                anchor_uri = str(metadata.get("support_anchor_uri", "")) or self._default_anchor_uri(
                    archive.user_id,
                    scene_key,
                )
                distribution = list(metadata.get("action_distribution", []) or [])
                total_opportunities = sum(max(0, int(item.get("count", 0) or 0)) for item in distribution)
                for item in distribution:
                    action = str(item.get("action", ""))
                    if not action:
                        continue
                    count = int(item.get("count", 0) or 0)
                    evidence = self._ensure_evidence(
                        evidence_by_key,
                        archive.user_id,
                        scene_key,
                        action,
                        support_anchor_uri=anchor_uri,
                    )
                    if evidence.support_anchor_uri != anchor_uri:
                        continue
                    evidence.opportunity_count += total_opportunities
                    evidence.activation_count += count
                    evidence.supported_behavior_pattern_uris.append(uri)
                    evidence.evidence_refs.append(uri)
        # 决策候选不能反过来证明自身正确；反馈计数只由 REWARD/PENALIZE 更新，避免双计数。
        for feedback in archive.feedback:
            reward = float(feedback.get("reward", feedback.get("reward_value", 0.0)) or 0.0)
            if reward <= 0 or not bool(feedback.get("explicit_authorized") or feedback.get("user_authorized")):
                continue
            scene_key = str(feedback.get("scene_key") or (scenes[0] if scenes else "default"))
            action = str(
                feedback.get("action")
                or feedback.get("selected_action")
                or feedback.get("executed_action")
                or feedback.get("actual_action")
                or ""
            )
            authorized_evidence = evidence_by_key.get((scene_key, canonical_action(action)))
            if authorized_evidence is not None:
                authorized_evidence.explicit_authorized = True
                authorized_evidence.evidence_refs.append(f"feedback:{archive.session_id}")
        for evidence in evidence_by_key.values():
            if not evidence.support_anchor_uri:
                evidence.support_anchor_uri = self._default_anchor_uri(archive.user_id, evidence.scene_key)
            evidence.evidence_refs = list(dict.fromkeys(evidence.evidence_refs))
            evidence.supported_behavior_pattern_uris = list(dict.fromkeys(evidence.supported_behavior_pattern_uris))
            evidence.required_resource_uris = list(dict.fromkeys(evidence.required_resource_uris))
            evidence.required_skill_uris = list(dict.fromkeys(evidence.required_skill_uris))
            evidence.required_resource_uris.extend(
                uri for uri in resource_uris if uri not in evidence.required_resource_uris
            )
            evidence.required_skill_uris.extend(
                uri for uri in skill_uris if uri not in evidence.required_skill_uris
            )
        return evidence_by_key

    def _ensure_evidence(
        self,
        evidence_by_key: dict[tuple[str, str], ActionPolicyEvidence],
        user_id: str,
        scene_key: str,
        action: str,
        *,
        support_anchor_uri: str | None = None,
    ) -> ActionPolicyEvidence:
        canonical = canonical_action(action)
        key = (scene_key, canonical)
        if key not in evidence_by_key:
            evidence_by_key[key] = ActionPolicyEvidence(
                user_id=user_id,
                scene_key=scene_key,
                action=canonical,
                support_anchor_uri=support_anchor_uri or self._default_anchor_uri(user_id, scene_key),
            )
        return evidence_by_key[key]

    def _stable_enough(self, evidence: ActionPolicyEvidence) -> bool:
        spec = action_spec(evidence.action)
        return bool(
            spec.predictable
            and evidence.action not in {"ask_user", "do_nothing"}
            and (
                evidence.supported_behavior_pattern_uris
                or evidence.opportunity_count >= 3
            )
        )

    def _scene_keys(self, archive: SessionArchive) -> list[str]:
        values: list[str] = []
        for observation in archive.observations:
            scene_key = observation.get("scene_key") or observation.get("scene")
            if scene_key:
                values.append(str(scene_key))
        return list(dict.fromkeys(values))

    def _behavior_metadata(self, user_id: str, scene_key: str) -> list[tuple[dict, str]]:
        if self.index_store is None:
            return []
        records: list[tuple[dict, str]] = []
        for context_type in (ContextType.BEHAVIOR_PATTERN,):
            for hit in self.index_store.search(
                scene_key,
                tenant_id=self._tenant_id(),
                filters={
                    "tenant_id": self._tenant_id(),
                    "owner_user_id": user_id,
                    "context_type": context_type.value,
                },
                limit=20,
            ):
                metadata = dict(hit.metadata)
                if self.source_store is not None:
                    try:
                        metadata = dict(self.source_store.read_object(hit.uri).metadata)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        pass
                records.append((metadata, hit.uri))
        return records

    def _read_existing_policy(self, user_id: str, scene_key: str, action: str) -> ActionPolicy | None:
        if self.source_store is None:
            return None
        uri = f"memoryos://user/{user_id}/action_policies/{scene_key}/{canonical_action(action)}"
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
        try:
            return ActionPolicy(**dict(obj.metadata))
        except (TypeError, ValueError, KeyError):
            try:
                content = self.source_store.read_content(uri)
                return ActionPolicy(**json.loads(content))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, json.JSONDecodeError, TypeError, ValueError, KeyError):
                return None

    def _anchor_exists(self, uri: str, user_id: str) -> bool:
        if self.source_store is None:
            return False
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False
        return bool(
            obj.context_type == ContextType.BEHAVIOR_SUPPORT
            and obj.owner_user_id == user_id
            and obj.lifecycle_state == LifecycleState.ACTIVE
            and isinstance(obj.metadata, Mapping)
            and str(obj.metadata.get("support_anchor_kind") or "") == "behavior"
        )

    def _tenant_id(self) -> str:
        return str(getattr(self.source_store, "tenant_id", "default") or "default")

    def _default_anchor_uri(self, user_id: str, scene_key: str) -> str:
        return f"memoryos://user/{user_id}/support/behavior/{scene_key}_anchor"

    def _used_uris(self, archive: SessionArchive, segment: str) -> list[str]:
        prefix = f"memoryos://{segment}/"
        values = [str(item.get("uri", "")) for item in [*archive.used_contexts, *archive.used_skills] if isinstance(item, dict)]
        return [value for value in dict.fromkeys(values) if value.startswith(prefix)]

    def _policy_uri_from_feedback(self, user_id: str, feedback: dict) -> str | None:
        scene_key = str(feedback.get("scene_key", "default"))
        action = canonical_action(str(feedback.get("action", feedback.get("selected_action", ""))))
        if not action:
            return None
        return f"memoryos://user/{user_id}/action_policies/{scene_key}/{action}"
