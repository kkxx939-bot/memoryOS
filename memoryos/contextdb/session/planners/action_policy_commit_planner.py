from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.update.action_policy_factory import ActionPolicyEvidence, ActionPolicyFactory
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import IndexStore, SourceStore
from memoryos.core.ids import stable_hash
from memoryos.memory.model.memory import Memory, MemoryAnchor, MemoryKind
from memoryos.memory.update.memory_updater import MemoryUpdater
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.security.action_risk import canonical_action


class ActionPolicyCommitPlanner:
    def __init__(self, index_store: IndexStore | None = None, source_store: SourceStore | None = None) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.memory_updater = MemoryUpdater()
        self.factory = ActionPolicyFactory()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        operations.extend(self._auto_policy_operations(archive))
        for feedback in archive.feedback:
            policy_uri = feedback.get("policy_uri") or feedback.get("action_policy_uri") or self._policy_uri_from_feedback(archive.user_id, feedback)
            reward = float(feedback.get("reward", feedback.get("reward_value", 0.0)) or 0.0)
            explicit_rule = str(feedback.get("explicit_rule", ""))
            if reward >= 0:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.REWARD,
                        target_uri=policy_uri,
                        payload={"reward": reward or 0.1, "signal_type": feedback.get("feedback_type", "implicit_positive")},
                        evidence=[{"source": "session_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
            else:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.PENALIZE,
                        target_uri=policy_uri,
                        payload={
                            "penalty": abs(reward),
                            "signal_type": feedback.get("feedback_type", "implicit_negative"),
                            "explicit_rule": explicit_rule,
                        },
                        evidence=[{"source": "session_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
            if explicit_rule:
                operations.append(self.memory_updater.policy_rule(self._policy_memory(archive.user_id, explicit_rule, policy_uri), evidence=[{"source": "explicit_negative_feedback"}]))
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.DISABLE,
                        target_uri=policy_uri,
                        payload={"explicit_rule": explicit_rule},
                        evidence=[{"source": "explicit_negative_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
        return operations

    def _auto_policy_operations(self, archive: SessionArchive) -> list[ContextOperation]:
        evidence_by_key = self._collect_evidence(archive)
        operations: list[ContextOperation] = []
        for evidence in evidence_by_key.values():
            if not evidence.memory_anchor_uri:
                continue
            if not self._stable_enough(evidence):
                continue
            if not self._anchor_exists(evidence.memory_anchor_uri):
                operations.append(
                    self.memory_updater.add_memory(
                        MemoryAnchor(
                            uri=evidence.memory_anchor_uri,
                            user_id=archive.user_id,
                            title=f"{evidence.scene_key} behavior anchor",
                            content=f"Recurring behavior theme for {evidence.scene_key}.",
                            anchor_key=f"{evidence.scene_key}_anchor",
                            supporting_behavior_uris=evidence.supported_behavior_pattern_uris,
                        ),
                        evidence=[{"source": "action_policy_auto_generation", "scene_key": evidence.scene_key}],
                    )
                )
            existing = self._read_existing_policy(archive.user_id, evidence.scene_key, evidence.action)
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
                            "memory_anchor_uri": evidence.memory_anchor_uri,
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
        for prediction in archive.predictions:
            scene_key = self._scene_from_prediction(prediction) or (scenes[0] if scenes else "default")
            for action in self._actions_from_prediction(prediction):
                evidence = self._ensure_evidence(evidence_by_key, archive.user_id, scene_key, action)
                evidence.opportunity_count += 1
                evidence.neutral_count += 1
                evidence.evidence_refs.append(f"prediction:{archive.session_id}")
                evidence.required_resource_uris.extend(resource_uris)
                evidence.required_skill_uris.extend(skill_uris)
        for feedback in archive.feedback:
            scene_key = str(feedback.get("scene_key") or (scenes[0] if scenes else "default"))
            action = str(feedback.get("action") or feedback.get("selected_action") or feedback.get("executed_action") or feedback.get("actual_action") or "")
            if not action:
                continue
            evidence = self._ensure_evidence(evidence_by_key, archive.user_id, scene_key, action)
            reward = float(feedback.get("reward", feedback.get("reward_value", 0.0)) or 0.0)
            evidence.opportunity_count += 1
            if reward > 0:
                evidence.positive_count += 1
                evidence.activation_count += 1
            elif reward < 0:
                evidence.negative_count += 1
            else:
                evidence.neutral_count += 1
            evidence.explicit_authorized = evidence.explicit_authorized or bool(feedback.get("explicit_authorized") or feedback.get("user_authorized"))
            evidence.evidence_refs.append(f"feedback:{archive.session_id}")
        for scene_key in scenes:
            for metadata, uri in self._behavior_metadata(archive.user_id, scene_key):
                anchor_uri = str(metadata.get("memory_anchor_uri", "")) or self._default_anchor_uri(archive.user_id, scene_key)
                for item in metadata.get("action_distribution", []) or []:
                    action = str(item.get("action", ""))
                    if not action:
                        continue
                    count = int(item.get("count", 0) or 0)
                    evidence = self._ensure_evidence(evidence_by_key, archive.user_id, scene_key, action)
                    evidence.memory_anchor_uri = evidence.memory_anchor_uri or anchor_uri
                    evidence.opportunity_count += count
                    evidence.positive_count += count
                    evidence.activation_count += count
                    evidence.supported_behavior_pattern_uris.append(uri)
                    evidence.evidence_refs.append(uri)
        for evidence in evidence_by_key.values():
            if not evidence.memory_anchor_uri:
                evidence.memory_anchor_uri = self._default_anchor_uri(archive.user_id, evidence.scene_key)
            evidence.evidence_refs = list(dict.fromkeys(evidence.evidence_refs))
            evidence.supported_behavior_pattern_uris = list(dict.fromkeys(evidence.supported_behavior_pattern_uris))
            evidence.required_resource_uris = list(dict.fromkeys(evidence.required_resource_uris))
            evidence.required_skill_uris = list(dict.fromkeys(evidence.required_skill_uris))
        return evidence_by_key

    def _ensure_evidence(
        self,
        evidence_by_key: dict[tuple[str, str], ActionPolicyEvidence],
        user_id: str,
        scene_key: str,
        action: str,
    ) -> ActionPolicyEvidence:
        canonical = canonical_action(action)
        key = (scene_key, canonical)
        if key not in evidence_by_key:
            evidence_by_key[key] = ActionPolicyEvidence(
                user_id=user_id,
                scene_key=scene_key,
                action=canonical,
                memory_anchor_uri=self._default_anchor_uri(user_id, scene_key),
            )
        return evidence_by_key[key]

    def _stable_enough(self, evidence: ActionPolicyEvidence) -> bool:
        return bool(evidence.supported_behavior_pattern_uris) or evidence.positive_count >= 2 or evidence.opportunity_count >= 3

    def _scene_keys(self, archive: SessionArchive) -> list[str]:
        values: list[str] = []
        for observation in archive.observations:
            scene_key = observation.get("scene_key") or observation.get("scene")
            if scene_key:
                values.append(str(scene_key))
        for prediction in archive.predictions:
            scene_key = self._scene_from_prediction(prediction)
            if scene_key:
                values.append(scene_key)
        return list(dict.fromkeys(values))

    def _scene_from_prediction(self, prediction: dict) -> str:
        observation = prediction.get("observation", {}) if isinstance(prediction, dict) else {}
        if isinstance(observation, dict) and observation.get("scene_key"):
            return str(observation["scene_key"])
        return str(prediction.get("scene_key", "")) if isinstance(prediction, dict) and prediction.get("scene_key") else ""

    def _actions_from_prediction(self, prediction: dict) -> list[str]:
        actions: list[str] = []
        if not isinstance(prediction, dict):
            return actions
        decision = prediction.get("decision", {})
        if isinstance(decision, dict) and decision.get("action"):
            actions.append(str(decision["action"]))
        for candidate in prediction.get("candidates", []) or []:
            if isinstance(candidate, dict) and candidate.get("action"):
                actions.append(str(candidate["action"]))
        return [canonical_action(action) for action in dict.fromkeys(actions) if action]

    def _behavior_metadata(self, user_id: str, scene_key: str) -> list[tuple[dict, str]]:
        if self.index_store is None:
            return []
        records: list[tuple[dict, str]] = []
        for context_type in (ContextType.BEHAVIOR_PATTERN, ContextType.BEHAVIOR_CLUSTER):
            for hit in self.index_store.search(scene_key, filters={"owner_user_id": user_id, "context_type": context_type.value}, limit=20):
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

    def _anchor_exists(self, uri: str) -> bool:
        if self.source_store is None:
            return False
        try:
            self.source_store.read_object(uri)
            return True
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False

    def _default_anchor_uri(self, user_id: str, scene_key: str) -> str:
        return f"memoryos://user/{user_id}/memories/anchors/{scene_key}_anchor"

    def _used_uris(self, archive: SessionArchive, segment: str) -> list[str]:
        prefix = f"memoryos://{segment}/"
        values = [str(item.get("uri", "")) for item in [*archive.used_contexts, *archive.used_skills] if isinstance(item, dict)]
        return [value for value in dict.fromkeys(values) if value.startswith(prefix)]

    def _policy_uri_from_feedback(self, user_id: str, feedback: dict) -> str:
        scene_key = str(feedback.get("scene_key", "default"))
        action = canonical_action(str(feedback.get("action", feedback.get("selected_action", "unknown"))))
        return f"memoryos://user/{user_id}/action_policies/{scene_key}/{action}"

    def _policy_memory(self, user_id: str, rule: str, policy_uri: str) -> Memory:
        digest = stable_hash([user_id, rule, policy_uri], length=16)
        return Memory(
            uri=f"memoryos://user/{user_id}/memories/policies/{digest}",
            user_id=user_id,
            title=rule[:48] or "policy memory",
            content=rule,
            kind=MemoryKind.POLICY,
            confidence=1.0,
            constrains_policy_uris=[policy_uri],
        )
