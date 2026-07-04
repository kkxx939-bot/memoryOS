from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexStore, SourceStore


class ActionPolicyRetriever:
    def __init__(self, index_store: IndexStore, source_store: SourceStore) -> None:
        self.index_store = index_store
        self.source_store = source_store

    def retrieve(
        self,
        user_id: str,
        available_actions: list[str],
        scene_key: str | None = None,
        limit: int = 20,
    ) -> list[ActionPolicy]:
        allowed_actions = {self._canonical(action) for action in available_actions}
        if not allowed_actions:
            return []
        queries = []
        if scene_key:
            queries.append(scene_key)
        queries.extend(sorted(allowed_actions))
        seen: set[str] = set()
        policies: list[ActionPolicy] = []
        for query in queries or [""]:
            hits = self.index_store.search(
                query,
                filters={"owner_user_id": user_id, "context_type": ContextType.ACTION_POLICY.value},
                limit=max(limit * 3, 20),
            )
            for hit in hits:
                if hit.uri in seen:
                    continue
                seen.add(hit.uri)
                policy = self._read_policy(hit.uri)
                if policy is None:
                    continue
                if policy.user_id != user_id:
                    continue
                if self._canonical(policy.action) not in allowed_actions:
                    continue
                if scene_key and policy.scene_key != scene_key:
                    # Scene matches are preferred, but non-matching policies can
                    # still be useful only after scene-specific policies are found
                    # insufficient. Keep them after exact matches via sort below.
                    pass
                if policy.status in {ActionPolicyStatus.DELETED, ActionPolicyStatus.OBSOLETE}:
                    continue
                policies.append(policy)
        policies.sort(
            key=lambda policy: (
                0 if scene_key and policy.scene_key == scene_key else 1,
                -policy.q_value,
                -policy.confidence,
                policy.action,
            )
        )
        return policies[:limit]

    def _read_policy(self, uri: str) -> ActionPolicy | None:
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, KeyError):
            return None
        if obj.context_type != ContextType.ACTION_POLICY:
            return None
        if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.OBSOLETE, LifecycleState.ARCHIVED}:
            return None
        data = dict(obj.metadata)
        if not data:
            try:
                content = self.source_store.read_content(uri)
                data = json.loads(content) if content else {}
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, json.JSONDecodeError):
                return None
        try:
            return ActionPolicy(**data)
        except (TypeError, ValueError, KeyError):
            return None

    def _canonical(self, action: str) -> str:
        from memoryos.security.action_risk import canonical_action

        return canonical_action(action)
