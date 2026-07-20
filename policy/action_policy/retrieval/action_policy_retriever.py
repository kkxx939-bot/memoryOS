"""动作策略里的动作策略检索器。"""

from __future__ import annotations

import json
from typing import Any

from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from policy.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus


class ActionPolicyRetriever:
    def __init__(self, index_store: IndexStore, source_store: SourceStore, hybrid_search: HybridSearch | None = None) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.hybrid_search = hybrid_search

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
        exact: list[ActionPolicy] = []
        fallback: list[ActionPolicy] = []
        seen_exact: set[str] = set()
        if scene_key:
            for policy in self._search_policies(user_id, [scene_key], limit=max(limit * 3, 20)):
                if policy.uri in seen_exact:
                    continue
                if not self._policy_allowed(policy, user_id, allowed_actions):
                    continue
                if policy.scene_key == scene_key:
                    policy.cross_scene_fallback = False
                    exact.append(policy)
                    seen_exact.add(policy.uri)
            exact.sort(key=self._exact_sort_key)
            if len(exact) >= limit or {policy.action for policy in exact} >= allowed_actions:
                return exact[:limit]
        seen = {policy.uri for policy in exact}
        for policy in self._search_policies(user_id, sorted(allowed_actions), limit=max(limit * 3, 20)):
            if policy.uri in seen:
                continue
            seen.add(policy.uri)
            if not self._policy_allowed(policy, user_id, allowed_actions):
                continue
            if scene_key and policy.scene_key == scene_key:
                policy.cross_scene_fallback = False
                exact.append(policy)
                continue
            policy.cross_scene_fallback = bool(scene_key)
            fallback.append(policy)
        exact.sort(key=self._exact_sort_key)
        fallback.sort(key=self._fallback_sort_key)
        # 跨场景策略只补齐当前场景缺少的动作，不能与同动作的精确场景策略竞争。
        selected_fallback: list[ActionPolicy] = []
        covered_actions = {policy.action for policy in exact}
        for policy in fallback:
            if policy.action in covered_actions:
                continue
            selected_fallback.append(policy)
            covered_actions.add(policy.action)
        return [*exact, *selected_fallback][:limit]

    def _search_policies(self, user_id: str, queries: list[str], limit: int) -> list[ActionPolicy]:
        policies: list[ActionPolicy] = []
        seen: set[str] = set()
        for query in queries or [""]:
            hits: list[Any]
            if self.hybrid_search is not None:
                hits = self.hybrid_search.search(
                    query,
                    filters={"tenant_id": self._tenant_id(), "owner_user_id": user_id},
                    namespace=f"memoryos://user/{user_id}/",
                    context_type=ContextType.ACTION_POLICY,
                    limit=max(limit * 3, 20),
                )
            else:
                hits = self.index_store.search(
                    query,
                    tenant_id=self._tenant_id(),
                    filters={
                        "tenant_id": self._tenant_id(),
                        "owner_user_id": user_id,
                        "context_type": ContextType.ACTION_POLICY.value,
                    },
                    limit=max(limit * 3, 20),
                )
            for hit in hits:
                if hit.uri in seen:
                    continue
                seen.add(hit.uri)
                policy = self._read_policy(hit.uri)
                if policy is None:
                    continue
                policies.append(policy)
        return policies

    def _policy_allowed(self, policy: ActionPolicy, user_id: str, allowed_actions: set[str]) -> bool:
        if policy.user_id != user_id:
            return False
        if self._canonical(policy.action) not in allowed_actions:
            return False
        return policy.status not in {
            ActionPolicyStatus.SUPPRESSED,
            ActionPolicyStatus.DELETED,
            ActionPolicyStatus.OBSOLETE,
        }

    def _exact_sort_key(self, policy: ActionPolicy) -> tuple[float, float, str]:
        return (-policy.q_value, -policy.confidence, policy.action)

    def _fallback_sort_key(self, policy: ActionPolicy) -> tuple[float, float, str]:
        return (-(policy.q_value * 0.7), -(policy.confidence * 0.7), policy.action)

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
        from policy.action_policy.risk import canonical_action

        return canonical_action(action)

    def _tenant_id(self) -> str:
        return str(getattr(self.source_store, "tenant_id", "default") or "default")
