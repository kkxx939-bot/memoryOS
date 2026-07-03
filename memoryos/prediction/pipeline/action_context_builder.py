from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.layers.context_packer import ContextPacker
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.source_store import IndexStore
from memoryos.prediction.model.action_context import ActionContext


class ActionContextBuilder:
    def __init__(self, index_store: IndexStore) -> None:
        self.index_store = index_store

    def build(
        self,
        user_id: str,
        top_candidates: list[ActionCandidate],
        policies: list[ActionPolicy],
        token_budget: int,
        resources: list[dict] | None = None,
        skills: list[dict] | None = None,
    ) -> ActionContext:
        actions = [candidate.action for candidate in top_candidates]
        policy_by_uri = {policy.uri: policy for policy in policies}
        sections = {
            "memory_rules": [],
            "memory_anchor": [],
            "behavior_pattern": [],
            "action_policy": [],
            "resource": resources or [],
            "skill": skills or [],
            "recent_session": [],
        }
        source_uris = []
        for candidate in top_candidates:
            policy = policy_by_uri.get(candidate.policy_uri)
            if not policy:
                continue
            source_uris.append(policy.uri)
            sections["action_policy"].append(
                {
                    "uri": policy.uri,
                    "content": policy.to_dict(),
                    "token_estimate": 120,
                }
            )
            sections["memory_anchor"].extend(self._hits(user_id, policy.memory_anchor_uri, ContextType.MEMORY))
            sections["memory_rules"].extend(self._hits(user_id, candidate.action, ContextType.MEMORY))
            sections["behavior_pattern"].extend(self._hits(user_id, policy.scene_key, ContextType.BEHAVIOR_PATTERN))
        packed = ContextPacker(
            token_budget,
            allocations={
                "memory_rules": 350,
                "memory_anchor": 200,
                "behavior_pattern": 350,
                "action_policy": 250,
                "resource": 250,
                "skill": 250,
                "recent_session": 100,
            },
        ).pack(sections)
        return ActionContext(user_id=user_id, candidate_actions=actions, packed_context=packed, source_uris=source_uris)

    def _hits(self, user_id: str, query: str, context_type: ContextType) -> list[dict]:
        return [
            {"uri": hit.uri, "content": hit.title, "token_estimate": 80}
            for hit in self.index_store.search(
                query,
                filters={"owner_user_id": user_id, "context_type": context_type.value},
                limit=4,
            )
        ]
