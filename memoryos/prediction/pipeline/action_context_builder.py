"""预测模块里的动作上下文组装。"""

from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.layers.context_packer import ContextPacker
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexHit, IndexStore, RelationStore, SourceStore
from memoryos.prediction.model.action_context import ActionContext


class ActionContextBuilder:
    relation_types = {
        "anchored_by": "memory_anchor",
        "constrained_by": "memory_rules",
        "supported_by": "behavior_pattern",
        "updated_by": "behavior_pattern",
        "requires_resource": "resource",
        "requires_skill": "skill",
        "uses_session": "recent_session",
        "evidence_for": "behavior_pattern",
    }

    def __init__(
        self,
        index_store: IndexStore,
        source_store: SourceStore | None = None,
        relation_store: RelationStore | None = None,
        context_packer: ContextPacker | None = None,
    ) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.relation_store = relation_store
        self.context_packer = context_packer

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
        for candidate in top_candidates:
            policy = policy_by_uri.get(candidate.policy_uri)
            if not policy:
                continue
            sections["action_policy"].append(
                {
                    "uri": policy.uri,
                    "content": policy.to_dict(),
                    "token_estimate": 120,
                    "layer": "metadata",
                }
            )
            relation_sections = self._relation_sections(policy.uri, user_id=user_id, token_budget_remaining=token_budget, candidate_score=candidate.score)
            for section, items in relation_sections.items():
                sections[section].extend(items)
            if not relation_sections.get("memory_anchor"):
                sections["memory_anchor"].extend(self._hits(user_id, policy.memory_anchor_uri, ContextType.MEMORY, token_budget_remaining=token_budget))
            if not relation_sections.get("memory_rules"):
                sections["memory_rules"].extend(self._hits(user_id, candidate.action, ContextType.MEMORY, token_budget_remaining=token_budget))
            if not relation_sections.get("behavior_pattern"):
                sections["behavior_pattern"].extend(self._hits(user_id, policy.scene_key, ContextType.BEHAVIOR_PATTERN, token_budget_remaining=token_budget))
        packer = self.context_packer or ContextPacker(
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
        )
        packed = packer.pack(sections)
        source_uris = self._packed_source_uris(packed)
        return ActionContext(user_id=user_id, candidate_actions=actions, packed_context=packed, source_uris=source_uris)

    def _relation_sections(
        self,
        policy_uri: str,
        user_id: str,
        token_budget_remaining: int,
        candidate_score: float,
    ) -> dict[str, list[dict]]:
        if self.relation_store is None:
            return {}
        sections: dict[str, list[dict]] = {}
        for relation in self.relation_store.relations_of(policy_uri, owner_user_id=user_id):
            if relation.source_uri != policy_uri:
                continue
            section = self.relation_types.get(relation.relation_type)
            if not section:
                continue
            item = self._object_context(
                relation.target_uri,
                user_id=user_id,
                section=section,
                token_budget_remaining=token_budget_remaining,
                candidate_score=candidate_score,
            )
            if item is None:
                if self.source_store is not None:
                    continue
                item = {
                    "uri": relation.target_uri,
                    "content": relation.metadata.get("summary", relation.relation_type),
                    "token_estimate": int(relation.metadata.get("token_estimate", 80)),
                    "layer": "fallback",
                }
            item["relation_type"] = relation.relation_type
            sections.setdefault(section, []).append(item)
        return sections

    def _hits(self, user_id: str, query: str, context_type: ContextType, token_budget_remaining: int) -> list[dict]:
        hits = self.index_store.search(query, filters={"owner_user_id": user_id, "context_type": context_type.value}, limit=4)
        items = []
        for hit in hits:
            item = self._hit_context(hit, user_id, token_budget_remaining=token_budget_remaining)
            if item is not None:
                items.append(item)
        return items

    def _hit_context(self, hit: IndexHit, user_id: str, token_budget_remaining: int) -> dict | None:
        item = self._object_context(
            hit.uri,
            user_id=user_id,
            section=self._section_for_type(hit.context_type),
            token_budget_remaining=token_budget_remaining,
            candidate_score=hit.score,
        )
        if item is not None:
            return item
        return {"uri": hit.uri, "content": hit.title, "token_estimate": 80, "score": hit.score, "layer": "fallback"}

    def _object_context(
        self,
        uri: str,
        user_id: str,
        section: str,
        token_budget_remaining: int = 1000,
        candidate_score: float = 0.0,
    ) -> dict | None:
        if self.source_store is None:
            return None
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
        if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.OBSOLETE}:
            return None
        if obj.owner_user_id not in {None, user_id} and not obj.uri.startswith(("memoryos://resources/", "memoryos://skills/")):
            return None
        layer_content, layer = self._read_best_layer(obj, section, token_budget_remaining, candidate_score=candidate_score)
        return {
            "uri": obj.uri,
            "title": obj.title,
            "context_type": obj.context_type.value,
            "content": layer_content,
            "layer": layer,
            "metadata": obj.metadata,
            "token_estimate": max(40, min(300, len(str(layer_content).split()) + 40)),
        }

    def _read_best_layer(self, obj, section: str, token_budget_remaining: int = 1000, candidate_score: float = 0.0):
        if section == "action_policy":
            return obj.metadata, "metadata"
        if token_budget_remaining <= 120:
            preferred = [(obj.layers.l0_uri, "l0"), (obj.layers.l1_uri, "l1")]
        else:
            preferred = [(obj.layers.l1_uri, "l1"), (obj.layers.l0_uri, "l0")]
        if section == "recent_session":
            preferred = [(obj.layers.l0_uri, "l0"), (obj.layers.l1_uri, "l1")]
        elif token_budget_remaining >= 1200 and candidate_score >= 0.85:
            preferred = [(obj.layers.l2_uri, "l2"), *preferred]
        for layer_uri, layer in preferred:
            if not layer_uri:
                continue
            try:
                content = self.source_store.read_content(layer_uri) if self.source_store is not None else obj.title
                return content, layer
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
        if obj.metadata.get("summary"):
            return obj.metadata["summary"], "metadata"
        return obj.title, "fallback"

    def _section_for_type(self, context_type: str) -> str:
        if context_type == ContextType.BEHAVIOR_PATTERN.value:
            return "behavior_pattern"
        if context_type == ContextType.RESOURCE.value:
            return "resource"
        if context_type == ContextType.SKILL.value:
            return "skill"
        if context_type == ContextType.ACTION_POLICY.value:
            return "action_policy"
        return "memory_rules"

    def _packed_source_uris(self, packed: dict) -> list[str]:
        uris: list[str] = []
        for section, payload in packed.get("slices", {}).items():
            for item in payload.get("items", []):
                uri = str(item.get("uri", ""))
                if not uri:
                    continue
                if section in {"resource", "skill"} and not item.get("context_type"):
                    continue
                uris.append(uri)
        return list(dict.fromkeys(uris))
