"""为 ActionPolicy 候选组装经过来源校验的决策上下文。"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from infrastructure.store.contracts.index import IndexHit, IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from policy.action_policy.decision.action_context import ActionContext
from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy


class ActionContextBuilder:
    section_limits = {
        "support_rules": 4,
        "support_anchor": 4,
        "behavior_pattern": 4,
        "action_policy": 4,
        "resource": 4,
        "skill": 4,
        "recent_session": 4,
    }
    relation_types = {
        "anchored_by": "support_anchor",
        "constrained_by": "support_rules",
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
    ) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.relation_store = relation_store

    def build(
        self,
        user_id: str,
        top_candidates: list[ActionCandidate],
        policies: list[ActionPolicy],
        resources: list[dict] | None = None,
        skills: list[dict] | None = None,
        *,
        tenant_id: str | None = None,
        verified_support_anchor_uris: Collection[str] | None = None,
    ) -> ActionContext:
        actions = [candidate.action for candidate in top_candidates]
        policy_by_uri = {policy.uri: policy for policy in policies if policy.user_id == user_id}
        expected_tenant = self._expected_tenant_id(tenant_id)
        derived_verified_anchors = self.verified_support_anchor_uris(
            user_id,
            list(policy_by_uri.values()),
            tenant_id=expected_tenant,
        )
        if verified_support_anchor_uris is not None:
            derived_verified_anchors &= {
                str(uri) for uri in verified_support_anchor_uris if str(uri)
            }
        sections = {
            "support_rules": [],
            "support_anchor": [],
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
                    "layer": "metadata",
                }
            )
            relation_sections = self._relation_sections(
                policy.uri,
                user_id=user_id,
                candidate_score=candidate.score,
                expected_anchor_uri=policy.support_anchor_uri,
                verified_anchor_uris=derived_verified_anchors,
                tenant_id=expected_tenant,
            )
            for section, items in relation_sections.items():
                sections[section].extend(items)
            if (
                policy.support_anchor_uri in derived_verified_anchors
                and not relation_sections.get("support_anchor")
            ):
                anchor = self._exact_support_anchor_context(
                    policy.support_anchor_uri,
                    user_id=user_id,
                    tenant_id=expected_tenant,
                    candidate_score=candidate.score,
                )
                if anchor is not None:
                    sections["support_anchor"].append(anchor)
            policy_rule_uris = {
                str(item.get("uri") or "") for item in relation_sections.get("support_rules", [])
            }
            for rule_uri in policy.constrained_by_support_uris:
                if not rule_uri or rule_uri in policy_rule_uris:
                    continue
                rule = self._exact_policy_rule_context(
                    rule_uri,
                    policy_uri=policy.uri,
                    user_id=user_id,
                    tenant_id=expected_tenant,
                    candidate_score=candidate.score,
                )
                if rule is not None:
                    sections["support_rules"].append(rule)
                    policy_rule_uris.add(rule_uri)
            if not policy_rule_uris:
                fallback_rules = self._hits(
                    user_id,
                    candidate.action,
                    ContextType.ACTION_POLICY_SUPPORT,
                    tenant_id=expected_tenant,
                )
                for item in fallback_rules:
                    exact_rule = self._exact_policy_rule_context(
                        str(item.get("uri") or ""),
                        policy_uri=policy.uri,
                        user_id=user_id,
                        tenant_id=expected_tenant,
                        candidate_score=candidate.score,
                    )
                    if exact_rule is not None:
                        sections["support_rules"].append(exact_rule)
                        policy_rule_uris.add(str(exact_rule.get("uri") or ""))
            if not relation_sections.get("behavior_pattern"):
                sections["behavior_pattern"].extend(
                    self._hits(
                        user_id,
                        policy.scene_key,
                        ContextType.BEHAVIOR_PATTERN,
                        tenant_id=expected_tenant,
                    )
                )
        packed = self._assemble_sections(sections)
        source_uris = self._packed_source_uris(packed)
        return ActionContext(user_id=user_id, candidate_actions=actions, packed_context=packed, source_uris=source_uris)

    def verified_support_anchor_uris(
        self,
        user_id: str,
        policies: list[ActionPolicy],
        *,
        tenant_id: str | None = None,
    ) -> set[str]:
        """只返回属于当前用户和租户、且仍为 active 的精确支持锚点。"""

        expected_tenant = self._expected_tenant_id(tenant_id)
        verified: set[str] = set()
        for policy in policies:
            uri = str(policy.support_anchor_uri or "")
            if policy.user_id != user_id or not uri:
                continue
            if self._read_verified_support_anchor(uri, user_id=user_id, tenant_id=expected_tenant) is not None:
                verified.add(uri)
        return verified

    def _relation_sections(
        self,
        policy_uri: str,
        user_id: str,
        candidate_score: float,
        expected_anchor_uri: str = "",
        verified_anchor_uris: Collection[str] = (),
        tenant_id: str = "default",
    ) -> dict[str, list[dict]]:
        if self.relation_store is None:
            return {}
        sections: dict[str, list[dict]] = {}
        for relation in self.relation_store.relations_of(
            policy_uri,
            tenant_id=tenant_id,
            owner_user_id=user_id,
        ):
            if relation.source_uri != policy_uri:
                continue
            section = self.relation_types.get(relation.relation_type)
            if not section:
                continue
            if section == "support_anchor":
                if (
                    relation.target_uri != expected_anchor_uri
                    or relation.target_uri not in verified_anchor_uris
                ):
                    continue
                item = self._exact_support_anchor_context(
                    relation.target_uri,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    candidate_score=candidate_score,
                )
            elif section == "support_rules":
                item = self._exact_policy_rule_context(
                    relation.target_uri,
                    policy_uri=policy_uri,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    candidate_score=candidate_score,
                )
            else:
                item = self._object_context(
                    relation.target_uri,
                    user_id=user_id,
                    section=section,
                    candidate_score=candidate_score,
                    tenant_id=tenant_id,
                )
            if item is None:
                if self.source_store is not None or section in {"support_anchor", "support_rules"}:
                    continue
                item = {
                    "uri": relation.target_uri,
                    "content": relation.metadata.get("summary", relation.relation_type),
                    "layer": "fallback",
                }
            item["relation_type"] = relation.relation_type
            sections.setdefault(section, []).append(item)
        return sections

    def _exact_support_anchor_context(
        self,
        uri: str,
        *,
        user_id: str,
        tenant_id: str,
        candidate_score: float,
    ) -> dict | None:
        obj = self._read_verified_support_anchor(uri, user_id=user_id, tenant_id=tenant_id)
        if obj is None:
            return None
        item = self._context_from_object(
            obj,
            section="support_anchor",
            candidate_score=candidate_score,
        )
        item["verified_exact_anchor"] = True
        item["verified_anchor_tenant_id"] = tenant_id
        return item

    def _exact_policy_rule_context(
        self,
        uri: str,
        *,
        policy_uri: str,
        user_id: str,
        tenant_id: str,
        candidate_score: float,
    ) -> dict | None:
        obj = self._read_verified_policy_rule(
            uri,
            policy_uri=policy_uri,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if obj is None:
            return None
        item = self._context_from_object(
            obj,
            section="support_rules",
            candidate_score=candidate_score,
        )
        item["verified_policy_rule"] = True
        item["verified_rule_tenant_id"] = tenant_id
        return item

    def _read_verified_support_anchor(self, uri: str, *, user_id: str, tenant_id: str):  # noqa: ANN202
        if self.source_store is None:
            return None
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
            return None
        if (
            obj.uri != uri
            or obj.context_type != ContextType.BEHAVIOR_SUPPORT
            or obj.owner_user_id != user_id
            or str(obj.tenant_id or "default") != tenant_id
            or not self._is_active_support_object(obj, expected_kind="behavior")
        ):
            return None
        return obj

    def _read_verified_policy_rule(
        self,
        uri: str,
        *,
        policy_uri: str,
        user_id: str,
        tenant_id: str,
    ):  # noqa: ANN202
        if self.source_store is None:
            return None
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
            return None
        constrained_payload = dict(obj.metadata or {}).get("constrains_policy_uris", [])
        if not isinstance(constrained_payload, (list, tuple, set)):
            return None
        constrained = {
            str(item)
            for item in constrained_payload
            if isinstance(item, str) and item
        }
        if (
            obj.uri != uri
            or obj.context_type != ContextType.ACTION_POLICY_SUPPORT
            or obj.owner_user_id != user_id
            or str(obj.tenant_id or "default") != tenant_id
            or policy_uri not in constrained
            or not self._is_active_support_object(obj, expected_kind="action_policy")
        ):
            return None
        return obj

    def _is_active_support_object(self, obj, *, expected_kind: str) -> bool:  # noqa: ANN001
        if obj.lifecycle_state != LifecycleState.ACTIVE or not isinstance(obj.metadata, Mapping):
            return False
        metadata = dict(obj.metadata)
        return str(metadata.get("support_anchor_kind") or "") == expected_kind

    def _expected_tenant_id(self, tenant_id: str | None) -> str:
        source_tenant = str(getattr(self.source_store, "tenant_id", "") or "")
        if source_tenant:
            return source_tenant
        return str(tenant_id or "default")

    def _hits(
        self,
        user_id: str,
        query: str,
        context_type: ContextType,
        tenant_id: str | None = None,
    ) -> list[dict]:
        if context_type in {ContextType.BEHAVIOR_SUPPORT, ContextType.ACTION_POLICY_SUPPORT} and self.source_store is None:
            return []
        expected_tenant = str(
            tenant_id or getattr(self.source_store, "tenant_id", "default") or "default"
        )
        filters = {
            "tenant_id": expected_tenant,
            "owner_user_id": user_id,
            "context_type": context_type.value,
        }
        hits = self.index_store.search(
            query,
            tenant_id=expected_tenant,
            filters=filters,
            limit=4,
        )
        items = []
        for hit in hits:
            item = self._hit_context(
                hit,
                user_id,
                tenant_id=tenant_id,
            )
            if item is not None:
                items.append(item)
        return items

    def _hit_context(
        self,
        hit: IndexHit,
        user_id: str,
        tenant_id: str | None = None,
    ) -> dict | None:
        item = self._object_context(
            hit.uri,
            user_id=user_id,
            section=self._section_for_type(hit.context_type),
            candidate_score=hit.score,
            tenant_id=tenant_id,
        )
        if item is not None:
            return item
        if self.source_store is not None:
            return None
        return {"uri": hit.uri, "content": hit.title, "score": hit.score, "layer": "fallback"}

    def _object_context(
        self,
        uri: str,
        user_id: str,
        section: str,
        candidate_score: float = 0.0,
        tenant_id: str | None = None,
    ) -> dict | None:
        if self.source_store is None:
            return None
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
        if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.OBSOLETE}:
            return None
        if section == "support_anchor" and (
            obj.context_type != ContextType.BEHAVIOR_SUPPORT
            or not self._is_active_support_object(obj, expected_kind="behavior")
        ):
            return None
        if section == "support_rules" and (
            obj.context_type != ContextType.ACTION_POLICY_SUPPORT
            or not self._is_active_support_object(obj, expected_kind="action_policy")
        ):
            return None
        if tenant_id is not None and str(obj.tenant_id or "default") != str(tenant_id):
            return None
        if obj.owner_user_id not in {None, user_id} and not obj.uri.startswith(("memoryos://resources/", "memoryos://skills/")):
            return None
        return self._context_from_object(
            obj,
            section=section,
            candidate_score=candidate_score,
        )

    def _context_from_object(
        self,
        obj,  # noqa: ANN001
        *,
        section: str,
        candidate_score: float,
    ) -> dict:
        layer_content, layer = self._read_best_layer(obj, section, candidate_score=candidate_score)
        return {
            "uri": obj.uri,
            "title": obj.title,
            "context_type": obj.context_type.value,
            "content": layer_content,
            "layer": layer,
            "metadata": obj.metadata,
        }

    def _read_best_layer(self, obj, section: str, candidate_score: float = 0.0):
        if section == "action_policy":
            return obj.metadata, "metadata"
        preferred = [(obj.layers.l1_uri, "l1"), (obj.layers.l0_uri, "l0")]
        if section == "recent_session":
            preferred = [(obj.layers.l0_uri, "l0"), (obj.layers.l1_uri, "l1")]
        elif candidate_score >= 0.85:
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
        if context_type == ContextType.BEHAVIOR_SUPPORT.value:
            return "support_anchor"
        return "support_rules"

    def _assemble_sections(self, sections: dict[str, list[dict]]) -> dict:
        """按每类条目上限形成动作上下文，不执行 token 估算或截断。"""

        slices: dict[str, dict[str, list[dict]]] = {}
        load_plan: list[dict] = []
        dropped: list[dict] = []
        for name, items in sections.items():
            limit = self.section_limits.get(name, 4)
            selected = items[:limit]
            slices[name] = {"items": selected}
            load_plan.extend(
                {
                    "uri": item.get("uri", ""),
                    "section": name,
                    "layer": item.get("layer", "fallback"),
                    "reason": "selected_by_section_limit",
                }
                for item in selected
            )
            dropped.extend(
                {
                    "uri": item.get("uri", ""),
                    "section": name,
                    "layer": item.get("layer", "fallback"),
                    "reason": "section_limit",
                }
                for item in items[limit:]
            )
        return {"slices": slices, "load_plan": load_plan, "dropped_contexts": dropped}

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
