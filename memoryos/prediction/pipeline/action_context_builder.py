"""预测模块里的动作上下文组装。"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.layers.context_packer import ContextPacker
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexHit,
    IndexStore,
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.memory.canonical.visibility import committed_content, read_committed_canonical
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
        *,
        tenant_id: str | None = None,
        verified_memory_anchor_uris: Collection[str] | None = None,
    ) -> ActionContext:
        actions = [candidate.action for candidate in top_candidates]
        policy_by_uri = {policy.uri: policy for policy in policies if policy.user_id == user_id}
        expected_tenant = self._expected_tenant_id(tenant_id)
        derived_verified_anchors = self.verified_memory_anchor_uris(
            user_id,
            list(policy_by_uri.values()),
            tenant_id=expected_tenant,
        )
        if verified_memory_anchor_uris is not None:
            derived_verified_anchors &= {
                str(uri) for uri in verified_memory_anchor_uris if str(uri)
            }
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
            relation_sections = self._relation_sections(
                policy.uri,
                user_id=user_id,
                token_budget_remaining=token_budget,
                candidate_score=candidate.score,
                expected_anchor_uri=policy.memory_anchor_uri,
                verified_anchor_uris=derived_verified_anchors,
                tenant_id=expected_tenant,
            )
            for section, items in relation_sections.items():
                sections[section].extend(items)
            if (
                policy.memory_anchor_uri in derived_verified_anchors
                and not relation_sections.get("memory_anchor")
            ):
                anchor = self._exact_memory_anchor_context(
                    policy.memory_anchor_uri,
                    user_id=user_id,
                    tenant_id=expected_tenant,
                    token_budget_remaining=token_budget,
                    candidate_score=candidate.score,
                )
                if anchor is not None:
                    sections["memory_anchor"].append(anchor)
            if not relation_sections.get("memory_rules"):
                sections["memory_rules"].extend(
                    self._hits(
                        user_id,
                        candidate.action,
                        ContextType.MEMORY,
                        token_budget_remaining=token_budget,
                        tenant_id=expected_tenant,
                    )
                )
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

    def verified_memory_anchor_uris(
        self,
        user_id: str,
        policies: list[ActionPolicy],
        *,
        tenant_id: str | None = None,
    ) -> set[str]:
        """Return only exact, active anchors authorized for this prediction boundary."""

        expected_tenant = self._expected_tenant_id(tenant_id)
        verified: set[str] = set()
        for policy in policies:
            uri = str(policy.memory_anchor_uri or "")
            if policy.user_id != user_id or not uri:
                continue
            if self._read_verified_memory_anchor(uri, user_id=user_id, tenant_id=expected_tenant) is not None:
                verified.add(uri)
        return verified

    def _relation_sections(
        self,
        policy_uri: str,
        user_id: str,
        token_budget_remaining: int,
        candidate_score: float,
        expected_anchor_uri: str = "",
        verified_anchor_uris: Collection[str] = (),
        tenant_id: str = "default",
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
            if section == "memory_anchor":
                if (
                    relation.target_uri != expected_anchor_uri
                    or relation.target_uri not in verified_anchor_uris
                ):
                    continue
                item = self._exact_memory_anchor_context(
                    relation.target_uri,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    token_budget_remaining=token_budget_remaining,
                    candidate_score=candidate_score,
                )
            else:
                item = self._object_context(
                    relation.target_uri,
                    user_id=user_id,
                    section=section,
                    token_budget_remaining=token_budget_remaining,
                    candidate_score=candidate_score,
                    tenant_id=tenant_id,
                )
            if item is None:
                if self.source_store is not None or section in {"memory_anchor", "memory_rules"}:
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

    def _exact_memory_anchor_context(
        self,
        uri: str,
        *,
        user_id: str,
        tenant_id: str,
        token_budget_remaining: int,
        candidate_score: float,
    ) -> dict | None:
        obj = self._read_verified_memory_anchor(uri, user_id=user_id, tenant_id=tenant_id)
        if obj is None:
            return None
        item = self._context_from_object(
            obj,
            section="memory_anchor",
            token_budget_remaining=token_budget_remaining,
            candidate_score=candidate_score,
        )
        item["verified_exact_anchor"] = True
        item["verified_anchor_tenant_id"] = tenant_id
        return item

    def _read_verified_memory_anchor(self, uri: str, *, user_id: str, tenant_id: str):  # noqa: ANN202
        if self.source_store is None:
            return None
        try:
            if is_canonical_memory_uri(uri):
                obj = read_committed_canonical(
                    self.source_store,
                    uri,
                    self.relation_store,
                ).object
            else:
                obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
            return None
        if is_canonical_memory_object(obj):
            try:
                committed = read_committed_canonical(self.source_store, uri, self.relation_store)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
                return None
            obj = committed.object
        if (
            obj.uri != uri
            or obj.context_type != ContextType.MEMORY
            or obj.owner_user_id != user_id
            or str(obj.tenant_id or "default") != tenant_id
            or not self._is_active_authoritative_anchor(obj, user_id=user_id, tenant_id=tenant_id)
        ):
            return None
        return obj

    def _is_active_authoritative_anchor(self, obj, *, user_id: str, tenant_id: str) -> bool:  # noqa: ANN001
        if obj.lifecycle_state != LifecycleState.ACTIVE or not isinstance(obj.metadata, Mapping):
            return False
        metadata = dict(obj.metadata)
        raw_admission = metadata.get("admission", {})
        if raw_admission is not None and not isinstance(raw_admission, Mapping):
            return False
        admission = dict(raw_admission or {})
        if admission.get("decision") in {"pending", "restricted", "archive_only", "reject"}:
            return False
        canonical_kind = str(metadata.get("canonical_kind") or "")
        if canonical_kind:
            if canonical_kind != "claim":
                return False
            if str(metadata.get("state") or metadata.get("claim_state") or "") != "ACTIVE":
                return False
            scope = metadata.get("scope", {})
            if not isinstance(scope, Mapping):
                return False
            scope = dict(scope)
            authority = metadata.get("authority") or scope.get("authority") or {}
            visibility = scope.get("visibility", {})
            if not isinstance(authority, Mapping) or not isinstance(visibility, Mapping):
                return False
            authority = dict(authority)
            visibility = dict(visibility)
            if not authority or bool(authority.get("inferred", False)):
                return False
            if str(authority.get("tenant_id") or tenant_id) != tenant_id:
                return False
            if str(visibility.get("tenant_id") or tenant_id) != tenant_id:
                return False
            principals = {
                str(item)
                for item in (
                    authority.get("allowed_principal_ids")
                    or authority.get("principal_ids")
                    or []
                )
            }
            visible_principals = {
                str(item) for item in visibility.get("allowed_principal_ids", []) or []
            }
            return (not principals or user_id in principals) and (
                not visible_principals or user_id in visible_principals
            )
        return str(metadata.get("memory_kind") or "") == "anchor_memory"

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
        token_budget_remaining: int,
        tenant_id: str | None = None,
    ) -> list[dict]:
        if context_type == ContextType.MEMORY and self.source_store is None:
            return []
        filters = {"owner_user_id": user_id, "context_type": context_type.value}
        if context_type == ContextType.MEMORY and tenant_id:
            filters["tenant_id"] = tenant_id
        hits = self.index_store.search(query, filters=filters, limit=4)
        items = []
        for hit in hits:
            item = self._hit_context(
                hit,
                user_id,
                token_budget_remaining=token_budget_remaining,
                tenant_id=tenant_id,
            )
            if item is not None:
                items.append(item)
        return items

    def _hit_context(
        self,
        hit: IndexHit,
        user_id: str,
        token_budget_remaining: int,
        tenant_id: str | None = None,
    ) -> dict | None:
        item = self._object_context(
            hit.uri,
            user_id=user_id,
            section=self._section_for_type(hit.context_type),
            token_budget_remaining=token_budget_remaining,
            candidate_score=hit.score,
            tenant_id=tenant_id,
        )
        if item is not None:
            return item
        if self.source_store is not None:
            return None
        return {"uri": hit.uri, "content": hit.title, "token_estimate": 80, "score": hit.score, "layer": "fallback"}

    def _object_context(
        self,
        uri: str,
        user_id: str,
        section: str,
        token_budget_remaining: int = 1000,
        candidate_score: float = 0.0,
        tenant_id: str | None = None,
    ) -> dict | None:
        if self.source_store is None:
            return None
        try:
            if is_canonical_memory_uri(uri):
                obj = read_committed_canonical(
                    self.source_store,
                    uri,
                    self.relation_store,
                ).object
            else:
                obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
        if obj.context_type == ContextType.MEMORY and isinstance(obj.metadata, Mapping):
            if is_canonical_memory_object(obj):
                try:
                    committed = read_committed_canonical(
                        self.source_store,
                        uri,
                        self.relation_store,
                    )
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError):
                    return None
                obj = committed.object
        if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.OBSOLETE}:
            return None
        if obj.context_type == ContextType.MEMORY and not self._is_active_authoritative_memory(obj):
            return None
        if (
            obj.context_type == ContextType.MEMORY
            and tenant_id is not None
            and str(obj.tenant_id or "default") != str(tenant_id)
        ):
            return None
        if obj.owner_user_id not in {None, user_id} and not obj.uri.startswith(("memoryos://resources/", "memoryos://skills/")):
            return None
        return self._context_from_object(
            obj,
            section=section,
            token_budget_remaining=token_budget_remaining,
            candidate_score=candidate_score,
        )

    def _context_from_object(
        self,
        obj,  # noqa: ANN001
        *,
        section: str,
        token_budget_remaining: int,
        candidate_score: float,
    ) -> dict:
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

    def _is_active_authoritative_memory(self, obj) -> bool:  # noqa: ANN001
        if obj.lifecycle_state != LifecycleState.ACTIVE:
            return False
        metadata = dict(obj.metadata or {})
        admission = dict(metadata.get("admission", {}) or {})
        if admission.get("decision") in {"pending", "restricted", "archive_only", "reject"}:
            return False
        if metadata.get("memory_kind") == "memory_candidate":
            return False
        canonical_kind = str(metadata.get("canonical_kind") or "")
        if canonical_kind == "pending_proposal":
            return False
        if canonical_kind and canonical_kind != "claim":
            return False
        claim_state = str(metadata.get("state") or metadata.get("claim_state") or "")
        if claim_state and claim_state != "ACTIVE":
            return False
        if canonical_kind == "claim":
            authority = dict(dict(metadata.get("scope", {}) or {}).get("authority", {}) or {})
            if not authority or bool(authority.get("inferred", False)):
                return False
        return True

    def _read_best_layer(self, obj, section: str, token_budget_remaining: int = 1000, candidate_score: float = 0.0):
        if section == "action_policy":
            return obj.metadata, "metadata"
        if (
            self.source_store is not None
            and dict(obj.metadata or {}).get("canonical_kind") == "claim"
        ):
            committed = read_committed_canonical(
                self.source_store,
                obj.uri,
                self.relation_store,
            )
            return committed_content(committed), "committed_l2"
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
