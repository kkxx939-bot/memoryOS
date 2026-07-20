from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from infrastructure.store.contracts.index import IndexHit
from policy.action_policy.decision.context_builder import ActionContextBuilder
from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from tests.support.persistence import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)


class ActionContextBuilderRelationFirstTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = FileSystemSourceStore(self.root)
        self.index = InMemoryIndexStore()
        self.relations = InMemoryRelationStore()
        self.anchor_uri = "memoryos://user/u1/support/behavior/home_comfort"
        self.rule_uri = "memoryos://user/u1/support/action-policy/no_auto"
        self.policy = ActionPolicy(
            user_id="u1",
            scene_key="hot_room",
            action="turn_on_ac",
            support_anchor_uri=self.anchor_uri,
            constrained_by_support_uris=[self.rule_uri],
        )
        self.source.write_object(self.policy.to_context_object(), content="action policy")
        self._write(
            self.anchor_uri,
            ContextType.BEHAVIOR_SUPPORT,
            "Home comfort support",
            "support anchor text",
            metadata={"support_anchor_kind": "behavior"},
        )
        self._write(
            self.rule_uri,
            ContextType.ACTION_POLICY_SUPPORT,
            "No auto AC",
            "do not automatically execute",
            metadata={
                "support_anchor_kind": "action_policy",
                "constrains_policy_uris": [self.policy.uri],
                "policy_rule_type": "action_auto_execute",
                "policy_rule_value": "forbidden",
            },
        )
        self._write(
            "memoryos://user/u1/behavior/patterns/hot_room/p1",
            ContextType.BEHAVIOR_PATTERN,
            "Hot room pattern",
            "pattern text",
        )
        self._write(
            "memoryos://resources/devices/ac-living-room",
            ContextType.RESOURCE,
            "Living room AC",
            "resource text",
            owner=None,
        )
        self._write(
            "memoryos://skills/smart_home/ac-control",
            ContextType.SKILL,
            "AC control",
            "skill text",
            owner=None,
        )
        for relation_type, target in (
            ("anchored_by", self.anchor_uri),
            ("constrained_by", self.rule_uri),
            ("supported_by", "memoryos://user/u1/behavior/patterns/hot_room/p1"),
            ("requires_resource", "memoryos://resources/devices/ac-living-room"),
            ("requires_skill", "memoryos://skills/smart_home/ac-control"),
        ):
            self.relations.add_relation(
                ContextRelation(
                    source_uri=self.policy.uri,
                    relation_type=relation_type,
                    target_uri=target,
                ),
                tenant_id="default",
            )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(
        self,
        uri: str,
        context_type: ContextType,
        title: str,
        content: str,
        owner: str | None = "u1",
        metadata: dict | None = None,
        *,
        tenant_id: str = "default",
        lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    ) -> ContextObject:
        obj = ContextObject(
            uri=uri,
            context_type=context_type,
            title=title,
            owner_user_id=owner,
            tenant_id=tenant_id,
            lifecycle_state=lifecycle_state,
            metadata=metadata or {},
        )
        self.source.write_object(obj, content=content)
        return obj

    def build(self):  # noqa: ANN201
        builder = ActionContextBuilder(
            self.index,
            source_store=self.source,
            relation_store=self.relations,
        )
        candidate = ActionCandidate(
            action=self.policy.action,
            score=0.9,
            policy_uri=self.policy.uri,
            reason="test",
        )
        return builder.build("u1", [candidate], [self.policy])

    def test_relation_first_fetches_verified_support_resource_and_skill(self) -> None:
        slices = self.build().packed_context["slices"]
        anchor = next(item for item in slices["support_anchor"]["items"] if item["uri"] == self.anchor_uri)
        rule = next(item for item in slices["support_rules"]["items"] if item["uri"] == self.rule_uri)

        self.assertIs(anchor["verified_exact_anchor"], True)
        self.assertIs(rule["verified_policy_rule"], True)
        self.assertTrue(any(item["uri"].startswith("memoryos://skills/") for item in slices["skill"]["items"]))
        self.assertTrue(any(item["uri"].startswith("memoryos://resources/") for item in slices["resource"]["items"]))
        self.assertTrue(any(item["context_type"] == ContextType.BEHAVIOR_PATTERN.value for item in slices["behavior_pattern"]["items"]))

    def test_semantic_fallback_cannot_substitute_for_missing_exact_support(self) -> None:
        self.source.delete_object(self.anchor_uri)
        fallback = ContextObject(
            uri="memoryos://user/u1/support/behavior/fallback",
            context_type=ContextType.BEHAVIOR_SUPPORT,
            title="fallback support",
            owner_user_id="u1",
            metadata={"support_anchor_kind": "behavior"},
        )
        self.source.write_object(fallback, content=self.anchor_uri)
        self.index.upsert_index(fallback, content=self.anchor_uri, tenant_id="default")
        builder = ActionContextBuilder(
            self.index,
            source_store=self.source,
            relation_store=InMemoryRelationStore(),
        )
        candidate = ActionCandidate(self.policy.action, 0.9, self.policy.uri, "test")

        context = builder.build("u1", [candidate], [self.policy])

        self.assertFalse(context.packed_context["slices"]["support_anchor"]["items"])

    def test_exact_support_verification_rejects_wrong_state_kind_owner_type_and_tenant(self) -> None:
        builder = ActionContextBuilder(
            self.index,
            source_store=self.source,
            relation_store=InMemoryRelationStore(),
        )
        cases = (
            (LifecycleState.PENDING, "behavior", "u1", ContextType.BEHAVIOR_SUPPORT, "default"),
            (LifecycleState.ACTIVE, "wrong", "u1", ContextType.BEHAVIOR_SUPPORT, "default"),
            (LifecycleState.ACTIVE, "behavior", "u2", ContextType.BEHAVIOR_SUPPORT, "default"),
            (LifecycleState.ACTIVE, "behavior", "u1", ContextType.RESOURCE, "default"),
            (LifecycleState.ACTIVE, "behavior", "u1", ContextType.BEHAVIOR_SUPPORT, "tenant-b"),
        )
        for lifecycle, kind, owner, context_type, tenant in cases:
            with self.subTest(lifecycle=lifecycle, kind=kind, owner=owner, context_type=context_type, tenant=tenant):
                obj = ContextObject(
                    uri=self.anchor_uri,
                    context_type=context_type,
                    title="candidate",
                    owner_user_id=owner,
                    tenant_id=tenant,
                    lifecycle_state=lifecycle,
                    metadata={"support_anchor_kind": kind},
                )
                if tenant == "default":
                    self.source.write_object(obj, content="candidate")
                else:
                    self.source.delete_object(self.anchor_uri)
                    FileSystemSourceStore(self.root, tenant_id=tenant).write_object(
                        obj,
                        content="candidate",
                    )
                self.assertEqual(builder.verified_support_anchor_uris("u1", [self.policy]), set())

    def test_policy_rule_requires_exact_policy_binding(self) -> None:
        rule = self.source.read_object(self.rule_uri)
        rule.metadata["constrains_policy_uris"] = ["memoryos://user/u1/action_policies/other/action"]
        self.source.write_object(rule, content="do not automatically execute")

        context = self.build()

        self.assertFalse(context.packed_context["slices"]["support_rules"]["items"])

    def test_stale_support_index_hit_cannot_bypass_source_validation(self) -> None:
        anchor = self.source.read_object(self.anchor_uri)
        anchor.lifecycle_state = LifecycleState.PENDING
        self.source.write_object(anchor, content="pending support behind stale index")

        class StaleSupportIndex(InMemoryIndexStore):
            def search(  # noqa: ANN201
                self,
                query,  # noqa: ANN001, ARG002
                *,
                tenant_id,  # noqa: ANN001, ARG002
                filters=None,  # noqa: ANN001
                limit=10,  # noqa: ARG002
            ):
                if dict(filters or {}).get("context_type") != ContextType.BEHAVIOR_SUPPORT.value:
                    return []
                return [
                    IndexHit(
                        uri=anchor.uri,
                        score=1.0,
                        context_type=ContextType.BEHAVIOR_SUPPORT.value,
                        title="stale support index hit",
                    )
                ]

        builder = ActionContextBuilder(
            StaleSupportIndex(),
            source_store=self.source,
            relation_store=InMemoryRelationStore(),
        )
        candidate = ActionCandidate(self.policy.action, 0.9, self.policy.uri, "test")
        context = builder.build("u1", [candidate], [self.policy])

        self.assertFalse(context.packed_context["slices"]["support_anchor"]["items"])

    def test_support_relations_fail_closed_without_source_store(self) -> None:
        builder = ActionContextBuilder(
            self.index,
            source_store=None,
            relation_store=self.relations,
        )
        candidate = ActionCandidate(self.policy.action, 0.9, self.policy.uri, "test")

        context = builder.build("u1", [candidate], [self.policy])

        self.assertFalse(context.packed_context["slices"]["support_anchor"]["items"])
        self.assertFalse(context.packed_context["slices"]["support_rules"]["items"])

    def test_behavior_fallback_binds_explicit_tenant_without_source_store(self) -> None:
        seen_tenants: list[str] = []

        class RecordingIndex(InMemoryIndexStore):
            def search(  # noqa: ANN201
                self,
                query,  # noqa: ANN001, ARG002
                *,
                tenant_id,  # noqa: ANN001
                filters=None,  # noqa: ANN001, ARG002
                limit=10,  # noqa: ARG002
            ):
                seen_tenants.append(str(tenant_id))
                return []

        candidate = ActionCandidate(self.policy.action, 0.9, self.policy.uri, "test")
        ActionContextBuilder(RecordingIndex()).build(
            "u1",
            [candidate],
            [self.policy],
            tenant_id="tenant-b",
        )

        self.assertEqual(seen_tenants, ["tenant-b"])

if __name__ == "__main__":
    unittest.main()
