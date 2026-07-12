from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder


class ActionContextBuilderRelationFirstTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = FileSystemSourceStore(self.root)
        self.index = InMemoryIndexStore()
        self.relations = InMemoryRelationStore()
        self.policy = ActionPolicy(
            user_id="u1",
            scene_key="hot_room",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/u1/memories/anchors/home_comfort",
        )
        self.source.write_object(self.policy.to_context_object(), content="action policy")
        self._write(
            "memoryos://user/u1/memories/anchors/home_comfort",
            ContextType.MEMORY,
            "Home comfort anchor",
            "anchor text",
            metadata={"memory_kind": "anchor_memory"},
        )
        self._write("memoryos://user/u1/memories/policies/no_auto", ContextType.MEMORY, "No auto AC", "policy memory text")
        self._write("memoryos://user/u1/behavior/patterns/hot_room/p1", ContextType.BEHAVIOR_PATTERN, "Hot room pattern", "pattern text")
        self._write("memoryos://resources/devices/ac-living-room", ContextType.RESOURCE, "Living room AC", "resource text", owner=None)
        self._write("memoryos://skills/smart_home/ac-control", ContextType.SKILL, "AC control", "skill text", owner=None)
        for relation_type, target in (
            ("anchored_by", "memoryos://user/u1/memories/anchors/home_comfort"),
            ("constrained_by", "memoryos://user/u1/memories/policies/no_auto"),
            ("supported_by", "memoryos://user/u1/behavior/patterns/hot_room/p1"),
            ("requires_resource", "memoryos://resources/devices/ac-living-room"),
            ("requires_skill", "memoryos://skills/smart_home/ac-control"),
        ):
            self.relations.add_relation(ContextRelation(source_uri=self.policy.uri, relation_type=relation_type, target_uri=target))

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
    ) -> None:
        obj = ContextObject(
            uri=uri,
            context_type=context_type,
            title=title,
            owner_user_id=owner,
            metadata=metadata or {},
        )
        self.source.write_object(obj, content=content)

    def build(self, budget: int = 2000):
        builder = ActionContextBuilder(self.index, source_store=self.source, relation_store=self.relations)
        candidate = ActionCandidate(action=self.policy.action, score=0.9, policy_uri=self.policy.uri, reason="test")
        return builder.build("u1", [candidate], [self.policy], token_budget=budget)

    def test_relation_first_fetches_anchor_policy_resource_and_skill(self) -> None:
        context = self.build()
        slices = context.packed_context["slices"]
        exact_anchor = next(
            item for item in slices["memory_anchor"]["items"] if item["uri"].endswith("/home_comfort")
        )
        self.assertIs(exact_anchor["verified_exact_anchor"], True)
        self.assertTrue(any(item["uri"].endswith("/no_auto") for item in slices["memory_rules"]["items"]))
        self.assertTrue(any(item["uri"].startswith("memoryos://skills/") for item in slices["skill"]["items"]))
        self.assertTrue(any(item["uri"].startswith("memoryos://resources/") for item in slices["resource"]["items"]))
        self.assertTrue(any(item["context_type"] == ContextType.BEHAVIOR_PATTERN.value for item in slices["behavior_pattern"]["items"]))

    def test_semantic_fallback_cannot_substitute_for_missing_exact_anchor(self) -> None:
        empty_relations = InMemoryRelationStore()
        self.source.delete_object(self.policy.memory_anchor_uri)
        anchor = ContextObject(
            uri="memoryos://user/u1/memories/anchors/fallback",
            context_type=ContextType.MEMORY,
            title="fallback anchor",
            owner_user_id="u1",
            metadata={"memory_kind": "anchor_memory"},
        )
        self.source.write_object(anchor, content=self.policy.memory_anchor_uri)
        self.index.upsert_index(anchor, content=self.policy.memory_anchor_uri)
        builder = ActionContextBuilder(self.index, source_store=self.source, relation_store=empty_relations)
        candidate = ActionCandidate(action=self.policy.action, score=0.9, policy_uri=self.policy.uri, reason="test")
        context = builder.build("u1", [candidate], [self.policy], token_budget=2000)
        self.assertFalse(context.packed_context["slices"]["memory_anchor"]["items"])

    def test_exact_anchor_verification_rejects_unconfirmed_state_owner_and_tenant(self) -> None:
        anchor_uri = self.policy.memory_anchor_uri
        builder = ActionContextBuilder(self.index, source_store=self.source, relation_store=InMemoryRelationStore())
        cases = (
            (
                "pending",
                LifecycleState.PENDING,
                {"memory_kind": "anchor_memory"},
                "u1",
                "default",
            ),
            (
                "proposed",
                LifecycleState.ACTIVE,
                {
                    "canonical_kind": "claim",
                    "state": "PROPOSED",
                    "scope": {
                        "authority": {"principal_ids": ["u1"], "inferred": False},
                        "visibility": {"tenant_id": "default", "allowed_principal_ids": ["u1"]},
                    },
                },
                "u1",
                "default",
            ),
            (
                "superseded",
                LifecycleState.ACTIVE,
                {
                    "canonical_kind": "claim",
                    "state": "SUPERSEDED",
                    "scope": {
                        "authority": {"principal_ids": ["u1"], "inferred": False},
                        "visibility": {"tenant_id": "default", "allowed_principal_ids": ["u1"]},
                    },
                },
                "u1",
                "default",
            ),
            (
                "restricted",
                LifecycleState.ACTIVE,
                {"memory_kind": "anchor_memory", "admission": {"decision": "restricted"}},
                "u1",
                "default",
            ),
            (
                "cross_user",
                LifecycleState.ACTIVE,
                {"memory_kind": "anchor_memory"},
                "u2",
                "default",
            ),
            (
                "cross_tenant",
                LifecycleState.ACTIVE,
                {"memory_kind": "anchor_memory"},
                "u1",
                "tenant-b",
            ),
        )
        for name, lifecycle, metadata, owner, tenant in cases:
            with self.subTest(name=name):
                self.source.write_object(
                    ContextObject(
                        uri=anchor_uri,
                        context_type=ContextType.MEMORY,
                        title=name,
                        owner_user_id=owner,
                        tenant_id=tenant,
                        lifecycle_state=lifecycle,
                        metadata=metadata,
                    ),
                    content=name,
                )
                self.assertEqual(
                    builder.verified_memory_anchor_uris("u1", [self.policy]),
                    set(),
                )

    def test_uncommitted_canonical_anchor_before_image_is_not_action_context_evidence(self) -> None:
        anchor_uri = self.policy.memory_anchor_uri
        scope = {
            "authority": {"principal_ids": ["u1"], "inferred": False},
            "visibility": {"tenant_id": "default", "allowed_principal_ids": ["u1"]},
        }
        before = ContextObject(
            uri=anchor_uri,
            context_type=ContextType.MEMORY,
            title="committed before image",
            owner_user_id="u1",
            metadata={"canonical_kind": "claim", "state": "ACTIVE", "scope": scope},
        )
        uncommitted = ContextObject(
            uri=anchor_uri,
            context_type=ContextType.MEMORY,
            title="uncommitted source revision",
            owner_user_id="u1",
            metadata={
                "canonical_kind": "claim",
                "state": "ACTIVE",
                "scope": scope,
                "canonical_idempotency_key": "anchor-update-key",
                "canonical_transaction_id": "anchor-update-tx",
            },
        )
        self.source.write_object(uncommitted, content="new uncommitted content")
        outbox = self.root / "system" / "outbox" / "anchor-update-tx.json"
        outbox.parent.mkdir(parents=True, exist_ok=True)
        outbox.write_text(
            json.dumps(
                {
                    "status": "source_committed",
                    "before_images": [
                        {
                            "uri": anchor_uri,
                            "exists": True,
                            "object": before.to_dict(),
                            "content": "old committed content",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        builder = ActionContextBuilder(
            self.index,
            source_store=self.source,
            relation_store=InMemoryRelationStore(),
        )

        self.assertEqual(builder.verified_memory_anchor_uris("u1", [self.policy]), set())

    def test_cross_tenant_memory_rule_relation_does_not_enter_action_context(self) -> None:
        cross_tenant_uri = "memoryos://user/u1/memories/policies/tenant-b-rule"
        self.source.write_object(
            ContextObject(
                uri=cross_tenant_uri,
                context_type=ContextType.MEMORY,
                title="tenant b rule",
                owner_user_id="u1",
                tenant_id="tenant-b",
                metadata={"memory_kind": "policy_memory"},
            ),
            content="do not automatically execute",
        )
        self.relations.add_relation(
            ContextRelation(
                source_uri=self.policy.uri,
                relation_type="constrained_by",
                target_uri=cross_tenant_uri,
            )
        )

        context = self.build()
        rule_uris = {
            item["uri"] for item in context.packed_context["slices"]["memory_rules"]["items"]
        }

        self.assertNotIn(cross_tenant_uri, rule_uris)

    def test_token_budget_limits_context(self) -> None:
        context = self.build(budget=160)
        self.assertLessEqual(context.packed_context["used"], 160)

    def test_cross_user_relation_target_is_not_read(self) -> None:
        self._write("memoryos://user/u2/memories/policies/private", ContextType.MEMORY, "private", "private text", owner="u2")
        self.relations.add_relation(ContextRelation(source_uri=self.policy.uri, relation_type="constrained_by", target_uri="memoryos://user/u2/memories/policies/private"))
        context = self.build()
        uris = [item["uri"] for item in context.packed_context["slices"]["memory_rules"]["items"]]
        self.assertNotIn("memoryos://user/u2/memories/policies/private", uris)

    def test_pending_memory_relations_never_enter_action_context(self) -> None:
        protected_uris = {
            "memoryos://user/u1/memories/anchors/home_comfort",
            "memoryos://user/u1/memories/policies/no_auto",
        }
        for lifecycle_state in (
            LifecycleState.PENDING,
            LifecycleState.RETRYABLE,
            LifecycleState.CONFIRMED,
            LifecycleState.ACTIVE,
        ):
            with self.subTest(lifecycle_state=lifecycle_state.value):
                for uri in protected_uris:
                    obj = self.source.read_object(uri)
                    obj.lifecycle_state = lifecycle_state
                    obj.metadata = {
                        "canonical_kind": "pending_proposal",
                        "admission": {"decision": "pending"},
                    }
                    self.source.write_object(obj, content="unconfirmed memory must stay out of action context")

                context = self.build()
                action_uris = {
                    item["uri"]
                    for section in ("memory_anchor", "memory_rules")
                    for item in context.packed_context["slices"][section]["items"]
                }
                self.assertTrue(protected_uris.isdisjoint(action_uris))

    def test_pending_direct_memory_anchor_uri_is_filtered_without_relations(self) -> None:
        anchor_uri = self.policy.memory_anchor_uri
        anchor = self.source.read_object(anchor_uri)
        anchor.lifecycle_state = LifecycleState.PENDING
        anchor.metadata = {
            "canonical_kind": "pending_proposal",
            "admission": {"decision": "pending"},
        }
        self.source.write_object(anchor, content="pending direct anchor")
        self.index.upsert_index(anchor, content=anchor_uri)
        builder = ActionContextBuilder(
            self.index,
            source_store=self.source,
            relation_store=InMemoryRelationStore(),
        )
        candidate = ActionCandidate(
            action=self.policy.action,
            score=0.9,
            policy_uri=self.policy.uri,
            reason="test",
        )

        context = builder.build("u1", [candidate], [self.policy], token_budget=2000)

        anchor_uris = {
            item["uri"] for item in context.packed_context["slices"]["memory_anchor"]["items"]
        }
        self.assertNotIn(anchor_uri, anchor_uris)

    def test_stale_pending_index_hit_cannot_bypass_source_validation(self) -> None:
        anchor_uri = self.policy.memory_anchor_uri
        anchor = self.source.read_object(anchor_uri)
        anchor.lifecycle_state = LifecycleState.PENDING
        anchor.metadata = {
            "canonical_kind": "pending_proposal",
            "admission": {"decision": "pending"},
        }
        self.source.write_object(anchor, content="pending source object behind stale index hit")

        class StalePendingIndex(InMemoryIndexStore):
            def search(self, query, filters=None, limit=10):  # noqa: ANN001, ANN201, ARG002
                if dict(filters or {}).get("context_type") != ContextType.MEMORY.value:
                    return []
                return [
                    IndexHit(
                        uri=anchor_uri,
                        score=1.0,
                        context_type=ContextType.MEMORY.value,
                        title="stale pending index hit",
                    )
                ]

        builder = ActionContextBuilder(
            StalePendingIndex(),
            source_store=self.source,
            relation_store=InMemoryRelationStore(),
        )
        candidate = ActionCandidate(
            action=self.policy.action,
            score=0.9,
            policy_uri=self.policy.uri,
            reason="test",
        )

        context = builder.build("u1", [candidate], [self.policy], token_budget=2000)

        memory_uris = {
            item["uri"]
            for section in ("memory_anchor", "memory_rules")
            for item in context.packed_context["slices"][section]["items"]
        }
        self.assertNotIn(anchor_uri, memory_uris)

    def test_memory_relations_fail_closed_without_source_store(self) -> None:
        builder = ActionContextBuilder(
            self.index,
            source_store=None,
            relation_store=self.relations,
        )
        candidate = ActionCandidate(
            action=self.policy.action,
            score=0.9,
            policy_uri=self.policy.uri,
            reason="test",
        )

        context = builder.build("u1", [candidate], [self.policy], token_budget=2000)

        self.assertFalse(context.packed_context["slices"]["memory_anchor"]["items"])
        self.assertFalse(context.packed_context["slices"]["memory_rules"]["items"])


if __name__ == "__main__":
    unittest.main()
