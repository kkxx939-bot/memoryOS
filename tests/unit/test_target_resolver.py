from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.target_resolver import TargetResolver


class TargetResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.source = FileSystemSourceStore(self.tmp.name)
        self.index = InMemoryIndexStore()
        self.resolver = TargetResolver(self.index, self.source)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_target_uri_is_resolved(self) -> None:
        obj = ContextObject(
            uri="memoryos://user/u1/memories/profile/a",
            context_type=ContextType.MEMORY,
            title="profile",
            owner_user_id="u1",
        )
        self.source.write_object(obj)
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, target_uri=obj.uri, payload={})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertTrue(result.resolved)
        self.assertEqual(op.status, OperationStatus.RESOLVED)

    def test_update_memory_without_target_resolves_by_query(self) -> None:
        obj = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature preference", owner_user_id="u1")
        self.source.write_object(obj)
        self.index.upsert_index(obj, content="prefers 26 degree room temperature")
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "26 degree room temperature"})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertTrue(result.resolved)
        self.assertEqual(op.target_uri, obj.uri)
        self.assertEqual([candidate.uri for candidate in result.candidates], [obj.uri])

    def test_delete_action_policy_without_target_resolves_by_scene_and_action(self) -> None:
        obj = ContextObject(uri="memoryos://user/u1/action_policies/hot_room/turn_on_ac", context_type=ContextType.ACTION_POLICY, title="hot_room turn_on_ac", owner_user_id="u1")
        self.source.write_object(obj)
        self.index.upsert_index(obj, content="hot_room turn_on_ac policy")
        op = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.DELETE, payload={"scene_key": "hot_room", "action": "turn_on_ac"})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertTrue(result.resolved)
        self.assertEqual(op.target_uri, obj.uri)

    def test_low_confidence_candidates_go_pending(self) -> None:
        class LowConfidenceIndex:
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                return [
                    IndexHit(
                        uri="memoryos://user/u1/memories/profile/name",
                        score=0.5,
                        context_type="memory",
                        title="name",
                        metadata={
                            "retrieval_scores": {"lexical": 0.5, "vector": 0.0, "identity": 0.0}
                        },
                    )
                ]

            def upsert_index(self, obj: ContextObject, content: str = "") -> None:
                return None

            def delete_index(self, uri: str) -> None:
                return None

            def indexed_uris(self) -> list[str]:
                return []

            def clear(self) -> None:
                return None

        obj = ContextObject(
            uri="memoryos://user/u1/memories/profile/name",
            context_type=ContextType.MEMORY,
            title="name",
            owner_user_id="u1",
        )
        self.source.write_object(obj)
        resolver = TargetResolver(LowConfidenceIndex(), self.source)
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "name"})
        result = resolver.resolve(op, user_id="u1")
        self.assertFalse(result.resolved)
        self.assertEqual(op.status, OperationStatus.PENDING)
        self.assertIn("target_candidates", op.payload)

    def test_cross_user_namespace_is_not_resolved(self) -> None:
        obj = ContextObject(uri="memoryos://user/u2/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature preference", owner_user_id="u2")
        self.source.write_object(obj)
        self.index.upsert_index(obj, content="prefers 26 degree room temperature")
        op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "26 degree room temperature"})
        result = self.resolver.resolve(op, user_id="u1")
        self.assertFalse(result.resolved)
        self.assertIsNone(op.target_uri)

    def test_close_top_candidates_are_ambiguous(self) -> None:
        for suffix in ("a", "b"):
            obj = ContextObject(
                uri=f"memoryos://user/u1/memories/{suffix}",
                context_type=ContextType.MEMORY,
                title="database backend",
                owner_user_id="u1",
            )
            self.source.write_object(obj)
            self.index.upsert_index(obj, content="PostgreSQL database backend")
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "PostgreSQL"},
        )

        result = self.resolver.resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_ambiguous")
        self.assertEqual(op.status, OperationStatus.PENDING)
        self.assertIsNone(op.target_uri)

    def test_automatic_target_requires_matching_tenant_workspace_memory_type_and_scope(self) -> None:
        def scoped(workspace: str) -> dict:
            return {
                "project_id": workspace,
                "visibility": {"tenant_id": "tenant-a"},
                "applicability": {
                    "all_of": [
                        {"namespace": "memoryos", "kind": "principal", "id": "u1"},
                        {"namespace": "memoryos", "kind": "workspace", "id": workspace},
                    ]
                },
            }

        rows = [
            ContextObject(
                uri="memoryos://user/u1/memories/wrong-tenant",
                context_type=ContextType.MEMORY,
                title="database backend",
                owner_user_id="u1",
                tenant_id="tenant-b",
                metadata={"memory_type": "project_rule", "scope": scoped("alpha")},
            ),
            ContextObject(
                uri="memoryos://user/u1/memories/wrong-workspace",
                context_type=ContextType.MEMORY,
                title="database backend",
                owner_user_id="u1",
                tenant_id="tenant-a",
                metadata={"memory_type": "project_rule", "scope": scoped("beta")},
            ),
            ContextObject(
                uri="memoryos://user/u1/memories/wrong-memory-type",
                context_type=ContextType.MEMORY,
                title="database backend",
                owner_user_id="u1",
                tenant_id="tenant-a",
                metadata={"memory_type": "preference", "scope": scoped("alpha")},
            ),
            ContextObject(
                uri="memoryos://user/u1/memories/correct",
                context_type=ContextType.MEMORY,
                title="database backend",
                owner_user_id="u1",
                tenant_id="tenant-a",
                metadata={"memory_type": "project_rule", "scope": scoped("alpha")},
            ),
        ]
        for row in rows:
            self.source.write_object(row)
            self.index.upsert_index(row, content="PostgreSQL database backend")
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={
                "query": "PostgreSQL",
                "tenant_id": "tenant-a",
                "project_id": "alpha",
                "memory_type": "project_rule",
                "applicability_scope_keys": ["memoryos:principal:u1", "memoryos:workspace:alpha"],
            },
        )

        result = self.resolver.resolve(op, user_id="u1")

        self.assertTrue(result.resolved)
        self.assertEqual(op.target_uri, "memoryos://user/u1/memories/correct")

    def test_high_hotness_with_zero_relevance_cannot_resolve_target(self) -> None:
        index = SQLiteIndexStore(f"{self.tmp.name}/target-index.sqlite3")
        unrelated = ContextObject(
            uri="memoryos://user/u1/memories/unrelated",
            context_type=ContextType.MEMORY,
            title="weather",
            owner_user_id="u1",
            hotness=1.0,
            semantic_hotness=1.0,
            behavior_support_hotness=1.0,
        )
        self.source.write_object(unrelated)
        index.upsert_index(unrelated, content="sunny outdoor activity")
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.UPDATE,
            payload={"query": "PostgreSQL"},
        )

        result = TargetResolver(index, self.source).resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_review_required")
        self.assertIsNone(op.target_uri)

    def test_automatic_resolution_never_targets_pending_proposal(self) -> None:
        pending = ContextObject(
            uri="memoryos://user/u1/memories/pending/p1",
            context_type=ContextType.MEMORY,
            title="PostgreSQL proposal",
            owner_user_id="u1",
            lifecycle_state=LifecycleState.PENDING,
            metadata={"canonical_kind": "pending_proposal", "admission": {"decision": "pending"}},
        )
        self.source.write_object(pending)

        class StaleIndex(InMemoryIndexStore):
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                return [
                    IndexHit(
                        uri=pending.uri,
                        score=10.0,
                        context_type="memory",
                        title=pending.title,
                        metadata={
                            "retrieval_scores": {
                                "lexical": 1.0,
                                "vector": 0.0,
                                "identity": 0.0,
                            }
                        },
                    )
                ]

        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "PostgreSQL"},
        )

        result = TargetResolver(StaleIndex(), self.source).resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertIsNone(op.target_uri)

    def test_explicit_cross_user_delete_is_rejected(self) -> None:
        victim = ContextObject(
            uri="memoryos://user/u2/memories/private",
            context_type=ContextType.MEMORY,
            title="private",
            owner_user_id="u2",
        )
        self.source.write_object(victim)
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=victim.uri,
            payload={},
        )

        result = self.resolver.resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_owner_mismatch")
        self.assertEqual(op.status, OperationStatus.REJECTED)

    def test_explicit_cross_tenant_target_is_rejected(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/other-tenant",
            context_type=ContextType.MEMORY,
            title="other tenant",
            owner_user_id="u1",
            tenant_id="tenant-b",
        )
        self.source.write_object(target)
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={"tenant_id": "tenant-a"},
        )

        result = self.resolver.resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_tenant_mismatch")
        self.assertEqual(op.status, OperationStatus.REJECTED)

    def test_payload_owner_mismatch_is_rejected(self) -> None:
        new_obj = ContextObject(
            uri="memoryos://user/u1/memories/pending/p1",
            context_type=ContextType.MEMORY,
            title="pending",
            owner_user_id="u2",
        )
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=new_obj.uri,
            payload={"context_object": new_obj.to_dict()},
        )

        result = self.resolver.resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "payload_owner_mismatch")
        self.assertEqual(op.status, OperationStatus.REJECTED)

    def test_valid_add_uri_is_validated_without_existing_source_object(self) -> None:
        new_obj = ContextObject(
            uri="memoryos://user/u1/memories/pending/p1",
            context_type=ContextType.MEMORY,
            title="pending",
            owner_user_id="u1",
            tenant_id="tenant-a",
        )
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=new_obj.uri,
            payload={"tenant_id": "tenant-a", "context_object": new_obj.to_dict()},
        )

        result = self.resolver.resolve(op, user_id="u1")

        self.assertTrue(result.resolved)
        self.assertEqual(op.status, OperationStatus.RESOLVED)

    def test_explicit_context_memory_type_workspace_and_scope_are_checked(self) -> None:
        scope = {
            "project_id": "beta",
            "visibility": {"tenant_id": "default"},
            "applicability": {
                "all_of": [
                    {"namespace": "memoryos", "kind": "principal", "id": "u1"},
                    {"namespace": "memoryos", "kind": "workspace", "id": "beta"},
                ]
            },
        }
        target = ContextObject(
            uri="memoryos://user/u1/memories/rule",
            context_type=ContextType.MEMORY,
            title="rule",
            owner_user_id="u1",
            metadata={"memory_type": "project_rule", "scope": scope},
        )
        self.source.write_object(target)

        wrong_memory_type = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={"memory_type": "preference"},
        )
        wrong_workspace = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={"memory_type": "project_rule", "project_id": "alpha"},
        )
        wrong_scope = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={
                "memory_type": "project_rule",
                "project_id": "beta",
                "applicability_scope_keys": ["memoryos:principal:u1", "memoryos:workspace:alpha"],
            },
        )
        wrong_context = ContextOperation(
            user_id="u1",
            context_type=ContextType.ACTION_POLICY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={},
        )

        self.assertEqual(self.resolver.resolve(wrong_memory_type, user_id="u1").reason, "target_memory_type_mismatch")
        self.assertEqual(self.resolver.resolve(wrong_workspace, user_id="u1").reason, "target_workspace_mismatch")
        self.assertEqual(self.resolver.resolve(wrong_scope, user_id="u1").reason, "payload_workspace_mismatch")
        self.assertEqual(self.resolver.resolve(wrong_context, user_id="u1").reason, "target_context_type_mismatch")

    def test_explicit_scoped_target_rejects_missing_boundary_dimensions(self) -> None:
        scope = {
            "project_id": "beta",
            "visibility": {"tenant_id": "default"},
            "applicability": {
                "all_of": [
                    {"namespace": "memoryos", "kind": "principal", "id": "u1"},
                    {"namespace": "memoryos", "kind": "workspace", "id": "beta"},
                ]
            },
        }
        target = ContextObject(
            uri="memoryos://user/u1/memories/scoped-rule",
            context_type=ContextType.MEMORY,
            title="scoped rule",
            owner_user_id="u1",
            metadata={"memory_type": "project_rule", "scope": scope},
        )
        self.source.write_object(target)

        missing_memory_type = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={},
        )
        missing_workspace = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={"memory_type": "project_rule"},
        )
        missing_scope = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={"memory_type": "project_rule", "project_id": "beta"},
        )

        memory_result = self.resolver.resolve(missing_memory_type, user_id="u1")
        workspace_result = self.resolver.resolve(missing_workspace, user_id="u1")
        scope_result = self.resolver.resolve(missing_scope, user_id="u1")

        self.assertEqual(memory_result.reason, "target_memory_type_unverified")
        self.assertEqual(workspace_result.reason, "target_workspace_unverified")
        self.assertEqual(scope_result.reason, "target_scope_unverified")
        self.assertEqual(missing_memory_type.status, OperationStatus.REJECTED)
        self.assertEqual(missing_workspace.status, OperationStatus.REJECTED)
        self.assertEqual(missing_scope.status, OperationStatus.REJECTED)

    def test_automatic_scoped_target_with_missing_boundary_goes_pending(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/scoped-rule",
            context_type=ContextType.MEMORY,
            title="PostgreSQL database rule",
            owner_user_id="u1",
            metadata={
                "memory_type": "project_rule",
                "scope": {
                    "project_id": "beta",
                    "visibility": {"tenant_id": "default"},
                    "applicability": {
                        "all_of": [
                            {"namespace": "memoryos", "kind": "principal", "id": "u1"},
                            {"namespace": "memoryos", "kind": "workspace", "id": "beta"},
                        ]
                    },
                },
            },
        )
        self.source.write_object(target)
        self.index.upsert_index(target, content="PostgreSQL database rule")
        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "PostgreSQL"},
        )

        result = self.resolver.resolve(operation, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_review_required")
        self.assertEqual(operation.status, OperationStatus.PENDING)
        self.assertIsNone(operation.target_uri)

    def test_explicit_unscoped_target_rejects_declared_operation_boundaries(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/unscoped",
            context_type=ContextType.MEMORY,
            title="unscoped target",
            owner_user_id="u1",
            metadata={},
        )
        self.source.write_object(target)
        operations = [
            (
                ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.DELETE,
                    target_uri=target.uri,
                    payload={"memory_type": "project_rule"},
                ),
                "target_memory_type_unverified",
            ),
            (
                ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.DELETE,
                    target_uri=target.uri,
                    payload={"project_id": "alpha"},
                ),
                "target_workspace_unverified",
            ),
            (
                ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.DELETE,
                    target_uri=target.uri,
                    payload={
                        "applicability_scope_keys": [
                            "memoryos:principal:u1",
                            "memoryos:workspace:alpha",
                        ]
                    },
                ),
                "target_workspace_unverified",
            ),
        ]

        for operation, expected_reason in operations:
            with self.subTest(expected_reason=expected_reason):
                result = self.resolver.resolve(operation, user_id="u1")
                self.assertFalse(result.resolved)
                self.assertEqual(result.reason, expected_reason)
                self.assertEqual(operation.status, OperationStatus.REJECTED)

    def test_automatic_resolution_filters_target_missing_declared_boundary(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/unscoped",
            context_type=ContextType.MEMORY,
            title="PostgreSQL database rule",
            owner_user_id="u1",
            metadata={},
        )
        self.source.write_object(target)
        self.index.upsert_index(target, content="PostgreSQL database rule")
        operations = [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                payload={"query": "PostgreSQL", "memory_type": "project_rule"},
            ),
            ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                payload={"query": "PostgreSQL", "project_id": "alpha"},
            ),
            ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                payload={
                    "query": "PostgreSQL",
                    "applicability_scope_keys": [
                        "memoryos:principal:u1",
                        "memoryos:workspace:alpha",
                    ],
                },
            ),
        ]

        for operation in operations:
            with self.subTest(payload=operation.payload):
                result = self.resolver.resolve(operation, user_id="u1")
                self.assertFalse(result.resolved)
                self.assertEqual(result.reason, "target_review_required")
                self.assertEqual(operation.status, OperationStatus.PENDING)
                self.assertIsNone(operation.target_uri)

    def test_commit_principal_is_an_implicit_scope_boundary(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/principal-scoped",
            context_type=ContextType.MEMORY,
            title="principal scoped",
            owner_user_id="u1",
            metadata={
                "scope": {
                    "visibility": {"tenant_id": "default"},
                    "applicability": {
                        "all_of": [{"namespace": "memoryos", "kind": "principal", "id": "u1"}]
                    },
                }
            },
        )
        self.source.write_object(target)
        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={},
        )

        result = self.resolver.resolve(operation, user_id="u1")

        self.assertTrue(result.resolved)
        self.assertEqual(operation.status, OperationStatus.RESOLVED)

    def test_automatic_resolution_ignores_invalid_stale_index_uri(self) -> None:
        class StaleIndex(InMemoryIndexStore):
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                return [
                    IndexHit(
                        uri="https://invalid.example/memory",
                        score=1.0,
                        context_type="memory",
                        title="invalid",
                        metadata={"retrieval_scores": {"lexical": 1.0, "vector": 0.0, "identity": 0.0}},
                    )
                ]

        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "invalid"},
        )

        result = TargetResolver(StaleIndex(), self.source).resolve(operation, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_review_required")
        self.assertEqual(operation.status, OperationStatus.PENDING)

    def test_explicit_update_without_source_store_fails_closed(self) -> None:
        resolver = TargetResolver(self.index)
        op = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.UPDATE,
            target_uri="memoryos://user/u1/memories/profile/a",
            payload={},
        )

        result = resolver.resolve(op, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_validation_unavailable")
        self.assertEqual(op.status, OperationStatus.REJECTED)

    def test_supersede_without_source_store_stays_pending_and_does_not_target_replacement(self) -> None:
        replacement = ContextObject(
            uri="memoryos://user/u1/memories/replacement",
            context_type=ContextType.MEMORY,
            title="replacement",
            owner_user_id="u1",
        )
        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.SUPERSEDE,
            payload={
                "query": "database decision",
                "context_object": replacement.to_dict(),
                "content": "new database decision",
            },
        )

        result = TargetResolver(self.index).resolve(operation, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_validation_unavailable")
        self.assertEqual(operation.status, OperationStatus.PENDING)
        self.assertIsNone(operation.target_uri)

    def test_explicit_supersede_validates_old_target_and_distinct_replacement_boundaries(self) -> None:
        scope = {
            "project_id": "alpha",
            "visibility": {"tenant_id": "tenant-a"},
            "applicability": {
                "all_of": [
                    {"namespace": "memoryos", "kind": "principal", "id": "u1"},
                    {"namespace": "memoryos", "kind": "workspace", "id": "alpha"},
                ]
            },
        }
        old = ContextObject(
            uri="memoryos://user/u1/memories/database/old",
            context_type=ContextType.MEMORY,
            title="PostgreSQL",
            owner_user_id="u1",
            tenant_id="tenant-a",
            metadata={"memory_type": "project_decision", "scope": scope},
        )
        replacement = ContextObject(
            uri="memoryos://user/u1/memories/database/new",
            context_type=ContextType.MEMORY,
            title="MySQL",
            owner_user_id="u1",
            tenant_id="tenant-a",
            metadata={"memory_type": "project_decision", "scope": scope},
        )
        self.source.write_object(old, content="PostgreSQL")
        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.SUPERSEDE,
            target_uri=old.uri,
            evidence=[{"event_id": "m1", "quoted_text": "Database formally changed to MySQL."}],
            payload={
                "target_uri": old.uri,
                "tenant_id": "tenant-a",
                "memory_type": "project_decision",
                "project_id": "alpha",
                "applicability_scope_keys": [
                    "memoryos:principal:u1",
                    "memoryos:workspace:alpha",
                ],
                "context_object": replacement.to_dict(),
                "content": "MySQL",
                "reason": "Database formally changed to MySQL.",
            },
        )

        result = self.resolver.resolve(operation, user_id="u1")

        self.assertTrue(result.resolved)
        self.assertEqual(operation.target_uri, old.uri)
        self.assertEqual(operation.payload["context_object"]["uri"], replacement.uri)
        self.assertEqual(operation.evidence[0]["event_id"], "m1")

    def test_explicit_supersede_rejects_reusing_target_as_replacement(self) -> None:
        old = ContextObject(
            uri="memoryos://user/u1/memories/database/old",
            context_type=ContextType.MEMORY,
            title="PostgreSQL",
            owner_user_id="u1",
        )
        self.source.write_object(old)
        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.SUPERSEDE,
            target_uri=old.uri,
            payload={"context_object": old.to_dict(), "reason": "replace"},
        )

        result = self.resolver.resolve(operation, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "supersede_replacement_matches_target")
        self.assertEqual(operation.status, OperationStatus.REJECTED)

    def test_limit_one_still_fetches_two_candidates_for_margin(self) -> None:
        seen_limits: list[int] = []
        objects = []
        hits = []
        for suffix, relevance in (("a", 0.90), ("b", 0.85)):
            obj = ContextObject(
                uri=f"memoryos://user/u1/memories/{suffix}",
                context_type=ContextType.MEMORY,
                title="database",
                owner_user_id="u1",
            )
            self.source.write_object(obj)
            objects.append(obj)
            hits.append(
                IndexHit(
                    uri=obj.uri,
                    score=relevance,
                    context_type="memory",
                    title=obj.title,
                    metadata={
                        "retrieval_scores": {
                            "lexical": relevance,
                            "vector": 0.0,
                            "identity": 0.0,
                        }
                    },
                )
            )

        class RecordingIndex(InMemoryIndexStore):
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                seen_limits.append(limit)
                return hits[:limit]

        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "database"},
        )

        result = TargetResolver(RecordingIndex(), self.source).resolve(operation, user_id="u1", limit=1)

        self.assertGreaterEqual(seen_limits[0], 2)
        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_ambiguous")
        self.assertIsNone(operation.target_uri)

    def test_nan_or_zero_base_relevance_never_resolves_even_with_high_index_score(self) -> None:
        objects = []
        for suffix in ("zero", "nan-score", "nan-base"):
            obj = ContextObject(
                uri=f"memoryos://user/u1/memories/{suffix}",
                context_type=ContextType.MEMORY,
                title="database",
                owner_user_id="u1",
            )
            self.source.write_object(obj)
            objects.append(obj)

        class MalformedScoreIndex(InMemoryIndexStore):
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                return [
                    IndexHit(
                        uri=objects[0].uri,
                        score=1000.0,
                        context_type="memory",
                        metadata={"retrieval_scores": {"lexical": 0.0, "vector": 0.0, "identity": 0.0}},
                    ),
                    IndexHit(
                        uri=objects[1].uri,
                        score=float("nan"),
                        context_type="memory",
                        metadata={"retrieval_scores": {"lexical": 1.0, "vector": 0.0, "identity": 0.0}},
                    ),
                    IndexHit(
                        uri=objects[2].uri,
                        score=1.0,
                        context_type="memory",
                        metadata={
                            "retrieval_scores": {"lexical": float("nan"), "vector": 0.0, "identity": 0.0}
                        },
                    ),
                ]

        operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "database"},
        )

        result = TargetResolver(MalformedScoreIndex(), self.source).resolve(operation, user_id="u1")

        self.assertFalse(result.resolved)
        self.assertEqual(result.reason, "target_review_required")
        self.assertEqual(result.candidates, [])
        self.assertIsNone(operation.target_uri)

    def test_payload_scope_keys_are_an_exact_target_boundary(self) -> None:
        def scoped(uri: str, all_of: list[dict]) -> ContextObject:
            return ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="scoped",
                owner_user_id="u1",
                metadata={"scope": {"applicability": {"all_of": all_of}}},
            )

        principal = {"namespace": "memoryos", "kind": "principal", "id": "u1"}
        workspace = {"namespace": "memoryos", "kind": "workspace", "id": "alpha"}
        exact = scoped("memoryos://user/u1/memories/exact", [principal, workspace])
        principal_only = scoped("memoryos://user/u1/memories/principal-only", [principal])
        self.source.write_object(exact)
        self.source.write_object(principal_only)
        payload = {
            "applicability_scope_keys": ["memoryos:principal:u1", "memoryos:workspace:alpha"]
        }

        exact_result = self.resolver.resolve(
            ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                target_uri=exact.uri,
                payload=dict(payload),
            ),
            user_id="u1",
        )
        subset_operation = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=principal_only.uri,
            payload=dict(payload),
        )
        subset_result = self.resolver.resolve(subset_operation, user_id="u1")

        self.assertTrue(exact_result.resolved)
        self.assertFalse(subset_result.resolved)
        self.assertIn(subset_result.reason, {"target_workspace_unverified", "target_scope_mismatch"})
        self.assertEqual(subset_operation.status, OperationStatus.REJECTED)

    def test_explicit_payload_uri_fields_fail_closed_for_malformed_or_mismatched_values(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/target",
            context_type=ContextType.MEMORY,
            title="target",
            owner_user_id="u1",
        )
        other = ContextObject(
            uri="memoryos://user/u1/memories/other",
            context_type=ContextType.MEMORY,
            title="other",
            owner_user_id="u1",
        )
        self.source.write_object(target)
        self.source.write_object(other)
        cases: list[tuple[dict[str, Any], str]] = [
            ({"target_uri": []}, "invalid_payload_target_uri"),
            ({"policy_uri": {}}, "invalid_payload_target_uri"),
            ({"context_object": {"uri": []}}, "invalid_payload_target_uri"),
            ({"context_object": other.to_dict()}, "payload_target_uri_mismatch"),
        ]

        for payload, reason in cases:
            with self.subTest(reason=reason):
                operation = ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.DELETE,
                    target_uri=target.uri,
                    payload=payload,
                )
                result = self.resolver.resolve(operation, user_id="u1")
                self.assertFalse(result.resolved)
                self.assertEqual(result.reason, reason)
                self.assertEqual(operation.status, OperationStatus.REJECTED)

    def test_malformed_nested_boundaries_are_rejected_without_exceptions(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/target",
            context_type=ContextType.MEMORY,
            title="target",
            owner_user_id="u1",
        )
        self.source.write_object(target)
        malformed_payloads: list[dict[str, Any]] = [
            {"owner_user_id": {}},
            {"tenant_id": []},
            {"workspace_id": {}},
            {"scope": {"visibility": {"tenant_id": []}}},
            {"scope": {"applicability": {"all_of": {"kind": "workspace", "id": "alpha"}}}},
            {"scope": {"applicability": {"all_of": [{"kind": [], "id": "alpha"}]}}},
            {"applicability_scope_keys": [{"kind": "workspace", "id": "alpha"}]},
            {"context_object": {"uri": target.uri, "metadata": {"fields": []}}},
        ]

        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                operation = ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.DELETE,
                    target_uri=target.uri,
                    payload=payload,
                )
                result = self.resolver.resolve(operation, user_id="u1")
                self.assertFalse(result.resolved)
                self.assertEqual(operation.status, OperationStatus.REJECTED)

    def test_malformed_source_scope_is_rejected_for_explicit_and_automatic_resolution(self) -> None:
        target = ContextObject(
            uri="memoryos://user/u1/memories/malformed",
            context_type=ContextType.MEMORY,
            title="PostgreSQL",
            owner_user_id="u1",
            metadata={
                "scope": {
                    "applicability": {"all_of": [{"namespace": "memoryos", "kind": [], "id": "alpha"}]}
                }
            },
        )
        self.source.write_object(target)
        self.index.upsert_index(target, content="PostgreSQL")

        explicit = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target.uri,
            payload={},
        )
        automatic = ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            payload={"query": "PostgreSQL"},
        )

        explicit_result = self.resolver.resolve(explicit, user_id="u1")
        automatic_result = self.resolver.resolve(automatic, user_id="u1")

        self.assertFalse(explicit_result.resolved)
        self.assertEqual(explicit_result.reason, "target_scope_invalid")
        self.assertFalse(automatic_result.resolved)
        self.assertEqual(automatic_result.reason, "target_review_required")


if __name__ == "__main__":
    unittest.main()
