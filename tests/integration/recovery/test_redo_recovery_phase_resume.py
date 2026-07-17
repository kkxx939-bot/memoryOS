from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryLockStore,
    InMemoryRelationStore,
)
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.core.ids import require_safe_path_segment
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoIntegrityError, RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class _FailOnceRelationStore(InMemoryRelationStore):
    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.add_calls = 0
        self.failed = False

    def add_relation(self, relation):  # noqa: ANN001, ANN201
        self.add_calls += 1
        if not self.failed and self.add_calls == self.fail_at:
            self.failed = True
            raise OSError("injected relation write failure")
        return super().add_relation(relation)


class RedoRecoveryPhaseResumeTest(unittest.TestCase):
    def test_artifact_identifiers_reject_path_escape_at_model_and_io_boundaries(self) -> None:
        unsafe_values = ("", ".", "..", "../escape", "..\\escape", "nul\x00escape")
        for value in unsafe_values:
            with self.subTest(value=value, boundary="helper"):
                with self.assertRaises(ValueError):
                    require_safe_path_segment(value, "artifact_id")
            if not value:
                continue
            with self.subTest(value=value, boundary="operation"):
                with self.assertRaises(ValueError):
                    ContextOperation(
                        user_id="u1",
                        context_type=ContextType.MEMORY,
                        action=OperationAction.ADD,
                        payload={},
                        operation_id=value,
                    )
            with self.subTest(value=value, boundary="diff"):
                with self.assertRaises(ValueError):
                    ContextDiff(user_id="u1", diff_id=value)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root, tenant_id="tenant-a")
            committer = OperationCommitter(source, InMemoryIndexStore(), str(root))
            uri = "memoryos://user/u1/memories/path-safety"
            obj = ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="path safety",
                owner_user_id="u1",
                tenant_id="tenant-a",
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=uri,
                payload={"tenant_id": "tenant-a", "context_object": obj.to_dict()},
                operation_id="safe-before-mutation",
            )
            operation.operation_id = "../../escaped-operation"

            with self.assertRaisesRegex(ValueError, "operation_id"):
                committer.commit("u1", [operation])
            with self.assertRaises(FileNotFoundError):
                source.read_object(uri)
            self.assertFalse((root / "tenants" / "tenant-a" / "system").exists())

            user_operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=uri,
                payload={"tenant_id": "tenant-a", "context_object": obj.to_dict()},
            )
            user_operation.user_id = "../../escaped-user"
            with self.assertRaisesRegex(ValueError, "commit user_id"):
                committer.commit("../../escaped-user", [user_operation])
            self.assertFalse((root / "tenants" / "tenant-a" / "system").exists())

            mutated_redo = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                payload={},
                operation_id="redo-safe",
            )
            mutated_redo.operation_id = "../redo-escape"
            with self.assertRaisesRegex(ValueError, "operation_id"):
                RedoLog(root / "redo-boundary").begin(mutated_redo)

            mutated_diff = ContextDiff(user_id="u1", diff_id="diff-safe")
            mutated_diff.diff_id = "../diff-escape"
            with self.assertRaisesRegex(ValueError, "diff_id"):
                DiffWriter(root / "diff-boundary").write(mutated_diff)
            with self.assertRaisesRegex(ValueError, "user_id"):
                AuditWriter(root / "audit-boundary").record("../audit-escape", "event", {})
            self.assertFalse((root / "redo-boundary" / "system").exists())
            self.assertFalse((root / "diff-boundary" / "system").exists())
            self.assertFalse((root / "audit-boundary" / "system").exists())

    def test_safe_extended_identifiers_keep_default_artifact_paths_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operation_id = require_safe_path_segment("op.v1_test-user@example", "operation_id")
            user_id = require_safe_path_segment("user.name@example.com", "user_id")
            diff_id = require_safe_path_segment("diff.v1_test-user@example", "diff_id")
            operation = ContextOperation(
                user_id=user_id,
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                payload={},
                operation_id=operation_id,
            )
            RedoLog(root).begin(operation)
            DiffWriter(root).write(ContextDiff(user_id=user_id, diff_id=diff_id))
            AuditWriter(root).record(user_id, "compatible_identifier", {})

            self.assertTrue((root / "system" / "redo" / f"{operation_id}.json").exists())
            self.assertTrue((root / "system" / "diffs" / f"{diff_id}.json").exists())
            self.assertTrue((root / "system" / "audit" / f"{user_id}.jsonl").exists())

    def test_runtime_queue_and_committer_artifacts_follow_tenant_namespace(self) -> None:
        from memoryos.runtime import RuntimeConfig, build_runtime_container

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = build_runtime_container(RuntimeConfig(root=str(root)))
            tenant = build_runtime_container(
                RuntimeConfig(root=str(root), tenant_id="tenant-a")
            )

            self.assertEqual(default.queue_store.path, root / "queues" / "jobs.sqlite3")  # type: ignore[attr-defined]
            self.assertEqual(default.committer.artifact_root, root)
            self.assertEqual(
                tenant.queue_store.path,  # type: ignore[attr-defined]
                root / "tenants" / "tenant-a" / "queues" / "jobs.sqlite3",
            )
            self.assertEqual(tenant.committer.artifact_root, root / "tenants" / "tenant-a")

    def test_committer_binds_source_tenant_and_rejects_cross_tenant_fresh_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root, tenant_id="tenant-a")
            committer = OperationCommitter(source, InMemoryIndexStore(), str(root))
            self.assertEqual(committer.tenant_id, "tenant-a")
            with self.assertRaisesRegex(ValueError, "does not match SourceStore tenant"):
                OperationCommitter(
                    source,
                    InMemoryIndexStore(),
                    str(root),
                    tenant_id="tenant-b",
                )

            for suffix, mutate in (
                ("object", lambda payload: payload["context_object"].__setitem__("tenant_id", "tenant-b")),
                ("payload", lambda payload: payload.__setitem__("tenant_id", "tenant-b")),
                (
                    "visibility",
                    lambda payload: payload["context_object"]["metadata"]["scope"]["visibility"].__setitem__(
                        "tenant_id", "tenant-b"
                    ),
                ),
                (
                    "authority",
                    lambda payload: payload["context_object"]["metadata"]["scope"]["authority"].__setitem__(
                        "tenant_id", "tenant-b"
                    ),
                ),
            ):
                uri = f"memoryos://user/u1/memories/cross-{suffix}"
                obj = ContextObject(
                    uri=uri,
                    context_type=ContextType.MEMORY,
                    title=suffix,
                    owner_user_id="u1",
                    tenant_id="tenant-a",
                    metadata={
                        "scope": {
                            "visibility": {"tenant_id": "tenant-a"},
                            "authority": {"tenant_id": "tenant-a", "principal_ids": ["u1"]},
                        }
                    },
                )
                payload = {"tenant_id": "tenant-a", "context_object": obj.to_dict()}
                mutate(payload)
                operation = ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.ADD,
                    target_uri=uri,
                    payload=payload,
                )
                with self.assertRaisesRegex(ValueError, "bound tenant"):
                    committer.commit("u1", [operation])
                with self.assertRaises(FileNotFoundError):
                    source.read_object(uri)

            self.assertFalse((root / "system").exists())

    def test_default_tenant_is_bound_when_regular_operation_omits_tenant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            committer = OperationCommitter(source, InMemoryIndexStore(), str(root))
            obj = ContextObject(
                uri="memoryos://user/u1/memories/implicit-default-tenant",
                context_type=ContextType.MEMORY,
                title="implicit default",
                owner_user_id="u1",
            ).to_dict()
            obj.pop("tenant_id")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=str(obj["uri"]),
                payload={"context_object": obj},
            )

            diff = committer.commit("u1", [operation])

            self.assertEqual([item.operation_id for item in diff.operations], [operation.operation_id])
            self.assertEqual(operation.payload["tenant_id"], "default")
            self.assertEqual(operation.payload["context_object"]["tenant_id"], "default")
            self.assertEqual(source.read_object(str(obj["uri"])).tenant_id, "default")
            self.assertEqual(committer.artifact_root, root)
            self.assertTrue(
                (root / "system" / "operations" / f"{operation.operation_id}.json").exists()
            )
            self.assertFalse((root / "tenants" / "default" / "system").exists())

    def test_nondefault_tenant_artifacts_are_physically_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_store = InMemoryLockStore()

            def build(tenant_id: str):  # noqa: ANN202
                source = FileSystemSourceStore(root, tenant_id=tenant_id)
                committer = OperationCommitter(
                    source,
                    InMemoryIndexStore(),
                    str(root),
                    lock_store=lock_store,
                )
                return source, committer

            source_a, tenant_a = build("tenant-a")
            source_b, tenant_b = build("tenant-b")
            uri = "memoryos://user/u1/memories/same-operation-id"

            def regular(tenant_id: str, title: str) -> ContextOperation:
                obj = ContextObject(
                    uri=uri,
                    context_type=ContextType.MEMORY,
                    title=title,
                    owner_user_id="u1",
                    tenant_id=tenant_id,
                )
                return ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.ADD,
                    target_uri=uri,
                    payload={"tenant_id": tenant_id, "context_object": obj.to_dict()},
                    operation_id="same-operation-id",
                )

            tenant_a.commit("u1", [regular("tenant-a", "tenant A")])
            tenant_b.commit("u1", [regular("tenant-b", "tenant B")])

            artifact_a = root / "tenants" / "tenant-a"
            artifact_b = root / "tenants" / "tenant-b"
            marker = Path("system/operations/same-operation-id.json")
            self.assertTrue((artifact_a / marker).exists())
            self.assertTrue((artifact_b / marker).exists())
            self.assertFalse((root / marker).exists())
            self.assertEqual(source_a.read_object(uri).title, "tenant A")
            self.assertEqual(source_b.read_object(uri).title, "tenant B")

            def canonical_marker(committer: OperationCommitter, tenant_id: str) -> ContextOperation:
                obj = ContextObject(
                    uri=f"memoryos://user/u1/memories/{tenant_id}-canonical-marker",
                    context_type=ContextType.MEMORY,
                    title=tenant_id,
                    owner_user_id="u1",
                    tenant_id=tenant_id,
                )
                operation = ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.ADD,
                    target_uri=obj.uri,
                    payload={
                        "tenant_id": tenant_id,
                        "context_object": obj.to_dict(),
                        "canonical_memory": True,
                        "transaction_id": "same-transaction",
                        "idempotency_key": "same-idempotency",
                        "commit_group_id": "same-commit-group",
                    },
                    operation_id="same-canonical-operation",
                )
                before_images = committer._capture_canonical_state([operation])
                before_by_uri = {
                    str(item["uri"]): item.get("object")
                    for item in before_images
                }
                relation_manifests = {
                    operation.operation_id: committer._build_canonical_relation_manifest(
                        operation,
                        before_by_uri.get(str(operation.target_uri or "")),
                    )
                }
                committer._write_outbox_event(
                    "same-transaction",
                    "same-idempotency",
                    [operation],
                    status="prepared",
                    before_images=before_images,
                    relation_manifests=relation_manifests,
                )
                committer.source_store.write_object(obj)
                committer._write_outbox_event(
                    "same-transaction",
                    "same-idempotency",
                    [operation],
                    status="source_committed",
                    before_images=before_images,
                    relation_manifests=relation_manifests,
                )
                committer._write_transaction_marker(
                    committer._transaction_marker("same-idempotency"),
                    ContextDiff(
                        user_id="u1",
                        operations=[operation],
                        diff_id=f"diff-{tenant_id}",
                    ),
                    [operation],
                    relation_manifests=relation_manifests,
                )
                committer._finalize_canonical_outbox(
                    "same-transaction",
                    "same-idempotency",
                    [operation],
                )
                return operation

            canonical_marker(tenant_a, "tenant-a")
            canonical_b = canonical_marker(tenant_b, "tenant-b")
            self.assertTrue((artifact_a / "system/outbox/same-transaction.json").exists())
            self.assertTrue((artifact_b / "system/outbox/same-transaction.json").exists())
            self.assertTrue((artifact_a / "system/transactions/same-idempotency.json").exists())
            self.assertTrue((artifact_b / "system/transactions/same-idempotency.json").exists())

            (artifact_a / "system/transactions/broken.json").write_text("{broken", encoding="utf-8")
            (artifact_a / "system/redo/broken.json").write_text("{broken", encoding="utf-8")
            committed = tenant_b.committed_canonical_diffs("u1", "same-commit-group")
            self.assertEqual(
                [operation.operation_id for diff in committed for operation in diff.operations],
                [canonical_b.operation_id],
            )

            recovery_uri = "memoryos://user/u1/memories/tenant-b-recovery"
            recovery_obj = ContextObject(
                uri=recovery_uri,
                context_type=ContextType.MEMORY,
                title="tenant B recovery",
                owner_user_id="u1",
                tenant_id="tenant-b",
            )
            recovery_operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=recovery_uri,
                payload={"tenant_id": "tenant-b", "context_object": recovery_obj.to_dict()},
                operation_id="tenant-b-recovery",
            )
            relation_manifest = tenant_b._build_regular_relation_manifest(recovery_operation)
            tenant_b.redo.begin(
                recovery_operation,
                phase="started",
                relation_manifest=relation_manifest,
            )
            recovered = RecoveryService(tenant_b.redo, tenant_b).recover("u1")
            self.assertEqual(recovered.operation_ids, [recovery_operation.operation_id])
            self.assertEqual(source_b.read_object(recovery_uri).title, "tenant B recovery")

    def test_cross_tenant_regular_and_canonical_redo_are_never_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root, tenant_id="tenant-a")
            committer = OperationCommitter(source, InMemoryIndexStore(), str(root))
            uri = "memoryos://user/u1/memories/cross-redo"
            obj = ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="cross redo",
                owner_user_id="u1",
                tenant_id="tenant-b",
            )
            regular = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=uri,
                payload={
                    "tenant_id": "tenant-b",
                    "context_object": obj.to_dict(),
                    "canonical_pending_proposal": True,
                    "commit_group_id": "cross-regular",
                },
            )
            with self.assertRaises(RedoIntegrityError):
                committer.resume("u1", regular, "started")
            committer.redo.begin(regular, phase="started")
            self.assertEqual(
                committer.recover_pending_regular_memory("u1", commit_group_id="cross-regular"),
                [],
            )

            canonical = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=uri,
                payload={
                    "tenant_id": "tenant-b",
                    "context_object": obj.to_dict(),
                    "canonical_memory": True,
                    "transaction_id": "cross-canonical",
                    "idempotency_key": "cross-canonical",
                    "commit_group_id": "cross-canonical-group",
                },
            )
            with self.assertRaises(RedoIntegrityError):
                committer.resume_canonical_batch(
                    "u1",
                    [SimpleNamespace(operation=canonical, source_effect=None, relation_manifest=None)],
                )
            committer.redo.begin(canonical, phase="started")
            self.assertEqual(
                committer.recover_pending_canonical("u1", commit_group_id="cross-canonical-group"),
                [],
            )
            self.assertEqual(
                {entry.operation_id for entry in committer.redo.pending_entries()},
                {regular.operation_id, canonical.operation_id},
            )
            with self.assertRaises(FileNotFoundError):
                source.read_object(uri)

    def test_context_relation_timestamp_round_trip_and_regular_commit_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            relations = InMemoryRelationStore()
            committer = OperationCommitter(
                source,
                InMemoryIndexStore(),
                str(root),
                relation_store=relations,
            )
            obj = ContextObject(
                uri="memoryos://user/u1/memories/relation-timestamp",
                context_type=ContextType.MEMORY,
                title="relation timestamp",
                owner_user_id="u1",
                relations=[
                    ContextRelation(
                        source_uri="memoryos://user/u1/memories/relation-timestamp",
                        relation_type="evidence_for",
                        target_uri="memoryos://user/u1/behaviors/b1",
                        metadata={"tenant_id": "default", "owner_user_id": "u1"},
                        created_at="2026-07-12T00:00:00+00:00",
                    )
                ],
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "stable relation"},
            )

            diff = committer.commit("u1", [operation])
            first = source.read_object(obj.uri).to_dict()
            second = source.read_object(obj.uri).to_dict()

            self.assertEqual(first, second)
            self.assertEqual(first["relations"][0]["created_at"], "2026-07-12T00:00:00+00:00")
            self.assertEqual([item.operation_id for item in diff.operations], [operation.operation_id])
            self.assertEqual(len(relations.relations), 1)

            legacy = obj.to_dict()
            legacy["relations"][0].pop("created_at")
            self.assertEqual(
                ContextObject.from_dict(legacy).to_dict()["relations"][0]["created_at"],
                "",
            )

    def test_started_refresh_layers_replays_stale_layers_under_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            uri = "memoryos://user/u1/memories/refresh-stale"
            obj = ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="refreshed title",
                owner_user_id="u1",
                layers=ContextLayers(
                    l0_uri=f"{uri}/.abstract.md",
                    l1_uri=f"{uri}/.overview.md",
                    l2_uri=f"{uri}/content.md",
                ),
            )
            source.write_object(obj)
            source.write_content(str(obj.layers.l0_uri), "STALE-L0")
            source.write_content(str(obj.layers.l1_uri), "STALE-L1")
            source.write_content(str(obj.layers.l2_uri), "fresh source")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.REFRESH_LAYERS,
                target_uri=uri,
                payload={"tenant_id": "default"},
            )
            manifest = committer._build_regular_relation_manifest(operation)
            committer.redo.begin(operation, phase="started", relation_manifest=manifest)

            recovered = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(recovered.operation_ids, [operation.operation_id])
            self.assertNotEqual(source.read_content(str(obj.layers.l0_uri)), "STALE-L0")
            self.assertFalse(committer.redo.pending_entries())
            self.assertIn(uri, index.indexed_uris())

    def test_relation_manifest_removes_old_managed_edge_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            relations = InMemoryRelationStore()
            committer = OperationCommitter(
                source,
                index,
                str(root),
                relation_store=relations,
            )
            uri = "memoryos://user/u1/memories/relation-update"
            initial = ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="initial",
                owner_user_id="u1",
                metadata={"supporting_behavior_uris": ["memoryos://user/u1/behaviors/b1"]},
            )
            committer.commit(
                "u1",
                [
                    ContextOperation(
                        user_id="u1",
                        context_type=ContextType.MEMORY,
                        action=OperationAction.ADD,
                        target_uri=uri,
                        payload={"context_object": initial.to_dict()},
                    )
                ],
            )
            desired = ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="updated",
                owner_user_id="u1",
            )
            update = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.UPDATE,
                target_uri=uri,
                payload={"context_object": desired.to_dict()},
            )
            manifest = committer._build_regular_relation_manifest(update)
            self.assertEqual(len(manifest["remove"]), 1)
            tampered = json.loads(json.dumps(manifest))
            tampered["remove"][0]["target_uri"] = "memoryos://user/u1/behaviors/forged"
            committer.redo.begin(update, phase="started", relation_manifest=tampered)

            failed = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(failed.recovered_count, 0)
            self.assertEqual(source.read_object(uri).title, "initial")
            self.assertEqual(len(relations.relations), 1)
            self.assertFalse((root / "system" / "operations" / f"{update.operation_id}.json").exists())
            self.assertEqual(committer.redo.pending_entries(), [])
            self.assertTrue(list((root / "system" / "quarantine" / "redo").glob("*.json")))

            committer.redo.begin(update, phase="started", relation_manifest=manifest)
            recovered = RecoveryService(committer.redo, committer).recover("u1")
            self.assertEqual(recovered.operation_ids, [update.operation_id])
            self.assertEqual(relations.relations, [])

    def test_same_target_policy_operation_is_marked_before_next_operation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            policy = ActionPolicy(
                user_id="u1",
                scene_key="sequential",
                action="turn_on_ac",
                memory_anchor_uri="memoryos://user/u1/memories/anchors/sequential",
            )
            source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
            reward = ContextOperation(
                user_id="u1",
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.REWARD,
                target_uri=policy.uri,
                payload={"reward": 1.0},
                operation_id="policy-first",
            )
            cooldown = ContextOperation(
                user_id="u1",
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.COOLDOWN,
                target_uri=policy.uri,
                payload={"cooldown_until": "2026-07-13T00:00:00+00:00"},
                operation_id="policy-second",
            )
            original_apply = committer._apply_source

            def fail_second(operation):  # noqa: ANN001, ANN202
                if operation.operation_id == cooldown.operation_id:
                    raise OSError("second policy operation failed")
                return original_apply(operation)

            committer._apply_source = fail_second  # type: ignore[method-assign]
            with self.assertRaisesRegex(OSError, "second policy operation failed"):
                committer.commit("u1", [reward, cooldown])

            self.assertTrue((root / "system" / "operations" / "policy-first.json").exists())
            self.assertTrue((root / "system" / "diffs" / "diff_policy-first.json").exists())
            self.assertEqual(
                [entry.operation_id for entry in committer.redo.pending_entries()],
                [cooldown.operation_id],
            )
            redo_payload = json.loads(
                (root / "system" / "redo" / f"{cooldown.operation_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("redo_relation_manifest", redo_payload)
            self.assertNotIn("redo_relation_manifest", redo_payload["payload"])
            self.assertNotIn(
                "redo_relation_manifest",
                (root / "system" / "operations" / "policy-first.json").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertNotIn(
                "redo_relation_manifest",
                (root / "system" / "diffs" / "diff_policy-first.json").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(reward.operation_id, source.read_object(policy.uri).metadata["applied_operation_ids"])

            committer._apply_source = original_apply  # type: ignore[method-assign]
            recovered = RecoveryService(committer.redo, committer).recover("u1")
            self.assertEqual(recovered.operation_ids, [cooldown.operation_id])
            final = source.read_object(policy.uri).metadata
            self.assertEqual(final["applied_operation_ids"], [reward.operation_id, cooldown.operation_id])

    def test_combined_regular_diff_preserves_marker_backed_pending_and_rejected_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            committer = OperationCommitter(source, InMemoryIndexStore(), str(root))
            obj = ContextObject(
                uri="memoryos://user/u1/memories/combined-diff",
                context_type=ContextType.MEMORY,
                title="combined diff",
                owner_user_id="u1",
            )
            committed = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict()},
                operation_id="combined-committed",
            )
            pending = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.UPDATE,
                payload={"query": "no matching target"},
                operation_id="combined-pending",
            )
            rejected = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                target_uri="memoryos://user/u2/memories/cross-owner",
                payload={},
                operation_id="combined-rejected",
            )

            diff = committer.commit("u1", [committed, pending, rejected])

            self.assertEqual([item.operation_id for item in diff.operations], [committed.operation_id])
            self.assertEqual(
                [item.operation_id for item in diff.pending_operations],
                [pending.operation_id],
            )
            self.assertEqual(
                [item.operation_id for item in diff.rejected_operations],
                [rejected.operation_id],
            )
            self.assertTrue(
                (root / "system" / "operations" / f"{committed.operation_id}.json").exists()
            )
            self.assertFalse(
                (root / "system" / "operations" / f"{pending.operation_id}.json").exists()
            )
            self.assertFalse(
                (root / "system" / "operations" / f"{rejected.operation_id}.json").exists()
            )

    def test_started_phase_replays_without_reacquiring_its_own_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(
                uri="memoryos://user/u1/memories/preferences/started",
                context_type=ContextType.MEMORY,
                title="started",
                owner_user_id="u1",
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "started recovery"},
            )
            committer.redo.begin(operation, phase="started")

            result = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(result.operation_ids, [operation.operation_id])
            self.assertEqual(source.read_content(obj.uri), "started recovery")
            self.assertFalse(committer.redo.pending_entries())

    def test_started_phase_adopts_exact_source_effect_after_crash_before_phase_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(
                uri="memoryos://user/u1/memories/preferences/source-before-phase",
                context_type=ContextType.MEMORY,
                title="source before phase",
                owner_user_id="u1",
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "durable before phase"},
            )
            original_advance = committer.redo.advance

            def crash_before_source_phase(operation_arg, phase, **kwargs):  # noqa: ANN001, ANN202
                if phase == "source_written":
                    raise SystemExit("crash before source phase advance")
                return original_advance(operation_arg, phase, **kwargs)

            committer.redo.advance = crash_before_source_phase  # type: ignore[assignment]
            with self.assertRaisesRegex(SystemExit, "source phase advance"):
                committer.commit("u1", [operation])
            self.assertEqual(committer.redo.pending_entries()[0].phase, "started")
            self.assertEqual(source.read_content(f"{obj.uri}/content.md"), "durable before phase")

            committer.redo.advance = original_advance  # type: ignore[method-assign]
            recovered = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(recovered.operation_ids, [operation.operation_id])
            self.assertIn(obj.uri, index.indexed_uris())
            self.assertFalse(committer.redo.pending_entries())

    def test_source_written_add_resumes_index_audit_and_diff_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature", owner_user_id="u1")
            op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.ADD, target_uri=obj.uri, payload={"context_object": obj.to_dict(), "content": "prefers 26"})
            committer._apply_source(op)
            committer.redo.begin(
                op,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(op),
            )
            RecoveryService(committer.redo, committer).recover("u1")
            RecoveryService(committer.redo, committer).recover("u1")
            self.assertTrue(index.search("26", filters={"owner_user_id": "u1", "context_type": "memory"}))
            audit_path = root / "system" / "audit" / "u1.jsonl"
            self.assertEqual(len(audit_path.read_text(encoding="utf-8").splitlines()), 1)
            diff_files = list((root / "system" / "diffs").glob(f"diff_{op.operation_id}.json"))
            self.assertEqual(len(diff_files), 1)

    def test_source_written_reward_and_penalty_do_not_reapply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
            source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
            index.upsert_index(policy.to_context_object(), content="policy")
            reward = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.REWARD, target_uri=policy.uri, payload={"reward": 1.0}, operation_id="reward-once")
            committer.commit("u1", [reward])
            first = source.read_object(policy.uri).metadata
            committer.redo.begin(
                reward,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(reward),
            )
            RecoveryService(committer.redo, committer).recover("u1")
            second = source.read_object(policy.uri).metadata
            self.assertEqual(first["success_count"], second["success_count"])
            penalty = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.PENALIZE, target_uri=policy.uri, payload={"penalty": 1.0}, operation_id="penalty-once")
            committer.commit("u1", [penalty])
            third = source.read_object(policy.uri).metadata
            committer.redo.begin(
                penalty,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(penalty),
            )
            RecoveryService(committer.redo, committer).recover("u1")
            fourth = source.read_object(policy.uri).metadata
            self.assertEqual(third["failure_count"], fourth["failure_count"])

    def test_source_effect_tampering_stops_recovery_before_index_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(
                uri="memoryos://user/u1/memories/preferences/integrity",
                context_type=ContextType.MEMORY,
                title="integrity",
                owner_user_id="u1",
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "expected"},
            )
            source.write_object(obj, content="expected")
            committer.redo.begin(
                operation,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(operation),
            )
            source.write_content(obj.uri, "tampered")

            result = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(result.recovered_count, 0)
            self.assertNotIn(obj.uri, index.indexed_uris())
            self.assertFalse((root / "system" / "redo" / f"{operation.operation_id}.json").exists())
            self.assertTrue(list((root / "system" / "quarantine" / "redo").glob("*.json")))
            self.assertFalse((root / "system" / "operations" / f"{operation.operation_id}.json").exists())
            self.assertFalse(list((root / "system" / "diffs").glob("*.json")))

    def test_source_written_delete_verifies_soft_delete_then_finishes_index_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(
                uri="memoryos://user/u1/memories/preferences/delete-me",
                context_type=ContextType.MEMORY,
                title="delete me",
                owner_user_id="u1",
            )
            source.write_object(obj, content="delete me")
            index.upsert_index(obj, content="delete me")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                target_uri=obj.uri,
                payload={"tenant_id": "default"},
            )
            committer.redo.begin(operation, phase="started")
            committer._apply_source(operation)
            committer.redo.advance(
                operation,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(operation),
            )

            recovered = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(recovered.operation_ids, [operation.operation_id])
            self.assertEqual(source.read_object(obj.uri).lifecycle_state.value, "deleted")
            self.assertNotIn(obj.uri, index.indexed_uris())
            self.assertFalse(committer.redo.pending_entries())

    def test_started_recovery_restores_first_or_partial_multi_relation_effect(self) -> None:
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = FileSystemSourceStore(root)
                index = InMemoryIndexStore()
                relations = _FailOnceRelationStore(fail_at)
                committer = OperationCommitter(
                    source,
                    index,
                    str(root),
                    relation_store=relations,
                )
                obj = ContextObject(
                    uri=f"memoryos://user/u1/memories/relation-crash-{fail_at}",
                    context_type=ContextType.MEMORY,
                    title="relation crash",
                    owner_user_id="u1",
                    metadata={
                        "supporting_behavior_uris": [
                            "memoryos://user/u1/behaviors/b1",
                            "memoryos://user/u1/behaviors/b2",
                        ]
                    },
                )
                operation = ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.ADD,
                    target_uri=obj.uri,
                    payload={"context_object": obj.to_dict(), "content": "relation effect"},
                )

                with self.assertRaisesRegex(OSError, "relation write failure"):
                    committer.commit("u1", [operation])

                self.assertEqual(committer.redo.pending_entries()[0].phase, "started")
                recovered = RecoveryService(committer.redo, committer).recover("u1")
                relation_keys = {
                    (item.source_uri, item.relation_type, item.target_uri)
                    for item in relations.relations
                }
                self.assertEqual(recovered.operation_ids, [operation.operation_id])
                self.assertEqual(
                    relation_keys,
                    {
                        (obj.uri, "evidence_for", "memoryos://user/u1/behaviors/b1"),
                        (obj.uri, "evidence_for", "memoryos://user/u1/behaviors/b2"),
                    },
                )
                self.assertEqual(len(relations.relations), 2)

    def test_started_supersede_recovery_restores_both_relation_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            relations = _FailOnceRelationStore(1)
            committer = OperationCommitter(
                source,
                index,
                str(root),
                relation_store=relations,
            )
            old = ContextObject(
                uri="memoryos://user/u1/memories/old-choice",
                context_type=ContextType.MEMORY,
                title="old",
                owner_user_id="u1",
            )
            new = ContextObject(
                uri="memoryos://user/u1/memories/new-choice",
                context_type=ContextType.MEMORY,
                title="new",
                owner_user_id="u1",
            )
            source.write_object(old, content="old")
            index.upsert_index(old, content="old")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.SUPERSEDE,
                target_uri=old.uri,
                payload={"context_object": new.to_dict(), "content": "new", "reason": "explicit"},
            )

            with self.assertRaisesRegex(OSError, "relation write failure"):
                committer.commit("u1", [operation])
            recovered = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(recovered.operation_ids, [operation.operation_id])
            self.assertEqual(
                {
                    (item.source_uri, item.relation_type, item.target_uri)
                    for item in relations.relations
                },
                {
                    (new.uri, "supersedes", old.uri),
                    (old.uri, "superseded_by", new.uri),
                },
            )

    def test_self_consistent_supersede_snapshot_with_wrong_new_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            old = ContextObject(
                uri="memoryos://user/u1/memories/preferences/old",
                context_type=ContextType.MEMORY,
                title="old",
                owner_user_id="u1",
            )
            desired = ContextObject(
                uri="memoryos://user/u1/memories/preferences/new",
                context_type=ContextType.MEMORY,
                title="expected new",
                owner_user_id="u1",
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.SUPERSEDE,
                target_uri=old.uri,
                payload={
                    "context_object": desired.to_dict(),
                    "content": "expected content",
                    "reason": "replacement",
                },
            )
            superseded_at = "2026-07-12T00:00:00Z"
            old.lifecycle_state = LifecycleState.OBSOLETE
            old.metadata = {
                "superseded_by": desired.uri,
                "superseded_at": superseded_at,
                "supersede_reason": "replacement",
            }
            forged = ContextObject.from_dict(desired.to_dict())
            forged.title = "forged new"
            forged.metadata = {
                "supersedes": old.uri,
                "superseded_at": superseded_at,
                "supersede_reason": "replacement",
            }
            source.write_object(old, content="old")
            source.write_object(forged, content="forged content")
            committer.redo.begin(
                operation,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(operation),
            )

            recovered = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(recovered.recovered_count, 0)
            self.assertNotIn(old.uri, index.indexed_uris())
            self.assertNotIn(desired.uri, index.indexed_uris())
            self.assertFalse((root / "system" / "redo" / f"{operation.operation_id}.json").exists())
            self.assertTrue(list((root / "system" / "quarantine" / "redo").glob("*.json")))

    def test_recovery_never_claims_another_users_redo_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(
                uri="memoryos://user/u2/memories/preferences/private",
                context_type=ContextType.MEMORY,
                title="private",
                owner_user_id="u2",
            )
            operation = ContextOperation(
                user_id="u2",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "u2 only"},
            )
            committer._apply_source(operation)
            committer.redo.begin(
                operation,
                phase="source_written",
                source_effect=committer._capture_regular_source_effect(operation),
            )

            skipped = RecoveryService(committer.redo, committer).recover("u1")

            self.assertEqual(skipped.recovered_count, 0)
            self.assertTrue((root / "system" / "redo" / f"{operation.operation_id}.json").exists())
            self.assertNotIn(obj.uri, index.indexed_uris())
            recovered = RecoveryService(committer.redo, committer).recover("u2")
            self.assertEqual(recovered.operation_ids, [operation.operation_id])
            self.assertIn(obj.uri, index.indexed_uris())


if __name__ == "__main__":
    unittest.main()
