"""通用事务主链和幂等控制记录测试。"""

from __future__ import annotations

import json
import tempfile
import unittest

from infrastructure.store.model.context import ContextObject, ContextType
from infrastructure.store.operation.redo import RedoLog
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.model import ContextOperation, OperationAction
from transaction.model.operation_status import OperationStatus
from transaction.resolver.conflict_resolver import ConflictResolver


class OperationPlaneTest(unittest.TestCase):
    def test_coalesces_add_update_and_add_delete(self) -> None:
        target = "memoryos://user/gulf/behavior_cases/home-comfort"
        add = ContextOperation(
            user_id="gulf",
            context_type=ContextType.BEHAVIOR_CASE,
            action=OperationAction.ADD,
            target_uri=target,
            payload={"title": "old"},
        )
        update = ContextOperation(
            user_id="gulf",
            context_type=ContextType.BEHAVIOR_CASE,
            action=OperationAction.UPDATE,
            target_uri=target,
            payload={"title": "new"},
        )
        resolved = ConflictResolver().resolve([add, update])
        self.assertEqual(len(resolved.accepted), 1)
        self.assertEqual(resolved.accepted[0].action, OperationAction.ADD)
        self.assertEqual(resolved.accepted[0].payload["title"], "new")

        delete = ContextOperation(
            user_id="gulf",
            context_type=ContextType.BEHAVIOR_CASE,
            action=OperationAction.DELETE,
            target_uri=target,
            payload={},
        )
        noop = ConflictResolver().resolve([add, delete])
        self.assertEqual(noop.accepted, [])
        self.assertEqual([item.status for item in noop.rejected], [OperationStatus.NOOP, OperationStatus.NOOP])

    def test_committer_writes_source_index_diff_audit_and_clears_redo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, tmp)
            obj = ContextObject(
                uri="memoryos://user/gulf/behavior_cases/home-comfort",
                context_type=ContextType.BEHAVIOR_CASE,
                title="Home comfort",
                owner_user_id="gulf",
            )
            op = ContextOperation(
                user_id="gulf",
                context_type=ContextType.BEHAVIOR_CASE,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "comfort"},
            )
            diff = committer.commit("gulf", [op])
            self.assertEqual(diff.operations[0].status.value, "committed")
            self.assertEqual(source.read_object(obj.uri).title, "Home comfort")
            self.assertEqual(
                index.search("comfort", tenant_id="default", filters={"owner_user_id": "gulf"})[0].uri,
                obj.uri,
            )
            self.assertEqual(RedoLog(tmp).pending(), [])

            audit_path = committer.audit.root / "system" / "audit" / "gulf.jsonl"
            audit = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
            single_diff = committer.diff_writer.read(f"diff_{op.operation_id}")
            marker = committer.marker_store.read(committer._operation_marker(op.operation_id))
            durable_control_records = [audit["payload"], single_diff, marker]
            encoded = json.dumps(durable_control_records, ensure_ascii=False)

            self.assertNotIn('"content"', encoded)
            self.assertNotIn('"evidence"', encoded)
            self.assertNotIn('"context_object"', encoded)
            self.assertNotIn("operation", marker)
            self.assertNotIn("diff", marker)

    def test_replayed_operation_does_not_publish_an_empty_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            committer = OperationCommitter(source, InMemoryIndexStore(), tmp)
            obj = ContextObject(
                uri="memoryos://user/gulf/behavior_cases/idempotent",
                context_type=ContextType.BEHAVIOR_CASE,
                title="Idempotent",
                owner_user_id="gulf",
            )
            operation = ContextOperation(
                user_id="gulf",
                context_type=ContextType.BEHAVIOR_CASE,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "once"},
            )

            first = committer.commit("gulf", [operation])
            second = committer.commit("gulf", [operation])

            self.assertEqual(second.diff_id, first.diff_id)
            self.assertEqual([item.operation_id for item in second.operations], [operation.operation_id])
            for path in (committer.artifact_root / "system" / "diffs").glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(
                    payload["operations"] or payload["pending_operations"] or payload["rejected_operations"]
                )


if __name__ == "__main__":
    unittest.main()
