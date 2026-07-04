from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.workers.recovery_worker import RecoveryWorker


def _committer(tmp_path):
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    worker = RecoveryWorker(RecoveryService(committer.redo, committer))
    return source, index, committer, worker


def _add_op(operation_id: str = "op-add") -> ContextOperation:
    obj = ContextObject(
        uri=f"memoryos://user/u1/memories/profile/{operation_id}",
        context_type=ContextType.MEMORY,
        title=operation_id,
        owner_user_id="u1",
    )
    return ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "alpha"},
        operation_id=operation_id,
    )


def test_recovery_worker_recovers_started_by_recommitting(tmp_path) -> None:
    source, index, committer, worker = _committer(tmp_path)
    op = _add_op("started")
    committer.redo.begin(op, phase="started")

    result = worker.process_pending("u1")

    assert result["recovered_count"] == 1
    assert op.target_uri is not None
    assert source.read_object(op.target_uri).title == "started"
    assert index.search("alpha", filters={"owner_user_id": "u1"})
    assert not (tmp_path / "system" / "redo" / f"{op.operation_id}.json").exists()


def test_recovery_worker_source_written_completes_index_audit_diff(tmp_path) -> None:
    source, index, committer, worker = _committer(tmp_path)
    op = _add_op("source-written")
    committer._apply_source(op)
    committer.redo.begin(op, phase="source_written")

    result = worker.process_pending("u1")

    assert result["operation_ids"] == [op.operation_id]
    assert index.search("alpha", filters={"owner_user_id": "u1"})
    assert (tmp_path / "system" / "audit" / "u1.jsonl").exists()
    assert (tmp_path / "system" / "diffs" / f"diff_{op.operation_id}.json").exists()


def test_recovery_worker_index_written_completes_audit_and_diff(tmp_path) -> None:
    _, index, committer, worker = _committer(tmp_path)
    op = _add_op("index-written")
    committer._apply_source(op)
    committer._apply_index(op)
    committer.redo.begin(op, phase="index_written")

    worker.process_pending("u1")

    assert index.search("alpha", filters={"owner_user_id": "u1"})
    assert (tmp_path / "system" / "audit" / "u1.jsonl").exists()
    assert (tmp_path / "system" / "diffs" / f"diff_{op.operation_id}.json").exists()


def test_recovery_worker_audit_written_completes_diff(tmp_path) -> None:
    _, _, committer, worker = _committer(tmp_path)
    op = _add_op("audit-written")
    committer._apply_source(op)
    committer._apply_index(op)
    committer.audit.record("u1", "context_operation_committed", op.to_dict())
    committer.redo.begin(op, phase="audit_written")

    worker.process_pending("u1")

    audit_lines = (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    assert (tmp_path / "system" / "diffs" / f"diff_{op.operation_id}.json").exists()


def test_recovery_worker_diff_written_only_commits_redo(tmp_path) -> None:
    _, _, committer, worker = _committer(tmp_path)
    op = _add_op("diff-written")
    committer.diff_writer.write(ContextDiff(user_id="u1", operations=[op], diff_id=f"diff_{op.operation_id}"))
    committer.redo.begin(op, phase="diff_written")

    result = worker.process_pending("u1")

    assert result["operation_ids"] == [op.operation_id]
    assert not (tmp_path / "system" / "redo" / f"{op.operation_id}.json").exists()


def test_recovery_worker_reward_and_penalize_do_not_apply_twice(tmp_path) -> None:
    source, index, committer, worker = _committer(tmp_path)
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    index.upsert_index(policy.to_context_object(), content="policy")

    reward = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.REWARD, target_uri=policy.uri, payload={"reward": 1.0}, operation_id="reward-recovery")
    committer.commit("u1", [reward])
    first = source.read_object(policy.uri).metadata
    committer.redo.begin(reward, phase="source_written")
    worker.process_pending("u1")
    assert source.read_object(policy.uri).metadata["success_count"] == first["success_count"]

    penalty = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.PENALIZE, target_uri=policy.uri, payload={"penalty": 1.0}, operation_id="penalty-recovery")
    committer.commit("u1", [penalty])
    second = source.read_object(policy.uri).metadata
    committer.redo.begin(penalty, phase="source_written")
    worker.process_pending("u1")
    assert source.read_object(policy.uri).metadata["failure_count"] == second["failure_count"]
