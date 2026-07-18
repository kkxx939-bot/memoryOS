from __future__ import annotations

import json

from memoryos.action_policy.integration.commit_registration import (
    register_default_action_policy_commit_handlers,
)
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


def _committer(tmp_path):  # noqa: ANN001, ANN202
    register_default_action_policy_commit_handlers()
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    return source, index, committer, RecoveryWorker(RecoveryService(committer.redo, committer))


def _add_op(operation_id: str) -> ContextOperation:
    obj = ContextObject(
        uri=f"memoryos://user/u1/behavior_cases/{operation_id}",
        context_type=ContextType.BEHAVIOR_CASE,
        title=operation_id,
        owner_user_id="u1",
    )
    return ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "alpha"},
        operation_id=operation_id,
    )


def test_worker_recovers_started_by_recommitting(tmp_path) -> None:  # noqa: ANN001
    source, index, committer, worker = _committer(tmp_path)
    operation = _add_op("started")
    committer.redo.begin(operation, phase="started")

    result = worker.process_pending("u1")

    assert result["operation_ids"] == [operation.operation_id]
    assert source.read_object(str(operation.target_uri)).title == "started"
    assert index.search("alpha", tenant_id="default", filters={"owner_user_id": "u1"})


def test_worker_resumes_each_post_source_phase_once(tmp_path) -> None:  # noqa: ANN001
    for phase in ("source_written", "index_written", "audit_written", "diff_written"):
        root = tmp_path / phase
        _source, index, committer, worker = _committer(root)
        operation = _add_op(phase)
        committer._apply_source(operation)
        source_effect = committer._capture_regular_source_effect(operation)
        if phase in {"index_written", "audit_written", "diff_written"}:
            committer._apply_index(operation)
        if phase in {"audit_written", "diff_written"}:
            committer.audit.record("u1", "context_operation_committed", operation.to_dict())
        if phase == "diff_written":
            committer.diff_writer.write(
                ContextDiff(user_id="u1", operations=[operation], diff_id=f"diff_{operation.operation_id}")
            )
        committer.redo.begin(operation, phase=phase, source_effect=source_effect)

        result = worker.process_pending("u1")

        assert result["operation_ids"] == [operation.operation_id]
        assert index.search("alpha", tenant_id="default", filters={"owner_user_id": "u1"})
        assert not committer.redo.pending_entries()
        assert (root / "system" / "operations" / f"{operation.operation_id}.json").exists()


def test_reward_recovery_does_not_apply_twice(tmp_path) -> None:  # noqa: ANN001
    source, index, committer, worker = _committer(tmp_path)
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_ac",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
    )
    source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    index.upsert_index(policy.to_context_object(), content="policy", tenant_id="default")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.ACTION_POLICY,
        action=OperationAction.REWARD,
        target_uri=policy.uri,
        payload={"reward": 1.0},
        operation_id="reward-recovery",
    )
    committer.commit("u1", [operation])
    count = source.read_object(policy.uri).metadata["success_count"]
    committer.redo.begin(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation),
    )

    worker.process_pending("u1")

    assert source.read_object(policy.uri).metadata["success_count"] == count
