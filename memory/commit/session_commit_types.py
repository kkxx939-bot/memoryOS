"""Session 提交消费者的共享失败类型。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class DerivedConsumerError(RuntimeError):
    """一个或多个独立派生消费者没有完成。"""

    def __init__(self, failures: Sequence[tuple[str, bool]]) -> None:
        self.failures = tuple(failures)
        self.retryable = bool(self.failures) and all(item[1] for item in self.failures)
        names = ",".join(item[0] for item in self.failures)
        super().__init__(f"Session derived consumers failed: {names}")


class ConsumerLeaseBusy(RuntimeError):
    retryable = True


class ConsumerTerminalError(RuntimeError):
    retryable = False


class _SessionCommitState:
    """拆分后的 Session 事务阶段共享同一个显式状态契约。"""

    archive_store: Any
    queue_store: Any
    committer: Any
    memory_planner: Any
    behavior_planner: Any
    action_policy_planner: Any
    context_planner: Any
    session_projector: Any
    commit_group_store: Any
    memory_committer: Any
    document_planner: Any
    projection_journal: Any
    _startup_recovery_group: Any

    def async_commit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _require_runtime_ready(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _project_session_archive(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _enqueue_session_commit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _record_projection(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _bind_archive_tenant(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _validate_persisted_memory_effects(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _startup_recovery_scope(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _tenant_id(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _is_sha256(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _validate_document_plan(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _effect_from_document_result(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _actor_binding(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _evidence_reference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _stabilize_operations(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _is_retryable(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


__all__ = [
    "ConsumerLeaseBusy",
    "ConsumerTerminalError",
    "DerivedConsumerError",
]
