"""记忆提交阶段的冲突异常。"""

from __future__ import annotations


class RevisionConflictError(RuntimeError):
    """提交计划使用的修订版本已经不再是当前版本。"""

    def __init__(self, message: str, *, committed_diff=None) -> None:  # noqa: ANN001
        self.committed_diff = committed_diff
        super().__init__(message)


__all__ = ["RevisionConflictError"]
