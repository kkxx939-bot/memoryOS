"""与传输方式无关的 Session 证据归档异常。"""


class EvidenceArchiveError(ValueError):
    """可观测证据归档错误的基类。"""


class EvidenceArchiveConflictError(EvidenceArchiveError):
    """内容寻址路径已存在，但其中字节与本次写入不一致。"""


class EvidenceArchiveIntegrityError(EvidenceArchiveError):
    """不可变证据与已经记录的摘要不一致。"""


class AsyncOutputIntegrityError(EvidenceArchiveIntegrityError):
    """已发布的异步输出代次不完整、混代或损坏。"""


__all__ = [
    "AsyncOutputIntegrityError",
    "EvidenceArchiveConflictError",
    "EvidenceArchiveError",
    "EvidenceArchiveIntegrityError",
]
