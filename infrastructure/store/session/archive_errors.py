"""与传输方式无关的 SessionArchive 异常。"""


class SessionArchiveError(ValueError):
    """可观测会话归档错误的基类。"""


class SessionArchiveConflictError(SessionArchiveError):
    """内容寻址路径已存在，但其中字节与本次写入不一致。"""


class SessionArchiveIntegrityError(SessionArchiveError):
    """会话归档内容与已经记录的摘要不一致。"""


class SessionAsyncOutputIntegrityError(SessionArchiveIntegrityError):
    """已发布的异步输出代次不完整、混代或损坏。"""


__all__ = [
    "SessionArchiveConflictError",
    "SessionArchiveError",
    "SessionArchiveIntegrityError",
    "SessionAsyncOutputIntegrityError",
]
