"""供运行时组合根选择的锁存储实现。"""

from infrastructure.store.locks.process_local import ProcessLocalLockStore

__all__ = ["ProcessLocalLockStore"]
