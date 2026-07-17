"""Lock adapter selection failures."""


class LockBackendUnavailableError(RuntimeError):
    """A named lock backend has no production implementation."""


__all__ = ["LockBackendUnavailableError"]
