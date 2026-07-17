"""Explicit vector adapter selection failures."""


class VectorBackendUnavailableError(RuntimeError):
    """A named backend has no installed production implementation."""


__all__ = ["VectorBackendUnavailableError"]
