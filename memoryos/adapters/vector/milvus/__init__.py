"""Milvus adapter selection boundary."""

from memoryos.adapters.vector.errors import VectorBackendUnavailableError


class MilvusStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise VectorBackendUnavailableError(
            "MilvusStore is not implemented in this distribution; configure an explicit VectorStore adapter"
        )


__all__ = ["MilvusStore"]
