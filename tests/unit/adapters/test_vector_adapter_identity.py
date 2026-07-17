from __future__ import annotations

import pytest

from memoryos.adapters.filesystem.fs_lock_store import FileSystemLockStore
from memoryos.adapters.locks.errors import LockBackendUnavailableError
from memoryos.adapters.vector import ChromaStore, InMemoryVectorStore, MilvusStore, QdrantStore
from memoryos.adapters.vector.errors import VectorBackendUnavailableError
from memoryos.contextdb.store.local_stores import InMemoryLockStore


@pytest.mark.parametrize("adapter", [QdrantStore, MilvusStore, ChromaStore])
def test_unimplemented_named_backends_are_not_in_memory_aliases_and_fail_fast(adapter: type) -> None:
    assert adapter is not InMemoryVectorStore
    assert not issubclass(adapter, InMemoryVectorStore)
    with pytest.raises(VectorBackendUnavailableError, match="not implemented"):
        adapter()


def test_filesystem_lock_name_is_not_an_in_memory_alias_and_fails_fast() -> None:
    assert FileSystemLockStore is not InMemoryLockStore
    assert not issubclass(FileSystemLockStore, InMemoryLockStore)
    with pytest.raises(LockBackendUnavailableError, match="not implemented"):
        FileSystemLockStore()
