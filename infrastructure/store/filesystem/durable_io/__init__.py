"""崩溃安全且独立于领域的本地持久化原语。"""

from infrastructure.store.filesystem.durable_io.atomic_file import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    read_regular_bytes,
)
from infrastructure.store.filesystem.durable_io.atomic_json import atomic_create_json, atomic_write_json

__all__ = [
    "ImmutableArtifactConflictError",
    "atomic_create_bytes",
    "atomic_replace_bytes",
    "atomic_create_json",
    "atomic_write_json",
    "read_regular_bytes",
]
