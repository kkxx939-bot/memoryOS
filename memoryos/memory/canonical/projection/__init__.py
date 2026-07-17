"""Revision-bound derived projections for canonical memory."""

from __future__ import annotations

from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
    ProjectionRecordStore,
    ProjectionStatus,
    ProjectionStepStatus,
)

from .models import ProjectionOutboxIntegrityError, ProjectionResult
from .service import CanonicalMemoryProjector
from .worker import MemoryProjectionWorker

# Preserve the original public import/pickle identity after the file-to-package split.
CanonicalMemoryProjector.__module__ = __name__
MemoryProjectionWorker.__module__ = __name__
ProjectionOutboxIntegrityError.__module__ = __name__
ProjectionResult.__module__ = __name__

__all__ = [
    "CanonicalMemoryProjector",
    "MemoryProjectionWorker",
    "ProjectionIntegrityError",
    "ProjectionOutboxIntegrityError",
    "ProjectionRecord",
    "ProjectionRecordStore",
    "ProjectionResult",
    "ProjectionStatus",
    "ProjectionStepStatus",
]
