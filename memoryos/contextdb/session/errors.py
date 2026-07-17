"""Transport-neutral session archive failure contracts."""


class EvidenceArchiveError(ValueError):
    """Base class for observable evidence archive failures."""


class EvidenceArchiveConflictError(EvidenceArchiveError):
    """A content-addressed path already exists with different bytes."""


class EvidenceArchiveIntegrityError(EvidenceArchiveError):
    """Immutable evidence no longer matches its recorded digest."""


class AsyncOutputIntegrityError(EvidenceArchiveIntegrityError):
    """A published async-output generation is incomplete, mixed, or corrupt."""


__all__ = [
    "AsyncOutputIntegrityError",
    "EvidenceArchiveConflictError",
    "EvidenceArchiveError",
    "EvidenceArchiveIntegrityError",
]
