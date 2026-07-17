"""Compatibility exports for durable control-file quarantine."""

from memoryos.core.durable_io.quarantine import (
    QuarantineRecord,
    list_quarantine_records,
    quarantine_control_file,
)

__all__ = ["QuarantineRecord", "list_quarantine_records", "quarantine_control_file"]
