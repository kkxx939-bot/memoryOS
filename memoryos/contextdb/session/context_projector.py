"""Compatibility exports for the application-owned Session projector."""

from memoryos.application.session.context_projector import (
    SessionContextProjector,
    SessionProjectionResult,
    workspace_id_from_session_metadata,
)

__all__ = [
    "SessionContextProjector",
    "SessionProjectionResult",
    "workspace_id_from_session_metadata",
]
