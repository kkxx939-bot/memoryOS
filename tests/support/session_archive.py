"""Explicit test composition for Memory-backed session evidence archives."""

from __future__ import annotations

from memoryos.action_policy.integration.commit_registration import (
    build_action_policy_commit_handlers,
)
from memoryos.adapters.persistence.filesystem.session_archive import SessionArchiveStore
from memoryos.contextdb.session.evidence_encoder import register_session_evidence_encoder
from memoryos.memory.integration.archive_reader import (
    register_session_evidence_archive_reader_factory,
)
from memoryos.memory.integration.commit_registration import build_memory_commit_handlers
from memoryos.memory.integration.session_evidence import CanonicalSessionEvidenceEncoder
from memoryos.operations.commit.domain_registry import (
    register_action_policy_commit_handlers,
    register_memory_commit_handlers,
)


def compose_domain_runtime_bindings() -> None:
    """Reproduce runtime's domain bindings for explicitly composed tests."""

    register_session_evidence_encoder(CanonicalSessionEvidenceEncoder())
    register_session_evidence_archive_reader_factory(SessionArchiveStore)
    register_memory_commit_handlers(
        build_memory_commit_handlers(session_evidence_reader_factory=SessionArchiveStore)
    )
    register_action_policy_commit_handlers(build_action_policy_commit_handlers())


def compose_memory_session_archive() -> None:
    """Compatibility name for the complete test composition boundary."""

    compose_domain_runtime_bindings()


__all__ = ["compose_domain_runtime_bindings", "compose_memory_session_archive"]
