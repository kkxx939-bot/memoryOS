"""Explicit test composition for immutable Session evidence archives."""

from __future__ import annotations

from memoryos.action_policy.integration.commit_registration import (
    build_action_policy_commit_handlers,
)
from memoryos.contextdb.session.evidence_encoder import register_session_evidence_encoder
from memoryos.memory.evidence import SessionEvidenceArchiveEncoder
from memoryos.operations.commit.domain_registry import (
    register_action_policy_commit_handlers,
)


def compose_domain_runtime_bindings() -> None:
    """Reproduce runtime's domain bindings for explicitly composed tests."""

    register_session_evidence_encoder(SessionEvidenceArchiveEncoder())
    register_action_policy_commit_handlers(build_action_policy_commit_handlers())


__all__ = ["compose_domain_runtime_bindings"]
