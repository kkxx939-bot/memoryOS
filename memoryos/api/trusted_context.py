"""Compatibility exports for trusted transport identity.

The transport-neutral implementation is owned by :mod:`memoryos.security.trusted_context`.
"""

from memoryos.security.trusted_context import (
    ATTEST_USER_INPUT,
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    COMMIT_SESSION,
    DEFAULT_AGENT_CAPABILITIES,
    KNOWN_CAPABILITIES,
    PRINCIPAL_ONLY_WORKSPACE,
    READ_CONTEXT,
    AuthenticationError,
    TrustedRequestContext,
    capabilities_from_csv,
    sanitize_ingress_messages,
    sanitize_ingress_tool_results,
    sanitize_session_provenance,
    sanitize_session_scope,
    scope_keys_from_csv,
    workspace_ids_from_csv,
    workspace_ids_from_metadata,
)

__all__ = [
    "ATTEST_USER_INPUT",
    "AUTHORITATIVE_FORGET",
    "AUTHORITATIVE_REMEMBER",
    "COMMIT_SESSION",
    "DEFAULT_AGENT_CAPABILITIES",
    "KNOWN_CAPABILITIES",
    "PRINCIPAL_ONLY_WORKSPACE",
    "READ_CONTEXT",
    "AuthenticationError",
    "TrustedRequestContext",
    "capabilities_from_csv",
    "sanitize_ingress_messages",
    "sanitize_ingress_tool_results",
    "sanitize_session_provenance",
    "sanitize_session_scope",
    "scope_keys_from_csv",
    "workspace_ids_from_csv",
    "workspace_ids_from_metadata",
]
