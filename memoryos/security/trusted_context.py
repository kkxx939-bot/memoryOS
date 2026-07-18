"""Small trusted caller boundary shared by HTTP and MCP transports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from memoryos.core.types import scope_key_from_payload

READ_CONTEXT = "context.read"
COMMIT_SESSION = "session.commit"
ATTEST_USER_INPUT = "session.attest_user_input"
AUTHORITATIVE_REMEMBER = "memory.authoritative.remember"
AUTHORITATIVE_FORGET = "memory.authoritative.forget"
HARD_ERASE_MEMORY = "memory.hard_erase"

DEFAULT_AGENT_CAPABILITIES = frozenset({READ_CONTEXT, COMMIT_SESSION})
PRINCIPAL_ONLY_WORKSPACE = "__memoryos_principal_only__"
KNOWN_CAPABILITIES = frozenset(
    {
        READ_CONTEXT,
        COMMIT_SESSION,
        ATTEST_USER_INPUT,
        AUTHORITATIVE_REMEMBER,
        AUTHORITATIVE_FORGET,
        HARD_ERASE_MEMORY,
    }
)


class AuthenticationError(PermissionError):
    """The transport could not authenticate a configured caller."""


@dataclass(frozen=True)
class TrustedRequestContext:
    """Identity and grants created by trusted process or bearer configuration."""

    tenant_id: str
    user_id: str
    actor_kind: str = "agent"
    actor_id: str = "generic_agent"
    capabilities: frozenset[str] = field(default_factory=lambda: DEFAULT_AGENT_CAPABILITIES)
    allowed_workspace_ids: frozenset[str] = field(default_factory=frozenset)
    authorized_scope_keys: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in ("tenant_id", "user_id", "actor_kind", "actor_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"trusted caller requires non-empty {name}")
        if self.actor_kind not in {"user", "agent", "service"}:
            raise ValueError("trusted caller actor_kind must be user, agent, or service")
        unknown = set(self.capabilities) - set(KNOWN_CAPABILITIES)
        if unknown:
            raise ValueError(f"unknown trusted caller capabilities: {','.join(sorted(unknown))}")
        for workspace_id in self.allowed_workspace_ids:
            if (
                not isinstance(workspace_id, str)
                or not workspace_id.strip()
                or workspace_id == PRINCIPAL_ONLY_WORKSPACE
            ):
                raise ValueError("trusted caller workspace IDs must be non-empty and non-reserved")
        normalized_scope_keys: set[str] = set()
        for scope_key in self.authorized_scope_keys:
            if not isinstance(scope_key, str) or not scope_key.strip() or "\x00" in scope_key:
                raise ValueError("trusted caller scope keys must be non-empty strings without NUL")
            normalized = scope_key.strip()
            normalized_scope_keys.add(normalized)
            principal_prefix = "memoryos:principal:"
            workspace_prefix = "memoryos:workspace:"
            if normalized.startswith(principal_prefix) and normalized != f"{principal_prefix}{self.user_id}":
                raise ValueError("trusted caller cannot authorize another principal scope")
            if normalized.startswith(workspace_prefix):
                workspace_id = normalized.removeprefix(workspace_prefix)
                if workspace_id not in self.allowed_workspace_ids:
                    raise ValueError("trusted caller workspace scope must match an allowed workspace")
        object.__setattr__(self, "authorized_scope_keys", frozenset(normalized_scope_keys))

    def require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise PermissionError(f"caller lacks capability: {capability}")

    def assert_identity(self, *, user_id: Any = None, tenant_id: Any = None) -> None:
        _assert_optional_identity("user_id", user_id, self.user_id)
        _assert_optional_identity("tenant_id", tenant_id, self.tenant_id)

    def assert_workspace(self, workspace_id: Any) -> str:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise PermissionError("caller request requires a concrete workspace")
        resolved = workspace_id.strip()
        if resolved not in self.allowed_workspace_ids:
            raise PermissionError("caller workspace is not authorized")
        return resolved

    def bind_read_workspace(self, workspace_id: Any = None) -> str:
        if workspace_id is not None and str(workspace_id).strip():
            if str(workspace_id).strip() == PRINCIPAL_ONLY_WORKSPACE and not self.allowed_workspace_ids:
                return PRINCIPAL_ONLY_WORKSPACE
            return self.assert_workspace(workspace_id)
        if len(self.allowed_workspace_ids) == 1:
            return next(iter(self.allowed_workspace_ids))
        if len(self.allowed_workspace_ids) > 1:
            raise PermissionError("caller must select one authorized workspace")
        return PRINCIPAL_ONLY_WORKSPACE

    def bind_write_workspace(self, workspace_id: Any = None) -> str:
        resolved = self.bind_read_workspace(workspace_id)
        if resolved == PRINCIPAL_ONLY_WORKSPACE:
            raise PermissionError("caller has no authorized workspace for this write")
        return resolved

    def retrieval_scope_keys(self, *, workspace_id: str | None = None) -> frozenset[str]:
        """Return the complete, trusted allow-list for one retrieval request.

        The principal and the selected workspace are derived from authenticated
        transport identity. Additional team/environment/asset grants may only
        come from trusted process configuration via ``authorized_scope_keys``.
        """

        selected_workspace = str(workspace_id or "").strip()
        if selected_workspace and selected_workspace != PRINCIPAL_ONLY_WORKSPACE:
            self.assert_workspace(selected_workspace)
        keys = {f"memoryos:principal:{self.user_id}", *self.authorized_scope_keys}
        if selected_workspace and selected_workspace != PRINCIPAL_ONLY_WORKSPACE:
            keys.add(f"memoryos:workspace:{selected_workspace}")
        return frozenset(keys)

    def assert_applicability_scope_keys(self, scope_keys: Any, *, workspace_id: str | None = None) -> None:
        """Reject user-declared applicability keys outside trusted grants."""

        if scope_keys is None:
            return
        if not isinstance(scope_keys, Sequence) or isinstance(scope_keys, str | bytes):
            raise PermissionError("applicability scope keys must be an array")
        requested: set[str] = set()
        for raw_key in scope_keys:
            if not isinstance(raw_key, str) or not raw_key.strip() or "\x00" in raw_key:
                raise PermissionError("applicability scope keys must contain non-empty strings")
            requested.add(raw_key.strip())
        unauthorized = requested - set(self.retrieval_scope_keys(workspace_id=workspace_id))
        if unauthorized:
            raise PermissionError("applicability scope keys exceed trusted caller grants")

    def assert_applicability_scopes(self, scopes: Any, *, workspace_id: str | None = None) -> None:
        if scopes is None:
            return
        if not isinstance(scopes, Sequence) or isinstance(scopes, str | bytes):
            raise PermissionError("applicability scopes must be an array")
        scope_keys: list[str] = []
        for item in scopes:
            if not isinstance(item, Mapping):
                raise PermissionError("applicability scopes must contain objects")
            try:
                scope_keys.append(scope_key_from_payload(item))
            except (TypeError, ValueError) as exc:
                raise PermissionError("applicability scopes contain an invalid scope") from exc
        self.assert_applicability_scope_keys(scope_keys, workspace_id=workspace_id)

    def bind_agent_connect_metadata(self, payload: Any) -> dict[str, Any] | None:
        """Bind adapter identity and non-action capabilities for an agent request."""

        if payload is not None and not isinstance(payload, Mapping):
            raise PermissionError("connect metadata must be an object")
        raw = dict(payload or {})
        claimed_adapter = raw.get("adapter_id")
        if claimed_adapter is not None and claimed_adapter != self.actor_id:
            raise PermissionError("caller adapter_id does not match trusted actor")
        return {
            "connect_type": "agent",
            "adapter_id": self.actor_id,
            "agent_instance_id": str(raw.get("agent_instance_id") or ""),
            "run_mode": "context_reduction",
            "world_domain": "digital",
            "source_kind": "coding_agent",
            "modality": list(raw.get("modality") or ["text"])
            if not isinstance(raw.get("modality"), str)
            else [str(raw["modality"])],
            "capabilities": {
                "can_write_memory": True,
                "can_search_context": True,
                "can_reduce_context": True,
                "can_predict_behavior": False,
                "can_generate_action": False,
                "can_execute_action": False,
                "can_use_external_tools": False,
            },
            "extra": dict(raw.get("extra", {}) or {}) if isinstance(raw.get("extra"), Mapping) else {},
        }


def capabilities_from_csv(raw: str | None, *, default: frozenset[str] = DEFAULT_AGENT_CAPABILITIES) -> frozenset[str]:
    if raw is None or not raw.strip():
        return default
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def workspace_ids_from_csv(raw: str | None) -> frozenset[str]:
    return frozenset(item.strip() for item in str(raw or "").split(",") if item.strip())


def scope_keys_from_csv(raw: str | None) -> frozenset[str]:
    """Parse trusted, deployment-configured non-principal scope grants."""

    return frozenset(item.strip() for item in str(raw or "").split(",") if item.strip())


def workspace_ids_from_metadata(metadata: Any) -> frozenset[str]:
    """Extract the persisted workspace boundary without guessing from prose."""

    if not isinstance(metadata, Mapping):
        raise ValueError("object metadata must be a mapping")
    raw_scope = metadata.get("scope", {})
    raw_fields = metadata.get("fields", {})
    if raw_scope is not None and not isinstance(raw_scope, Mapping):
        raise ValueError("object scope must be a mapping")
    if raw_fields is not None and not isinstance(raw_fields, Mapping):
        raise ValueError("object fields must be a mapping")
    scope = dict(raw_scope or {})
    fields = dict(raw_fields or {})
    values = {
        str(value).strip()
        for value in (
            metadata.get("workspace_id"),
            metadata.get("project_id"),
            scope.get("workspace_id"),
            scope.get("project_id"),
            fields.get("workspace_id"),
            fields.get("project_id"),
        )
        if value is not None and str(value).strip() and str(value).strip() != PRINCIPAL_ONLY_WORKSPACE
    }
    applicability = scope.get("applicability", {})
    if applicability is not None and not isinstance(applicability, Mapping):
        raise ValueError("scope applicability must be a mapping")
    all_of = dict(applicability or {}).get("all_of", [])
    if not isinstance(all_of, Sequence) or isinstance(all_of, str | bytes):
        raise ValueError("scope applicability all_of must be an array")
    for item in all_of:
        if not isinstance(item, Mapping):
            raise ValueError("scope applicability entries must be mappings")
        kind = str(item.get("kind") or "").strip().casefold()
        if kind == "workspace":
            identifier = str(item.get("id") or "").strip()
            if not identifier:
                raise ValueError("workspace scope requires an id")
            values.add(identifier)
    raw_scope_keys = metadata.get("scope_keys", [])
    if not isinstance(raw_scope_keys, Sequence) or isinstance(raw_scope_keys, str | bytes):
        raise ValueError("scope_keys must be an array")
    for raw_key in raw_scope_keys:
        key = str(raw_key)
        parts = key.split(":", 2)
        if len(parts) == 3 and parts[1] == "workspace" and parts[2]:
            values.add(parts[2])
    if len(values) > 1:
        raise ValueError("object declares multiple workspace boundaries")
    return frozenset(values)


def sanitize_ingress_messages(
    messages: list[dict[str, Any]] | None,
    caller: TrustedRequestContext,
) -> list[dict[str, Any]]:
    """Bind archived message roles and actors before they become evidence."""

    sanitized: list[dict[str, Any]] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            raise ValueError("messages must contain objects")
        row = dict(raw)
        claimed_role = str(row.get("role") or "assistant").strip().casefold()
        user_attested = caller.actor_kind == "user" or ATTEST_USER_INPUT in caller.capabilities
        system_attested = caller.actor_kind == "service" and ATTEST_USER_INPUT in caller.capabilities
        if claimed_role == "user" and user_attested:
            role, actor_id, attested = "user", caller.user_id, True
        elif claimed_role == "system" and system_attested:
            role, actor_id, attested = "system", caller.actor_id, True
        elif claimed_role == "tool":
            role, actor_id, attested = "tool", caller.actor_id, True
        elif claimed_role == "assistant":
            role, actor_id, attested = "assistant", caller.actor_id, True
        else:
            role, actor_id, attested = "assistant", caller.actor_id, False

        metadata = dict(row.get("metadata", {}) or {}) if isinstance(row.get("metadata"), dict) else {}
        for key in (
            "actor_id",
            "actor_kind",
            "asserted_by",
            "authority",
            "effect_authority",
            "source_role",
            "structured_memory_command",
            "subjects",
        ):
            row.pop(key, None)
            metadata.pop(key, None)
        metadata.update(
            {
                "ingress_actor_kind": caller.actor_kind,
                "ingress_actor_id": caller.actor_id,
                "actor_attested": attested,
            }
        )
        if role != claimed_role:
            metadata["claimed_role"] = claimed_role
        row.update({"role": role, "actor_id": actor_id, "metadata": metadata})
        sanitized.append(row)
    return sanitized


def sanitize_ingress_tool_results(
    tool_results: list[dict[str, Any]] | None,
    caller: TrustedRequestContext,
) -> list[dict[str, Any]]:
    """Tool evidence is always attributed to the trusted tool transport actor."""

    sanitized: list[dict[str, Any]] = []
    for raw in tool_results or []:
        if not isinstance(raw, dict):
            raise ValueError("tool_results must contain objects")
        row = dict(raw)
        metadata = dict(row.get("metadata", {}) or {}) if isinstance(row.get("metadata"), dict) else {}
        for key in (
            "actor_id",
            "actor_kind",
            "asserted_by",
            "authority",
            "effect_authority",
            "source_role",
            "structured_memory_command",
            "subjects",
        ):
            row.pop(key, None)
            metadata.pop(key, None)
        metadata.update(
            {
                "ingress_actor_kind": caller.actor_kind,
                "ingress_actor_id": caller.actor_id,
                "actor_attested": True,
            }
        )
        row.update({"role": "tool", "actor_id": caller.actor_id, "metadata": metadata})
        sanitized.append(row)
    return sanitized


def sanitize_session_scope(
    scope: dict[str, Any] | None,
    caller: TrustedRequestContext,
    *,
    project_id: str,
    session_key: str,
) -> dict[str, Any]:
    raw = dict(scope or {})
    caller.assert_identity(user_id=raw.get("user_id"), tenant_id=raw.get("tenant_id"))
    for key in (
        "user_id",
        "tenant_id",
        "project_id",
        "session_key",
        "subjects",
        "authority",
        "origin",
        "actor_id",
        "actor_kind",
        "asserted_by",
        "effect_authority",
        "source_role",
    ):
        raw.pop(key, None)
    return {
        **raw,
        "user_id": caller.user_id,
        "tenant_id": caller.tenant_id,
        "project_id": project_id,
        "session_key": session_key,
    }


def sanitize_session_provenance(
    provenance: dict[str, Any] | None,
    caller: TrustedRequestContext,
    *,
    native_session_id: str,
) -> dict[str, Any]:
    raw = dict(provenance or {})
    for key in (
        "native_session_id",
        "user_id",
        "tenant_id",
        "actor_id",
        "actor_kind",
        "asserted_by",
        "authority",
        "effect_authority",
        "source_role",
    ):
        raw.pop(key, None)
    return {
        **raw,
        "native_session_id": native_session_id,
        "tenant_id": caller.tenant_id,
        "user_id": caller.user_id,
        "actor_kind": caller.actor_kind,
        "actor_id": caller.actor_id,
    }


def _assert_optional_identity(name: str, provided: Any, expected: str) -> None:
    if provided is None:
        return
    if not isinstance(provided, str) or provided != expected:
        raise PermissionError(f"caller {name} does not match trusted identity")
