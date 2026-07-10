from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.canonical.scope import MemoryScope, ScopeRef


def canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"[\s_-]+", "-", text).strip("-")


class AliasRegistry:
    def __init__(self, aliases: Mapping[str, Mapping[str, str]] | None = None) -> None:
        self._aliases: dict[str, dict[str, str]] = {}
        for namespace, values in dict(aliases or {}).items():
            self._aliases[str(namespace)] = {
                canonical_text(alias): str(identifier) for alias, identifier in values.items()
            }

    def resolve(self, namespace: str, value: Any) -> str:
        normalized = canonical_text(value)
        return self._aliases.get(namespace, {}).get(normalized, normalized)

    def canonical_scope(self, scope: ScopeRef) -> ScopeRef:
        identifier = self.resolve(f"scope:{scope.kind}", scope.id)
        return ScopeRef(scope.namespace, scope.kind, identifier, scope.parent_id, scope.attributes)


@dataclass(frozen=True)
class ResolvedMemoryIdentity:
    slot_id: str
    slot_uri: str
    claim_id: str
    claim_uri: str
    slot_identity: Mapping[str, Any]
    canonical_value: str
    scope_keys: tuple[str, ...]


class StableMemoryIdentityResolver:
    """Schema-field identity only; content, title, time, and embeddings are excluded."""

    SLOT_FIELDS = {
        "profile": ("attribute_key",),
        "preference": ("subject", "dimension"),
        "entity": ("entity_type", "canonical_entity_id"),
        "project_rule": ("rule_topic",),
        "project_decision": ("decision_topic",),
        "event": ("event_key",),
        "agent_experience": ("task_pattern", "environment_signature"),
    }

    def __init__(self, aliases: AliasRegistry | None = None) -> None:
        self.aliases = aliases or AliasRegistry()

    def resolve(
        self,
        proposal: MemorySemanticProposal,
        memory_scope: MemoryScope,
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> ResolvedMemoryIdentity:
        expected = self.SLOT_FIELDS.get(proposal.memory_type)
        if expected is None:
            raise ValueError(f"no identity schema for memory type: {proposal.memory_type}")
        missing = [field for field in expected if not proposal.identity_fields.get(field)]
        if missing:
            raise ValueError(f"missing stable identity fields: {','.join(missing)}")
        slot_identity = {
            field: self.aliases.resolve(f"{proposal.memory_type}:{field}", proposal.identity_fields[field])
            for field in expected
        }
        scopes = tuple(sorted(self.aliases.canonical_scope(scope).key for scope in memory_scope.applicability.all_of))
        slot_id = stable_hash([tenant_id, proposal.memory_type, scopes, slot_identity], length=32)
        canonical_value = self._canonical_value(proposal)
        claim_id = stable_hash([slot_id, canonical_value], length=32)
        root = f"memoryos://user/{owner_user_id}/memories/canonical/slots/{slot_id}"
        return ResolvedMemoryIdentity(
            slot_id=slot_id,
            slot_uri=root,
            claim_id=claim_id,
            claim_uri=f"{root}/claims/{claim_id}",
            slot_identity=slot_identity,
            canonical_value=canonical_value,
            scope_keys=scopes,
        )

    def _canonical_value(self, proposal: MemorySemanticProposal) -> str:
        for key in (
            "canonical_value",
            "value",
            "decision",
            "rule",
            "preference",
            "name",
            "event",
            "outcome",
        ):
            value = proposal.value_fields.get(key)
            if value not in {None, ""}:
                return self.aliases.resolve(f"{proposal.memory_type}:value", value)
        if not proposal.value_fields:
            raise ValueError("claim requires at least one value field")
        return stable_hash(dict(proposal.value_fields), length=32)
