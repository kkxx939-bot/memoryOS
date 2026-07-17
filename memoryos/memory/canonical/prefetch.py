"""记忆系统里的预取。"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import materialized_current_revision_payload
from memoryos.memory.canonical.visibility import (
    committed_content,
    committed_relations,
    read_committed_canonical,
)


@dataclass(frozen=True)
class PrefetchedMemory:
    """负责 PrefetchedMemory 这部分逻辑。"""

    uri: str
    memory_type: str
    state: str
    revision: int
    slot_id: str
    claim_id: str
    canonical_value: str
    identity_fields: dict[str, Any]
    scope: dict[str, Any]
    l0: str
    l1: str
    l2: str = ""
    relations: tuple[dict[str, Any], ...] = ()


class ExistingMemoryPrefetcher:
    """提取前先取回相关 Slot、Claim 和当前 Revision。"""

    def __init__(
        self,
        source_store: SourceStore | None,
        index_store: IndexStore | None,
        relation_store: RelationStore | None = None,
        hybrid_search: HybridSearch | None = None,
        *,
        top_k: int = 20,
        token_budget: int = 4000,
        timeout_ms: int = 250,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.hybrid_search = hybrid_search
        self.top_k = min(20, max(1, int(top_k)))
        self.token_budget = max(100, int(token_budget))
        self.timeout_ms = max(10, int(timeout_ms))

    def prefetch(self, episode: EvidenceEpisode, *, owner_user_id: str) -> tuple[PrefetchedMemory, ...]:
        """处理 prefetch 这一步。"""

        if self.source_store is None or self.index_store is None:
            return ()
        started = monotonic()
        query = " ".join(event.text() for event in episode.events)[:1000]
        filters = {
            "tenant_id": episode.tenant_id,
            "owner_user_id": owner_user_id,
            "context_type": ContextType.MEMORY.value,
        }
        hits = self._recall(query, filters, episode)
        remaining = self.token_budget
        results = []
        legal_scope_keys = {scope.key for scope in episode.legal_scope_candidates()}
        type_hints = {
            str(item)
            for event in episode.events
            for item in (event.metadata.get("memory_types", []) or [])
        }
        for hit in hits:
            if (monotonic() - started) * 1000 > self.timeout_ms:
                break
            source_uri = self._canonical_source_uri(hit)
            if not source_uri:
                continue
            try:
                committed = read_committed_canonical(self.source_store, source_uri, self.relation_store)
                obj = committed.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if source_uri != str(hit.uri) and not self._current_slot_binding_is_valid(hit, obj):
                continue
            if str(obj.tenant_id or "default") != episode.tenant_id:
                continue
            metadata = dict(obj.metadata or {})
            if metadata.get("canonical_kind") == "slot":
                continue
            if type_hints and str(metadata.get("memory_type", "")) not in type_hints:
                continue
            if not self._visible(metadata, episode.tenant_id, owner_user_id):
                continue
            if not self._authority_permits(metadata):
                continue
            if not self._applicable(metadata, legal_scope_keys):
                continue
            if metadata.get("canonical_kind") == "claim" and metadata.get("state") not in {
                "ACTIVE",
                "PROPOSED",
                "CONFLICTED",
            }:
                continue
            current_revision = materialized_current_revision_payload(metadata)
            l0 = obj.title
            l1 = str(
                {
                    "identity_fields": metadata.get("identity_fields", {}),
                    "value_fields": current_revision.get("value_fields", {}),
                    "qualifiers": current_revision.get("qualifiers", {}),
                }
            )
            required = self._tokens(l0) + self._tokens(l1)
            if required > remaining:
                continue
            remaining -= required
            l2 = ""
            full = committed_content(committed)
            full_tokens = self._tokens(full)
            if full and full_tokens <= remaining:
                l2 = full
                remaining -= full_tokens
            results.append(
                PrefetchedMemory(
                    uri=obj.uri,
                    memory_type=str(metadata.get("memory_type", "")),
                    state=str(metadata.get("state", metadata.get("lifecycle_state", "ACTIVE"))),
                    revision=int(metadata.get("revision", 0)),
                    slot_id=str(metadata.get("slot_id", "")),
                    claim_id=str(metadata.get("claim_id", "")),
                    canonical_value=str(metadata.get("canonical_value", "")),
                    identity_fields=dict(metadata.get("identity_fields", {}) or {}),
                    scope=dict(metadata.get("scope", {}) or {}),
                    l0=l0,
                    l1=l1,
                    l2=l2,
                    relations=tuple(
                        relation.to_dict()
                        for relation in committed_relations(committed)
                    ),
                )
            )
        return tuple(results)

    @staticmethod
    def _canonical_source_uri(hit: Any) -> str:
        """Resolve a rebuildable Current Slot row to its authoritative Claim.

        ``current_slot`` is the CURRENT serving identity, so its public URI is
        deliberately not a SourceStore object URI.  Extraction prefetch still
        needs the exact Claim in order to bind an LLM's opaque candidate ref
        to a canonical Claim/Slot identity.  The Catalog may only supply that
        bounded pointer; the Claim and Slot are verified from canonical Source
        below before the candidate is exposed to the model.
        """

        metadata = dict(getattr(hit, "metadata", {}) or {})
        record_kind = str(metadata.get("record_kind") or "")
        canonical_kind = str(metadata.get("canonical_kind") or "")
        if record_kind != "current_slot" and canonical_kind != "current_slot_projection":
            return str(getattr(hit, "uri", "") or "")
        if str(metadata.get("canonical_state") or "ACTIVE").upper() != "ACTIVE":
            return ""
        return str(metadata.get("active_claim_uri") or metadata.get("canonical_claim_uri") or "")

    def _current_slot_binding_is_valid(self, hit: Any, claim_obj: Any) -> bool:
        """Fail closed when a Current Slot pointer lags or crosses identity."""

        if self.source_store is None:
            return False
        metadata = dict(getattr(hit, "metadata", {}) or {})
        claim_metadata = dict(getattr(claim_obj, "metadata", {}) or {})
        claim_id = str(claim_metadata.get("claim_id") or "")
        slot_id = str(claim_metadata.get("slot_id") or "")
        claim_uri = str(getattr(claim_obj, "uri", "") or "")
        advertised_claim_ids = {
            str(value)
            for value in (metadata.get("active_claim_id"), metadata.get("canonical_claim_id"))
            if value
        }
        advertised_claim_uris = {
            str(value)
            for value in (metadata.get("active_claim_uri"), metadata.get("canonical_claim_uri"))
            if value
        }
        if (
            not claim_id
            or not slot_id
            or not claim_uri
            or claim_metadata.get("canonical_kind") != "claim"
            or claim_metadata.get("state") != "ACTIVE"
            or advertised_claim_ids != {claim_id}
            or advertised_claim_uris != {claim_uri}
        ):
            return False
        advertised_claim_revisions = tuple(
            value
            for value in (metadata.get("active_claim_revision"), metadata.get("canonical_revision"))
            if value is not None
        )
        try:
            claim_revision = int(claim_metadata.get("revision", 0))
            if not advertised_claim_revisions or any(
                int(revision) != claim_revision for revision in advertised_claim_revisions
            ):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        slot_uri = str(
            metadata.get("canonical_slot_uri")
            or metadata.get("slot_uri")
            or claim_metadata.get("slot_uri")
            or str(getattr(claim_obj, "uri", "")).rsplit("/claims/", 1)[0]
        )
        if not slot_uri:
            return False
        try:
            slot = read_committed_canonical(self.source_store, slot_uri, self.relation_store).object
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False
        slot_metadata = dict(slot.metadata or {})
        advertised_slot_revisions = tuple(
            value
            for value in (metadata.get("projection_source_revision"), metadata.get("slot_revision"))
            if value is not None
        )
        try:
            slot_revision = int(slot_metadata.get("revision", 0))
            if not advertised_slot_revisions or any(
                int(revision) != slot_revision for revision in advertised_slot_revisions
            ):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(
            slot_metadata.get("canonical_kind") == "slot"
            and str(slot.tenant_id or "default") == str(claim_obj.tenant_id or "default")
            and str(slot_metadata.get("slot_id") or "") == slot_id
            and str(slot_metadata.get("active_claim_id") or "") == claim_id
            and str(metadata.get("canonical_slot_id") or metadata.get("slot_id") or slot_id) == slot_id
        )

    def _recall(self, query: str, filters: dict[str, Any], episode: EvidenceEpisode) -> list[Any]:
        exact_uris: list[str] = []
        # 适配器可以带入已经解析好的规范身份，但模型不能控制这些字段。
        for event in episode.events:
            exact_uris.extend(str(uri) for uri in event.metadata.get("canonical_memory_uris", []) or [])
        exact_hits = []
        for uri in dict.fromkeys(exact_uris):
            try:
                obj = (
                    read_committed_canonical(self.source_store, uri, self.relation_store).object
                    if self.source_store is not None
                    else None
                )
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                obj = None
            if obj is not None:
                exact_hits.append(type("ExactHit", (), {"uri": uri, "score": 100.0})())
        recalled: list[Any]
        if self.hybrid_search is not None:
            recalled = list(self.hybrid_search.search(
                query,
                filters=filters,
                context_type=ContextType.MEMORY,
                limit=self.top_k,
            ))
        else:
            assert self.index_store is not None
            recalled = list(self.index_store.search(query, filters=filters, limit=self.top_k))
        seen: set[str] = set()
        results = []
        for hit in [*exact_hits, *recalled]:
            if hit.uri in seen:
                continue
            seen.add(hit.uri)
            results.append(hit)
        return results[: self.top_k]

    def _applicable(self, metadata: dict[str, Any], legal_scope_keys: set[str]) -> bool:
        scope = self._canonical_scope(metadata)
        if scope is None:
            return False
        required = {item.key for item in scope.applicability.all_of}
        return required.issubset(legal_scope_keys)

    def _visible(self, metadata: dict[str, Any], tenant_id: str, principal_id: str) -> bool:
        scope = self._canonical_scope(metadata)
        return bool(
            scope is not None
            and scope.visibility.permits(tenant_id=tenant_id, principal_id=principal_id)
        )

    def _authority_permits(self, metadata: dict[str, Any]) -> bool:
        scope = self._canonical_scope(metadata)
        if scope is None or scope.authority.inferred:
            return False
        if not scope.authority.principal_ids and not scope.authority.service_ids:
            return True
        return bool(
            str(metadata.get("asserted_by") or "") in set(scope.authority.principal_ids)
            or str(metadata.get("asserted_by_service") or "") in set(scope.authority.service_ids)
        )

    def _canonical_scope(self, metadata: dict[str, Any]) -> MemoryScope | None:
        raw_scope = metadata.get("scope")
        if not isinstance(raw_scope, dict):
            return None
        try:
            scope = MemoryScope.from_dict(raw_scope)
        except (KeyError, TypeError, ValueError):
            return None
        return scope if scope.canonical_subject is not None else None

    def _tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0
