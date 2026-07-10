from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.visibility import read_committed_canonical, relation_is_committed


@dataclass(frozen=True)
class PrefetchedMemory:
    uri: str
    memory_type: str
    state: str
    revision: int
    scope: dict[str, Any]
    l0: str
    l1: str
    l2: str = ""
    relations: tuple[dict[str, Any], ...] = ()


class ExistingMemoryPrefetcher:
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
        if self.source_store is None or self.index_store is None:
            return ()
        started = monotonic()
        query = " ".join(event.text() for event in episode.events)[:1000]
        filters = {
            "tenant_id": episode.tenant_id,
            "owner_user_id": owner_user_id,
            "context_type": ContextType.MEMORY.value,
        }
        try:
            hits = self._recall(query, filters, episode)
        except Exception:
            return ()
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
            try:
                committed = read_committed_canonical(self.source_store, hit.uri)
                obj = committed.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if str(obj.tenant_id or "default") != episode.tenant_id:
                continue
            metadata = dict(obj.metadata or {})
            if type_hints and str(metadata.get("memory_type", "")) not in type_hints:
                continue
            if not self._visible(metadata, episode.tenant_id, owner_user_id):
                continue
            if not self._applicable(metadata, legal_scope_keys):
                continue
            if metadata.get("canonical_kind") == "claim" and metadata.get("state") not in {
                "ACTIVE",
                "PROPOSED",
                "PENDING",
                "CONFLICTED",
                "VALIDATED",
                "OBSERVED",
            }:
                continue
            l0 = self._read(obj.layers.l0_uri) or obj.title
            l1 = self._read(obj.layers.l1_uri) or l0
            required = self._tokens(l0) + self._tokens(l1)
            if required > remaining:
                continue
            remaining -= required
            l2 = ""
            full = (
                committed.content_override
                if committed.content_override is not None and not obj.layers.l2_uri
                else self._read(obj.layers.l2_uri or obj.uri)
            )
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
                    scope=dict(metadata.get("scope", {}) or {}),
                    l0=l0,
                    l1=l1,
                    l2=l2,
                    relations=tuple(
                        relation.to_dict()
                        for relation in (
                            self.relation_store.relations_of(
                                obj.uri,
                                tenant_id=episode.tenant_id,
                                owner_user_id=owner_user_id,
                            )
                            if self.relation_store is not None
                            else []
                        )
                        if relation_is_committed(self.source_store, relation)
                    ),
                )
            )
        return tuple(results)

    def _recall(self, query: str, filters: dict[str, Any], episode: EvidenceEpisode) -> list[Any]:
        exact_uris: list[str] = []
        # Adapters may provide already-resolved canonical identities. The model never controls these fields.
        for event in episode.events:
            exact_uris.extend(str(uri) for uri in event.metadata.get("canonical_memory_uris", []) or [])
        exact_hits = []
        for uri in dict.fromkeys(exact_uris):
            try:
                obj = read_committed_canonical(self.source_store, uri).object if self.source_store is not None else None
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
        scope = dict(metadata.get("scope", {}) or {})
        applicability = dict(scope.get("applicability", {}) or {})
        required = {
            f"{item.get('namespace', 'memoryos')}:{item.get('kind')}:{item.get('id')}"
            for item in applicability.get("all_of", []) or []
            if isinstance(item, dict) and item.get("kind") and item.get("id")
        }
        return not required or required.issubset(legal_scope_keys)

    def _visible(self, metadata: dict[str, Any], tenant_id: str, principal_id: str) -> bool:
        scope = dict(metadata.get("scope", {}) or {})
        visibility = dict(scope.get("visibility", {}) or {})
        if visibility and str(visibility.get("tenant_id", "default")) != tenant_id:
            return False
        allowed = {str(item) for item in visibility.get("allowed_principal_ids", []) or []}
        return not allowed or principal_id in allowed

    def _read(self, uri: str | None) -> str:
        if not uri or self.source_store is None:
            return ""
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""

    def _tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0
