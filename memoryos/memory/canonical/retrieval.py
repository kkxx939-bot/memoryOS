from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore
from memoryos.memory.canonical.visibility import read_committed_canonical, relation_is_committed


class CanonicalQueryIntent(str, Enum):
    CURRENT = "CURRENT"
    OPTIONS = "OPTIONS"
    HISTORY = "HISTORY"
    CONFLICTS = "CONFLICTS"


@dataclass(frozen=True)
class CanonicalMemoryQuery:
    text: str
    tenant_id: str
    principal_id: str | None = None
    service_id: str | None = None
    applicability_scope_keys: tuple[str, ...] = ()
    memory_types: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    intent: CanonicalQueryIntent | None = None
    claim_uris: tuple[str, ...] = ()
    slot_uris: tuple[str, ...] = ()
    expand_relations: bool = True
    limit: int = 10


class CanonicalMemoryRetriever:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None = None,
        hybrid_search: HybridSearch | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.hybrid_search = hybrid_search

    def search(self, query: CanonicalMemoryQuery) -> list[dict[str, Any]]:
        intent = query.intent or self.classify_intent(query.text)
        allowed_states = set(query.states or self._states_for(intent))
        filters: dict[str, Any] = {
            "tenant_id": query.tenant_id,
            "context_type": ContextType.MEMORY.value,
        }
        if query.principal_id:
            filters["owner_user_id"] = query.principal_id
        hits = self._recall(query, filters)
        results = []
        result_uris: set[str] = set()
        for hit in hits:
            try:
                committed = read_committed_canonical(self.source_store, hit.uri)
                obj = committed.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            state = self._accepted_state(obj, query, allowed_states)
            if state is None:
                continue
            results.append(self._payload(obj, hit.score, state, content_override=committed.content_override))
            result_uris.add(obj.uri)
        if query.expand_relations:
            results.extend(self._expand_relations(results, result_uris, query, allowed_states))
        results.sort(key=lambda item: self._rank(item, intent), reverse=True)
        return results[: max(0, query.limit)]

    def _recall(self, query: CanonicalMemoryQuery, filters: dict[str, Any]) -> list[Any]:
        exact_uris = list(query.claim_uris)
        for slot_uri in query.slot_uris:
            try:
                slot = read_committed_canonical(self.source_store, slot_uri).object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            for claim_id in dict(slot.metadata or {}).get("claim_ids", []) or []:
                exact_uris.append(f"{slot_uri}/claims/{claim_id}")
        hits: list[Any] = []
        for uri in dict.fromkeys(exact_uris):
            hits.append(type("ExactHit", (), {"uri": uri, "score": 100.0})())
        recalled = (
            self.hybrid_search.search(
                query.text,
                filters=filters,
                context_type=ContextType.MEMORY,
                limit=max(query.limit * 5, 20),
            )
            if self.hybrid_search is not None
            else self.index_store.search(query.text, filters=filters, limit=max(query.limit * 5, 20))
        )
        seen = set()
        results = []
        for hit in [*hits, *recalled]:
            if hit.uri in seen:
                continue
            seen.add(hit.uri)
            results.append(hit)
        return results

    def _accepted_state(
        self,
        obj: Any,
        query: CanonicalMemoryQuery,
        allowed_states: set[str],
    ) -> str | None:
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "claim":
            return None
        if str(obj.tenant_id or "default") != query.tenant_id:
            return None
        if query.memory_types and str(metadata.get("memory_type", "")) not in query.memory_types:
            return None
        state = str(metadata.get("state", ""))
        if state not in allowed_states or not self._visible(metadata, query):
            return None
        if not self._applicable(metadata, query.applicability_scope_keys):
            return None
        return state

    def _expand_relations(
        self,
        primary: list[dict[str, Any]],
        seen: set[str],
        query: CanonicalMemoryQuery,
        allowed_states: set[str],
    ) -> list[dict[str, Any]]:
        if self.relation_store is None:
            return []
        expanded = []
        for item in primary:
            for relation in item.get("relations", []) or []:
                source = str(relation.get("source_uri", ""))
                target = str(relation.get("target_uri", ""))
                related_uri = target if source == item["uri"] else source
                if not related_uri or related_uri in seen or "/claims/" not in related_uri:
                    continue
                try:
                    committed = read_committed_canonical(self.source_store, related_uri)
                    obj = committed.object
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    continue
                state = self._accepted_state(obj, query, allowed_states)
                if state is None:
                    continue
                seen.add(related_uri)
                payload = self._payload(
                    obj,
                    max(0.0, float(item.get("score", 0.0)) * 0.75),
                    state,
                    content_override=committed.content_override,
                )
                payload["retrieval_source"] = "canonical_relation_expansion"
                expanded.append(payload)
        return expanded

    def classify_intent(self, text: str) -> CanonicalQueryIntent:
        normalized = str(text).casefold()
        if any(token in normalized for token in ("history", "historical", "previous", "历史", "曾经", "之前")):
            return CanonicalQueryIntent.HISTORY
        if any(token in normalized for token in ("conflict", "contradiction", "冲突", "矛盾")):
            return CanonicalQueryIntent.CONFLICTS
        if any(
            token in normalized
            for token in ("option", "alternative", "consider", "candidate", "方案", "候选", "考虑", "评估")
        ):
            return CanonicalQueryIntent.OPTIONS
        return CanonicalQueryIntent.CURRENT

    def _states_for(self, intent: CanonicalQueryIntent) -> tuple[str, ...]:
        if intent == CanonicalQueryIntent.CURRENT:
            return ("ACTIVE", "VALIDATED", "OBSERVED")
        if intent == CanonicalQueryIntent.OPTIONS:
            return ("ACTIVE", "VALIDATED", "OBSERVED", "PROPOSED", "PENDING", "CONFLICTED")
        if intent == CanonicalQueryIntent.CONFLICTS:
            return ("CONFLICTED",)
        return (
            "ACTIVE",
            "VALIDATED",
            "OBSERVED",
            "PROPOSED",
            "PENDING",
            "SUPERSEDED",
            "RETRACTED",
            "STALE",
            "ARCHIVED",
            "CONFLICTED",
        )

    def _visible(self, metadata: dict[str, Any], query: CanonicalMemoryQuery) -> bool:
        scope = dict(metadata.get("scope", {}) or {})
        visibility = dict(scope.get("visibility", {}) or {})
        if str(visibility.get("tenant_id", "default")) != query.tenant_id:
            return False
        principals = {str(item) for item in visibility.get("allowed_principal_ids", []) or []}
        services = {str(item) for item in visibility.get("allowed_service_ids", []) or []}
        private = bool(visibility.get("private", False))
        if not private and not principals and not services:
            return True
        return bool(
            (query.principal_id and query.principal_id in principals)
            or (query.service_id and query.service_id in services)
        )

    def _applicable(self, metadata: dict[str, Any], available_scope_keys: Sequence[str]) -> bool:
        scope = dict(metadata.get("scope", {}) or {})
        applicability = dict(scope.get("applicability", {}) or {})
        required = {
            f"{item.get('namespace', 'memoryos')}:{item.get('kind')}:{item.get('id')}"
            for item in applicability.get("all_of", []) or []
            if isinstance(item, dict) and item.get("kind") and item.get("id")
        }
        if not required:
            return False
        return required.issubset(set(available_scope_keys))

    def _payload(
        self,
        obj: Any,
        score: float,
        state: str,
        *,
        content_override: str | None = None,
    ) -> dict[str, Any]:
        metadata = dict(obj.metadata or {})
        if content_override is not None:
            text = content_override
        else:
            try:
                text = self.source_store.read_content(obj.layers.l2_uri or obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                text = str(metadata.get("canonical_value", obj.title))
        relations = []
        if self.relation_store is not None:
            relations = [
                relation.to_dict()
                for relation in self.relation_store.relations_of(
                    obj.uri,
                    tenant_id=obj.tenant_id,
                    owner_user_id=obj.owner_user_id,
                )
                if relation_is_committed(self.source_store, relation)
            ]
        return {
            "uri": obj.uri,
            "score": float(score),
            "context_type": obj.context_type.value,
            "title": obj.title,
            "text": text,
            "layer": "canonical_index",
            "metadata": metadata,
            "memory_state": state,
            "memory_category": self._category(state, str(metadata.get("epistemic_status", ""))),
            "relations": relations,
            "layer_texts": self._layer_texts(obj, content_override=content_override),
            "retrieval_source": "canonical_structured_lexical",
        }

    def _layer_texts(self, obj: Any, *, content_override: str | None = None) -> dict[str, str]:
        values = {}
        for name, uri in (
            ("L0", obj.layers.l0_uri),
            ("L1", obj.layers.l1_uri),
            ("L2", obj.layers.l2_uri or obj.uri),
        ):
            if not uri:
                continue
            if name == "L2" and content_override is not None and not obj.layers.l2_uri:
                values[name] = content_override
                continue
            try:
                values[name] = self.source_store.read_content(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
        return values

    def _category(self, state: str, epistemic: str) -> str:
        if state == "CONFLICTED":
            return "conflict"
        if state in {"PROPOSED", "PENDING"}:
            return "candidate"
        if state in {"SUPERSEDED", "RETRACTED", "STALE", "ARCHIVED"}:
            return "history"
        if epistemic in {"INFERRED", "HYPOTHESIZED"}:
            return "inference"
        return "current"

    def _rank(self, item: dict[str, Any], intent: CanonicalQueryIntent) -> float:
        state = str(item.get("memory_state", ""))
        state_bonus = {
            CanonicalQueryIntent.CURRENT: {"ACTIVE": 5.0, "VALIDATED": 4.0, "OBSERVED": 3.0},
            CanonicalQueryIntent.OPTIONS: {"ACTIVE": 4.0, "PROPOSED": 3.0, "PENDING": 2.0},
            CanonicalQueryIntent.HISTORY: {"SUPERSEDED": 5.0, "RETRACTED": 4.0, "STALE": 3.0},
            CanonicalQueryIntent.CONFLICTS: {"CONFLICTED": 5.0},
        }[intent].get(state, 0.0)
        return float(item.get("score", 0.0)) + state_bonus
