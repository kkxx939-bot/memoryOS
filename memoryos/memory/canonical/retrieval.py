"""Offline/admin validation retrieval for canonical memory.

This module intentionally contains the exhaustive canonical snapshot reader
used by repair, audit, and migration tooling.  Product recall is owned by the
bounded Unified Context orchestrator under ``memoryos.contextdb.retrieval``;
the offline reader must never be wired into SDK, HTTP, MCP, or context
assembly entrypoints.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.sqlite_index_store import lexical_match_count, lexical_terms
from memoryos.memory.canonical.event import canonical_digest, canonical_json
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
    ProjectionRecordStore,
)
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.state import (
    CanonicalMemoryInvariantError,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalSnapshot,
    capture_committed_canonical_snapshot,
    committed_content,
    committed_relations,
)


class CanonicalInvariantViolation(RuntimeError):
    """Canonical Slot/Claim state is internally inconsistent and must be repaired."""


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


class OfflineCanonicalMemoryRetriever:
    """Exhaustive canonical reader restricted to offline repair/admin work.

    ``search()`` deliberately captures a complete committed Slot/Claim
    snapshot so that audits can validate global invariants.  That O(N)
    behavior is forbidden on the online retrieval path.  The explicit class
    name and required ``offline_admin=True`` acknowledgement prevent this
    implementation from becoming an accidental second product retrieval
    chain.
    """

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None = None,
        hybrid_search: HybridSearch | None = None,
        projection_store: ProjectionRecordStore | None = None,
        *,
        offline_admin: bool,
    ) -> None:
        if offline_admin is not True:
            raise PermissionError("OfflineCanonicalMemoryRetriever is restricted to offline admin/audit use")
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.hybrid_search = hybrid_search
        root = getattr(source_store, "root", None)
        # FileSystemSourceStore.root is the shared storage root. Production
        # callers inject a tenant-bound ProjectionRecordStore explicitly; the
        # fallback must still match a directly constructed projector using the
        # same root rather than deriving a second, incompatible location.
        inferred_root = Path(root) if root is not None else None
        self.projection_store = projection_store or (
            ProjectionRecordStore(inferred_root) if inferred_root is not None else None
        )

    def search(self, query: CanonicalMemoryQuery) -> list[dict[str, Any]]:
        snapshot = capture_committed_canonical_snapshot(
            self.source_store,
            self.relation_store,
            kinds=("slot", "claim"),
        )
        intent = query.intent or self._intent_for_states(query.states) or self.classify_intent(query.text)
        allowed_states = set(query.states or self._states_for(intent))
        filters: dict[str, Any] = {
            "tenant_id": query.tenant_id,
            "context_type": ContextType.MEMORY.value,
            "claim_state": tuple(sorted(allowed_states)),
        }
        if query.memory_types:
            filters["memory_type"] = query.memory_types
        hits = self._recall(query, filters, intent, snapshot)
        results: list[dict[str, Any]] = []
        result_uris: set[str] = set()
        validated_slots: dict[str, Any] = {}
        for hit in hits:
            try:
                committed = snapshot.get(hit.uri)
                if committed is None:
                    continue
                obj = committed.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if not self._accepted_base(obj, query):
                continue
            slot = self._validated_slot(obj, query, validated_slots, snapshot)
            if slot is None:
                continue
            if intent == CanonicalQueryIntent.HISTORY:
                historical = self._history_payloads(obj, slot, hit.score, query, allowed_states, snapshot)
                results.extend(historical)
                result_uris.update(str(item["uri"]) for item in historical)
                continue
            state = str(dict(obj.metadata or {}).get("state", ""))
            if state not in allowed_states:
                continue
            if intent == CanonicalQueryIntent.CURRENT and not self._is_slot_current(obj, slot):
                continue
            projection = self._current_projection(obj, snapshot)
            if not self._hit_revision_is_current(hit, obj, projection):
                continue
            results.append(
                self._payload(
                    obj,
                    hit.score,
                    state,
                    projection=projection,
                    revision_payload=self._current_revision_payload(obj),
                    snapshot=snapshot,
                )
            )
            result_uris.add(obj.uri)
        if query.expand_relations and intent != CanonicalQueryIntent.HISTORY:
            results.extend(
                self._expand_relations(
                    results,
                    result_uris,
                    query,
                    allowed_states,
                    intent,
                    validated_slots,
                    snapshot,
                )
            )
        results.sort(key=lambda item: self._rank(item, intent), reverse=True)
        if intent == CanonicalQueryIntent.CURRENT:
            results = self._one_current_per_slot(results)
        return results[: max(0, query.limit)]

    def _recall(
        self,
        query: CanonicalMemoryQuery,
        filters: dict[str, Any],
        intent: CanonicalQueryIntent,
        snapshot: CommittedCanonicalSnapshot,
    ) -> list[Any]:
        exact_uris = list(query.claim_uris)
        for slot_uri in query.slot_uris:
            try:
                committed_slot = snapshot.get(slot_uri)
                if committed_slot is None:
                    continue
                slot = committed_slot.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            for claim_id in dict(slot.metadata or {}).get("claim_ids", []) or []:
                exact_uris.append(f"{slot_uri}/claims/{claim_id}")
        hits: list[Any] = []
        for uri in dict.fromkeys(exact_uris):
            hits.append(
                type(
                    "ExactHit",
                    (),
                    {"uri": uri, "score": 100.0, "source": "canonical_exact", "metadata": {}},
                )()
            )
        terms = lexical_terms(query.text)
        allowed_states = set(str(item) for item in filters.get("claim_state", []) or [])
        allowed_uris: set[str] = set()
        candidate_uris = tuple(
            uri
            for uri, committed in snapshot.records.items()
            if dict(committed.object.metadata or {}).get("canonical_kind") == "claim"
        )
        for uri in candidate_uris:
            try:
                committed = snapshot.get(uri)
                if committed is None:
                    continue
                obj = committed.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if not self._accepted_base(obj, query):
                continue
            metadata = dict(obj.metadata or {})
            if intent != CanonicalQueryIntent.HISTORY and str(metadata.get("state", "")) not in allowed_states:
                continue
            if intent == CanonicalQueryIntent.HISTORY and not metadata.get("revisions"):
                continue
            allowed_uris.add(obj.uri)
            searchable_metadata: dict[str, Any]
            if intent == CanonicalQueryIntent.HISTORY:
                searchable_metadata = metadata
            else:
                current = self._current_revision_payload(obj)
                current_values = dict(current.get("value_fields", {}) or {})
                searchable_metadata = {
                    "canonical_value": current_values.get("canonical_value", current_values.get("value", "")),
                    "identity_fields": metadata.get("identity_fields", {}),
                    "memory_type": metadata.get("memory_type", ""),
                    "value_fields": current.get("value_fields", {}),
                    "qualifiers": current.get("qualifiers", {}),
                }
            title = obj.title if intent == CanonicalQueryIntent.HISTORY else ""
            haystack = " ".join((title, str(searchable_metadata))).casefold()
            score = float(lexical_match_count(query.text, haystack)) if terms else 0.1
            if score > 0:
                hits.append(
                    type(
                        "SourceHit",
                        (),
                        {
                            "uri": obj.uri,
                            "score": score,
                            "source": "canonical_source",
                            "metadata": {"source_revision": int(metadata.get("revision", 0))},
                        },
                    )()
                )
        filters["allowed_uris"] = tuple(sorted(allowed_uris))
        recall_filters = dict(filters)
        if intent == CanonicalQueryIntent.HISTORY:
            recall_filters.pop("claim_state", None)
        recalled = (
            self.hybrid_search.search(
                query.text,
                filters=recall_filters,
                context_type=ContextType.MEMORY,
                limit=max(query.limit * 5, 20),
                source_snapshot={uri: snapshot.records[uri].object for uri in allowed_uris},
            )
            if self.hybrid_search is not None
            else self.index_store.search(
                query.text,
                filters=recall_filters,
                limit=max(query.limit * 5, 20),
            )
        )
        seen: set[str] = set()
        results = []
        for hit in [*hits, *recalled]:
            if hit.uri in seen:
                continue
            seen.add(hit.uri)
            results.append(hit)
        return results

    def _accepted_base(self, obj: Any, query: CanonicalMemoryQuery) -> bool:
        metadata = dict(obj.metadata or {})
        if obj.lifecycle_state != LifecycleState.ACTIVE:
            return False
        if metadata.get("canonical_kind") != "claim":
            return False
        if metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2:
            return False
        try:
            current = materialized_current_revision_payload(metadata)
        except CanonicalMemoryInvariantError as exc:
            raise CanonicalInvariantViolation(str(exc)) from exc
        if (
            metadata.get("state") != current.get("state")
            or metadata.get("epistemic_status") != current.get("epistemic_status")
            or metadata.get("semantic_relation") != current.get("relation")
        ):
            raise CanonicalInvariantViolation("canonical Claim current-state mirror is inconsistent")
        if str(obj.tenant_id or "default") != query.tenant_id:
            return False
        if query.memory_types and str(metadata.get("memory_type", "")) not in query.memory_types:
            return False
        if not self._visible(metadata, query):
            return False
        if not self._authority_permits(metadata, query):
            return False
        return self._applicable(metadata, query.applicability_scope_keys)

    def _validated_slot(
        self,
        claim_obj: Any,
        query: CanonicalMemoryQuery,
        cache: dict[str, Any],
        snapshot: CommittedCanonicalSnapshot,
    ) -> Any | None:
        metadata = dict(claim_obj.metadata or {})
        slot_uri = str(metadata.get("slot_uri") or claim_obj.uri.rsplit("/claims/", 1)[0])
        if slot_uri in cache:
            return cache[slot_uri]
        try:
            committed = snapshot.get(slot_uri)
            if committed is None:
                cache[slot_uri] = None
                return None
            slot = committed.object
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            cache[slot_uri] = None
            return None
        slot_metadata = dict(slot.metadata or {})
        if (
            slot_metadata.get("canonical_kind") != "slot"
            or slot_metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
            or str(slot.tenant_id or "default") != query.tenant_id
            or not self._visible(slot_metadata, query)
            or not self._authority_permits(slot_metadata, query)
            or not self._applicable(slot_metadata, query.applicability_scope_keys)
        ):
            cache[slot_uri] = None
            return None
        active_ids: list[str] = []
        missing_ids: list[str] = []
        for claim_id in slot_metadata.get("claim_ids", []) or []:
            uri = f"{slot_uri}/claims/{claim_id}"
            try:
                committed_candidate = snapshot.get(uri)
                if committed_candidate is None:
                    missing_ids.append(str(claim_id))
                    continue
                candidate = committed_candidate.object
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                missing_ids.append(str(claim_id))
                continue
            candidate_metadata = dict(candidate.metadata or {})
            if (
                str(candidate_metadata.get("slot_id", "")) != str(slot_metadata.get("slot_id", ""))
                or candidate_metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
                or str(candidate.tenant_id or "default") != query.tenant_id
            ):
                raise CanonicalInvariantViolation(f"slot {slot_uri} contains a Claim outside its canonical boundary")
            if (
                candidate.lifecycle_state == LifecycleState.ACTIVE
                and candidate_metadata.get("canonical_kind") == "claim"
                and candidate_metadata.get("state") == "ACTIVE"
            ):
                active_ids.append(str(candidate_metadata.get("claim_id") or claim_id))
        if missing_ids:
            raise CanonicalInvariantViolation(f"slot {slot_uri} is missing canonical Claims: {sorted(missing_ids)}")
        if len(active_ids) > 1:
            raise CanonicalInvariantViolation(f"slot {slot_uri} contains multiple ACTIVE claims: {sorted(active_ids)}")
        declared = str(slot_metadata.get("active_claim_id") or "")
        if active_ids and declared != active_ids[0]:
            raise CanonicalInvariantViolation(f"slot {slot_uri} active_claim_id does not match its ACTIVE claim")
        if declared and not active_ids:
            raise CanonicalInvariantViolation(f"slot {slot_uri} declares an ACTIVE claim that is not ACTIVE")
        cache[slot_uri] = slot
        return slot

    def _is_slot_current(self, claim_obj: Any, slot: Any) -> bool:
        claim_metadata = dict(claim_obj.metadata or {})
        slot_metadata = dict(slot.metadata or {})
        revision = self._current_revision_payload(claim_obj)
        return bool(
            claim_metadata.get("state") == "ACTIVE"
            and str(slot_metadata.get("active_claim_id") or "") == str(claim_metadata.get("claim_id") or "")
            and self._revision_is_effective(revision)
        )

    def _hit_revision_is_current(
        self,
        hit: Any,
        obj: Any,
        projection: ProjectionRecord | None,
    ) -> bool:
        source = str(getattr(hit, "source", ""))
        if source in {"canonical_source", "canonical_exact", "canonical_relation"}:
            return True
        current_revision = int(dict(obj.metadata or {}).get("revision", 0))
        hit_metadata = dict(getattr(hit, "metadata", {}) or {})
        advertised = hit_metadata.get("projection_source_revision")
        if advertised is None:
            advertised = hit_metadata.get("source_revision")
        if advertised is not None and int(advertised) != current_revision:
            return False
        if (
            projection is None
            or not projection.current
            or projection.source_revision != current_revision
            or projection.projection_revision != current_revision
        ):
            return False
        expected = {
            ("projection_attempt_id",): projection.projection_attempt_id,
            ("input_effect_hash", "projection_input_effect_hash"): projection.input_effect_hash,
            ("publish_token", "projection_publish_token"): projection.publish_token,
            ("projected_content_digest", "projection_content_digest"): projection.projected_content_digest,
            ("projected_relation_digest", "projection_relation_digest"): projection.projected_relation_digest,
        }
        return all(any(hit_metadata.get(key) == value for key in aliases) for aliases, value in expected.items())

    def _history_payloads(
        self,
        obj: Any,
        slot: Any,
        score: float,
        query: CanonicalMemoryQuery,
        allowed_states: set[str],
        snapshot: CommittedCanonicalSnapshot,
    ) -> list[dict[str, Any]]:
        metadata = dict(obj.metadata or {})
        current_revision = int(materialized_current_revision_payload(metadata)["revision"])
        current_claim = self._is_slot_current(obj, slot)
        results = []
        for item in metadata.get("revisions", []) or []:
            if not isinstance(item, dict):
                continue
            revision = int(item.get("revision", 0))
            state = str(item.get("state", ""))
            if query.states and state not in allowed_states:
                continue
            if current_claim and revision == current_revision and state == "ACTIVE":
                continue
            projection = self._projection_for_revision(obj.uri, revision)
            results.append(
                self._payload(
                    obj,
                    score,
                    state,
                    projection=projection,
                    revision_payload=dict(item),
                    historical=True,
                    snapshot=snapshot,
                )
            )
        return results

    def _expand_relations(
        self,
        primary: list[dict[str, Any]],
        seen: set[str],
        query: CanonicalMemoryQuery,
        allowed_states: set[str],
        intent: CanonicalQueryIntent,
        validated_slots: dict[str, Any],
        snapshot: CommittedCanonicalSnapshot,
    ) -> list[dict[str, Any]]:
        expanded = []
        for item in primary:
            for relation in item.get("relations", []) or []:
                source = str(relation.get("source_uri", ""))
                target = str(relation.get("target_uri", ""))
                related_uri = target if source == item["uri"] else source
                if not related_uri or related_uri in seen or "/claims/" not in related_uri:
                    continue
                relation_metadata = dict(relation.get("metadata", {}) or {})
                relation_revision = relation_metadata.get("source_revision")
                try:
                    committed_related = snapshot.get(related_uri)
                    if committed_related is None:
                        continue
                    obj = committed_related.object
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    continue
                if relation_revision is not None:
                    relation_source_revision = (
                        int(item.get("source_revision", 0))
                        if source == item["uri"]
                        else int(dict(obj.metadata or {}).get("revision", 0))
                    )
                    if int(relation_revision) != relation_source_revision:
                        continue
                if not self._accepted_base(obj, query):
                    continue
                slot = self._validated_slot(obj, query, validated_slots, snapshot)
                if slot is None:
                    continue
                state = str(dict(obj.metadata or {}).get("state", ""))
                if state not in allowed_states:
                    continue
                if intent == CanonicalQueryIntent.CURRENT and not self._is_slot_current(obj, slot):
                    continue
                seen.add(related_uri)
                payload = self._payload(
                    obj,
                    max(0.0, float(item.get("score", 0.0)) * 0.75),
                    state,
                    projection=self._current_projection(obj, snapshot),
                    revision_payload=self._current_revision_payload(obj),
                    snapshot=snapshot,
                )
                payload["retrieval_source"] = "canonical_relation_expansion"
                expanded.append(payload)
        return expanded

    def classify_intent(self, text: str) -> CanonicalQueryIntent:
        normalized = str(text).casefold()
        negated = (
            r"(?:no|not|without|do\s+not\s+(?:show|include)|don't\s+(?:show|include))\s+"
            r"(?:history|historical|conflicts?|contradictions?|options?|alternatives?|candidates?)"
            r"|(?:没有|无|不看|不要(?:显示|包含)?)(?:历史|冲突|矛盾|方案|候选|选项)"
        )
        if re.search(negated, normalized):
            return CanonicalQueryIntent.CURRENT
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

    def _intent_for_states(self, states: tuple[str, ...]) -> CanonicalQueryIntent | None:
        requested = {str(state).upper() for state in states}
        if requested & {"SUPERSEDED", "RETRACTED"}:
            return CanonicalQueryIntent.HISTORY
        if requested == {"CONFLICTED"}:
            return CanonicalQueryIntent.CONFLICTS
        if requested & {"PROPOSED", "CONFLICTED"} and "ACTIVE" not in requested:
            return CanonicalQueryIntent.OPTIONS
        return CanonicalQueryIntent.CURRENT if requested == {"ACTIVE"} else None

    def _states_for(self, intent: CanonicalQueryIntent) -> tuple[str, ...]:
        if intent == CanonicalQueryIntent.CURRENT:
            return ("ACTIVE",)
        if intent == CanonicalQueryIntent.OPTIONS:
            return ("PROPOSED", "CONFLICTED")
        if intent == CanonicalQueryIntent.CONFLICTS:
            return ("CONFLICTED",)
        return ("PROPOSED", "ACTIVE", "SUPERSEDED", "CONFLICTED", "RETRACTED")

    def _visible(self, metadata: dict[str, Any], query: CanonicalMemoryQuery) -> bool:
        scope = self._canonical_scope(metadata)
        return bool(
            scope is not None
            and scope.visibility.permits(
                tenant_id=query.tenant_id,
                principal_id=query.principal_id,
                service_id=query.service_id,
            )
        )

    def _authority_permits(self, metadata: dict[str, Any], query: CanonicalMemoryQuery) -> bool:
        scope = self._canonical_scope(metadata)
        if scope is None or scope.authority.inferred:
            return False
        principals = set(scope.authority.principal_ids)
        services = set(scope.authority.service_ids)
        if not principals and not services:
            return True
        provenance = dict(metadata.get("provenance", {}) or {})
        asserted_by = str(metadata.get("asserted_by") or provenance.get("asserted_by") or "")
        asserted_by_service = str(metadata.get("asserted_by_service") or provenance.get("asserted_by_service") or "")
        if not asserted_by and not asserted_by_service:
            return False
        return bool(asserted_by in principals or asserted_by_service in services)

    def _applicable(self, metadata: dict[str, Any], available_scope_keys: Sequence[str]) -> bool:
        scope = self._canonical_scope(metadata)
        if scope is None:
            return False
        required = tuple(item.key for item in scope.applicability.all_of)
        available = set(available_scope_keys)
        return all(scope_key in available for scope_key in required)

    def _canonical_scope(self, metadata: dict[str, Any]) -> MemoryScope | None:
        raw_scope = metadata.get("scope")
        if not isinstance(raw_scope, dict):
            return None
        try:
            scope = MemoryScope.from_dict(raw_scope)
        except (KeyError, TypeError, ValueError):
            return None
        return scope if scope.canonical_subject is not None else None

    def _payload(
        self,
        obj: Any,
        score: float,
        state: str,
        *,
        projection: ProjectionRecord | None,
        revision_payload: dict[str, Any],
        snapshot: CommittedCanonicalSnapshot,
        historical: bool = False,
    ) -> dict[str, Any]:
        canonical_metadata = dict(obj.metadata or {})
        revision = int(revision_payload.get("revision", canonical_metadata.get("revision", 0)) or 0)
        canonical_source_revision = int(canonical_metadata.get("revision", revision))
        values = dict(revision_payload.get("value_fields", {}) or {})
        value = str(
            values.get("canonical_value") or values.get("value") or canonical_metadata.get("canonical_value", obj.title)
        )
        layers = self._layer_texts(projection)
        text = str(layers.get("L2") or value)
        metadata = {
            **canonical_metadata,
            "state": state,
            "revision": revision if historical else canonical_source_revision,
            "current_revision": revision,
            "epistemic_status": revision_payload.get(
                "epistemic_status", canonical_metadata.get("epistemic_status", "")
            ),
            "semantic_relation": revision_payload.get("relation", canonical_metadata.get("semantic_relation", "")),
        }
        if projection is not None:
            metadata.update(
                {
                    "projection_pending": False,
                    "projection_revision": projection.projection_revision,
                    "projection_source_revision": projection.source_revision,
                    "projection_manifest_uri": projection.manifest_uri,
                    "projection_record_path": str(
                        self.projection_store.record_path(obj.uri, projection.source_revision)
                    )
                    if self.projection_store is not None
                    else "",
                }
            )
        committed = snapshot.get(obj.uri)
        if committed is None:
            raise CanonicalInvariantViolation(
                f"canonical snapshot lost Claim during payload materialization: {obj.uri}"
            )
        relations = [relation.to_dict() for relation in committed_relations(committed)]
        slot_uri = obj.uri.rsplit("/claims/", 1)[0]
        revision_uri = f"{obj.uri}#revision-{revision}"
        source_revision = (
            projection.source_revision
            if projection is not None
            else (revision if historical else canonical_source_revision)
        )
        try:
            normalized_score = float(score)
        except (TypeError, ValueError):
            normalized_score = 0.0
        if not math.isfinite(normalized_score):
            normalized_score = 0.0
        return {
            "uri": obj.uri,
            "tenant_id": str(obj.tenant_id or "default"),
            "owner_user_id": obj.owner_user_id,
            "revision_uri": revision_uri,
            "retrieval_identity": revision_uri if historical else obj.uri,
            "slot_uri": slot_uri,
            "slot_id": canonical_metadata.get("slot_id"),
            "claim_id": canonical_metadata.get("claim_id"),
            "revision": revision,
            "source_revision": source_revision,
            "projection_revision": projection.projection_revision if projection is not None else None,
            "score": normalized_score,
            "context_type": obj.context_type.value,
            "title": obj.title,
            "text": text,
            "layer": "canonical_projection" if projection is not None else "canonical_source",
            "metadata": metadata,
            "memory_state": state,
            "memory_category": "history"
            if historical
            else self._category(state, str(metadata.get("epistemic_status", ""))),
            "relations": relations,
            "layer_texts": layers,
            "layer_revisions": {
                name: projection.projection_revision if projection is not None else source_revision for name in layers
            },
            "projection_record": projection.to_dict() if projection is not None else None,
            "retrieval_source": "canonical_history" if historical else "canonical_structured_lexical",
        }

    def _layer_texts(self, projection: ProjectionRecord | None) -> dict[str, str]:
        if projection is None or not projection.usable:
            return {}
        values: dict[str, str] = {}
        for name, uri in (
            ("L0", projection.l0_uri),
            ("L1", projection.l1_uri),
            ("L2", projection.l2_uri),
        ):
            if not uri:
                continue
            try:
                values[name] = self.source_store.read_content(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return {}
        if set(values) != {"L0", "L1", "L2"} or projection.projected_content_digest != canonical_digest(values):
            return {}
        return values

    def _current_projection(
        self,
        obj: Any,
        snapshot: CommittedCanonicalSnapshot,
    ) -> ProjectionRecord | None:
        if self.projection_store is None:
            return None
        metadata = dict(obj.metadata or {})
        revision = int(metadata.get("revision", 0))
        current_claim_revision = int(materialized_current_revision_payload(metadata)["revision"])
        try:
            record = self.projection_store.load_current(obj.uri, source_revision=revision)
        except ProjectionIntegrityError:
            return None
        if (
            record is None
            or record.projection_revision != revision
            or record.current_claim_revision != current_claim_revision
            or not self._projection_refs_match(record)
        ):
            return None
        try:
            committed = snapshot.get(obj.uri)
            if committed is None:
                return None
            relations = sorted(
                (relation.to_dict() for relation in committed_relations(committed)),
                key=canonical_json,
            )
            expected_input = canonical_digest(
                {
                    "claim_uri": obj.uri,
                    "source_revision": revision,
                    "object": committed.object.to_dict(),
                    "content": committed_content(committed),
                    "relations": relations,
                }
            )
            layers = {
                "L0": self.source_store.read_content(record.l0_uri),
                "L1": self.source_store.read_content(record.l1_uri),
                "L2": self.source_store.read_content(record.l2_uri),
            }
            relation_payload = json.loads(self.source_store.read_content(record.relations_uri))
            manifest = json.loads(self.source_store.read_content(record.manifest_uri))
            index_metadata = self.index_store.get_index_metadata(obj.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError, json.JSONDecodeError):
            return None
        expected_identity = {
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
        }
        if (
            record.input_effect_hash != expected_input
            or record.projected_content_digest != canonical_digest(layers)
            or not isinstance(relation_payload, dict)
            or relation_payload.get("relations") != relations
            or record.projected_relation_digest != canonical_digest(relations)
            or index_metadata is None
            or index_metadata.get("index_content_digest")
            != canonical_digest("\n".join((layers["L0"], layers["L1"], layers["L2"])))
            or any(manifest.get(field) != value for field, value in expected_identity.items())
            or any(
                index_metadata.get(
                    {
                        "source_revision": "projection_source_revision",
                        "input_effect_hash": "projection_input_effect_hash",
                        "publish_token": "projection_publish_token",
                        "projected_content_digest": "projection_content_digest",
                        "projected_relation_digest": "projection_relation_digest",
                    }.get(field, field)
                )
                != value
                for field, value in expected_identity.items()
            )
        ):
            return None
        return record

    def _projection_for_revision(self, claim_uri: str, revision: int) -> ProjectionRecord | None:
        if self.projection_store is None:
            return None
        try:
            record = self.projection_store.load(claim_uri, revision)
        except ProjectionIntegrityError:
            return None
        if (
            record is None
            or not record.usable
            or record.projection_revision != revision
            or record.current_claim_revision != revision
            or not self._projection_refs_match(record)
        ):
            return None
        layers = self._layer_texts(record)
        if not layers:
            return None
        try:
            relation_payload = json.loads(self.source_store.read_content(record.relations_uri))
            manifest = json.loads(self.source_store.read_content(record.manifest_uri))
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(relation_payload, dict) or not isinstance(manifest, dict):
            return None
        relations = relation_payload.get("relations")
        expected_identity = {
            "claim_uri": record.claim_uri,
            "source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "input_effect_hash": record.input_effect_hash,
            "publish_token": record.publish_token,
            "projected_content_digest": record.projected_content_digest,
            "projected_relation_digest": record.projected_relation_digest,
        }
        if (
            not isinstance(relations, list)
            or record.projected_relation_digest != canonical_digest(relations)
            or any(relation_payload.get(field) != value for field, value in expected_identity.items())
            or any(manifest.get(field) != value for field, value in expected_identity.items())
        ):
            return None
        return record

    def _projection_refs_match(self, record: ProjectionRecord) -> bool:
        marker = f"/projections/rev-{record.source_revision}/"
        return all(
            marker in uri
            for uri in (
                record.l0_uri,
                record.l1_uri,
                record.l2_uri,
                record.relations_uri,
                record.manifest_uri,
            )
        )

    def _current_revision_payload(self, obj: Any) -> dict[str, Any]:
        metadata = dict(obj.metadata or {})
        return materialized_current_revision_payload(metadata)

    def _revision_is_effective(self, revision: dict[str, Any]) -> bool:
        valid_from = revision.get("valid_from")
        valid_to = revision.get("valid_to")
        if not valid_from:
            return False
        try:
            start = datetime.fromisoformat(str(valid_from).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(valid_to).replace("Z", "+00:00")) if valid_to else None
        except ValueError:
            return False
        if start.tzinfo is None or (end is not None and end.tzinfo is None):
            return False
        now = datetime.now(timezone.utc)
        return start.astimezone(timezone.utc) <= now and (end is None or now < end.astimezone(timezone.utc))

    def _one_current_per_slot(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = []
        seen: set[str] = set()
        for item in results:
            slot_uri = str(item.get("slot_uri", ""))
            if slot_uri in seen:
                continue
            seen.add(slot_uri)
            selected.append(item)
        return selected

    def _category(self, state: str, epistemic: str) -> str:
        if state == "CONFLICTED":
            return "conflict"
        if state == "PROPOSED":
            return "candidate"
        if state in {"SUPERSEDED", "RETRACTED"}:
            return "history"
        if epistemic in {"INFERRED", "HYPOTHESIZED"}:
            return "inference"
        return "current"

    def _rank(self, item: dict[str, Any], intent: CanonicalQueryIntent) -> float:
        state = str(item.get("memory_state", ""))
        state_bonus = {
            CanonicalQueryIntent.CURRENT: {"ACTIVE": 5.0},
            CanonicalQueryIntent.OPTIONS: {"PROPOSED": 3.0},
            CanonicalQueryIntent.HISTORY: {
                "SUPERSEDED": 5.0,
                "RETRACTED": 4.0,
                "ACTIVE": 2.0,
                "PROPOSED": 1.0,
            },
            CanonicalQueryIntent.CONFLICTS: {"CONFLICTED": 5.0},
        }[intent].get(state, 0.0)
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        return score + state_bonus


__all__ = [
    "CanonicalInvariantViolation",
    "CanonicalMemoryQuery",
    "CanonicalQueryIntent",
    "OfflineCanonicalMemoryRetriever",
]
