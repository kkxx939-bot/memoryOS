"""行为模块里的相似行为检索。"""

from __future__ import annotations

from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.store.source_store import IndexHit, IndexStore, RelationStore, SourceStore


class SimilarBehaviorRetriever:
    relation_types = {
        "anchored_by",
        "supported_by",
        "aggregated_from",
        "updates_policy",
        "constrained_by",
        "requires_resource",
        "requires_skill",
    }

    def __init__(
        self,
        index_store: IndexStore,
        source_store: SourceStore | None = None,
        relation_store: RelationStore | None = None,
        hybrid_search: HybridSearch | None = None,
    ) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.relation_store = relation_store
        self.hybrid_search = hybrid_search

    def retrieve(self, user_id: str, observation: Observation, limit: int = 8) -> dict:
        query = " ".join([observation.raw_text, observation.location, observation.activity, *observation.signals])
        search_query = query or observation.scene_key
        namespace = f"memoryos://user/{user_id}/"
        trace: dict[str, dict] = {}
        patterns = self._fallback_hits(user_id, search_query, ContextType.BEHAVIOR_PATTERN, limit, trace, weight=1.0, namespace=namespace)
        clusters = self._fallback_hits(user_id, search_query, ContextType.BEHAVIOR_CLUSTER, limit, trace, weight=0.75, namespace=namespace)
        memory_anchors: list[dict] = []
        policy_refs: list[dict] = []
        relation_cases: list[dict] = []
        relation_uris = [item["uri"] for item in [*patterns, *clusters]]
        for uri in relation_uris:
            related = self._relation_results(uri, user_id=user_id, trace=trace)
            memory_anchors.extend(related["memory_anchors"])
            policy_refs.extend(related["policy_refs"])
            relation_cases.extend(related["cases"])
        if not memory_anchors:
            memory_anchors = self._fallback_hits(user_id, observation.scene_key or search_query, ContextType.MEMORY, limit, trace, weight=0.65, namespace=namespace)
        case_hits = relation_cases + self._fallback_hits(user_id, search_query, ContextType.BEHAVIOR_CASE, limit, trace, weight=0.45, namespace=namespace)
        representative_cases = self._representative_cases(case_hits)
        indexed_policy_refs = self._fallback_hits(user_id, observation.scene_key or search_query, ContextType.ACTION_POLICY, limit, trace, weight=0.55, namespace=namespace)
        policy_refs = self._dedupe([*policy_refs, *indexed_policy_refs])
        patterns = self._dedupe(patterns)
        clusters = self._dedupe(clusters)
        memory_anchors = self._dedupe(memory_anchors)
        hits = [*patterns, *clusters, *representative_cases, *memory_anchors]
        similarity_scores: dict[str, float] = {}
        for item in patterns:
            similarity_scores[item["uri"]] = max(similarity_scores.get(item["uri"], 0.0), float(item.get("score", 0.0)) * 1.0)
        for item in clusters:
            similarity_scores[item["uri"]] = max(similarity_scores.get(item["uri"], 0.0), float(item.get("score", 0.0)) * 0.75)
        for item in representative_cases:
            similarity_scores[item["uri"]] = max(similarity_scores.get(item["uri"], 0.0), float(item.get("score", 0.0)) * 0.45)
        for item in policy_refs:
            similarity_scores[item["uri"]] = max(similarity_scores.get(item["uri"], 0.0), float(item.get("score", 0.5)))
        return {
            "query": query,
            "scene_key": observation.scene_key,
            "patterns": patterns[:limit],
            "clusters": clusters[:limit],
            "representative_cases": representative_cases[:3],
            "memory_anchors": memory_anchors[:limit],
            "policy_refs": policy_refs[:limit],
            "hits": hits[:limit],
            "similarity_scores": {uri: min(1.0, score) for uri, score in similarity_scores.items()},
            "retrieval_trace": trace,
        }

    def _fallback_hits(
        self,
        user_id: str,
        query: str,
        context_type: ContextType,
        limit: int,
        trace: dict[str, dict],
        weight: float,
        namespace: str,
    ) -> list[dict]:
        if self.hybrid_search is not None:
            hybrid_hits = self.hybrid_search.search(
                query,
                filters={"owner_user_id": user_id},
                namespace=namespace,
                context_type=context_type,
                limit=limit,
            )
            items = [
                {
                    "uri": hit.uri,
                    "title": hit.title,
                    "context_type": hit.context_type,
                    "score": min(1.0, float(hit.score) * weight),
                    "source": hit.source,
                    "metadata": dict(hit.metadata),
                }
                for hit in hybrid_hits
            ]
        else:
            index_hits = self.index_store.search(
                query,
                filters={"owner_user_id": user_id, "context_type": context_type.value},
                limit=limit,
            )
            items = [self._hit_item(hit, source="index_fallback", weight=weight) for hit in index_hits]
        for item in items:
            trace.setdefault(str(item["uri"]), {"source": item.get("source", "index_fallback"), "context_type": context_type.value})
        return items

    def _relation_results(self, uri: str, user_id: str, trace: dict[str, dict]) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {"memory_anchors": [], "policy_refs": [], "cases": []}
        if self.relation_store is None:
            return result
        for relation in self.relation_store.relations_of(uri, owner_user_id=user_id):
            if relation.relation_type not in self.relation_types:
                continue
            target = relation.target_uri if relation.source_uri == uri else relation.source_uri
            item = self._object_item(target, relation_type=relation.relation_type)
            if item is None:
                item = {
                    "uri": target,
                    "title": relation.metadata.get("summary", target.rsplit("/", 1)[-1]),
                    "context_type": "",
                    "score": float(relation.weight),
                    "source": "relation",
                    "relation_type": relation.relation_type,
                }
            trace[item["uri"]] = {"source": "relation", "relation_type": relation.relation_type, "via": uri}
            if item.get("context_type") == ContextType.ACTION_POLICY.value or relation.relation_type == "updates_policy":
                result["policy_refs"].append(item)
            elif item.get("context_type") == ContextType.BEHAVIOR_CASE.value or relation.relation_type == "aggregated_from":
                result["cases"].append(item)
            elif item.get("context_type") == ContextType.MEMORY.value or relation.relation_type in {"anchored_by", "constrained_by"}:
                result["memory_anchors"].append(item)
        return result

    def _hit_item(self, hit: IndexHit, source: str, weight: float) -> dict:
        return {
            "uri": hit.uri,
            "title": hit.title,
            "context_type": hit.context_type,
            "score": min(1.0, float(hit.score) * weight),
            "source": source,
            "metadata": dict(hit.metadata),
        }

    def _object_item(self, uri: str, relation_type: str) -> dict | None:
        if self.source_store is None:
            return None
        try:
            obj = self.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
        if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.OBSOLETE, LifecycleState.ARCHIVED}:
            return None
        return {
            "uri": obj.uri,
            "title": obj.title,
            "context_type": obj.context_type.value,
            "score": max(0.1, obj.hotness, obj.semantic_hotness, obj.behavior_support_hotness),
            "source": "relation",
            "relation_type": relation_type,
            "metadata": dict(obj.metadata),
        }

    def _representative_cases(self, cases: list[dict]) -> list[dict]:
        deduped = self._dedupe(cases)
        positives = [item for item in deduped if float(item.get("metadata", {}).get("reward", 0.0) or 0.0) > 0]
        negatives = [item for item in deduped if float(item.get("metadata", {}).get("reward", 0.0) or 0.0) < 0]
        selected = []
        for group in (positives, negatives):
            if group:
                selected.append(sorted(group, key=lambda item: str(item.get("metadata", {}).get("created_at", "")), reverse=True)[0])
        if deduped:
            selected.append(sorted(deduped, key=lambda item: float(item.get("score", 0.0)), reverse=True)[0])
        return self._dedupe(selected)[:3]

    def _dedupe(self, items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        result: list[dict] = []
        for item in items:
            uri = str(item.get("uri", ""))
            if not uri or uri in seen:
                continue
            seen.add(uri)
            result.append(item)
        return result
