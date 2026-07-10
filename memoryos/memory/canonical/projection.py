from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexStore, QueueJob, QueueStore, SourceStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.core.time import utc_now
from memoryos.memory.canonical.visibility import read_committed_canonical
from memoryos.providers.embedding import EmbeddingProvider, HashingEmbeddingProvider


@dataclass(frozen=True)
class ProjectionResult:
    claim_uri: str
    source_revision: int
    status: str


class CanonicalMemoryProjector:
    GENERATOR = "deterministic-template-v1"
    PROMPT_VERSION = "none"

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str | Path,
        *,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.root = Path(root)
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()

    def project(self, claim_uri: str, source_revision: int | None = None) -> ProjectionResult:
        committed = read_committed_canonical(self.source_store, claim_uri)
        obj = committed.object
        if committed.from_before_image:
            return ProjectionResult(claim_uri, int(obj.metadata.get("revision", 0)), "skipped_uncommitted")
        metadata = dict(obj.metadata or {})
        if metadata.get("canonical_kind") != "claim":
            return ProjectionResult(claim_uri, int(metadata.get("revision", 0)), "skipped_non_claim")
        current_revision = int(metadata.get("revision", 0))
        requested = current_revision if source_revision is None else int(source_revision)
        if requested < current_revision:
            return ProjectionResult(claim_uri, current_revision, "skipped_stale")
        if requested > current_revision:
            raise ValueError("projection source revision is newer than canonical claim")
        state = str(metadata.get("state", ""))
        memory_type = str(metadata.get("memory_type", "memory"))
        revision = self._revision_payload(metadata, current_revision)
        revision_values = dict(revision.get("value_fields", {}) or {})
        value = str(
            revision_values.get("canonical_value")
            or revision_values.get("value")
            or metadata.get("canonical_value", obj.title)
        )
        l0 = f"{value} [{state}]"
        qualifiers = dict(revision.get("qualifiers", {}) or {})
        l1_lines = [
            f"# {value}",
            f"- type: {memory_type}",
            f"- state: {state}",
            f"- revision: {current_revision}",
            f"- epistemic: {revision.get('epistemic_status', '')}",
            f"- relation: {revision.get('relation', '')}",
        ]
        if qualifiers:
            l1_lines.append(f"- qualifiers: {json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)}")
        l1 = "\n".join(l1_lines)
        try:
            l2 = self.source_store.read_content(claim_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            l2 = json.dumps(revision, ensure_ascii=False, indent=2, sort_keys=True)

        base = f"{claim_uri}/projections/rev-{current_revision}"
        l0_uri = f"{base}/l0.md"
        l1_uri = f"{base}/l1.md"
        l2_uri = f"{base}/l2.json"
        manifest_uri = f"{base}/manifest.json"
        self.source_store.write_content(l0_uri, l0)
        self.source_store.write_content(l1_uri, l1)
        self.source_store.write_content(l2_uri, l2)
        projection_created_at = utc_now()
        manifest = {
            "memory_id": metadata.get("claim_id"),
            "claim_id": metadata.get("claim_id"),
            "source_revision": current_revision,
            "projection_levels": ["L0", "L1", "L2"],
            "projections": [
                {
                    "memory_id": metadata.get("claim_id"),
                    "claim_id": metadata.get("claim_id"),
                    "source_revision": current_revision,
                    "projection_level": level,
                    "uri": uri,
                    "generator": self.GENERATOR,
                    "model_id": None,
                    "prompt_version": self.PROMPT_VERSION,
                    "created_at": projection_created_at,
                }
                for level, uri in (("L0", l0_uri), ("L1", l1_uri), ("L2", l2_uri))
            ],
            "generator": self.GENERATOR,
            "model_id": None,
            "prompt_version": self.PROMPT_VERSION,
            "created_at": projection_created_at,
        }
        self.source_store.write_content(manifest_uri, json.dumps(manifest, ensure_ascii=False, indent=2))
        obj.layers = ContextLayers(l0_uri=l0_uri, l1_uri=l1_uri, l2_uri=l2_uri)
        obj.metadata = {
            **metadata,
            "projection_pending": False,
            "projection_revision": current_revision,
            "projection_manifest_uri": manifest_uri,
        }
        self.source_store.write_object(obj)
        self.index_store.upsert_index(obj, content="\n".join((l0, l1, l2)))
        self._write_views(obj, current_revision)
        self._project_vector(obj, "\n".join((l0, l1)))
        return ProjectionResult(claim_uri, current_revision, "projected")

    def rebuild(self, *, clear_views: bool = True) -> dict[str, int]:
        if clear_views:
            for name in ("scope", "taxonomy"):
                path = self.root / "views" / name
                if path.exists():
                    shutil.rmtree(path)
        projected = 0
        skipped = 0
        for obj in self.source_store.list_objects():
            if dict(obj.metadata or {}).get("canonical_kind") != "claim":
                continue
            result = self.project(obj.uri)
            if result.status == "projected":
                projected += 1
            else:
                skipped += 1
        return {"projected": projected, "skipped": skipped}

    def _revision_payload(self, metadata: dict[str, Any], revision: int) -> dict[str, Any]:
        revisions = [
            dict(item) for item in metadata.get("revisions", []) or [] if int(item.get("revision", 0)) == revision
        ]
        if not revisions:
            raise ValueError("canonical claim revision payload is missing")
        return revisions[-1]

    def _project_vector(self, obj: ContextObject, content: str) -> None:
        if self.vector_store is None:
            return
        embedding = self.embedding_provider.embed(content)
        self.vector_store.upsert_vector(
            obj.uri,
            embedding,
            metadata={
                "tenant_id": obj.tenant_id,
                "owner_user_id": obj.owner_user_id,
                "claim_id": obj.metadata.get("claim_id"),
                "source_revision": obj.metadata.get("revision"),
                "embedding_model": self.embedding_provider.model_name,
                "schema_version": "canonical_vector_projection_v1",
            },
        )

    def _write_views(self, obj: ContextObject, revision: int) -> None:
        metadata = dict(obj.metadata or {})
        reference = {
            "claim_uri": obj.uri,
            "claim_id": metadata.get("claim_id"),
            "source_revision": revision,
        }
        scope = dict(metadata.get("scope", {}) or {})
        applicability = dict(scope.get("applicability", {}) or {})
        for scope_ref in applicability.get("all_of", []) or []:
            if not isinstance(scope_ref, dict):
                continue
            path = (
                self.root
                / "views"
                / "scope"
                / self._segment(scope_ref.get("kind", "unknown"))
                / self._segment(scope_ref.get("id", "unknown"))
                / f"{metadata.get('claim_id')}.json"
            )
            self._write_json_atomic(path, reference)
        taxonomy = self._taxonomy_path(metadata)
        self._write_json_atomic(
            self.root / "views" / "taxonomy" / taxonomy / f"{metadata.get('claim_id')}.json",
            reference,
        )

    def _taxonomy_path(self, metadata: dict[str, Any]) -> Path:
        memory_type = str(metadata.get("memory_type", "memory"))
        revisions = metadata.get("revisions", []) or []
        current = dict(revisions[-1]) if revisions else {}
        values = dict(current.get("value_fields", {}) or {})
        identity = dict(metadata.get("identity_fields", {}) or {})
        category = {
            "project_decision": "decisions",
            "project_rule": "rules",
            "preference": "preferences",
            "agent_experience": "experiences",
            "profile": "profiles",
            "entity": "entities",
            "event": "events",
        }.get(memory_type, "memory")
        topic = str(
            identity.get("decision_topic")
            or identity.get("rule_topic")
            or identity.get("dimension")
            or identity.get("task_pattern")
            or identity.get("attribute_key")
            or identity.get("canonical_entity_id")
            or metadata.get("canonical_value")
            or values.get("topic")
            or values.get("dimension")
            or "general"
        )
        return Path(category) / self._segment(topic)

    def _segment(self, value: Any) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._:-]+", "-", str(value)).strip("-.")
        return cleaned[:120] or "unknown"

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            if int(current.get("source_revision", 0)) > int(payload.get("source_revision", 0)):
                return
            if current == payload:
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


class MemoryProjectionWorker:
    def __init__(self, projector: CanonicalMemoryProjector, queue_store: QueueStore) -> None:
        self.projector = projector
        self.queue_store = queue_store

    def process_pending(self, limit: int = 10) -> dict[str, list[str]]:
        self.dispatch_outbox()
        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        for job in self.queue_store.lease("memory_projection", limit=limit):
            try:
                outbox = json.loads(Path(str(job.payload["outbox_path"])).read_text(encoding="utf-8"))
                for item in outbox.get("claim_revisions", []) or []:
                    result = self.projector.project(str(item["uri"]), int(item["revision"]))
                    if result.status == "skipped_stale":
                        stale.append(job.job_id)
            except Exception as exc:
                retry = getattr(self.queue_store, "retry", None)
                if callable(retry):
                    retry(job.job_id, str(exc), max_retries=3, retryable=True)
                else:
                    self.queue_store.fail(job.job_id, str(exc))
                failed.append(job.job_id)
                continue
            self.queue_store.ack(job.job_id)
            processed.append(job.job_id)
        return {"processed": processed, "stale": stale, "failed": failed}

    def dispatch_outbox(self) -> list[str]:
        dispatched: list[str] = []
        outbox_root = self.projector.root / "system" / "outbox"
        if not outbox_root.exists():
            return dispatched
        for path in sorted(outbox_root.glob("*.json")):
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if event.get("event_type") != "MemoryCommitted" or event.get("status") != "committed":
                continue
            transaction_id = str(event.get("transaction_id", ""))
            if not transaction_id:
                continue
            claim_revisions = event.get("claim_revisions", []) or []
            target_uri = str(claim_revisions[0].get("uri", "")) if claim_revisions else ""
            self.queue_store.enqueue(
                QueueJob(
                    job_id=f"outbox_{transaction_id}",
                    queue_name="memory_projection",
                    action="project_memory_committed",
                    target_uri=target_uri,
                    payload={
                        "transaction_id": transaction_id,
                        "outbox_path": str(path),
                        "operation_ids": [str(item) for item in event.get("operation_ids", []) or []],
                    },
                )
            )
            dispatched.append(transaction_id)
        return dispatched
