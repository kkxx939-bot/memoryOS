"""Idempotent SessionArchive projection into the Unified Context Catalog.

The archive remains the immutable evidence source.  Records produced here are
sanitized, bounded serving projections and can be deleted and rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone, tzinfo
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    catalog_vector_metadata,
)
from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.projection_equivalence import (
    MAX_EQUIVALENCE_RECORDS,
    ProjectionEquivalenceProof,
    build_projection_equivalence_proof,
)
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.vector import VectorStore, vector_row_id
from memoryos.memory.evidence import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.security.context_projection import ContextProjectionSanitizer
from memoryos.security.workspace_identity import normalize_workspace_id, repository_workspace_id

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_COLLECTION_KIND = {
    "message": CatalogRecordKind.MESSAGE.value,
    "tool_result": CatalogRecordKind.TOOL_RESULT.value,
    "observation": CatalogRecordKind.OBSERVATION.value,
    "action_result": CatalogRecordKind.ACTION_RESULT.value,
    "feedback": CatalogRecordKind.EVENT.value,
    "session": CatalogRecordKind.EVENT.value,
}


def workspace_id_from_session_metadata(metadata: Mapping[str, Any]) -> str:
    """Resolve the stable workspace projection used by Session Catalog rows.

    The same deterministic resolver is shared by the durable projection
    frontier so retrieval health is scoped to exactly the workspace that can
    observe the pending Session projection.
    """

    connect = metadata.get("connect")
    connect_map = dict(connect) if isinstance(connect, Mapping) else {}
    extra = connect_map.get("extra")
    extra_map = dict(extra) if isinstance(extra, Mapping) else {}
    logical = (
        metadata.get("workspace_id")
        or metadata.get("project_id")
        or connect_map.get("project_id")
        or extra_map.get("project_id")
    )
    if logical:
        return normalize_workspace_id(logical)
    repository = extra_map.get("repo") or connect_map.get("repo")
    if repository:
        return repository_workspace_id(
            repo_root=repository,
            cwd=extra_map.get("cwd") or connect_map.get("cwd") or "",
            git_remote=extra_map.get("git_remote") or connect_map.get("git_remote") or "",
        )
    return ""


class CatalogProjectionStore(Protocol):
    def upsert_catalog(self, record: CatalogRecord, *, tenant_id: str) -> None: ...


@dataclass(frozen=True)
class SessionProjectionResult:
    archive_uri: str
    source_digest: str
    projected: int
    record_keys: tuple[str, ...]
    vector_eligible: int
    vectors_projected: int = 0
    tombstoned_records: int = 0
    equivalence_proof: ProjectionEquivalenceProof | None = None


class SessionContextProjector:
    """Project one verified archive into rebuildable serving records."""

    def __init__(
        self,
        catalog_store: CatalogProjectionStore,
        *,
        sanitizer: ContextProjectionSanitizer | None = None,
        semantic_segment_size: int = 8,
        vectorize_important_events: bool = False,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if semantic_segment_size < 1 or semantic_segment_size > 64:
            raise ValueError("semantic_segment_size must be between 1 and 64")
        self.catalog_store = catalog_store
        self.sanitizer = sanitizer or ContextProjectionSanitizer()
        self.semantic_segment_size = semantic_segment_size
        self.vectorize_important_events = vectorize_important_events
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def project(
        self,
        archive: SessionArchive,
        *,
        async_outputs: Mapping[str, Any] | None = None,
        respect_applied_tombstones: bool = False,
    ) -> SessionProjectionResult:
        if not archive.archive_digest or not archive.manifest_digest:
            raise ValueError("SessionArchive must be durably written before projection")
        tenant_id = str(archive.metadata.get("tenant_id") or "default")
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        expected_records = self.build_records(
            archive,
            episode=episode,
            async_outputs=async_outputs,
        )
        records = expected_records
        if respect_applied_tombstones:
            selector = getattr(self.catalog_store, "rebuildable_catalog_records", None)
            if not callable(selector):
                raise RuntimeError("Session Catalog rebuild requires durable tombstone filtering")
            selected: Any = selector(expected_records, tenant_id=tenant_id)
            if not isinstance(selected, Sequence) or any(
                not isinstance(record, CatalogRecord) for record in selected
            ):
                raise TypeError("Session Catalog tombstone filter returned invalid records")
            records = tuple(selected)
        vector_rows = self._prepare_vector_rows(records)
        pending_records = (
            tuple(replace(record, projection_status=CatalogProjectionStatus.PENDING.value) for record in records)
            if vector_rows
            else records
        )
        batch = getattr(self.catalog_store, "upsert_catalog_batch", None)
        if callable(batch):
            batch(pending_records, tenant_id=tenant_id)
        else:
            for record in pending_records:
                self.catalog_store.upsert_catalog(record, tenant_id=tenant_id)
        projected_vector_uris: list[str] = []
        try:
            if self.vector_store is not None:
                for uri, embedding, metadata in vector_rows:
                    self.vector_store.upsert_vector(uri, embedding, metadata)
                    projected_vector_uris.append(uri)
        except Exception as exc:
            cleanup_error = ""
            if self.vector_store is not None:
                for uri in projected_vector_uris:
                    try:
                        self.vector_store.delete_vector(uri)
                    except Exception as cleanup_exc:
                        # Catalog stays explicitly DEGRADED and the durable
                        # Session job will replay both vector writes/deletes.
                        cleanup_error = type(cleanup_exc).__name__
            degraded = tuple(
                replace(
                    record,
                    projection_status=CatalogProjectionStatus.DEGRADED.value,
                    metadata={
                        **dict(record.metadata),
                        "projection_error": type(exc).__name__,
                        **({"projection_cleanup_error": cleanup_error} if cleanup_error else {}),
                    },
                )
                for record in records
            )
            if callable(batch):
                batch(degraded, tenant_id=tenant_id)
            else:
                for record in degraded:
                    self.catalog_store.upsert_catalog(record, tenant_id=tenant_id)
            raise
        if vector_rows:
            completed = tuple(
                replace(record, projection_status=CatalogProjectionStatus.PROJECTED.value) for record in records
            )
            if callable(batch):
                batch(completed, tenant_id=tenant_id)
            else:
                for record in completed:
                    self.catalog_store.upsert_catalog(record, tenant_id=tenant_id)
        proof = self.prove_projection(
            archive,
            expected_records=records,
            async_outputs=async_outputs,
        )
        return SessionProjectionResult(
            archive_uri=archive.archive_uri,
            source_digest=archive.archive_digest,
            projected=len(records),
            record_keys=tuple(record.record_key for record in records),
            vector_eligible=sum(bool(record.metadata.get("vector_eligible")) for record in records),
            vectors_projected=len(vector_rows),
            tombstoned_records=len(expected_records) - len(records),
            equivalence_proof=proof,
        )

    def prove_projection(
        self,
        archive: SessionArchive,
        *,
        expected_records: Sequence[CatalogRecord] | None = None,
        async_outputs: Mapping[str, Any] | None = None,
    ) -> ProjectionEquivalenceProof | None:
        """Prove one archive projection through an exact evidence lookup.

        The expected set is rebuilt from immutable SessionArchive evidence.
        The actual set comes from the Catalog's evidence identity index, not
        from online search or retrieval ranking.
        """

        lookup = getattr(self.catalog_store, "list_catalog_projection_records", None)
        if not callable(lookup):
            return None
        expected = tuple(
            self.build_records(archive, async_outputs=async_outputs)
            if expected_records is None
            else expected_records
        )
        raw_actual: Any = lookup(
            tenant_id=str(archive.metadata.get("tenant_id") or "default"),
            source_uri=archive.archive_uri,
            projection_effect_hash=archive.manifest_digest,
            limit=MAX_EQUIVALENCE_RECORDS + 1,
        )
        if not isinstance(raw_actual, Sequence) or any(not isinstance(record, CatalogRecord) for record in raw_actual):
            raise TypeError("Catalog projection proof lookup returned invalid records")
        actual = tuple(cast(Sequence[CatalogRecord], raw_actual))
        return build_projection_equivalence_proof(
            plane="session_archive",
            source_identity=archive.archive_uri,
            evidence_digest=archive.archive_digest,
            expected_records=expected,
            actual_records=actual,
            sanitizer=self.sanitizer,
        )

    def _prepare_vector_rows(
        self,
        records: Sequence[CatalogRecord],
    ) -> tuple[tuple[str, list[float], dict[str, Any]], ...]:
        """Embed only bounded, sanitized serving text selected by policy.

        ``build_records`` has already passed every record through the
        ContextProjectionSanitizer.  Preparing all embeddings before mutating
        Catalog/Vector state also makes provider failures retryable without
        publishing a Catalog row that falsely claims its vector is ready.
        """

        if self.vector_store is None or self.embedding_provider is None:
            return ()
        prepared: list[tuple[str, list[float], dict[str, Any]]] = []
        for record in records:
            if not bool(record.metadata.get("vector_eligible")):
                continue
            text = "\n".join(part for part in (record.title, record.l0_text, record.l1_text) if part)
            if not text:
                raise ValueError("vector-eligible Session projection has no sanitized text")
            embedding = [float(value) for value in self.embedding_provider.embed(text)]
            if not embedding or any(not math.isfinite(value) for value in embedding):
                raise ValueError("Session embedding provider returned an invalid vector")
            prepared.append(
                (
                    vector_row_id(record.tenant_id, record.record_key),
                    embedding,
                    {
                        **catalog_vector_metadata(record, sanitizer=self.sanitizer),
                        "public_uri": record.uri,
                        "source_manifest_digest": record.projection_effect_hash,
                        "embedding_model": self.embedding_provider.model_name,
                        "schema_version": "unified_context_vector_v1",
                    },
                )
            )
        return tuple(prepared)

    def build_records(
        self,
        archive: SessionArchive,
        *,
        episode: EvidenceEpisode | None = None,
        async_outputs: Mapping[str, Any] | None = None,
    ) -> tuple[CatalogRecord, ...]:
        episode = episode or SessionArchiveEpisodeAdapter().adapt(archive)
        tenant_id = episode.tenant_id
        owner_user_id = archive.user_id
        workspace_id = workspace_id_from_session_metadata(archive.metadata)
        adapter_id = episode.origin.adapter_id
        base_paths = self._base_paths(
            archive,
            event_time=episode.started_at,
            workspace_id=workspace_id,
            adapter_id=adapter_id,
        )
        created_at = self._iso(archive.created_at)
        event_texts = [self._event_text(event.content) for event in episode.events]
        joined = "\n".join(text for text in event_texts if text)
        abstract = l0_abstract(joined or f"Session {archive.session_id}")
        overview = l1_overview(
            f"Session {archive.session_id}",
            [
                f"messages: {len(archive.messages)}",
                f"tool_results: {len(archive.tool_results)}",
                f"observations: {len(archive.observations)}",
                f"action_results: {len(archive.action_results)}",
                f"used_contexts: {len(archive.used_contexts)}",
                f"used_skills: {len(archive.used_skills)}",
            ],
        )
        summary_metadata: dict[str, Any] = {"summary_source": "session_archive"}
        if async_outputs is not None:
            head = async_outputs.get("head")
            manifest = async_outputs.get("manifest")
            if not isinstance(head, Mapping) or not isinstance(manifest, Mapping):
                raise TypeError("Session async outputs require verified head and manifest metadata")
            if (
                str(head.get("task_id") or "") != archive.task_id
                or str(manifest.get("task_id") or "") != archive.task_id
                or str(head.get("archive_uri") or "") != archive.archive_uri
                or str(manifest.get("archive_uri") or "") != archive.archive_uri
            ):
                raise ValueError("Session async outputs are detached from their archive")
            async_abstract = async_outputs.get("abstract")
            async_overview = async_outputs.get("overview")
            if not isinstance(async_abstract, str) or not isinstance(async_overview, str):
                raise TypeError("Session async summaries must be text")
            abstract = async_abstract
            overview = async_overview
            manifest_digest = str(manifest.get("manifest_digest") or "")
            if len(manifest_digest) != 64:
                raise ValueError("Session async output manifest digest is invalid")
            summary_metadata = {
                "summary_source": "session_async_outputs",
                "async_output_manifest_digest": manifest_digest,
            }
        common = {
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "workspace_id": workspace_id,
            "session_id": archive.session_id,
            "adapter_id": adapter_id,
            "context_type": "session",
            "lifecycle_state": "active",
            "created_at": created_at,
            "updated_at": created_at,
            "transaction_time": created_at,
            "l2_uri": archive.archive_uri,
            "source_revision": 1,
            "projection_effect_hash": archive.manifest_digest,
        }
        root_uri = f"{archive.archive_uri.rstrip('/')}/context/root"
        records: list[CatalogRecord] = [
            self._record(
                **common,
                record_key=self._key(archive, "root", archive.archive_digest),
                uri=root_uri,
                source_kind="session_root",
                record_kind=CatalogRecordKind.SESSION_ROOT.value,
                parent_uri="",
                tree_paths=base_paths,
                event_time=self._iso(episode.started_at),
                ingested_at=created_at,
                title=f"Session {archive.session_id}",
                l0_text=abstract,
                l1_text=overview,
                source_uri=archive.archive_uri,
                source_digest=archive.archive_digest,
                metadata={
                    "archive_uri": archive.archive_uri,
                    "manifest_digest": archive.manifest_digest,
                    "vector_eligible": True,
                    "projection_source": "session_archive",
                    **summary_metadata,
                },
            ),
            self._record(
                **common,
                record_key=self._key(archive, "l0", archive.archive_digest),
                uri=f"{archive.archive_uri.rstrip('/')}/context/l0",
                source_kind="session_abstract",
                record_kind=CatalogRecordKind.SESSION_L0.value,
                parent_uri=root_uri,
                tree_paths=base_paths,
                event_time=self._iso(episode.started_at),
                ingested_at=created_at,
                title=f"Session {archive.session_id} abstract",
                l0_text=abstract,
                l1_text="",
                source_uri=archive.archive_uri,
                source_digest=archive.archive_digest,
                metadata={
                    "archive_uri": archive.archive_uri,
                    "vector_eligible": False,
                    **summary_metadata,
                },
            ),
            self._record(
                **common,
                record_key=self._key(archive, "l1", archive.archive_digest),
                uri=f"{archive.archive_uri.rstrip('/')}/context/l1",
                source_kind="session_overview",
                record_kind=CatalogRecordKind.SESSION_L1.value,
                parent_uri=root_uri,
                tree_paths=base_paths,
                event_time=self._iso(episode.started_at),
                ingested_at=created_at,
                title=f"Session {archive.session_id} overview",
                l0_text=abstract,
                l1_text=overview,
                source_uri=archive.archive_uri,
                source_digest=archive.archive_digest,
                metadata={
                    "archive_uri": archive.archive_uri,
                    "vector_eligible": False,
                    **summary_metadata,
                },
            ),
        ]

        event_records: list[CatalogRecord] = []
        for ordinal, event in enumerate(episode.events):
            category = str(event.metadata.get("category") or "event").casefold()
            record_kind = _COLLECTION_KIND.get(category, CatalogRecordKind.EVENT.value)
            raw = dict(event.content)
            text = self._event_text(raw)
            paths = self._event_paths(
                archive,
                raw.get("occurred_at")
                or raw.get("event_time")
                or raw.get("created_at")
                or event.occurred_at,
                base_paths=base_paths,
                raw=raw,
            )
            metadata = self._event_metadata(archive, raw, category=category, event_id=event.event_id)
            important = bool(raw.get("important") or raw.get("salient") or raw.get("pinned"))
            metadata["vector_eligible"] = bool(
                category not in {"message", "tool_result"} and important and self.vectorize_important_events
            )
            title = self._event_title(category, raw, ordinal)
            event_record = self._record(
                **common,
                record_key=self._key(archive, category, event.digest),
                uri=f"{archive.archive_uri.rstrip('/')}/context/{record_kind}/{event.digest[:20]}",
                source_kind=category,
                record_kind=record_kind,
                parent_uri=root_uri,
                tree_paths=paths,
                event_time=self._iso(event.occurred_at),
                ingested_at=self._iso(event.ingested_at or event.occurred_at),
                title=title,
                l0_text=l0_abstract(text or title),
                l1_text=text,
                source_uri=archive.archive_uri,
                source_digest=event.digest,
                metadata=metadata,
            )
            event_records.append(event_record)
            if category == "tool_result" and event_record.metadata.get("resource_name"):
                event_records.append(
                    self._resource_record(
                        archive,
                        event_record,
                        raw,
                        common=common,
                        root_uri=root_uri,
                    )
                )
        records.extend(event_records)
        records.extend(self._semantic_segments(archive, event_records, common=common, root_uri=root_uri))
        records.extend(
            self._reference_records(
                archive,
                archive.used_contexts,
                kind=CatalogRecordKind.USED_CONTEXT,
                common=common,
                root_uri=root_uri,
                base_paths=base_paths,
                fallback_event_time=episode.started_at,
            )
        )
        records.extend(
            self._reference_records(
                archive,
                archive.used_skills,
                kind=CatalogRecordKind.USED_SKILL,
                common=common,
                root_uri=root_uri,
                base_paths=base_paths,
                fallback_event_time=episode.started_at,
            )
        )
        keys = [record.record_key for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("session projection produced duplicate record keys")
        return tuple(records)

    def _record(self, **kwargs: Any) -> CatalogRecord:
        return CatalogRecord(**kwargs).with_sanitized_projection(self.sanitizer)

    def _semantic_segments(
        self,
        archive: SessionArchive,
        event_records: Sequence[CatalogRecord],
        *,
        common: Mapping[str, Any],
        root_uri: str,
    ) -> list[CatalogRecord]:
        source_events = [
            record for record in event_records if record.record_kind != CatalogRecordKind.RESOURCE_REFERENCE.value
        ]
        # A semantic segment has one structured ``event_time``.  Never let it
        # span multiple local Timeline days, otherwise one row would advertise
        # a first-event timestamp while being reachable through a different
        # day's path.  This remains deterministic and bounded by the existing
        # maximum segment size.
        chunks: list[list[CatalogRecord]] = []
        chunk: list[CatalogRecord] = []
        chunk_timeline = ""
        for record in source_events:
            timeline = next((path for path in record.tree_paths if path.startswith("timeline/")), "")
            if not timeline:
                raise ValueError("session event projection has no Timeline path")
            if chunk and (len(chunk) >= self.semantic_segment_size or timeline != chunk_timeline):
                chunks.append(chunk)
                chunk = []
            if not chunk:
                chunk_timeline = timeline
            chunk.append(record)
        if chunk:
            chunks.append(chunk)

        segments: list[CatalogRecord] = []
        for segment_index, chunk in enumerate(chunks):
            digest = self.sanitizer.digest([record.source_digest for record in chunk])
            text = "\n".join(record.l1_text or record.l0_text for record in chunk)
            paths = tuple(dict.fromkeys(path for record in chunk for path in record.tree_paths))[:8]
            segments.append(
                self._record(
                    **common,
                    record_key=self._key(archive, "semantic_segment", digest),
                    uri=f"{archive.archive_uri.rstrip('/')}/context/segments/{segment_index}",
                    source_kind="semantic_segment",
                    record_kind=CatalogRecordKind.SEMANTIC_SEGMENT.value,
                    parent_uri=root_uri,
                    tree_paths=paths,
                    event_time=chunk[0].event_time,
                    ingested_at=chunk[-1].ingested_at,
                    title=f"Session {archive.session_id} segment {segment_index + 1}",
                    l0_text=l0_abstract(text),
                    l1_text=text,
                    source_uri=archive.archive_uri,
                    source_digest=digest,
                    metadata={
                        "archive_uri": archive.archive_uri,
                        "event_source_digests": [record.source_digest for record in chunk],
                        "vector_eligible": True,
                    },
                )
            )
        return segments

    def _resource_record(
        self,
        archive: SessionArchive,
        tool_record: CatalogRecord,
        raw: Mapping[str, Any],
        *,
        common: Mapping[str, Any],
        root_uri: str,
    ) -> CatalogRecord:
        name = str(tool_record.metadata.get("resource_name") or "resource")
        location = str(tool_record.metadata.get("resource_location") or "external")
        digest = self.sanitizer.digest(
            {"tool": tool_record.source_digest, "resource_name": name, "resource_location": location}
        )
        paths = tuple(dict.fromkeys((*tool_record.tree_paths, f"resources/{self._segment(location)}")))[:8]
        return self._record(
            **common,
            record_key=self._key(archive, "resource", digest),
            uri=f"{archive.archive_uri.rstrip('/')}/context/resources/{digest[:20]}",
            source_kind="resource_reference",
            record_kind=CatalogRecordKind.RESOURCE_REFERENCE.value,
            parent_uri=root_uri,
            tree_paths=paths,
            event_time=tool_record.event_time,
            ingested_at=tool_record.ingested_at,
            title=name,
            l0_text=f"{location} resource: {name}",
            l1_text=str(raw.get("description") or raw.get("summary") or name),
            source_uri=tool_record.source_uri,
            source_digest=tool_record.source_digest,
            metadata={
                "archive_uri": archive.archive_uri,
                "resource_name": name,
                "resource_location": location,
                "tool_result_record_key": tool_record.record_key,
                "vector_eligible": bool(
                    self.vectorize_important_events
                    and (raw.get("important") or raw.get("salient") or raw.get("pinned"))
                ),
            },
        )

    def _reference_records(
        self,
        archive: SessionArchive,
        values: Iterable[Mapping[str, Any]],
        *,
        kind: CatalogRecordKind,
        common: Mapping[str, Any],
        root_uri: str,
        base_paths: tuple[str, ...],
        fallback_event_time: object,
    ) -> list[CatalogRecord]:
        result: list[CatalogRecord] = []
        for index, value in enumerate(values):
            raw = dict(value)
            digest = self.sanitizer.digest(raw)
            text = self._event_text(raw)
            title = str(raw.get("title") or raw.get("name") or raw.get("skill_name") or f"{kind.value} {index + 1}")
            reference_event_time = raw.get("event_time") or raw.get("occurred_at") or fallback_event_time
            paths = [
                self._timeline_path(reference_event_time, archive.metadata),
                *(path for path in base_paths if not path.startswith("timeline/")),
            ]
            if kind is CatalogRecordKind.USED_SKILL:
                paths.append(f"skills/{self._segment(raw.get('skill_name') or raw.get('name') or title)}")
            result.append(
                self._record(
                    **common,
                    record_key=self._key(archive, kind.value, digest),
                    uri=f"{archive.archive_uri.rstrip('/')}/context/{kind.value}/{digest[:20]}",
                    source_kind=kind.value,
                    record_kind=kind.value,
                    parent_uri=root_uri,
                    tree_paths=tuple(dict.fromkeys(paths))[:8],
                    event_time=self._iso(reference_event_time),
                    ingested_at=self._iso(raw.get("ingested_at") or archive.created_at),
                    title=title,
                    l0_text=l0_abstract(text or title),
                    l1_text=text,
                    source_uri=archive.archive_uri,
                    source_digest=digest,
                    metadata={
                        "archive_uri": archive.archive_uri,
                        "source_reference_uri": str(raw.get("source_uri") or raw.get("uri") or ""),
                        "vector_eligible": False,
                    },
                )
            )
        return result

    def _event_paths(
        self,
        archive: SessionArchive,
        event_time: object,
        *,
        base_paths: tuple[str, ...],
        raw: Mapping[str, Any],
    ) -> tuple[str, ...]:
        # Event nodes belong to the day on which the event occurred.  A
        # Session root may also have an archive/episode timeline path, but
        # copying that write-time path onto every event makes one Tool Result
        # appear on multiple, incorrect days.
        paths = [
            self._timeline_path(event_time, archive.metadata),
            *(path for path in base_paths if not path.startswith("timeline/")),
        ]
        resource_name, location = self.sanitizer._resource_identity(raw)  # noqa: SLF001 - one policy boundary.
        if resource_name:
            paths.append(f"resources/{self._segment(location or 'external')}")
        return tuple(dict.fromkeys(paths))[:8]

    def _base_paths(
        self,
        archive: SessionArchive,
        *,
        event_time: object,
        workspace_id: str,
        adapter_id: str,
    ) -> tuple[str, ...]:
        paths = [
            f"sessions/{self._segment(archive.session_id)}",
            self._timeline_path(event_time, archive.metadata),
        ]
        if workspace_id:
            paths.append(f"projects/{self._segment(workspace_id)}")
        if adapter_id:
            paths.append(f"agents/{self._segment(adapter_id)}")
        return tuple(dict.fromkeys(paths))

    def _event_metadata(
        self,
        archive: SessionArchive,
        raw: Mapping[str, Any],
        *,
        category: str,
        event_id: str,
    ) -> dict[str, Any]:
        allowed = {
            key: raw[key]
            for key in (
                "tool_name",
                "name",
                "status",
                "result_status",
                "resource_uri",
                "file_uri",
                "path",
                "file_path",
                "absolute_path",
                "file_name",
                "filename",
                "resource_name",
                "resource_location",
                "important",
                "salient",
            )
            if key in raw
        }
        return {
            **allowed,
            "archive_uri": archive.archive_uri,
            "event_id": event_id,
            "event_category": category,
            "projection_source": "session_archive_event",
        }

    @staticmethod
    def _event_text(raw: Mapping[str, Any]) -> str:
        for key in ("content", "text", "raw_text", "scene", "output", "result", "summary", "description"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, Mapping | list) and value:
                return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return json.dumps(dict(raw), ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _event_title(category: str, raw: Mapping[str, Any], ordinal: int) -> str:
        return str(
            raw.get("title")
            or raw.get("file_name")
            or raw.get("filename")
            or raw.get("resource_name")
            or raw.get("tool_name")
            or f"{category.replace('_', ' ').title()} {ordinal + 1}"
        )

    @staticmethod
    def _key(archive: SessionArchive, kind: str, digest: str) -> str:
        return f"session:{archive.session_id}:{archive.manifest_digest}:{kind}:{digest[:32]}"

    @staticmethod
    def _segment(value: object) -> str:
        text = str(value or "unknown").strip()
        if _SAFE_SEGMENT.fullmatch(text):
            return text
        return "id-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def _timeline_path(cls, value: object, metadata: Mapping[str, Any]) -> str:
        parsed = cls._datetime(value)
        timezone_name = str(metadata.get("timezone") or metadata.get("time_zone") or "")
        if timezone_name:
            try:
                zone: tzinfo = ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                zone = timezone.utc
        else:
            # Preserve an explicit offset carried by the evidence when the
            # Session has no configured IANA timezone.  Structured event_time
            # remains UTC; only the controlled logical day path is localized.
            zone = parsed.tzinfo or timezone.utc
        local = parsed.astimezone(zone)
        return f"timeline/{local.year:04d}/{local.month:02d}/{local.day:02d}"

    @staticmethod
    def _datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("session projection timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("session projection timestamp must include timezone")
        return parsed

    @classmethod
    def _iso(cls, value: object) -> str:
        return cls._datetime(value).astimezone(timezone.utc).isoformat()


__all__ = ["SessionContextProjector", "SessionProjectionResult"]
