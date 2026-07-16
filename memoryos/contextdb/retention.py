"""Safe lifecycle, compaction, and GC for rebuildable serving projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    normalize_tree_path,
)
from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.store.vector_store import vector_row_id
from memoryos.security.context_projection import ContextProjectionSanitizer


class RetentionCatalogStore(Protocol):
    def scan_catalog_batch(
        self,
        *,
        after_record_key: str = "",
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
    ) -> list[CatalogRecord]: ...

    def get_catalog(self, record_key: str, *, tenant_id: str | None = None) -> CatalogRecord | None: ...

    def upsert_catalog(self, record: CatalogRecord) -> None: ...

    def enqueue_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str = "",
        source_revision: int = 0,
        tombstone_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def mark_tombstone_applied(self, tombstone_id: str) -> dict[str, Any] | None: ...

    def mark_tombstone_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None: ...

    def gc_orphan_paths(self, *, limit: int = 256) -> int: ...

    def gc_applied_tombstones(self, *, updated_before: str, limit: int = 256) -> int: ...


class VectorDeleteStore(Protocol):
    def delete_vector(self, uri: str) -> None: ...

    def get_vector_metadata(self, uri: str) -> dict[str, Any] | None: ...


class TombstoneProcessor(Protocol):
    def process_pending(self, *, limit: int = 100) -> Any: ...


@dataclass(frozen=True)
class RetentionPolicy:
    """Safe defaults tier serving data without deleting immutable evidence."""

    hot_for: timedelta = timedelta(days=7)
    warm_for: timedelta = timedelta(days=30)
    cold_for: timedelta = timedelta(days=180)
    tombstone_journal_for: timedelta = timedelta(days=30)
    vectorize_warm: bool = False
    batch_size: int = 256
    max_compaction_sources: int = 256

    def __post_init__(self) -> None:
        if not timedelta(0) <= self.hot_for <= self.warm_for <= self.cold_for:
            raise ValueError("retention tier durations must be monotonically increasing")
        if self.tombstone_journal_for <= timedelta(0):
            raise ValueError("tombstone_journal_for must be positive")
        if not 1 <= self.batch_size <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        if not 1 <= self.max_compaction_sources <= 1_000:
            raise ValueError("max_compaction_sources must be between 1 and 1000")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> RetentionPolicy:
        """Build a validated policy from a small, explicit runtime schema."""

        values = dict(config or {})
        allowed = {
            "hot_days",
            "warm_days",
            "cold_days",
            "tombstone_journal_days",
            "vectorize_warm",
            "batch_size",
            "max_compaction_sources",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unsupported retention policy fields: {', '.join(unknown)}")

        def duration(name: str, default: timedelta) -> timedelta:
            raw = values.get(name)
            if raw is None:
                return default
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise ValueError(f"{name} must be a finite number of days")
            days = float(raw)
            if days != days or days in {float("inf"), float("-inf")}:
                raise ValueError(f"{name} must be a finite number of days")
            return timedelta(days=days)

        def integer(name: str, default: int) -> int:
            raw = values.get(name, default)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"{name} must be an integer")
            return raw

        vectorize_warm = values.get("vectorize_warm", False)
        if not isinstance(vectorize_warm, bool):
            raise ValueError("vectorize_warm must be a boolean")

        defaults = cls()
        return cls(
            hot_for=duration("hot_days", defaults.hot_for),
            warm_for=duration("warm_days", defaults.warm_for),
            cold_for=duration("cold_days", defaults.cold_for),
            tombstone_journal_for=duration(
                "tombstone_journal_days",
                defaults.tombstone_journal_for,
            ),
            vectorize_warm=vectorize_warm,
            batch_size=integer("batch_size", defaults.batch_size),
            max_compaction_sources=integer("max_compaction_sources", defaults.max_compaction_sources),
        )


@dataclass(frozen=True)
class RetentionRunResult:
    scanned: int = 0
    tier_changes: int = 0
    stale_projections: int = 0
    vectors_deleted: int = 0
    orphan_paths_deleted: int = 0
    tombstones_deleted: int = 0
    tombstones_enqueued: int = 0
    tombstones_applied: int = 0
    tombstones_failed: int = 0


class CatalogRetentionManager:
    """Manage serving tiers while preserving SourceStore and archive evidence."""

    def __init__(
        self,
        catalog_store: RetentionCatalogStore,
        *,
        vector_store: VectorDeleteStore | None = None,
        tombstone_service: TombstoneProcessor | None = None,
        policy: RetentionPolicy | None = None,
        sanitizer: ContextProjectionSanitizer | None = None,
    ) -> None:
        self.catalog_store = catalog_store
        self.vector_store = vector_store
        self.policy = policy or RetentionPolicy()
        self.sanitizer = sanitizer or ContextProjectionSanitizer()
        if tombstone_service is None and vector_store is not None:
            # Local/direct manager users still get the same durable journal as
            # the runtime.  RuntimeContainer injects the full service with
            # Source/Relation consumers.
            from memoryos.contextdb.tombstone import ProjectionTombstoneService

            tombstone_service = ProjectionTombstoneService(
                catalog_store,
                vector_store=vector_store,  # type: ignore[arg-type]
            )
        self.tombstone_service = tombstone_service

    def apply_serving_tiers(
        self,
        *,
        tenant_id: str,
        now: datetime | None = None,
    ) -> RetentionRunResult:
        effective_now = self._utc(now or datetime.now(timezone.utc))
        scanned = 0
        changes = 0
        tombstone_ids: list[str] = []
        cursor = ""
        while True:
            batch = self.catalog_store.scan_catalog_batch(
                after_record_key=cursor,
                filters={"tenant_id": tenant_id, "include_inactive": True},
                limit=self.policy.batch_size,
            )
            if not batch:
                break
            for record in batch:
                scanned += 1
                target = self.tier_for(record, now=effective_now)
                if target.value != record.serving_tier:
                    metadata = dict(record.metadata)
                    metadata["serving_tier_changed_at"] = effective_now.isoformat()
                    updated = replace(
                        record,
                        serving_tier=target.value,
                        updated_at=effective_now.isoformat(),
                        metadata=metadata,
                    )
                    self.catalog_store.upsert_catalog(updated)
                    if not self._retains_vector(updated):
                        tombstone_id = self._enqueue_vector_delete(updated, reason="retention-vector-delete")
                        if tombstone_id:
                            tombstone_ids.append(tombstone_id)
                    changes += 1
                cursor = record.record_key
        return RetentionRunResult(
            scanned=scanned,
            tier_changes=changes,
            tombstones_enqueued=len(tombstone_ids),
        )

    def tier_for(self, record: CatalogRecord, *, now: datetime) -> ServingTier:
        if record.record_kind == CatalogRecordKind.CURRENT_SLOT.value:
            return ServingTier.HOT
        reference = self._reference_time(record)
        if reference is None:
            # Missing structured time is unsafe to archive automatically.
            return ServingTier.HOT
        age = max(timedelta(0), self._utc(now) - reference)
        if age <= self.policy.hot_for:
            return ServingTier.HOT
        if age <= self.policy.warm_for:
            return ServingTier.WARM
        if age <= self.policy.cold_for:
            return ServingTier.COLD
        return ServingTier.ARCHIVED

    def compact_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
        owner_user_id: str = "",
        now: datetime | None = None,
    ) -> CatalogRecord | None:
        filters: dict[str, Any] = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "include_inactive": True,
        }
        if owner_user_id:
            filters["owner_user_id"] = owner_user_id
        records = self.catalog_store.scan_catalog_batch(
            filters=filters,
            limit=self.policy.max_compaction_sources,
        )
        sources = [
            record
            for record in records
            if record.record_kind
            in {
                CatalogRecordKind.SEMANTIC_SEGMENT.value,
                CatalogRecordKind.MESSAGE.value,
                CatalogRecordKind.TOOL_RESULT.value,
                CatalogRecordKind.OBSERVATION.value,
                CatalogRecordKind.ACTION_RESULT.value,
                CatalogRecordKind.EVENT.value,
            }
        ]
        if not sources:
            return None
        owners = {record.owner_user_id for record in sources if record.owner_user_id}
        if owner_user_id:
            owners.add(owner_user_id)
        if len(owners) != 1:
            raise ValueError("session compaction requires exactly one owner")
        owner = next(iter(owners))
        effective_now = self._utc(now or datetime.now(timezone.utc)).isoformat()
        sources.sort(key=lambda record: (record.event_time, record.record_key))
        summary_lines = [record.l0_text or record.title for record in sources if record.l0_text or record.title]
        source_digest = self.sanitizer.digest(
            [(record.record_key, record.source_digest, record.source_revision) for record in sources]
        )
        session_path = f"sessions/{self._safe_path_segment(session_id)}"
        timeline_paths = [path for record in sources for path in record.tree_paths if path.startswith("timeline/")]
        tree_paths = tuple(dict.fromkeys((session_path, *timeline_paths)))[:8]
        uri = f"memoryos://user/{owner}/sessions/history/{self._safe_path_segment(session_id)}/context/compacted"
        compacted = CatalogRecord(
            record_key=f"compaction:session:{tenant_id}:{owner}:{session_id}",
            uri=uri,
            tenant_id=tenant_id,
            owner_user_id=owner,
            workspace_id=next((record.workspace_id for record in sources if record.workspace_id), ""),
            session_id=session_id,
            adapter_id=next((record.adapter_id for record in sources if record.adapter_id), ""),
            context_type="session",
            source_kind="session_compaction",
            record_kind=CatalogRecordKind.SESSION_L1.value,
            lifecycle_state="active",
            tree_paths=tree_paths,
            created_at=min(record.created_at for record in sources if record.created_at),
            updated_at=effective_now,
            event_time=min(record.event_time for record in sources if record.event_time),
            ingested_at=effective_now,
            transaction_time=effective_now,
            title=f"Session {session_id} compacted overview",
            l0_text=l0_abstract(" ".join(summary_lines)),
            l1_text=l1_overview(
                f"Session {session_id}",
                summary_lines,
                max_bullets=12,
            ),
            l2_uri=next((record.l2_uri for record in sources if record.l2_uri), ""),
            source_uri=next((record.source_uri for record in sources if record.source_uri), ""),
            source_digest=source_digest,
            source_revision=max(record.source_revision for record in sources),
            serving_tier=ServingTier.WARM.value,
            projection_status=CatalogProjectionStatus.PROJECTED.value,
            metadata={
                "compaction_kind": "session_segment",
                "source_count": len(sources),
                "source_record_digest": self.sanitizer.digest([record.record_key for record in sources]),
                "vector_eligible": False,
            },
        ).with_sanitized_projection(self.sanitizer)
        self.catalog_store.upsert_catalog(compacted)
        for source in sources:
            target = (
                ServingTier.WARM if source.record_kind == CatalogRecordKind.SEMANTIC_SEGMENT.value else ServingTier.COLD
            )
            if source.serving_tier != target.value:
                updated = replace(source, serving_tier=target.value, updated_at=effective_now)
                self.catalog_store.upsert_catalog(updated)
                if not self._retains_vector(updated):
                    self._enqueue_vector_delete(updated, reason="retention-vector-delete")
        return compacted

    def compact_timeline(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        timeline_path: str,
        now: datetime | None = None,
    ) -> CatalogRecord | None:
        normalized_path = normalize_tree_path(timeline_path)
        if not normalized_path.startswith("timeline/") or len(normalized_path.split("/")) != 4:
            raise ValueError("timeline compaction requires a day path")
        sources = self.catalog_store.scan_catalog_batch(
            filters={
                "tenant_id": tenant_id,
                "owner_user_id": owner_user_id,
                "target_paths": (normalized_path,),
                "include_inactive": True,
            },
            limit=self.policy.max_compaction_sources,
        )
        sources = [
            record
            for record in sources
            if record.record_kind != CatalogRecordKind.TREE_OVERVIEW.value and (record.l0_text or record.title)
        ]
        if not sources:
            return None
        effective_now = self._utc(now or datetime.now(timezone.utc)).isoformat()
        sources.sort(key=lambda record: (record.event_time, record.record_key))
        bullets = [record.l0_text or record.title for record in sources]
        path_digest = self.sanitizer.digest((tenant_id, owner_user_id, normalized_path))
        record = CatalogRecord(
            record_key=f"compaction:timeline:{path_digest}",
            uri=f"memoryos://user/{owner_user_id}/catalog/timeline/{path_digest[:20]}",
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            context_type="session",
            source_kind="timeline_compaction",
            record_kind=CatalogRecordKind.TREE_OVERVIEW.value,
            lifecycle_state="active",
            tree_paths=(normalized_path,),
            created_at=min(item.created_at for item in sources if item.created_at),
            updated_at=effective_now,
            event_time=min(item.event_time for item in sources if item.event_time),
            ingested_at=effective_now,
            transaction_time=effective_now,
            title=f"Timeline overview {normalized_path.removeprefix('timeline/')}",
            l0_text=l0_abstract(" ".join(bullets)),
            l1_text=l1_overview("Timeline overview", bullets, max_bullets=12),
            source_uri=next((item.source_uri for item in sources if item.source_uri), ""),
            source_digest=self.sanitizer.digest([(item.record_key, item.source_digest) for item in sources]),
            source_revision=max(item.source_revision for item in sources),
            serving_tier=ServingTier.WARM.value,
            projection_status=CatalogProjectionStatus.PROJECTED.value,
            metadata={
                "compaction_kind": "timeline_overview",
                "source_count": len(sources),
                "vector_eligible": False,
            },
        ).with_sanitized_projection(self.sanitizer)
        self.catalog_store.upsert_catalog(record)
        return record

    def restore_cold_record(
        self,
        record_key: str,
        *,
        tenant_id: str,
        now: datetime | None = None,
    ) -> CatalogRecord:
        record = self.catalog_store.get_catalog(record_key, tenant_id=tenant_id)
        if record is None:
            raise KeyError(record_key)
        if record.serving_tier not in {ServingTier.COLD.value, ServingTier.ARCHIVED.value}:
            return record
        effective_now = self._utc(now or datetime.now(timezone.utc)).isoformat()
        metadata = dict(record.metadata)
        metadata.update({"cold_restored_at": effective_now, "vector_eligible": False})
        restored = replace(
            record,
            serving_tier=ServingTier.WARM.value,
            lifecycle_state="active",
            projection_status=CatalogProjectionStatus.PROJECTED.value,
            updated_at=effective_now,
            metadata=metadata,
        )
        self.catalog_store.upsert_catalog(restored)
        return restored

    def gc_stale_projections(self, *, tenant_id: str) -> RetentionRunResult:
        cursor = ""
        tombstone_ids: list[str] = []
        while True:
            batch = self.catalog_store.scan_catalog_batch(
                after_record_key=cursor,
                filters={"tenant_id": tenant_id, "include_inactive": True},
                limit=self.policy.batch_size,
            )
            if not batch:
                break
            for record in batch:
                cursor = record.record_key
                stale = (
                    record.lifecycle_state in {"deleted", "obsolete"}
                    or record.projection_status == CatalogProjectionStatus.TOMBSTONED.value
                )
                if not stale:
                    continue
                tombstone = self.catalog_store.enqueue_tombstone(
                    tenant_id=tenant_id,
                    record_key=record.record_key,
                    uri=record.uri,
                    reason="retention-stale-projection",
                    source_revision=record.source_revision,
                    payload={
                        "record_kind": record.record_kind,
                        # This manager only tombstones rebuildable serving
                        # rows already marked deleted/obsolete.  It never
                        # deletes their SourceStore or SessionArchive evidence.
                        "gc_safe": True,
                    },
                )
                if str(tombstone.get("status") or "") in {"PENDING", "FAILED", "CLEANING"}:
                    tombstone_ids.append(str(tombstone["tombstone_id"]))
        applied, failed = self._drain_tombstones(tombstone_ids)
        return RetentionRunResult(
            stale_projections=applied,
            vectors_deleted=applied if self.vector_store is not None else 0,
            tombstones_enqueued=len(tombstone_ids),
            tombstones_applied=applied,
            tombstones_failed=failed,
        )

    def gc_vectors(self, *, tenant_id: str) -> RetentionRunResult:
        if self.vector_store is None:
            return RetentionRunResult()
        cursor = ""
        tombstone_ids: list[str] = []
        scanned = 0
        while True:
            batch = self.catalog_store.scan_catalog_batch(
                after_record_key=cursor,
                filters={"tenant_id": tenant_id, "include_inactive": True},
                limit=self.policy.batch_size,
            )
            if not batch:
                break
            for record in batch:
                cursor = record.record_key
                scanned += 1
                if self._retains_vector(record):
                    continue
                metadata_getter = getattr(self.vector_store, "get_vector_metadata", None)
                if callable(metadata_getter) and metadata_getter(self._vector_uri(record)) is None:
                    continue
                tombstone_id = self._enqueue_vector_delete(record, reason="retention-vector-delete")
                if tombstone_id:
                    tombstone_ids.append(tombstone_id)
        applied, failed = self._drain_tombstones(tombstone_ids)
        return RetentionRunResult(
            scanned=scanned,
            vectors_deleted=applied,
            tombstones_enqueued=len(tombstone_ids),
            tombstones_applied=applied,
            tombstones_failed=failed,
        )

    def gc_auxiliary_state(
        self,
        *,
        now: datetime | None = None,
    ) -> RetentionRunResult:
        effective_now = self._utc(now or datetime.now(timezone.utc))
        path_count = self.catalog_store.gc_orphan_paths(limit=self.policy.batch_size)
        tombstone_count = self.catalog_store.gc_applied_tombstones(
            updated_before=(effective_now - self.policy.tombstone_journal_for).isoformat(),
            limit=self.policy.batch_size,
        )
        return RetentionRunResult(
            orphan_paths_deleted=path_count,
            tombstones_deleted=tombstone_count,
        )

    def _vector_uri(self, record: CatalogRecord) -> str:
        row_id = vector_row_id(record.tenant_id, record.record_key)
        if self.vector_store is None:
            return row_id
        metadata_getter = getattr(self.vector_store, "get_vector_metadata", None)
        if not callable(metadata_getter) or metadata_getter(row_id) is not None:
            return row_id
        # Bounded compatibility cleanup for pre-namespace local rows.  Never
        # accept a legacy row owned by another tenant or Catalog record.
        legacy = metadata_getter(record.uri)
        if not isinstance(legacy, Mapping):
            return row_id
        actual_tenant = str(legacy.get("tenant_id") or "")
        actual_key = str(legacy.get("catalog_record_key") or "")
        if actual_tenant and actual_tenant != record.tenant_id:
            return row_id
        if actual_key and actual_key != record.record_key:
            return row_id
        return record.uri

    def _retains_vector(self, record: CatalogRecord) -> bool:
        tier_allows_vector = record.serving_tier == ServingTier.HOT.value or (
            record.serving_tier == ServingTier.WARM.value and self.policy.vectorize_warm
        )
        # Canonical projectors own their deterministic vector eligibility and
        # intentionally do not trust a free-form metadata flag.  CurrentSlot
        # is always HOT; Claim revisions retain vectors only while their
        # lifecycle tier permits it.  Session/generic Context rows continue
        # to use the explicit sanitized ``vector_eligible`` policy bit.
        canonical_vector = record.record_kind in {
            CatalogRecordKind.CURRENT_SLOT.value,
            CatalogRecordKind.CLAIM_REVISION.value,
        }
        return tier_allows_vector and (
            canonical_vector or bool(record.metadata.get("vector_eligible"))
        )

    def _enqueue_vector_delete(self, record: CatalogRecord, *, reason: str) -> str:
        if self.vector_store is None:
            return ""
        metadata_getter = getattr(self.vector_store, "get_vector_metadata", None)
        if callable(metadata_getter) and metadata_getter(self._vector_uri(record)) is None:
            return ""
        identity = self.sanitizer.digest(
            (
                record.tenant_id,
                record.record_key,
                record.updated_at,
                record.serving_tier,
                self._vector_uri(record),
                reason,
            )
        )
        row = self.catalog_store.enqueue_tombstone(
            tenant_id=record.tenant_id,
            # A synthetic key makes this a projection-consumer outbox item;
            # applying it must not delete the live Catalog row.
            record_key=f"vector-gc:{identity}",
            uri="",
            reason=f"{reason}:{record.serving_tier}",
            source_revision=record.source_revision,
            payload={
                "projection_action": "vector_delete",
                "vector_uris": [self._vector_uri(record)],
                "catalog_record_key": record.record_key,
                "expected_source_revision": record.source_revision,
                "expected_projection_effect_hash": record.projection_effect_hash,
                "expected_updated_at": record.updated_at,
                "expected_serving_tier": record.serving_tier,
                "gc_safe": True,
            },
        )
        return str(row["tombstone_id"]) if str(row.get("status") or "") in {"PENDING", "FAILED", "CLEANING"} else ""

    def _drain_tombstones(self, tombstone_ids: Sequence[str]) -> tuple[int, int]:
        targets = tuple(dict.fromkeys(str(item) for item in tombstone_ids if item))
        if not targets:
            return 0, 0
        if self.tombstone_service is None:
            raise RuntimeError("retention cleanup requires a durable tombstone processor")
        exact_processor = getattr(self.tombstone_service, "process_tombstones", None)
        if callable(exact_processor):
            result = exact_processor(targets)
        else:
            result = self.tombstone_service.process_pending(limit=min(1_000, max(len(targets), 1)))
        processed = set(getattr(result, "processed", ()) or ())
        stale = set(getattr(result, "stale", ()) or ())
        failed = set(getattr(result, "failed", ()) or ())
        completed = (processed | stale) & set(targets)
        target_failures = failed & set(targets)
        incomplete = set(targets) - completed - target_failures
        if target_failures or incomplete:
            raise RuntimeError("retention tombstone cleanup is durable but incomplete; retry the retention cycle")
        return len(completed), len(target_failures)

    @staticmethod
    def _safe_path_segment(value: str) -> str:
        # Session ids normally originate in a validated memoryos URI.  A
        # digest fallback keeps compaction paths inside the controlled tree.
        if value and all(character.isalnum() or character in "._:-" for character in value):
            return value[:160]
        return "id-" + ContextProjectionSanitizer().digest(value)[:20]

    @classmethod
    def _reference_time(cls, record: CatalogRecord) -> datetime | None:
        for value in (record.event_time, record.created_at, record.transaction_time):
            if value:
                return cls._parse_time(value)
        return None

    @staticmethod
    def _parse_time(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("retention timestamps must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("retention timestamps must include timezone")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("retention clock must include timezone")
        return value.astimezone(timezone.utc)


__all__ = [
    "CatalogRetentionManager",
    "RetentionPolicy",
    "RetentionRunResult",
]
