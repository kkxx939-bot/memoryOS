"""通过耐久墓碑清理可重建的上下文投影，并支持失败重放。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.contracts.vector import VectorStore, vector_capabilities, vector_row_id
from infrastructure.store.model.catalog import CatalogRecord
from infrastructure.store.model.context.lifecycle import LifecycleState


@dataclass(frozen=True)
class TombstoneRunResult:
    processed: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()


class ProjectionTombstoneService:
    """只有耐久墓碑存在时才删除派生 Serving 状态。

    本服务不物理删除 SourceStore。普通上下文只有进入非服务生命周期后才能清理；
    Session 或事实源 URI 的投影退役可以显式保留不可变证据。
    """

    def __init__(
        self,
        index_store: Any,
        *,
        source_store: SourceStore | None = None,
        vector_store: VectorStore | None = None,
        relation_store: RelationStore | None = None,
    ) -> None:
        required = (
            "enqueue_tombstone",
            "get_pending_tombstones",
            "get_pending_tombstones_for_uri",
            "begin_tombstone_cleanup",
            "finish_tombstone_cleanup",
        )
        if any(not callable(getattr(index_store, name, None)) for name in required):
            raise TypeError("ProjectionTombstoneService requires a durable catalog tombstone store")
        self.index_store = index_store
        self.source_store = source_store
        self.vector_store = vector_store
        self.relation_store = relation_store

    def drain_pending(
        self,
        *,
        tenant_id: str,
        max_rounds: int = 1_000,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """有界重放墓碑，直到当前 tenant 没有待处理项目。"""

        processed = stale = 0
        for _ in range(max_rounds):
            result = self.process_pending(tenant_id=tenant_id, limit=batch_size)
            if result.failed:
                raise RuntimeError("generic projection tombstone replay is incomplete")
            processed += len(result.processed)
            stale += len(result.stale)
            if not result.processed and not result.stale:
                return {"processed": processed, "stale": stale}
        raise RuntimeError("generic projection tombstone replay exceeded its startup bound")

    def enqueue_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        reason: str,
        require_source_retired: bool = True,
    ) -> tuple[str, ...]:
        return self._enqueue_all(
            filters={
                "tenant_id": tenant_id,
                "target_uris": (uri,),
                "include_inactive": True,
            },
            tenant_id=tenant_id,
            reason=reason,
            require_source_retired=require_source_retired,
            fallback_uri=uri,
        )

    def enqueue_source_uri(
        self,
        source_uri: str,
        *,
        tenant_id: str,
        reason: str,
        require_source_retired: bool = False,
    ) -> tuple[str, ...]:
        return self._enqueue_all(
            filters={
                "tenant_id": tenant_id,
                "source_uris": (source_uri,),
                "include_inactive": True,
            },
            tenant_id=tenant_id,
            reason=reason,
            require_source_retired=require_source_retired,
            fallback_uri=source_uri,
        )

    def enqueue_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        reason: str,
    ) -> tuple[str, ...]:
        if not str(session_id).strip():
            raise ValueError("Session projection deletion requires session_id")
        queued = list(
            self._enqueue_all(
                filters={
                    "tenant_id": tenant_id,
                    "session_ids": (session_id,),
                    "include_inactive": True,
                },
                tenant_id=tenant_id,
                reason=reason,
                require_source_retired=False,
            )
        )
        # 记录级墓碑删除当前全部投影；稳定 Session 屏障还会阻止新版投影器在重建时复活证据。
        session_digest = hashlib.sha256(f"{tenant_id}\0{session_id}".encode()).hexdigest()
        barrier = self.index_store.enqueue_tombstone(
            tenant_id=tenant_id,
            record_key=f"session-delete-barrier:{session_digest}",
            reason=reason,
            payload={
                "record_kind": "session_delete_barrier",
                "session_id": str(session_id),
                "gc_safe": False,
            },
        )
        queued.append(str(barrier["tombstone_id"]))
        return tuple(dict.fromkeys(queued))

    def _enqueue_all(
        self,
        *,
        filters: Mapping[str, Any],
        tenant_id: str,
        reason: str,
        require_source_retired: bool,
        fallback_uri: str = "",
    ) -> tuple[str, ...]:
        """返回前使用 Keyset 分页遍历所有匹配投影。

        普通 Catalog 查询单次最多返回 1000 条；删除流程不能继承这个限制，否则大型
        Session 会在仍有可检索记录时被错误报告为已经删除。
        """

        scanner = getattr(self.index_store, "scan_catalog_batch", None)
        if not callable(scanner):
            raise TypeError("projection cleanup requires keyset-paginated catalog scanning")
        queued: list[str] = []
        if fallback_uri:
            after_tombstone_id = ""
            while True:
                unfinished = self.index_store.get_pending_tombstones_for_uri(
                    fallback_uri,
                    tenant_id=tenant_id,
                    after_tombstone_id=after_tombstone_id,
                    limit=1_000,
                )
                if not isinstance(unfinished, Sequence) or isinstance(unfinished, str | bytes):
                    raise TypeError("projection cleanup URI journal returned an invalid batch")
                if not unfinished:
                    break
                for row in unfinished:
                    if not isinstance(row, Mapping) or not str(row.get("tombstone_id") or ""):
                        raise TypeError("projection cleanup URI journal returned an invalid row")
                    queued.append(str(row["tombstone_id"]))
                next_tombstone_id = str(unfinished[-1]["tombstone_id"])
                if next_tombstone_id <= after_tombstone_id:
                    raise RuntimeError("projection cleanup URI journal pagination did not advance")
                after_tombstone_id = next_tombstone_id
        after_record_key = ""
        while True:
            raw_records: Any = scanner(
                tenant_id=tenant_id,
                after_record_key=after_record_key,
                filters=filters,
                limit=1_000,
            )
            if not isinstance(raw_records, Sequence) or isinstance(raw_records, str | bytes):
                raise TypeError("catalog keyset scanner returned an invalid batch")
            records = tuple(record for record in raw_records if isinstance(record, CatalogRecord))
            if len(records) != len(raw_records):
                raise TypeError("catalog keyset scanner returned a non-Catalog record")
            if not records:
                break
            queued.extend(
                self._enqueue_records(
                    records,
                    tenant_id=tenant_id,
                    reason=reason,
                    require_source_retired=require_source_retired,
                )
            )
            next_record_key = str(records[-1].record_key)
            if next_record_key <= after_record_key:
                raise RuntimeError("catalog keyset pagination did not advance")
            after_record_key = next_record_key
        queued = list(dict.fromkeys(queued))
        if not queued and fallback_uri:
            digest = hashlib.sha256(f"{tenant_id}\0{fallback_uri}".encode()).hexdigest()
            row = self.index_store.enqueue_tombstone(
                tenant_id=tenant_id,
                record_key=f"orphan-projection:{digest}",
                uri=fallback_uri,
                reason=reason,
                payload={
                    "source_uri": fallback_uri,
                    "projection_uri": fallback_uri,
                    "record_kind": "orphan_projection_cleanup",
                    "require_source_retired": require_source_retired,
                },
            )
            queued.append(str(row["tombstone_id"]))
        return tuple(queued)

    def _enqueue_records(
        self,
        records: Sequence[Any],
        *,
        tenant_id: str,
        reason: str,
        require_source_retired: bool,
    ) -> tuple[str, ...]:
        queued: list[str] = []
        for record in records:
            if not isinstance(record, CatalogRecord):
                raise TypeError("projection cleanup received a non-Catalog record")
            if record.tenant_id != tenant_id:
                raise PermissionError("projection cleanup crossed the requested tenant boundary")
            row = self.index_store.enqueue_tombstone(
                tenant_id=tenant_id,
                record_key=record.record_key,
                uri=record.uri,
                reason=reason,
                source_revision=record.source_revision,
                payload={
                    "source_uri": record.source_uri,
                    "projection_uri": record.uri,
                    "record_kind": record.record_kind,
                    "require_source_retired": require_source_retired,
                    "expected_source_digest": record.source_digest,
                    "expected_projection_effect_hash": record.projection_effect_hash,
                    "expected_updated_at": record.updated_at,
                },
            )
            queued.append(str(row["tombstone_id"]))
        return tuple(queued)

    def process_pending(self, *, tenant_id: str, limit: int = 100) -> TombstoneRunResult:
        rows = self.index_store.get_pending_tombstones(tenant_id=tenant_id, limit=limit)
        return self._process_rows(rows, tenant_id=tenant_id)

    def process_tombstones(
        self,
        tombstone_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> TombstoneRunResult:
        """只执行一次公开删除请求创建的精确墓碑集合。"""

        requested = tuple(dict.fromkeys(str(item) for item in tombstone_ids if str(item)))
        if not requested:
            return TombstoneRunResult()
        getter = getattr(self.index_store, "get_tombstones", None)
        if not callable(getter):
            raise TypeError("projection cleanup requires exact durable tombstone reads")
        raw_rows: Any = getter(requested, tenant_id=tenant_id)
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, str | bytes):
            raise TypeError("durable tombstone store returned an invalid batch")
        rows = tuple(row for row in raw_rows if isinstance(row, Mapping))
        if len(rows) != len(raw_rows):
            raise TypeError("durable tombstone store returned an invalid row")
        found = {str(row.get("tombstone_id") or "") for row in rows}
        missing = tuple(tombstone_id for tombstone_id in requested if tombstone_id not in found)
        if missing:
            return TombstoneRunResult(failed=missing)
        return self._process_rows(rows, tenant_id=tenant_id)

    def _process_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        tenant_id: str,
    ) -> TombstoneRunResult:
        processed: list[str] = []
        failed: list[str] = []
        stale: list[str] = []
        for row in rows:
            if self._row_tenant(row) != tenant_id:
                raise PermissionError("tombstone batch crossed the requested tenant boundary")
            tombstone_id = str(row.get("tombstone_id") or "")
            cleanup_started = str(row.get("status") or "") == "CLEANING"
            try:
                self._validate_source_boundary(row)
                begun = self.index_store.begin_tombstone_cleanup(
                    tombstone_id,
                    tenant_id=tenant_id,
                )
                if not isinstance(begun, Mapping):
                    raise RuntimeError("durable tombstone disappeared before cleanup")
                status = str(begun.get("status") or "")
                if status == "STALE":
                    stale.append(tombstone_id)
                    continue
                if status == "APPLIED":
                    processed.append(tombstone_id)
                    continue
                if status != "CLEANING":
                    raise RuntimeError(f"unexpected tombstone cleanup status: {status}")
                cleanup_started = True
                row = begun
                self._delete_vector(row)
                self._delete_relations(row)
                applied = self.index_store.finish_tombstone_cleanup(
                    tombstone_id,
                    tenant_id=tenant_id,
                )
                if not isinstance(applied, Mapping):
                    raise RuntimeError("durable tombstone disappeared during application")
                status = str(applied.get("status") or "")
                if status == "STALE":
                    stale.append(tombstone_id)
                elif status == "APPLIED":
                    processed.append(tombstone_id)
                else:
                    raise RuntimeError(f"unexpected tombstone terminal status: {status}")
            except Exception as exc:
                marker_name = "mark_tombstone_cleanup_failed" if cleanup_started else "mark_tombstone_failed"
                marker = getattr(self.index_store, marker_name, None)
                if callable(marker):
                    marker(
                        tombstone_id,
                        f"{type(exc).__name__}: {exc}",
                        tenant_id=tenant_id,
                    )
                failed.append(tombstone_id)
        return TombstoneRunResult(tuple(processed), tuple(failed), tuple(stale))

    def _validate_source_boundary(self, row: Mapping[str, Any]) -> None:
        payload = dict(row.get("payload_json") or {})
        if not payload.get("require_source_retired") or self.source_store is None:
            return
        source_uri = str(payload.get("source_uri") or row.get("uri") or "")
        if not source_uri:
            raise RuntimeError("tombstone has no Source identity")
        try:
            obj = self.source_store.read_object(source_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return
        if obj.lifecycle_state not in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            raise RuntimeError("Source remains active; projection delete is not eligible")

    def _delete_vector(self, row: Mapping[str, Any]) -> None:
        if self.vector_store is None:
            return
        payload = dict(row.get("payload_json") or {})
        if payload.get("projection_action") == "vector_delete":
            if not self._vector_delete_is_current(row, payload):
                return
            raw_uris = payload.get("vector_uris")
            if not isinstance(raw_uris, Sequence) or isinstance(raw_uris, str | bytes):
                raise ValueError("vector-delete tombstone requires a bounded URI list")
            uris = tuple(dict.fromkeys(str(value) for value in raw_uris if value))
            if not uris or len(uris) > 16:
                raise ValueError("vector-delete tombstone URI list is empty or unbounded")
            catalog_record_key = str(payload.get("catalog_record_key") or "")
            tenant_id = self._row_tenant(row)
            expected_row_id = vector_row_id(tenant_id, catalog_record_key)
            if any(uri != expected_row_id for uri in uris):
                raise ValueError("vector-delete tombstone contains a non-Catalog row ID")
            if self._vector_delete_matches(
                row,
                expected_row_id,
                expected_record_key=catalog_record_key,
                expected_revision=int(payload.get("expected_source_revision") or 0),
                expected_effect=str(payload.get("expected_projection_effect_hash") or ""),
            ):
                self.vector_store.delete_vector(expected_row_id)
            return
        record_key = str(row.get("record_key") or "")
        tenant_id = self._row_tenant(row)
        if not record_key:
            raise ValueError("projection tombstone has no Catalog vector identity")
        if str(payload.get("record_kind") or "") == "orphan_projection_cleanup":
            self._delete_orphan_vectors(row, payload)
            return
        row_id = vector_row_id(tenant_id, record_key)
        if self._vector_delete_matches(row, row_id):
            self.vector_store.delete_vector(row_id)

    def _delete_orphan_vectors(
        self,
        row: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        """Catalog 键消失后，根据可信元数据删除孤立向量。

        租户限定的向量行 ID 来自原始 Catalog 键；Catalog 记录消失后不能构造替代键。
        后端必须支持精确元数据删除，否则墓碑保持失败可重试，不能伪报成功。
        """

        assert self.vector_store is not None
        capabilities = vector_capabilities(self.vector_store)
        deleter = getattr(self.vector_store, "delete_by_filter", None)
        if not capabilities.supports_delete_by_filter or not callable(deleter):
            raise RuntimeError("orphan vector cleanup requires exact delete-by-filter capability")
        tenant_id = self._row_tenant(row)
        catalog_record_key = str(payload.get("catalog_record_key") or "")
        source_uri = str(payload.get("source_uri") or row.get("uri") or "")
        if catalog_record_key:
            selector = {"tenant_id": tenant_id, "catalog_record_key": catalog_record_key}
        elif source_uri:
            selector = {"tenant_id": tenant_id, "source_uri": source_uri}
        else:
            raise ValueError("orphan vector cleanup requires a durable projection identity")
        deleted = deleter(selector)
        if not isinstance(deleted, int) or isinstance(deleted, bool) or deleted < 0:
            raise TypeError("vector delete-by-filter returned an invalid deletion count")

    def _delete_relations(self, row: Mapping[str, Any]) -> None:
        payload = dict(row.get("payload_json") or {})
        if payload.get("projection_action") == "vector_delete":
            return
        if self.relation_store is None:
            return
        uris = tuple(
            dict.fromkeys(
                str(value)
                for value in (row.get("uri"), payload.get("projection_uri"), payload.get("source_uri"))
                if value
            )
        )
        orphan_cleanup = str(payload.get("record_kind") or "") == "orphan_projection_cleanup"
        for uri in uris:
            deleter_name = "delete_uri_relations" if orphan_cleanup else "delete_projection_relations"
            deleter = getattr(self.relation_store, deleter_name, None)
            if not callable(deleter):
                raise TypeError("relation cleanup requires bounded projection ownership deletion")
            while True:
                kwargs: dict[str, Any] = {
                    "tenant_id": self._row_tenant(row),
                    "limit": 1_000,
                }
                if not orphan_cleanup:
                    kwargs["catalog_record_key"] = str(row.get("record_key") or "")
                deleted = deleter(uri, **kwargs)
                if not isinstance(deleted, int) or isinstance(deleted, bool):
                    raise TypeError("relation cleanup returned an invalid deletion count")
                if deleted < 0 or deleted > 1_000:
                    raise RuntimeError("relation cleanup exceeded its bounded batch")
                if deleted == 0:
                    break

    def _vector_delete_matches(
        self,
        row: Mapping[str, Any],
        uri: str,
        *,
        expected_record_key: str = "",
        expected_revision: int | None = None,
        expected_effect: str = "",
    ) -> bool:
        """比较身份后，只删除属于当前 Catalog 修订的向量。"""

        if self.vector_store is None:
            return False
        getter = getattr(self.vector_store, "get_vector_metadata", None)
        if not callable(getter):
            return True
        metadata_value = getter(uri)
        if metadata_value is None:
            return False
        if not isinstance(metadata_value, Mapping):
            return False
        metadata = metadata_value
        expected_record_key = expected_record_key or str(row.get("record_key") or "")
        actual_record_key = str(metadata.get("catalog_record_key") or "")
        if not expected_record_key or actual_record_key != expected_record_key:
            return False
        actual_tenant = str(metadata.get("tenant_id") or "")
        expected_tenant = self._row_tenant(row)
        if actual_tenant != expected_tenant:
            return False
        expected_revision = (
            int(row.get("source_revision") or 0) if expected_revision is None else int(expected_revision)
        )
        try:
            actual_revision = int(metadata.get("source_revision") or 0)
        except (TypeError, ValueError):
            return False
        if expected_revision and actual_revision > expected_revision:
            return False
        payload = dict(row.get("payload_json") or {})
        expected_effect = expected_effect or str(payload.get("expected_projection_effect_hash") or "")
        actual_effect = str(metadata.get("projection_effect_hash") or "")
        if expected_effect and actual_effect and actual_effect != expected_effect:
            return False
        return True

    def _vector_delete_is_current(
        self,
        row: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        """冷热层级发生竞争时把旧墓碑按空操作消费，不能删除已恢复向量。"""

        catalog_record_key = str(payload.get("catalog_record_key") or "")
        if not catalog_record_key:
            raise ValueError("vector-delete tombstone has no Catalog identity")
        getter = getattr(self.index_store, "get_catalog", None)
        if not callable(getter):
            raise TypeError("vector-delete tombstone requires exact Catalog reads")
        current = getter(
            catalog_record_key,
            tenant_id=self._row_tenant(row),
        )
        if current is None:
            # Catalog 记录被其他耐久墓碑删除后，孤立向量仍需继续清理。
            return True
        if not isinstance(current, CatalogRecord):
            raise TypeError("vector-delete tombstone Catalog read returned an invalid record")
        expected_updated_at = str(payload.get("expected_updated_at") or "")
        expected_serving_tier = str(payload.get("expected_serving_tier") or "")
        return bool(current.updated_at == expected_updated_at and current.serving_tier == expected_serving_tier)

    @staticmethod
    def _row_tenant(row: Mapping[str, Any]) -> str:
        tenant_id = str(row.get("tenant_id") or "").strip()
        if not tenant_id:
            raise ValueError("tombstone row is missing tenant_id")
        return tenant_id


__all__ = ["ProjectionTombstoneService", "TombstoneRunResult"]
