"""Committed visibility backed by an immutable receipt and atomic current head."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.source_store import (
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    artifact_root_for,
    load_current_head,
    load_current_head_member,
    receipt_history_contains_uri,
    receipt_history_effects,
    snapshot_object,
    validate_current_head_set,
    validate_current_head_set_path,
)
from memoryos.memory.canonical.event import canonical_digest, canonical_json
from memoryos.operations.commit.effect_marker import marker_proves_relation, normalized_relation
from memoryos.operations.commit.receipt import ReceiptIntegrityError, receipt_snapshot
from memoryos.operations.commit.redo_log import RedoControlFileError, RedoLog


@dataclass(frozen=True)
class CommittedCanonicalRead:
    object: ContextObject
    content_override: str | None = None
    from_before_image: bool = False
    head: dict | None = None
    receipt: dict | None = None


@dataclass(frozen=True)
class CommittedCanonicalSnapshot:
    """One query-local immutable mapping captured from atomic head-sets."""

    records: Mapping[str, CommittedCanonicalRead]

    def get(self, uri: str) -> CommittedCanonicalRead | None:
        return self.records.get(uri)

    @property
    def uris(self) -> tuple[str, ...]:
        return tuple(self.records)


class CommittedStateIntegrityError(RuntimeError):
    """Live Source diverged from its head without a valid in-flight redo proof."""


def committed_content(committed: CommittedCanonicalRead) -> str:
    """Return L2 bytes from the immutable snapshot selected by the head."""

    if committed.content_override is not None:
        return committed.content_override
    if committed.receipt is None:
        raise FileNotFoundError(f"committed content receipt is missing: {committed.object.uri}")
    try:
        snapshot = receipt_snapshot(committed.receipt, committed.object.uri)
    except ReceiptIntegrityError as exc:
        raise FileNotFoundError(f"committed content snapshot is missing: {committed.object.uri}") from exc
    content = snapshot.get("content")
    if not isinstance(content, str):
        raise FileNotFoundError(f"committed content is invalid: {committed.object.uri}")
    return content


def committed_relations(
    committed: CommittedCanonicalRead,
    *,
    source_uri: str | None = None,
) -> tuple[ContextRelation, ...]:
    """Return formal outgoing relations from the immutable current receipt.

    RelationStore is a rebuildable projection.  Business reads must not expose
    an in-flight or corrupt row from that store, so the current receipt is the
    sole relation source of truth.
    """

    if committed.receipt is None:
        return tuple(committed.object.relations)
    wanted = source_uri or committed.object.uri
    try:
        snapshot = receipt_snapshot(committed.receipt, committed.object.uri)
    except ReceiptIntegrityError as exc:
        raise FileNotFoundError(f"committed relation snapshot is missing: {committed.object.uri}") from exc
    raw = snapshot.get("relation_snapshot")
    if not isinstance(raw, list):
        raise FileNotFoundError(f"committed relation snapshot is invalid: {committed.object.uri}")
    relations: list[ContextRelation] = []
    for item in raw:
        if not isinstance(item, dict) or str(item.get("source_uri") or "") != wanted:
            continue
        relations.append(
            ContextRelation(
                source_uri=str(item.get("source_uri") or ""),
                relation_type=str(item.get("relation_type") or item.get("type") or ""),
                target_uri=str(item.get("target_uri") or ""),
                weight=float(item.get("weight", 1.0)),
                metadata=dict(item.get("metadata", {}) or {}),
                created_at="",
            )
        )
    return tuple(sorted(relations, key=lambda item: canonical_json(normalized_relation(item))))


def _canonical_uri(uri: str) -> bool:
    return is_canonical_memory_uri(uri)


def _live_state_is_proved_inflight(
    source_store: SourceStore,
    artifact_root: object,
    uri: str,
    live: ContextObject | None,
    live_content: str,
    *,
    transaction_id: str = "",
) -> bool:
    """Accept a Source-ahead state only when one intact redo owns that exact effect."""

    if live is None:
        return False
    try:
        entries = RedoLog(artifact_root).pending_entries()  # type: ignore[arg-type]
    except (RedoControlFileError, OSError, ValueError):
        return False
    pre_head_phases = {
        "begin",
        "started",
        "source_written",
        "index_written",
        "audit_written",
        "diff_written",
    }
    head_published_transactions = {
        str(entry.operation.payload.get("transaction_id") or entry.operation.operation_id)
        for entry in entries
        if entry.phase == "head_published"
    }
    tenant_id = str(getattr(source_store, "tenant_id", "default"))
    for entry in entries:
        operation = entry.operation
        operation_transaction_id = str(operation.payload.get("transaction_id") or operation.operation_id)
        if entry.phase not in pre_head_phases or operation_transaction_id in head_published_transactions:
            continue
        if str(operation.target_uri or "") != uri:
            continue
        if transaction_id and operation_transaction_id != transaction_id:
            continue
        if str(operation.payload.get("tenant_id") or "default") != tenant_id:
            continue
        raw = operation.payload.get("context_object")
        if not isinstance(raw, dict):
            continue
        try:
            desired = ContextObject.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            continue
        if canonical_digest(desired.to_dict()) == canonical_digest(live.to_dict()) and canonical_digest(
            str(operation.payload.get("content", ""))
        ) == canonical_digest(live_content):
            return True
    return False


def _read_live_bundle(source_store: SourceStore, uri: str) -> tuple[ContextObject | None, str]:
    try:
        live = source_store.read_object(uri)
        return live, source_store.read_content(live.layers.l2_uri or live.uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, RuntimeError, ValueError):
        return None, ""


def _validate_head_enumeration_coverage(
    source_store: SourceStore,
    artifact_root: Any,
    captured_heads: Mapping[str, dict[str, Any]],
    *,
    kinds: Iterable[str],
) -> None:
    """Prove that head enumeration covers immutable committed history.

    A legal concurrent transaction may publish a receipt immediately before
    its head.  In that window an existing prior head remains the committed
    snapshot; a first revision remains invisible only while an intact redo
    proves the exact live bundle.  Without either proof, missing or stale
    enumeration is an integrity failure, never an empty search result.
    """

    try:
        history = receipt_history_effects(artifact_root, kinds=kinds)
    except CurrentHeadIntegrityError as exc:
        _fail_head_integrity(
            source_store,
            str(artifact_root),
            f"immutable receipt history is invalid during current-head enumeration: {exc}",
        )
    for uri, rows in history.items():
        latest = rows[-1]
        captured = captured_heads.get(uri)
        if captured is not None and (
            int(captured.get("current_revision", -1)) == int(latest["after_revision"])
            and str(captured.get("receipt_digest") or "") == str(latest["receipt_digest"])
        ):
            continue

        # The head may have advanced after the query captured its immutable
        # snapshot.  Accept that race without replacing the query-local view.
        try:
            current, _receipt, _snapshot = load_current_head(artifact_root, uri)
        except FileNotFoundError:
            current = None
        except CurrentHeadIntegrityError as exc:
            _fail_head_integrity(
                source_store,
                uri,
                f"canonical current head is invalid during enumeration: {uri}: {exc}",
            )
        if current is not None and (
            int(current.get("current_revision", -1)) == int(latest["after_revision"])
            and str(current.get("receipt_digest") or "") == str(latest["receipt_digest"])
        ):
            continue

        live, live_content = _read_live_bundle(source_store, uri)
        inflight = _live_state_is_proved_inflight(
            source_store,
            artifact_root,
            uri,
            live,
            live_content,
            transaction_id=str(latest["transaction_id"]),
        )
        if inflight:
            if captured is None and int(latest["before_revision"]) == 0:
                # First publication has no earlier committed state to return.
                continue
            if captured is not None and len(rows) >= 2:
                previous = rows[-2]
                if int(captured.get("current_revision", -1)) == int(previous["after_revision"]) and str(
                    captured.get("receipt_digest") or ""
                ) == str(previous["receipt_digest"]):
                    continue
        _fail_head_integrity(
            source_store,
            uri,
            f"required current head is missing or stale during enumeration: {uri}",
        )


def _fail_live_integrity(source_store: SourceStore, uri: str) -> NoReturn:
    reason = f"current head and live Source bundle disagree without an in-flight redo proof: {uri}"
    readiness = getattr(source_store, "readiness", None)
    mark_not_ready = getattr(readiness, "mark_not_ready", None)
    if callable(mark_not_ready):
        mark_not_ready(reason, details={"uri": uri, "artifact": "canonical_source"})
    raise CommittedStateIntegrityError(reason)


def _fail_head_integrity(source_store: SourceStore, uri: str, reason: str) -> NoReturn:
    readiness = getattr(source_store, "readiness", None)
    mark_not_ready = getattr(readiness, "mark_not_ready", None)
    if callable(mark_not_ready):
        mark_not_ready(reason, details={"uri": uri, "artifact": "canonical_current_head"})
    raise CommittedStateIntegrityError(reason)


def read_committed_canonical(
    source_store: SourceStore,
    uri: str,
    relation_store: RelationStore | None = None,
) -> CommittedCanonicalRead:
    """Return exactly the state selected by the current head.

    If Source publication is ahead of the atomic head-set (or a bundle is
    temporarily unavailable), the immutable snapshot referenced by the old
    head remains visible.  Historical receipts are never checked against the
    latest Source state.
    """

    del relation_store  # relation proof is independently validated below.
    artifact_root = artifact_root_for(source_store)
    if artifact_root is None:
        if _canonical_uri(uri):
            raise FileNotFoundError(f"canonical object has no tenant artifact root: {uri}")
        obj = source_store.read_object(uri)
        if is_canonical_memory_object(obj):
            raise FileNotFoundError(f"canonical object has no tenant artifact root: {uri}")
        return CommittedCanonicalRead(obj)
    try:
        head, receipt, snapshot = load_current_head(artifact_root, uri)
    except FileNotFoundError:
        if _canonical_uri(uri):
            live: ContextObject | None = None
            live_content = ""
            try:
                live = source_store.read_object(uri)
                live_content = source_store.read_content(live.layers.l2_uri or live.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, RuntimeError, ValueError):
                live = None
            if live is not None and _live_state_is_proved_inflight(
                source_store,
                artifact_root,
                uri,
                live,
                live_content,
            ):
                raise FileNotFoundError(f"canonical object is not committed yet: {uri}") from None
            try:
                was_committed = receipt_history_contains_uri(artifact_root, uri)
            except CurrentHeadIntegrityError as exc:
                _fail_head_integrity(source_store, uri, f"immutable receipt history is invalid: {uri}: {exc}")
            if was_committed:
                _fail_head_integrity(source_store, uri, f"required current head is missing: {uri}")
            raise FileNotFoundError(
                f"canonical object is not committed: no committed transaction proof/current head: {uri}"
            ) from None
        obj = source_store.read_object(uri)
        if is_canonical_memory_object(obj):
            raise FileNotFoundError(f"canonical object has no current head transaction proof: {uri}") from None
        return CommittedCanonicalRead(obj)
    except CurrentHeadIntegrityError as exc:
        _fail_head_integrity(source_store, uri, f"canonical object current proof is invalid: {uri}: {exc}")

    return _read_committed_from_proof(
        source_store,
        artifact_root,
        uri,
        head,
        receipt,
        snapshot,
        captured=False,
    )


def _read_committed_from_proof(
    source_store: SourceStore,
    artifact_root: Any,
    uri: str,
    head: dict[str, Any],
    receipt: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    captured: bool,
) -> CommittedCanonicalRead:
    committed_obj = snapshot_object(snapshot)
    if str(committed_obj.tenant_id or "default") != str(head["tenant_id"]):
        raise FileNotFoundError(f"canonical receipt snapshot crosses tenant boundary: {uri}")
    if str(committed_obj.owner_user_id or "") != str(head["owner_user_id"]):
        raise FileNotFoundError(f"canonical receipt snapshot crosses owner boundary: {uri}")
    snapshot_content = str(snapshot.get("content", ""))
    try:
        live = source_store.read_object(uri)
        live_content = source_store.read_content(live.layers.l2_uri or live.uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, RuntimeError, ValueError):
        live = None
        live_content = ""
    live_matches = bool(
        live is not None
        and canonical_digest(live.to_dict()) == head["object_digest"]
        and canonical_digest(live_content) == head["content_digest"]
        and canonical_digest(sorted((relation.to_dict() for relation in live.relations), key=canonical_json))
        == head["bundle_relation_digest"]
    )
    if live_matches:
        return CommittedCanonicalRead(committed_obj, head=head, receipt=receipt)
    if captured:
        try:
            latest_head, _latest_receipt, _latest_snapshot = load_current_head(artifact_root, uri)
        except (FileNotFoundError, CurrentHeadIntegrityError):
            latest_head = None
        if latest_head is not None and latest_head.get("head_digest") != head.get("head_digest"):
            return CommittedCanonicalRead(
                committed_obj,
                content_override=snapshot_content,
                from_before_image=True,
                head=head,
                receipt=receipt,
            )
    if not _live_state_is_proved_inflight(source_store, artifact_root, uri, live, live_content):
        _fail_live_integrity(source_store, uri)
    return CommittedCanonicalRead(
        committed_obj,
        content_override=snapshot_content,
        from_before_image=True,
        head=head,
        receipt=receipt,
    )


def capture_committed_canonical_snapshot(
    source_store: SourceStore,
    relation_store: RelationStore | None = None,
    *,
    kinds: Iterable[str] = ("slot", "claim", "pending_proposal"),
) -> CommittedCanonicalSnapshot:
    """Capture every selected head-set once and validate its committed effects.

    Subsequent query stages use only this mapping.  A concurrent legal head
    advance may make live Source newer than the captured proof; in that case
    the immutable captured receipt snapshot remains the query result.
    """

    del relation_store
    artifact_root = artifact_root_for(source_store)
    if artifact_root is None:
        return CommittedCanonicalSnapshot(MappingProxyType({}))
    requested = set(kinds)
    head_root = artifact_root / "system" / "current-heads"
    records: dict[str, CommittedCanonicalRead] = {}
    for path in sorted(head_root.glob("*.json")) if head_root.exists() else ():
        try:
            head_set = validate_current_head_set(json.loads(path.read_text(encoding="utf-8")))
            validate_current_head_set_path(artifact_root, path, head_set)
        except (OSError, UnicodeError, json.JSONDecodeError, CurrentHeadIntegrityError) as exc:
            _fail_head_integrity(
                source_store,
                str(path),
                f"canonical snapshot contains an invalid current head-set: {path.name}: {exc}",
            )
        for uri, raw_head in dict(head_set["heads"]).items():
            kind = str(dict(raw_head).get("canonical_kind") or "")
            if requested and kind not in requested:
                continue
            if uri in records:
                _fail_head_integrity(
                    source_store,
                    str(uri),
                    f"canonical URI appears in multiple current head-sets: {uri}",
                )
            try:
                head, receipt, effect_snapshot = load_current_head_member(
                    artifact_root,
                    head_set,
                    str(uri),
                )
            except (FileNotFoundError, CurrentHeadIntegrityError) as exc:
                _fail_head_integrity(
                    source_store,
                    str(uri),
                    f"canonical snapshot current proof is invalid: {uri}: {exc}",
                )
            records[str(uri)] = _read_committed_from_proof(
                source_store,
                artifact_root,
                str(uri),
                head,
                receipt,
                effect_snapshot,
                captured=True,
            )
    _validate_head_enumeration_coverage(
        source_store,
        artifact_root,
        {uri: dict(record.head or {}) for uri, record in records.items()},
        kinds=requested,
    )
    return CommittedCanonicalSnapshot(MappingProxyType(records))


def read_committed_pending(
    source_store: SourceStore,
    uri: str,
    relation_store: RelationStore | None = None,
) -> CommittedCanonicalRead:
    committed = read_committed_canonical(source_store, uri, relation_store)
    if dict(committed.object.metadata or {}).get("canonical_kind") != "pending_proposal":
        raise FileNotFoundError(f"URI is not a committed pending proposal: {uri}")
    return committed


def list_committed_canonical(
    source_store: SourceStore,
    relation_store: RelationStore | None = None,
    *,
    kinds: Iterable[str] = ("slot", "claim", "pending_proposal"),
) -> tuple[CommittedCanonicalRead, ...]:
    snapshot = capture_committed_canonical_snapshot(
        source_store,
        relation_store,
        kinds=kinds,
    )
    return tuple(snapshot.records.values())


def reconcile_committed_relation_store(
    source_store: SourceStore,
    relation_store: RelationStore,
) -> dict[str, int]:
    """Rebuild canonical outgoing rows from current receipt/head proofs.

    The relation database is disposable.  Missing, stale, or partially
    published canonical rows are deterministically replaced without changing
    any immutable receipt or current head.
    """

    records = list_committed_canonical(source_store, relation_store)
    tenant_id = str(getattr(source_store, "tenant_id", None) or "default")
    if any(str(record.object.tenant_id or "default") != tenant_id for record in records):
        raise RuntimeError("canonical relation reconcile Source snapshot crosses its tenant")
    expected_by_source: dict[str, dict[tuple[str, str, str], ContextRelation]] = {}
    for committed in records:
        source_uri = committed.object.uri
        expected_by_source[source_uri] = {
            (relation.source_uri, relation.relation_type, relation.target_uri): relation
            for relation in committed_relations(committed)
        }

    deleted = 0
    written = 0
    enumerate_relations = getattr(relation_store, "all_relations", None)
    raw_relations = enumerate_relations() if callable(enumerate_relations) else ()
    all_relations = raw_relations if isinstance(raw_relations, Iterable) else ()
    for relation in all_relations:
        if str(relation.metadata.get("tenant_id") or "default") != tenant_id:
            continue
        if _canonical_uri(relation.source_uri) and relation.source_uri not in expected_by_source:
            relation_store.delete_relation(
                relation.source_uri,
                relation.relation_type,
                relation.target_uri,
                tenant_id=tenant_id,
            )
            deleted += 1
    for source_uri, expected in expected_by_source.items():
        actual = [
            relation
            for relation in relation_store.relations_of(source_uri, tenant_id=tenant_id)
            if relation.source_uri == source_uri
        ]
        actual_by_identity: dict[tuple[str, str, str], list[ContextRelation]] = {}
        for relation in actual:
            identity = (relation.source_uri, relation.relation_type, relation.target_uri)
            actual_by_identity.setdefault(identity, []).append(relation)
        for identity, rows in actual_by_identity.items():
            wanted = expected.get(identity)
            if (
                wanted is None
                or len(rows) != 1
                or canonical_json(normalized_relation(rows[0])) != canonical_json(normalized_relation(wanted))
            ):
                relation_store.delete_relation(*identity, tenant_id=tenant_id)
                deleted += len(rows)
        for identity, wanted in expected.items():
            current = [
                relation
                for relation in relation_store.relations_of(source_uri, tenant_id=tenant_id)
                if (relation.source_uri, relation.relation_type, relation.target_uri) == identity
            ]
            if len(current) == 1 and canonical_json(normalized_relation(current[0])) == canonical_json(
                normalized_relation(wanted)
            ):
                continue
            if current:
                relation_store.delete_relation(*identity, tenant_id=tenant_id)
                deleted += len(current)
            relation_store.add_relation(wanted)
            written += 1

    # A second pass is an integrity assertion, not best-effort cleanup.
    for source_uri, expected in expected_by_source.items():
        actual_by_key = {
            (relation.source_uri, relation.relation_type, relation.target_uri): relation
            for relation in relation_store.relations_of(source_uri, tenant_id=tenant_id)
            if relation.source_uri == source_uri
        }
        if set(actual_by_key) != set(expected) or any(
            canonical_json(normalized_relation(actual_by_key[identity])) != canonical_json(normalized_relation(wanted))
            for identity, wanted in expected.items()
        ):
            raise RuntimeError(f"canonical RelationStore cannot be reconciled from current proof: {source_uri}")
    return {
        "objects": len(expected_by_source),
        "deleted": deleted,
        "written": written,
    }


def list_committed_relations(
    source_store: SourceStore,
    uri: str,
    relation_store: RelationStore | None = None,
) -> tuple[ContextRelation, ...]:
    """Enumerate receipt-proved canonical relations incident to one URI."""

    by_identity: dict[tuple[str, str, str], ContextRelation] = {}
    for committed in list_committed_canonical(source_store, relation_store):
        for relation in committed_relations(committed):
            if relation.source_uri != uri and relation.target_uri != uri:
                continue
            identity = (relation.source_uri, relation.relation_type, relation.target_uri)
            existing = by_identity.get(identity)
            if existing is not None and canonical_json(normalized_relation(existing)) != canonical_json(
                normalized_relation(relation)
            ):
                raise RuntimeError(f"committed canonical relation has a current fork: {identity}")
            by_identity[identity] = relation
    return tuple(by_identity[identity] for identity in sorted(by_identity))


def relation_is_committed(
    source_store: SourceStore,
    relation: ContextRelation,
    relation_store: RelationStore | None = None,
) -> bool:
    """Prove a relation from the current source head's immutable receipt."""

    del relation_store
    artifact_root = artifact_root_for(source_store)
    if artifact_root is None:
        return False
    candidates = (relation.source_uri, relation.target_uri)
    for uri in candidates:
        if not _canonical_uri(uri):
            continue
        try:
            head, receipt, snapshot = load_current_head(artifact_root, uri)
        except (FileNotFoundError, CurrentHeadIntegrityError):
            continue
        if not marker_proves_relation(receipt, relation):
            continue
        relations = snapshot.get("relation_snapshot", [])
        if not isinstance(relations, list):
            return False
        # The per-object relation snapshot and head digest form the current
        # proof even while the disposable RelationStore is being recovered.
        if canonical_digest(sorted(relations, key=canonical_json)) != head["relation_digest"]:
            return False
        return True
    return False
