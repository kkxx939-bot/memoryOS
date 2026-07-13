"""Atomic current-head sets for committed canonical and pending memory."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.path_safety import DurablePathIntegrityError, require_safe_artifact_path
from memoryos.memory.canonical.event import canonical_digest
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    ReceiptIntegrityError,
    load_transaction_receipt,
    receipt_snapshot,
)

CURRENT_HEAD_SCHEMA_VERSION = "canonical_current_head_v1"
CURRENT_HEAD_SET_SCHEMA_VERSION = "canonical_current_head_set_v1"


class CurrentHeadIntegrityError(RuntimeError):
    """Current state cannot be proven by its head and immutable receipt."""


class CurrentHeadConflictError(CurrentHeadIntegrityError):
    """A receipt would rewind or fork an already published current head."""


def artifact_root_for(source_store: Any) -> Path | None:
    root = getattr(source_store, "root", None)
    tenant_id = getattr(source_store, "tenant_id", "default")
    if root is None or not isinstance(tenant_id, str) or not tenant_id.strip():
        return None
    if tenant_id in {".", ".."} or "/" in tenant_id or "\\" in tenant_id:
        return None
    root_path = Path(root)
    return root_path if tenant_id == "default" else root_path / "tenants" / tenant_id


def scope_uri_for(uri: str, canonical_kind: str = "") -> str:
    if canonical_kind == "pending_proposal" or "/memories/pending/" in uri:
        return uri
    if "/claims/" in uri:
        return uri.rsplit("/claims/", 1)[0]
    return uri


def head_set_path(artifact_root: Path, scope_uri: str) -> Path:
    digest = hashlib.sha256(scope_uri.encode("utf-8")).hexdigest()
    return artifact_root / "system" / "current-heads" / f"{digest}.json"


def _receipt_relative_path(artifact_root: Path, receipt_path: Path) -> str:
    try:
        safe = require_safe_artifact_path(
            artifact_root,
            receipt_path,
            label="canonical receipt",
        )
        return str(safe.relative_to(Path(artifact_root).absolute()))
    except (ValueError, DurablePathIntegrityError) as exc:
        raise CurrentHeadIntegrityError("receipt path is outside the tenant artifact root") from exc


def _head_from_snapshot(
    snapshot: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: str,
) -> dict[str, Any]:
    object_payload = dict(snapshot.get("object", {}) or {})
    metadata = dict(object_payload.get("metadata", {}) or {})
    kind = str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "")
    revision = metadata.get("lifecycle_revision", metadata.get("revision", 0))
    core: dict[str, Any] = {
        "schema_version": CURRENT_HEAD_SCHEMA_VERSION,
        "uri": str(snapshot["uri"]),
        "tenant_id": str(receipt["tenant_id"]),
        "owner_user_id": str(object_payload.get("owner_user_id") or receipt["user_id"]),
        "canonical_kind": kind,
        "current_revision": revision,
        "current_lifecycle_state": str(metadata.get("lifecycle_state") or object_payload.get("lifecycle_state") or ""),
        "current_transaction_id": str(receipt["transaction_id"]),
        "current_operation_id": str(snapshot.get("operation_id") or ""),
        "current_idempotency_key": str(receipt["idempotency_key"]),
        "receipt_path": receipt_path,
        "receipt_digest": str(receipt["receipt_digest"]),
        "object_digest": str(snapshot["object_digest"]),
        "content_digest": str(snapshot["content_digest"]),
        "bundle_relation_digest": str(snapshot["bundle_relation_digest"]),
        "relation_digest": str(snapshot["relation_digest"]),
        "proposal_fingerprint": str(metadata.get("proposal_fingerprint") or ""),
        "updated_at": str(object_payload.get("updated_at") or receipt["created_at"]),
    }
    return {**core, "head_digest": canonical_digest(core)}


def validate_current_head(head: object) -> dict[str, Any]:
    if not isinstance(head, dict) or head.get("schema_version") != CURRENT_HEAD_SCHEMA_VERSION:
        raise CurrentHeadIntegrityError("current head schema is unsupported")
    digest = head.get("head_digest")
    core = {key: value for key, value in head.items() if key != "head_digest"}
    if not isinstance(digest, str) or digest != canonical_digest(core):
        raise CurrentHeadIntegrityError("current head digest is corrupt")
    required = (
        "uri",
        "tenant_id",
        "owner_user_id",
        "canonical_kind",
        "current_transaction_id",
        "current_operation_id",
        "current_idempotency_key",
        "receipt_path",
        "receipt_digest",
        "object_digest",
        "content_digest",
        "bundle_relation_digest",
        "relation_digest",
    )
    if any(not isinstance(head.get(key), str) or not str(head.get(key)) for key in required):
        raise CurrentHeadIntegrityError("current head identity is incomplete")
    revision = head.get("current_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise CurrentHeadIntegrityError("current head revision is invalid")
    if not isinstance(head.get("current_lifecycle_state"), str):
        raise CurrentHeadIntegrityError("current head lifecycle state is invalid")
    if not isinstance(head.get("updated_at"), str) or not str(head.get("updated_at")):
        raise CurrentHeadIntegrityError("current head update time is invalid")
    return head


def validate_current_head_set(payload: object, *, scope_uri: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != CURRENT_HEAD_SET_SCHEMA_VERSION:
        raise CurrentHeadIntegrityError("current head-set schema is unsupported")
    digest = payload.get("head_set_digest")
    core = {key: value for key, value in payload.items() if key != "head_set_digest"}
    if not isinstance(digest, str) or digest != canonical_digest(core):
        raise CurrentHeadIntegrityError("current head-set digest is corrupt")
    if scope_uri is not None and payload.get("scope_uri") != scope_uri:
        raise CurrentHeadIntegrityError("current head-set scope does not match")
    heads = payload.get("heads")
    if not isinstance(heads, dict) or not heads:
        raise CurrentHeadIntegrityError("current head-set has no members")
    for uri, head in heads.items():
        validated = validate_current_head(head)
        if str(uri) != str(validated["uri"]):
            raise CurrentHeadIntegrityError("current head-set member URI is inconsistent")
        if scope_uri_for(str(uri), str(validated["canonical_kind"])) != str(payload["scope_uri"]):
            raise CurrentHeadIntegrityError("current head-set crosses a canonical scope")
    return payload


def validate_current_head_set_path(
    artifact_root: Path,
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Require one unique content-addressed path for every current head-set."""

    expected = head_set_path(artifact_root, str(payload.get("scope_uri") or ""))
    try:
        safe_path = require_safe_artifact_path(
            artifact_root,
            path,
            label="current head-set",
        )
        safe_expected = require_safe_artifact_path(
            artifact_root,
            expected,
            label="expected current head-set",
        )
    except DurablePathIntegrityError as exc:
        raise CurrentHeadIntegrityError(f"current head-set path identity is invalid: {path.name}") from exc
    if safe_path != safe_expected or path.name != expected.name:
        raise CurrentHeadIntegrityError(f"current head-set path identity is invalid: {path.name}")


def _validate_receipt_path_identity(
    artifact_root: Path,
    relative: str,
    receipt: dict[str, Any],
    *,
    actual_path: Path | None = None,
) -> None:
    """Bind a head to the receipt's one legal immutable artifact location."""

    try:
        idempotency_key = require_safe_path_segment(
            receipt.get("idempotency_key"),
            "canonical receipt idempotency_key",
        )
        operation_ids = receipt.get("operation_ids")
        operations = receipt.get("operations")
        pending_only = bool(
            isinstance(operations, list)
            and len(operations) == 1
            and isinstance(operations[0], dict)
            and isinstance(operations[0].get("payload"), dict)
            and operations[0]["payload"].get("canonical_pending_proposal") is True
            and operations[0]["payload"].get("canonical_memory") is not True
        )
        allowed: set[str]
        if pending_only and isinstance(operation_ids, list) and len(operation_ids) == 1:
            operation_id = require_safe_path_segment(
                operation_ids[0],
                "pending receipt operation_id",
            )
            allowed = {
                (Path("system") / "operations" / f"{operation_id}.json").as_posix(),
            }
        else:
            allowed = {
                (Path("system") / "transactions" / f"{idempotency_key}.json").as_posix(),
            }
    except (TypeError, ValueError) as exc:
        raise CurrentHeadIntegrityError("current head receipt path identity is invalid") from exc
    normalized = Path(relative)
    normalized_relative = normalized.as_posix()
    candidate = actual_path or artifact_root / normalized
    try:
        require_safe_artifact_path(
            artifact_root,
            candidate,
            label="current head receipt",
        )
    except DurablePathIntegrityError as exc:
        raise CurrentHeadIntegrityError("current head receipt path identity is invalid") from exc
    if normalized.is_absolute() or normalized_relative not in allowed:
        raise CurrentHeadIntegrityError("current head receipt path identity is invalid")


def publish_current_head_sets(
    artifact_root: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    uris: Sequence[str] | None = None,
) -> list[Path]:
    """Overlay changed heads, publishing one atomic pointer per Slot/pending URI.

    Canonical Slot sets are published before pending lifecycle sets.  Therefore
    a CONFIRM_AND_APPLY crash can expose a committed canonical effect while the
    pending remains CONFIRMED, never RESOLVED without canonical proof.
    """

    try:
        stored_receipt = load_transaction_receipt(receipt_path)
    except ReceiptIntegrityError as exc:
        raise CurrentHeadIntegrityError("current head cannot reference an invalid receipt") from exc
    if stored_receipt.get("receipt_digest") != receipt.get("receipt_digest"):
        raise CurrentHeadIntegrityError("current head receipt payload differs from its immutable file")
    receipt = stored_receipt
    relative_receipt = _receipt_relative_path(artifact_root, receipt_path)
    _validate_receipt_path_identity(
        artifact_root,
        relative_receipt,
        receipt,
        actual_path=receipt_path,
    )
    selected = set(uris or ())
    snapshots = [
        dict(item)
        for item in receipt.get("effect_snapshots", [])
        if isinstance(item, dict) and (not selected or str(item.get("uri") or "") in selected)
    ]
    if selected and {str(item.get("uri") or "") for item in snapshots} != selected:
        raise CurrentHeadIntegrityError("receipt cannot publish all requested current heads")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        uri = str(snapshot.get("uri") or "")
        kind = str(snapshot.get("canonical_kind") or "")
        grouped.setdefault(scope_uri_for(uri, kind), []).append(snapshot)
    ordered_scopes = sorted(grouped, key=lambda value: ("/memories/pending/" in value, value))
    published: list[Path] = []
    for scope_uri in ordered_scopes:
        path = head_set_path(artifact_root, scope_uri)
        try:
            require_safe_artifact_path(
                artifact_root,
                path,
                label="current head-set",
            )
        except DurablePathIntegrityError as exc:
            raise CurrentHeadIntegrityError("existing current head-set path cannot traverse a symbolic link") from exc
        existing_heads: dict[str, Any] = {}
        if path.is_symlink():
            raise CurrentHeadIntegrityError("existing current head-set path cannot be a symbolic link")
        if path.exists():
            try:
                existing = validate_current_head_set(
                    __import__("json").loads(path.read_text(encoding="utf-8")),
                    scope_uri=scope_uri,
                )
            except (OSError, UnicodeError, ValueError, CurrentHeadIntegrityError) as exc:
                raise CurrentHeadIntegrityError("existing current head-set is corrupt") from exc
            existing_heads = {str(uri): dict(head) for uri, head in dict(existing["heads"]).items()}
        proposed = {
            str(snapshot["uri"]): _head_from_snapshot(
                snapshot,
                receipt,
                receipt_path=relative_receipt,
            )
            for snapshot in grouped[scope_uri]
        }
        by_uri = {str(snapshot["uri"]): snapshot for snapshot in grouped[scope_uri]}
        historical_replay = any(
            uri in existing_heads and int(existing_heads[uri]["current_revision"]) > int(head["current_revision"])
            for uri, head in proposed.items()
        )
        if historical_replay:
            for uri, head in proposed.items():
                existing_head = existing_heads.get(uri)
                if existing_head is None or int(existing_head["current_revision"]) < int(head["current_revision"]):
                    raise CurrentHeadConflictError("historical receipt replay sees a partially rewound head-set")
                if int(existing_head["current_revision"]) == int(head["current_revision"]) and existing_head.get(
                    "receipt_digest"
                ) != head.get("receipt_digest"):
                    raise CurrentHeadConflictError("historical receipt replay conflicts at the same revision")
            # A valid old receipt remains immutable history; it must never
            # replace or prune any member of the newer current head-set.
            continue
        changed = False
        for uri, head in proposed.items():
            snapshot = by_uri[uri]
            current = existing_heads.get(uri)
            after_revision = int(head["current_revision"])
            before_revision = int(snapshot.get("before_revision", 0) or 0)
            if current is None:
                if before_revision != 0 or after_revision not in {0, 1}:
                    raise CurrentHeadConflictError("new current head does not start from an absent revision")
                existing_heads[uri] = head
                changed = True
                continue
            current_revision = int(current["current_revision"])
            if current_revision == after_revision:
                if current.get("head_digest") != head.get("head_digest"):
                    raise CurrentHeadConflictError("current head has a same-revision transaction fork")
                continue
            if current_revision != before_revision or after_revision != before_revision + 1:
                raise CurrentHeadConflictError("current head publication violates revision compare-and-swap")
            existing_heads[uri] = head
            changed = True
        if not changed:
            continue
        # Current Claim heads are historical domain members.  Replacement and
        # retraction publish a new Claim revision; no transaction may prune a
        # previously committed Claim merely by omitting it from a Slot
        # payload.  The final-state validator rejects such an omission, and
        # startup verifies exact Slot/head membership independently.
        updated_at = str(receipt.get("created_at") or "")
        core: dict[str, Any] = {
            "schema_version": CURRENT_HEAD_SET_SCHEMA_VERSION,
            "scope_uri": scope_uri,
            "heads": {uri: existing_heads[uri] for uri in sorted(existing_heads)},
            "updated_at": updated_at,
        }
        atomic_write_json(
            path,
            {**core, "head_set_digest": canonical_digest(core)},
            artifact_root=artifact_root,
        )
        published.append(path)
    return published


def _safe_receipt_path(artifact_root: Path, relative: str) -> Path:
    candidate = artifact_root / relative
    try:
        return require_safe_artifact_path(
            artifact_root,
            candidate,
            label="current head receipt",
        )
    except DurablePathIntegrityError as exc:
        raise CurrentHeadIntegrityError("current head receipt path escapes its tenant root") from exc


def load_current_head(
    artifact_root: Path,
    uri: str,
    *,
    canonical_kind: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scope_uri = scope_uri_for(uri, canonical_kind)
    path = head_set_path(artifact_root, scope_uri)
    try:
        require_safe_artifact_path(
            artifact_root,
            path,
            label="current head-set",
        )
    except DurablePathIntegrityError as exc:
        raise CurrentHeadIntegrityError(f"current head-set path cannot be a symbolic link: {scope_uri}") from exc
    if not path.exists():
        raise FileNotFoundError(f"current head is missing: {uri}")
    try:
        import json

        head_set = validate_current_head_set(json.loads(path.read_text(encoding="utf-8")), scope_uri=scope_uri)
        validate_current_head_set_path(artifact_root, path, head_set)
    except (OSError, UnicodeError, ValueError, CurrentHeadIntegrityError) as exc:
        raise CurrentHeadIntegrityError(f"current head-set is invalid: {scope_uri}") from exc
    return load_current_head_member(artifact_root, head_set, uri)


def load_current_head_member(
    artifact_root: Path,
    head_set: dict[str, Any],
    uri: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve one member from an already captured, validated head-set."""

    validated_set = validate_current_head_set(head_set)
    raw_head = dict(validated_set["heads"]).get(uri)
    if raw_head is None:
        raise FileNotFoundError(f"current head does not contain URI: {uri}")
    head = validate_current_head(raw_head)
    receipt_relative = str(head["receipt_path"])
    receipt_path = _safe_receipt_path(artifact_root, receipt_relative)
    try:
        receipt = load_transaction_receipt(receipt_path)
    except ReceiptIntegrityError as exc:
        raise CurrentHeadIntegrityError("current head receipt is invalid") from exc
    _validate_receipt_path_identity(
        artifact_root,
        receipt_relative,
        receipt,
        actual_path=artifact_root / receipt_relative,
    )
    if (
        receipt.get("receipt_digest") != head.get("receipt_digest")
        or receipt.get("transaction_id") != head.get("current_transaction_id")
        or receipt.get("idempotency_key") != head.get("current_idempotency_key")
        or receipt.get("tenant_id") != head.get("tenant_id")
        or receipt.get("user_id") != head.get("owner_user_id")
    ):
        raise CurrentHeadIntegrityError("current head crosses its immutable receipt boundary")
    try:
        snapshot = receipt_snapshot(receipt, uri)
    except ReceiptIntegrityError as exc:
        raise CurrentHeadIntegrityError("current head receipt has no matching object snapshot") from exc
    for field in (
        "object_digest",
        "content_digest",
        "bundle_relation_digest",
        "relation_digest",
    ):
        if snapshot.get(field) != head.get(field):
            raise CurrentHeadIntegrityError(f"current head {field} does not match its receipt")
    return head, receipt, snapshot


def iter_current_head_uris(
    artifact_root: Path,
    *,
    kinds: Iterable[str] = (),
) -> tuple[str, ...]:
    requested = set(kinds)
    root = artifact_root / "system" / "current-heads"
    try:
        require_safe_artifact_path(
            artifact_root,
            root,
            label="current head-set directory",
        )
    except DurablePathIntegrityError as exc:
        raise CurrentHeadIntegrityError("current head-set directory is invalid") from exc
    if not root.exists():
        return ()
    result: list[str] = []
    import json

    for path in sorted(root.glob("*.json")):
        try:
            require_safe_artifact_path(
                artifact_root,
                path,
                label="current head-set",
            )
            payload = validate_current_head_set(json.loads(path.read_text(encoding="utf-8")))
            validate_current_head_set_path(artifact_root, path, payload)
        except (
            OSError,
            UnicodeError,
            ValueError,
            DurablePathIntegrityError,
            CurrentHeadIntegrityError,
        ) as exc:
            raise CurrentHeadIntegrityError(f"current head-set is invalid: {path.name}") from exc
        for uri, head in dict(payload["heads"]).items():
            kind = str(dict(head).get("canonical_kind") or "")
            if not requested or kind in requested:
                result.append(str(uri))
    return tuple(dict.fromkeys(result))


def receipt_history_contains_uri(artifact_root: Path, uri: str) -> bool:
    """Return whether immutable receipt history proves ``uri`` was committed.

    The scan is deliberately independent from current-head enumeration so a
    deleted pointer cannot turn established committed state into apparent
    absence.  A corrupt receipt fails closed instead of being skipped.
    """

    import json

    for directory_name in ("transactions", "operations"):
        directory = artifact_root / "system" / directory_name
        try:
            require_safe_artifact_path(
                artifact_root,
                directory,
                label="receipt history directory",
            )
        except DurablePathIntegrityError as exc:
            raise CurrentHeadIntegrityError("receipt history directory is invalid") from exc
        for path in sorted(directory.glob("*.json")) if directory.exists() else ():
            try:
                require_safe_artifact_path(
                    artifact_root,
                    path,
                    label="receipt history artifact",
                )
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                DurablePathIntegrityError,
            ) as exc:
                raise CurrentHeadIntegrityError(f"receipt artifact is unreadable: {path.name}") from exc
            if not isinstance(raw, dict) or raw.get("schema_version") != TRANSACTION_RECEIPT_SCHEMA_VERSION:
                continue
            try:
                receipt = load_transaction_receipt(path)
            except ReceiptIntegrityError as exc:
                raise CurrentHeadIntegrityError(f"receipt artifact is corrupt: {path.name}") from exc
            if any(
                isinstance(snapshot, dict) and str(snapshot.get("uri") or "") == uri
                for snapshot in receipt.get("effect_snapshots", []) or []
            ):
                return True
    return False


def receipt_history_effects(
    artifact_root: Path,
    *,
    kinds: Iterable[str] = (),
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return validated canonical receipt effects grouped by URI and revision.

    Current-head files are mutable pointers, so they cannot be their own
    enumeration authority.  Business enumeration uses this independent
    immutable inventory to detect a deleted head-set instead of silently
    turning committed memory into an empty result.

    This deliberately validates only immutable receipt identity/history.  It
    does not consult Source state or require a current head, which keeps the
    artifact dependency graph acyclic and lets the visibility layer account
    for the narrow receipt-before-head redo window.
    """

    import json

    requested = set(kinds)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for directory_name in ("transactions", "operations"):
        directory = artifact_root / "system" / directory_name
        try:
            require_safe_artifact_path(
                artifact_root,
                directory,
                label="receipt history directory",
            )
        except DurablePathIntegrityError as exc:
            raise CurrentHeadIntegrityError("receipt history directory is invalid") from exc
        for path in sorted(directory.glob("*.json")) if directory.exists() else ():
            try:
                require_safe_artifact_path(
                    artifact_root,
                    path,
                    label="receipt history artifact",
                )
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                DurablePathIntegrityError,
            ) as exc:
                raise CurrentHeadIntegrityError(f"receipt artifact is unreadable: {path.name}") from exc
            if not isinstance(raw, dict) or raw.get("schema_version") != TRANSACTION_RECEIPT_SCHEMA_VERSION:
                continue
            try:
                receipt = load_transaction_receipt(path)
            except ReceiptIntegrityError as exc:
                raise CurrentHeadIntegrityError(f"receipt artifact is corrupt: {path.name}") from exc
            for snapshot in receipt.get("effect_snapshots", []) or []:
                if not isinstance(snapshot, dict):
                    continue
                object_payload = snapshot.get("object")
                metadata = dict(object_payload.get("metadata", {}) or {}) if isinstance(object_payload, dict) else {}
                kind = str(snapshot.get("canonical_kind") or metadata.get("canonical_kind") or "")
                if kind not in {"slot", "claim", "pending_proposal"}:
                    continue
                if requested and kind not in requested:
                    continue
                uri = str(snapshot.get("uri") or "")
                if not uri:
                    raise CurrentHeadIntegrityError("receipt history contains an empty canonical URI")
                grouped.setdefault(uri, []).append(
                    {
                        "uri": uri,
                        "canonical_kind": kind,
                        "before_revision": int(snapshot.get("before_revision", 0) or 0),
                        "after_revision": int(snapshot.get("after_revision", 0) or 0),
                        "transaction_id": str(receipt.get("transaction_id") or ""),
                        "receipt_digest": str(receipt.get("receipt_digest") or ""),
                        "created_at": str(receipt.get("created_at") or ""),
                    }
                )

    validated: dict[str, tuple[dict[str, Any], ...]] = {}
    for uri, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                int(row["after_revision"]),
                int(row["before_revision"]),
                str(row["created_at"]),
                str(row["receipt_digest"]),
            )
        )
        seen_after: dict[int, str] = {}
        previous_after: int | None = None
        for index, row in enumerate(rows):
            before = int(row["before_revision"])
            after = int(row["after_revision"])
            digest = str(row["receipt_digest"])
            existing = seen_after.setdefault(after, digest)
            if existing != digest:
                raise CurrentHeadIntegrityError(f"receipt history has a same-revision fork: {uri}#{after}")
            if index == 0:
                if before != 0 or after not in {0, 1}:
                    raise CurrentHeadIntegrityError(f"receipt history does not start at revision zero/one: {uri}")
            elif before != previous_after or after != before + 1:
                raise CurrentHeadIntegrityError(f"receipt history is non-contiguous: {uri}")
            previous_after = after
        validated[uri] = tuple(rows)
    return validated


def snapshot_object(snapshot: dict[str, Any]) -> ContextObject:
    payload = snapshot.get("object")
    if not isinstance(payload, dict):
        raise CurrentHeadIntegrityError("current receipt snapshot has no object")
    return ContextObject.from_dict(payload)
