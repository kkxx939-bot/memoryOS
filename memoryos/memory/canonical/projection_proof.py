"""Immutable publication receipts and completion proofs for memory projection.

Projection records and index/vector/view rows describe the *current* derived
state.  They therefore cannot also serve as immutable evidence that an older
commit group completed successfully.  This module keeps the two roles
separate, mirroring the transaction-receipt/current-head split used by
canonical Source objects.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memoryos.core.durable_io import atomic_create_json
from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError as _ProjectionIntegrityError,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionRecord,
)

try:  # pragma: no cover - supported production POSIX platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION = "projection_publication_receipt_v1"
PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION = "projection_completion_proof_v2"


class AuthoritativeProjectionIntegrityError(_ProjectionIntegrityError):
    """An immutable projection proof is corrupt or conflicts with its identity."""


# Keep the implementation's existing validation sites concise while making
# proof failures distinguishable from rebuildable projection/index/view state.
ProjectionIntegrityError = AuthoritativeProjectionIntegrityError

_MUTABLE_RECORD_FIELDS = frozenset(
    {
        "current",
        "failure_reason",
        "retryable",
        "status",
        "updated_at",
    }
)


def projection_publication_record_digest(record: ProjectionRecord) -> str:
    """Digest the durable projection effect while excluding current-pointer state.

    A later canonical revision legitimately retires an old attempt by changing
    ``current``, ``status`` and failure bookkeeping.  All publication identity,
    artifact locations, effect hashes and component completion statuses remain
    covered by this digest.
    """

    stable = {key: value for key, value in asdict(record).items() if key not in _MUTABLE_RECORD_FIELDS}
    return canonical_digest(stable)


class ProjectionProofStore:
    """Create-only projection proof artifacts under one tenant artifact root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def publication_path(self, transaction_id: str) -> Path:
        safe = require_safe_path_segment(transaction_id, "projection publication transaction_id")
        return self.root / "system" / "projection-publications" / f"{safe}.json"

    def completion_path(self, transaction_id: str) -> Path:
        safe = require_safe_path_segment(transaction_id, "projection completion transaction_id")
        return self.root / "system" / "projection-completions" / f"{safe}.json"

    def load_publication(self, transaction_id: str) -> dict[str, Any] | None:
        path = self.publication_path(transaction_id)
        if path.is_symlink():
            raise ProjectionIntegrityError("projection publication receipt path cannot be a symbolic link")
        if not path.exists():
            return None
        payload = self.validate_publication(self._read(path), transaction_id=transaction_id)
        if path.resolve() != self.publication_path(str(payload["transaction_id"])).resolve():
            raise ProjectionIntegrityError("projection publication receipt path identity mismatch")
        return payload

    def load_completion(self, transaction_id: str) -> dict[str, Any] | None:
        path = self.completion_path(transaction_id)
        if path.is_symlink():
            raise ProjectionIntegrityError("projection completion proof path cannot be a symbolic link")
        if not path.exists():
            return None
        payload = self.validate_completion(self._read(path), transaction_id=transaction_id)
        if path.resolve() != self.completion_path(str(payload["transaction_id"])).resolve():
            raise ProjectionIntegrityError("projection completion proof path identity mismatch")
        return payload

    def ensure_publication(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        validated = self.validate_publication(dict(payload))
        path = self.publication_path(str(validated["transaction_id"]))
        with self._artifact_lock(path):
            self._create_or_match(path, validated, self.validate_publication)
        loaded = self.load_publication(str(validated["transaction_id"]))
        assert loaded is not None
        return loaded

    def ensure_completion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        validated = self.validate_completion(dict(payload))
        path = self.completion_path(str(validated["transaction_id"]))
        with self._artifact_lock(path):
            self._create_or_match(path, validated, self.validate_completion)
        loaded = self.load_completion(str(validated["transaction_id"]))
        assert loaded is not None
        return loaded

    def iter_publications(self) -> tuple[dict[str, Any], ...]:
        root = self.root / "system" / "projection-publications"
        result: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.json")) if root.exists() else ():
            if path.is_symlink():
                raise ProjectionIntegrityError("projection publication receipt path cannot be a symbolic link")
            payload = self.validate_publication(self._read(path))
            if path.resolve() != self.publication_path(str(payload["transaction_id"])).resolve():
                raise ProjectionIntegrityError("projection publication receipt path identity mismatch")
            result.append(payload)
        return tuple(result)

    def iter_completions(self) -> tuple[dict[str, Any], ...]:
        root = self.root / "system" / "projection-completions"
        result: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.json")) if root.exists() else ():
            if path.is_symlink():
                raise ProjectionIntegrityError("projection completion proof path cannot be a symbolic link")
            payload = self.validate_completion(self._read(path))
            if path.resolve() != self.completion_path(str(payload["transaction_id"])).resolve():
                raise ProjectionIntegrityError("projection completion proof path identity mismatch")
            result.append(payload)
        return tuple(result)

    def validate_all(self) -> dict[str, int]:
        publication_items = self.iter_publications()
        completion_items = self.iter_completions()
        publications = {str(payload["transaction_id"]): payload for payload in publication_items}
        completions = {str(payload["transaction_id"]): payload for payload in completion_items}
        if len(publications) != len(publication_items) or len(completions) != len(completion_items):
            raise ProjectionIntegrityError("projection proof transaction identity is duplicated")
        for transaction_id, completion in completions.items():
            publication = publications.get(transaction_id)
            if publication is None:
                raise ProjectionIntegrityError("projection completion proof has no publication receipt")
            for key in (
                "commit_group_id",
                "transaction_id",
                "job_id",
                "tenant_id",
                "user_id",
                "queue_identity_digest",
                "outbox_digest",
                "receipt_digest",
                "prepared_intent_digest",
                "operation_ids",
                "claim_revisions",
                "claims",
            ):
                if completion.get(key) != publication.get(key):
                    raise ProjectionIntegrityError("projection completion proof differs from its publication receipt")
            if completion.get("publication_digest") != publication.get("publication_digest"):
                raise ProjectionIntegrityError("projection completion proof publication digest is inconsistent")
        return {
            "publications": len(publications),
            "completions": len(completions),
        }

    @staticmethod
    def validate_publication(
        payload: object,
        *,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProjectionIntegrityError("projection publication receipt must be an object")
        if payload.get("schema_version") != PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION:
            raise ProjectionIntegrityError("projection publication receipt schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "publication_digest"}
        if payload.get("publication_digest") != canonical_digest(core):
            raise ProjectionIntegrityError("projection publication receipt digest is corrupt")
        ProjectionProofStore._validate_boundary(payload, transaction_id=transaction_id)
        claims = ProjectionProofStore._validate_claims(payload.get("claims"))
        if payload.get("claim_revisions") != [
            {"uri": item["claim_uri"], "revision": item["source_revision"]} for item in claims
        ]:
            raise ProjectionIntegrityError("projection publication claim set is inconsistent")
        return payload

    @staticmethod
    def validate_completion(
        payload: object,
        *,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProjectionIntegrityError("projection completion proof must be an object")
        if payload.get("schema_version") != PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION:
            raise ProjectionIntegrityError("projection completion proof schema is unsupported")
        core = {key: value for key, value in payload.items() if key != "proof_digest"}
        if payload.get("proof_digest") != canonical_digest(core):
            raise ProjectionIntegrityError("projection completion proof digest is corrupt")
        ProjectionProofStore._validate_boundary(payload, transaction_id=transaction_id)
        if payload.get("queue_status") != "done":
            raise ProjectionIntegrityError("projection completion proof has no terminal queue state")
        if not ProjectionProofStore._is_digest(payload.get("publication_digest")):
            raise ProjectionIntegrityError("projection completion proof has no publication receipt")
        claims = ProjectionProofStore._validate_claims(payload.get("claims"))
        if payload.get("claim_revisions") != [
            {"uri": item["claim_uri"], "revision": item["source_revision"]} for item in claims
        ]:
            raise ProjectionIntegrityError("projection completion claim set is inconsistent")
        return payload

    @staticmethod
    def _validate_boundary(payload: Mapping[str, Any], *, transaction_id: str | None) -> None:
        declared_transaction = str(payload.get("transaction_id") or "")
        try:
            require_safe_path_segment(declared_transaction, "projection proof transaction_id")
        except (TypeError, ValueError) as exc:
            raise ProjectionIntegrityError("projection proof transaction identity is invalid") from exc
        if transaction_id is not None and declared_transaction != transaction_id:
            raise ProjectionIntegrityError("projection proof transaction identity mismatch")
        required_strings = (
            "commit_group_id",
            "job_id",
            "tenant_id",
            "user_id",
        )
        if any(not isinstance(payload.get(key), str) or not payload.get(key) for key in required_strings):
            raise ProjectionIntegrityError("projection proof boundary identity is incomplete")
        if payload.get("job_id") != f"outbox_{declared_transaction}":
            raise ProjectionIntegrityError("projection proof queue identity is inconsistent")
        for key in (
            "queue_identity_digest",
            "outbox_digest",
            "receipt_digest",
            "prepared_intent_digest",
        ):
            if not ProjectionProofStore._is_digest(payload.get(key)):
                raise ProjectionIntegrityError(f"projection proof {key} is invalid")
        operation_ids = payload.get("operation_ids")
        if (
            not isinstance(operation_ids, list)
            or not operation_ids
            or any(not isinstance(item, str) or not item for item in operation_ids)
            or len(operation_ids) != len(set(operation_ids))
        ):
            raise ProjectionIntegrityError("projection proof operation set is invalid")

    @staticmethod
    def _validate_claims(raw_claims: object) -> list[dict[str, Any]]:
        if not isinstance(raw_claims, list):
            raise ProjectionIntegrityError("projection proof claims are invalid")
        claims: list[dict[str, Any]] = []
        identities: set[tuple[str, int]] = set()
        for raw in raw_claims:
            if not isinstance(raw, dict):
                raise ProjectionIntegrityError("projection claim proof must be an object")
            core = {key: value for key, value in raw.items() if key != "claim_proof_digest"}
            if raw.get("claim_proof_digest") != canonical_digest(core):
                raise ProjectionIntegrityError("projection claim proof digest is corrupt")
            claim_uri = str(raw.get("claim_uri") or "")
            revision = raw.get("source_revision")
            if not claim_uri or isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
                raise ProjectionIntegrityError("projection claim proof identity is invalid")
            identity = (claim_uri, revision)
            if identity in identities:
                raise ProjectionIntegrityError("projection claim proof identity is duplicated")
            identities.add(identity)
            for key in (
                "projection_attempt_id",
                "input_effect_hash",
                "publish_token",
                "projected_content_digest",
                "projected_relation_digest",
                "record_digest",
                "publication_record_digest",
                "relation_artifact_digest",
                "manifest_digest",
                "index_metadata_digest",
            ):
                value = raw.get(key)
                if not isinstance(value, str) or not value:
                    raise ProjectionIntegrityError(f"projection claim proof {key} is missing")
            for key in (
                "input_effect_hash",
                "projected_content_digest",
                "projected_relation_digest",
                "record_digest",
                "publication_record_digest",
                "relation_artifact_digest",
                "manifest_digest",
                "index_metadata_digest",
            ):
                if not ProjectionProofStore._is_digest(raw.get(key)):
                    raise ProjectionIntegrityError(f"projection claim proof {key} is invalid")
            vector_digest = raw.get("vector_metadata_digest")
            if vector_digest not in (None, "") and not ProjectionProofStore._is_digest(vector_digest):
                raise ProjectionIntegrityError("projection claim proof vector digest is invalid")
            layer_uris = raw.get("layer_uris")
            layer_digests = raw.get("layer_digests")
            if (
                not isinstance(layer_uris, dict)
                or set(layer_uris) != {"L0", "L1", "L2", "manifest", "relations"}
                or any(not isinstance(value, str) or not value for value in layer_uris.values())
                or not isinstance(layer_digests, dict)
                or set(layer_digests) != {"L0", "L1", "L2"}
                or any(not ProjectionProofStore._is_digest(value) for value in layer_digests.values())
            ):
                raise ProjectionIntegrityError("projection claim artifact set is invalid")
            for key in ("scope_view_digests", "taxonomy_view_digests"):
                values = raw.get(key)
                if (
                    not isinstance(values, list)
                    or not values
                    or any(not ProjectionProofStore._is_digest(value) for value in values)
                ):
                    raise ProjectionIntegrityError(f"projection claim proof {key} is invalid")
            domain_identity = raw.get("domain_identity")
            if not isinstance(domain_identity, dict) or domain_identity.get("claim_uri") != claim_uri:
                raise ProjectionIntegrityError("projection claim domain identity is invalid")
            claims.append(raw)
        return claims

    @staticmethod
    def _is_digest(value: object) -> bool:
        return (
            isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectionIntegrityError(f"immutable projection proof is unreadable: {path.name}") from exc
        if not isinstance(payload, dict):
            raise ProjectionIntegrityError("immutable projection proof is not an object")
        return payload

    def _create_or_match(self, path: Path, payload: dict[str, Any], validator: Any) -> None:
        if path.is_symlink():
            raise ProjectionIntegrityError("immutable projection proof path cannot be a symbolic link")
        if path.exists():
            existing = validator(ProjectionProofStore._read(path))
            if existing != payload:
                raise ProjectionIntegrityError("immutable projection proof conflicts with an existing identity")
            return
        atomic_create_json(path, payload, artifact_root=self.root)

    @contextmanager
    def _artifact_lock(self, path: Path) -> Iterator[None]:
        lock_path = path.parent / ".locks" / f"{path.name}.lock"
        descriptor = open_private_lock(lock_path, root=self.root)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


__all__ = [
    "AuthoritativeProjectionIntegrityError",
    "PROJECTION_COMPLETION_PROOF_SCHEMA_VERSION",
    "PROJECTION_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "ProjectionProofStore",
    "projection_publication_record_digest",
]
