"""Stable canonical projection facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from memoryos.contextdb.catalog import (
    CatalogRecord,
)
from memoryos.contextdb.extensions import ContextDomainClassifier
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.store.vector import VectorStore
from memoryos.memory.canonical.projection_proof import (
    ProjectionProofStore,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionRecord,
    ProjectionRecordStore,
)
from memoryos.memory.canonical.visibility import (
    CommittedCanonicalRead,
)
from memoryos.memory.integration.context_overlay import CanonicalMemoryContextOverlay
from memoryos.security.context_projection import ContextProjectionSanitizer

from . import catalog as _catalog
from . import materialization as _materialization
from . import service_logic as _service_logic
from . import validation as _validation
from . import vector as _vector
from . import views as _views
from .embedding import DeterministicProjectionEmbedding
from .models import ProjectionResult


class _DomainAwareStore(Protocol):
    domain_classifier: ContextDomainClassifier


class CanonicalMemoryProjector:
    """Build disposable projections without ever writing a canonical object."""

    GENERATOR = "deterministic-template-v2"
    PROMPT_VERSION = "none"

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str | Path,
        *,
        relation_store: RelationStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        record_store: ProjectionRecordStore | None = None,
        test_hook: Callable[[str, str, int], None] | None = None,
        status_callback: Callable[[ProjectionRecord], None] | None = None,
        sanitizer: ContextProjectionSanitizer | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.root = Path(root)
        self.relation_store = relation_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider or DeterministicProjectionEmbedding()
        self.record_store = record_store or ProjectionRecordStore(self.root)
        self.test_hook = test_hook
        self.status_callback = status_callback
        self.sanitizer = sanitizer or ContextProjectionSanitizer()
        domain_overlay = CanonicalMemoryContextOverlay()
        for store in (source_store, index_store, relation_store):
            if store is not None and hasattr(store, "domain_classifier"):
                cast(_DomainAwareStore, store).domain_classifier = domain_overlay

    def project(
        self,
        claim_uri: str,
        source_revision: int | None = None,
        *,
        force: bool = False,
    ) -> ProjectionResult:
        return _service_logic.project(self, claim_uri, source_revision, force=force)

    def rebuild(self, *, clear_views: bool = True) -> dict[str, int]:
        return _service_logic.rebuild(self, clear_views=clear_views)

    def _verified_rebuild_claim_proofs(
        self,
        proof_store: ProjectionProofStore,
    ) -> dict[tuple[str, int], tuple[dict[str, Any], ProjectionRecord]]:
        return _validation._verified_rebuild_claim_proofs(self, proof_store)

    def _verified_publication_receipt(self, publication: dict[str, Any]) -> dict[str, Any]:
        return _validation._verified_publication_receipt(self, publication)

    def _verified_projection_record_from_publication(
        self,
        claim_proof: dict[str, Any],
        *,
        receipt: dict[str, Any],
    ) -> ProjectionRecord:
        return _validation._verified_projection_record_from_publication(self, claim_proof, receipt=receipt)

    def _rebuild_claim_revision_catalog(
        self,
        claim_uri: str,
        proofs: dict[tuple[str, int], tuple[dict[str, Any], ProjectionRecord]],
    ) -> int:
        return _catalog._rebuild_claim_revision_catalog(self, claim_uri, proofs)

    def _layers(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        revision: dict[str, Any],
        source_revision: int,
    ) -> tuple[str, str, str]:
        return _materialization._layers(self, obj, metadata, revision, source_revision)

    def _sanitized_revision_layers(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        revision: dict[str, Any],
        source_revision: int,
    ) -> tuple[str, str, str]:
        return _materialization._sanitized_revision_layers(self, obj, metadata, revision, source_revision)

    @staticmethod
    def _bounded_claim_revisions(metadata: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        return _materialization._bounded_claim_revisions(metadata)

    def _revision_payload(self, metadata: dict[str, Any], revision: int) -> dict[str, Any]:
        return _materialization._revision_payload(self, metadata, revision)

    def _projection_domain_identity(
        self,
        committed: CommittedCanonicalRead,
        current_revision: dict[str, Any],
    ) -> dict[str, Any]:
        return _validation._projection_domain_identity(self, committed, current_revision)

    def _projection_object(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        record: ProjectionRecord,
        *,
        domain_identity: dict[str, Any],
        layers: ContextLayers,
    ) -> ContextObject:
        return _materialization._projection_object(
            self, obj, metadata, record, domain_identity=domain_identity, layers=layers
        )

    def _claim_revision_catalog_record(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        record: ProjectionRecord,
        revision: dict[str, Any],
        *,
        proof_metadata: dict[str, Any],
        l0_text: str,
        l1_text: str,
        l2_text: str,
    ) -> CatalogRecord:
        return _catalog._claim_revision_catalog_record(
            self,
            obj,
            metadata,
            record,
            revision,
            proof_metadata=proof_metadata,
            l0_text=l0_text,
            l1_text=l1_text,
            l2_text=l2_text,
        )

    def _reconcile_claim_catalog_projections(
        self,
        obj: ContextObject,
        metadata: dict[str, Any],
        *,
        published_revision: int,
    ) -> None:
        return _catalog._reconcile_claim_catalog_projections(self, obj, metadata, published_revision=published_revision)

    def _revision_bound_projection_proof(self, existing: CatalogRecord) -> ProjectionRecord:
        return _catalog._revision_bound_projection_proof(self, existing)

    def _refresh_claim_vector(
        self,
        record: CatalogRecord,
        *,
        proof: ProjectionRecord | None = None,
    ) -> None:
        return _vector._refresh_claim_vector(self, record, proof=proof)

    def _publish_catalog_vector(
        self,
        catalog_record: CatalogRecord,
        embedding: list[float],
        record: ProjectionRecord,
    ) -> None:
        return _vector._publish_catalog_vector(self, catalog_record, embedding, record)

    def _publish_vector(
        self,
        obj: ContextObject,
        embedding: list[float],
        record: ProjectionRecord,
    ) -> None:
        return _vector._publish_vector(self, obj, embedding, record)

    @staticmethod
    def _claim_catalog_record_key(metadata: Any, source_revision: int) -> str:
        return _catalog._claim_catalog_record_key(metadata, source_revision)

    def _canonical_tree_paths(self, metadata: dict[str, Any]) -> tuple[str, ...]:
        return _catalog._canonical_tree_paths(self, metadata)

    def _canonical_path_segment(self, value: Any) -> str:
        return _catalog._canonical_path_segment(self, value)

    def _write_scope_views(self, obj: ContextObject, record: ProjectionRecord) -> None:
        return _views._write_scope_views(self, obj, record)

    def _write_taxonomy_view(self, obj: ContextObject, record: ProjectionRecord) -> None:
        return _views._write_taxonomy_view(self, obj, record)

    def _write_revisioned_view(self, directory: Path, payload: dict[str, Any]) -> None:
        return _views._write_revisioned_view(self, directory, payload)

    def _publish_view_currents(self, record: ProjectionRecord) -> None:
        return _views._publish_view_currents(self, record)

    def _view_reference(self, obj: ContextObject, record: ProjectionRecord) -> dict[str, Any]:
        return _views._view_reference(self, obj, record)

    def _taxonomy_path(self, metadata: dict[str, Any]) -> Path:
        return _views._taxonomy_path(self, metadata)

    def _manifest(
        self,
        record: ProjectionRecord,
        metadata: dict[str, Any],
        relations_uri: str,
        *,
        domain_identity: dict[str, Any],
    ) -> dict[str, Any]:
        return _materialization._manifest(self, record, metadata, relations_uri, domain_identity=domain_identity)

    def _is_current(self, claim_uri: str, revision: int, expected_effect_hash: str) -> bool:
        return _validation._is_current(self, claim_uri, revision, expected_effect_hash)

    def _remove_view_currents(self, record: ProjectionRecord) -> None:
        return _views._remove_view_currents(self, record)

    def _input_effect_hash(
        self,
        committed: CommittedCanonicalRead,
        source_revision: int,
    ) -> str:
        return _validation._input_effect_hash(self, committed, source_revision)

    def _notify(self, stage: str, claim_uri: str, revision: int) -> None:
        return _service_logic._notify(self, stage, claim_uri, revision)

    def _result(self, record: ProjectionRecord, status: str) -> ProjectionResult:
        return _service_logic._result(self, record, status)

    def _emit(self, record: ProjectionRecord) -> None:
        return _service_logic._emit(self, record)

    def _segment(self, value: Any) -> str:
        return _views._segment(self, value)

    def _read_json_optional(self, path: Path) -> dict[str, Any] | None:
        return _views._read_json_optional(self, path)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        return _views._write_json_atomic(self, path, payload)
