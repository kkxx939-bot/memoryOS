"""存储、模型和检索基础对象的装配。"""

from __future__ import annotations

from dataclasses import dataclass

from config import RuntimeMode
from foundation.readiness import RuntimeReadiness, RuntimeReadinessState
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.model import build_model_client
from infrastructure.store.contracts.vector import require_production_vector_capabilities
from infrastructure.store.filesystem import FileSystemSourceStore
from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory import RuntimeLayout
from infrastructure.store.sqlite import SQLiteIndexStore, SQLiteLockStore, SQLiteQueueStore, SQLiteRelationStore
from runtime.config import RuntimeConfig
from runtime.container import StoreRuntime
from runtime.dependencies import RuntimeDependencies


@dataclass(frozen=True)
class StoreAssembly:
    """基础存储装配的结果，以及必须先探测的 Markdown Store。"""

    layout: RuntimeLayout
    readiness: RuntimeReadiness
    stores: StoreRuntime
    document_store: FileSystemMemoryDocumentStore


def wire_stores(config: RuntimeConfig, dependencies: RuntimeDependencies) -> StoreAssembly:
    """先验证 RuntimeLayout 和文件能力，再创建任何 SQLite serving 状态。"""

    root = config.root_path
    layout = RuntimeLayout.open(root, tenant_id=config.tenant_id)
    layout_details = layout.initialize_or_validate()
    readiness = RuntimeReadiness()
    readiness.transition(
        RuntimeReadinessState.RECOVERING,
        details={"runtime_layout": layout_details},
    )
    document_store = FileSystemMemoryDocumentStore(
        root,
        max_file_bytes=config.memory_document_max_bytes,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
        max_scan_files=config.memory_scan_max_files,
    )
    document_store.probe_write_capabilities(config.tenant_id)

    source = dependencies.source_store or FileSystemSourceStore(root, tenant_id=config.tenant_id)
    source_tenant = str(getattr(source, "tenant_id", config.tenant_id))
    if source_tenant != config.tenant_id:
        raise ValueError("SourceStore tenant does not match RuntimeConfig tenant_id")
    if hasattr(source, "__dict__"):
        vars(source)["readiness"] = readiness

    index_root = layout.tenant_root / "indexes"
    index = dependencies.index_store or SQLiteIndexStore(index_root / "context.sqlite3")
    relation = dependencies.relation_store or SQLiteRelationStore(index_root / "relations.sqlite3")
    queue = dependencies.queue_store or SQLiteQueueStore(layout.tenant_root / "queues" / "jobs.sqlite3")
    lock = dependencies.lock_store or SQLiteLockStore(layout.tenant_root / "system" / "locks.sqlite3")
    model_client = dependencies.model_client
    if model_client is None and config.model.enabled:
        model_client = build_model_client(config.model)
    if config.mode == RuntimeMode.SERVER and dependencies.vector_store is not None:
        require_production_vector_capabilities(dependencies.vector_store)
    search = dependencies.hybrid_search or HybridSearch(
        index,  # pyright: ignore[reportArgumentType]
        vector_store=dependencies.vector_store,
        embedding_provider=dependencies.embedding_provider,
        source_store=source,
    )
    stores = StoreRuntime(
        source=source,
        index=index,
        relation=relation,
        queue=queue,
        lock=lock,
        vector=dependencies.vector_store,
        embedding=dependencies.embedding_provider,
        hybrid_search=search,
        reranker=dependencies.reranker,
        model_client=model_client,
    )
    return StoreAssembly(layout=layout, readiness=readiness, stores=stores, document_store=document_store)


__all__ = ["StoreAssembly", "wire_stores"]
