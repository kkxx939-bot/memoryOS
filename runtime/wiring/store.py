"""存储、模型和检索基础对象的装配。"""

from __future__ import annotations

from dataclasses import dataclass

from config import RuntimeMode
from foundation.readiness import RuntimeReadiness, RuntimeReadinessState
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.model import build_model_client
from infrastructure.store.contracts.vector import require_production_vector_capabilities
from infrastructure.store.filesystem import FileSystemSourceStore
from infrastructure.store.runtime_layout import RuntimeLayout
from infrastructure.store.sqlite import SQLiteIndexStore, SQLiteLockStore, SQLiteQueueStore, SQLiteRelationStore
from runtime.config import RuntimeConfig
from runtime.container import StoreRuntime
from runtime.dependencies import RuntimeDependencies


@dataclass(frozen=True)
class StoreAssembly:
    """基础存储装配结果。"""

    layout: RuntimeLayout
    readiness: RuntimeReadiness
    stores: StoreRuntime


def wire_stores(config: RuntimeConfig, dependencies: RuntimeDependencies) -> StoreAssembly:
    """先验证 RuntimeLayout，再创建 SQLite serving 状态。"""

    root = config.root_path
    layout = RuntimeLayout.open(root, tenant_id=config.tenant_id)
    layout_details = layout.initialize_or_validate()
    readiness = RuntimeReadiness()
    readiness.transition(
        RuntimeReadinessState.RECOVERING,
        details={"runtime_layout": layout_details},
    )
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
    return StoreAssembly(layout=layout, readiness=readiness, stores=stores)


__all__ = ["StoreAssembly", "wire_stores"]
