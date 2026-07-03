# ContextDB

ContextDB provides:

- `memoryos://` URI validation and user/resource/skill namespaces.
- ContextObject metadata with L0/L1/L2 layer URIs.
- ContextRelation for anchors, evidence, resource requirements, and skill requirements.
- SourceStore as the fact source.
- IndexStore, VectorStore, RelationStore, QueueStore, and LockStore as derived or coordination stores.
- Hierarchical retrieval through QueryPlan, L0 hits, L1 selection, and L2 on-demand loading.
- Token budget packing for action-context retrieval.

Source is authoritative. Indexes are rebuildable.
