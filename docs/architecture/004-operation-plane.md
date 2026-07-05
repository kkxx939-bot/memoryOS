# Operation Plane

All durable writes use ContextOperation.

Supported actions include add, update, delete, supersede, merge, confirm, reject, reward, penalize, cooldown, suppress, disable, archive, compress, refresh_layers, and reindex.

Commit flow:

1. Build ContextOperation.
2. Coalesce operations by target.
3. Resolve conflicts.
4. Write redo marker.
5. Write SourceStore.
6. Update derived index.
7. Write diff and audit.
8. Clear redo marker.

LLM calls, embeddings, and layer generation do not run inside write locks.
