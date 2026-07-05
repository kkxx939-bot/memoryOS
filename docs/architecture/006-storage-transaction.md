# Storage And Transaction

Filesystem SourceStore stores true context objects and content. SQLite adapters provide persistent index, relation, queue, and metadata stores. Vector adapters expose local and external-vector replacement points.

RedoLog supports recovery of interrupted writes. QueueStore supports semantic refresh, embedding refresh, reindex, and session commit jobs.
