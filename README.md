# m2bOS

m2bOS is currently undergoing a full memory-architecture refactor.

## Current state

The previous long-term memory tree, document-based write path, editing commands, projections, workers, and public write APIs have been removed without a compatibility layer.

The replacement memory tree, conversation parser, Memory Editor, durable-memory schema, indexing strategy, and conversation compression policy have not been implemented yet.

The repository currently retains:

- Conversation and SessionArchive persistence;
- strict session roles for messages and tool interactions;
- generic Context projection, retrieval, and exact reads;
- generic SourceStore, SQLite, Queue, Vector, Relation, locking, and atomic-file infrastructure;
- Behavior and ActionPolicy capabilities that do not write long-term user memory.

Durable long-term memory write, edit, rename, merge, forget, restore, and review interfaces are intentionally unavailable during this stage.

## Development checks

Until the new test suite is designed, production-source changes are checked with Python compilation, Ruff, MyPy, Pyright, runtime/storage smoke checks, and the existing TypeScript integration builds.
