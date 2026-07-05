# MemoryOS Benchmark

This directory is intentionally outside the core `memoryos` package.

Current scope:

- `smoke/`: tiny import and runtime checks that call public APIs only.

Future benchmark directions:

- LoCoMo-style long-context memory retrieval evaluation.
- LongMemEval-style memory persistence and recall checks.
- RAG memory evaluation for context assembly and ranking.

Benchmarks must not be imported by the core package and should avoid heavyweight dependencies unless a dedicated benchmark environment is added.
