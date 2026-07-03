# Product Positioning

MemoryOS is a Predictive Context Database for AI Agents.

It combines a ContextDB substrate with Memory, Behavior, ActionPolicy, PredictionEngine, and asynchronous commit. Memory stores user facts, preferences, rules, anchors, and candidates. Behavior stores evidence chains. ActionPolicy stores q_value, reward, penalty, cooldown, suppress, and execution state.

Non-negotiable invariants:

- Memory, Behavior, and ActionPolicy are semantically independent.
- They share ContextDB, Operation, Relation, L0/L1/L2, Diff, Redo, lifecycle, and index infrastructure.
- BehaviorPattern and ActionPolicy must bind to a MemoryAnchor.
- Prediction results do not directly write durable Memory.
- Automatic execution must pass PolicyGate.
