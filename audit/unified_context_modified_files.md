# Unified Context 主链修改文件清单

本清单以当前差异审计阶段的 `git status --short` 为准。每个变更路径只列一次；`M` 表示已跟踪文件被修改，`??` 表示当前工作区新增且尚未跟踪。链路标签说明文件位于读链、写链、删除链、迁移/运行时、测试或文档中的位置，不代表最终质量门结果。

## 1. 仓库约束、架构、审计与基准

- `AGENTS.md` (`??`) — [文档/约束] 固化 Context First、Canonical State Overlay、统一在线检索、Tombstone 和可回滚迁移等仓库级开发不变量。
- `audit/unified_context_requirement_traceability.md` (`??`) — [审计] 将需求绑定到实现、测试和验收状态，作为第一轮需求追踪矩阵。
- `audit/unified_context_self_checks.md` (`??`) — [审计] 记录四轮独立自检的发现、修复证据及最终全量质量门结果。
- `audit/unified_context_source_call_graph.md` (`??`) — [审计/全链] 从 SDK、HTTP、MCP 重追统一读链，并记录 Session、Canonical、删除、迁移和 Retention 写链。
- `audit/unified_context_modified_files.md` (`??`) — [审计] 枚举本次工作区全部修改路径及各自在主链中的职责。
- `benchmark/unified_context_100k.py` (`??`) — [性能] 提供可独立运行的 100,000 条 Catalog 数据有界查询基准和候选计数输出。
- `docs/architecture/008-production-closure.md` (`M`) — [文档] 校正无 FTS5 时只保留 exact/structured、禁止 contains-scan fallback 的生产降级语义。
- `docs/architecture/MemoryOS第一主链-记忆形成与更新设计.md` (`??`) — [文档/写链] 重建第一主链设计，明确晋升、单 ACTIVE Claim、双投影与 Outbox/Tombstone 边界。
- `docs/architecture/canonical-memory-change-checklist.md` (`??`) — [文档/验收] 给出 Canonical 变更对 Identity、Evidence、Projection、检索、迁移和回滚的逐项检查表。
- `docs/architecture/canonical-memory-pipeline.md` (`??`) — [文档/Canonical] 定义 Canonical 形成、事务、Receipt/Head、Claim Revision 和 Current Slot 投影不变量。
- `docs/architecture/unified-context-storage-retrieval.md` (`??`) — [文档/全链] 完整说明 Evidence、Catalog/Tree、Session 投影、统一检索、Retention、迁移及 OpenViking 吸收边界。

## 2. 公共 API、可信边界与安全投影

- `memoryos/adapters/agent_hooks/events.py` (`M`) — [Agent/Workspace 安全] 复用稳定 Workspace identity 计算，不把仓库绝对路径直接带入 Project/Serving identity。
- `memoryos/__init__.py` (`M`) — [公开 API] 在顶层稳定导出结构化 RetrievalOptions 与 RetrievalQueryPlan。
- `memoryos/api/http/app.py` (`M`) — [HTTP 读链] 解析共享 RetrievalOptions，绑定可信 Workspace/Scope/ACL 后调用同一 SDK 检索入口。
- `memoryos/api/mcp/config.py` (`M`) — [MCP 安全边界] 从受信运行配置加载可授权 applicability scope keys，并注入 MCP caller context。
- `memoryos/api/mcp/errors.py` (`M`) — [MCP 失败语义] 将有界检索不可用映射为可重试 `NOT_READY`，保留 sanitized degraded modes 而不伪装空成功。
- `memoryos/api/mcp/schemas.py` (`M`) — [MCP 契约] 复用共享 RetrievalOptions JSON Schema，并让新短工具名与旧工具名保持同一 Schema。
- `memoryos/api/mcp/tools.py` (`M`) — [MCP 读链] 将 search/assemble 别名汇入同一 handler，转换结构化选项并在调用 SDK 前拒绝 Scope 扩权。
- `memoryos/api/sdk/__init__.py` (`M`) — [SDK 契约] 从 SDK 包稳定导出结构化检索选项和查询计划。
- `memoryos/api/sdk/client.py` (`M`) — [SDK/全链] 将 search、assemble、archive_search 汇入统一 Orchestrator；兼容旧参数，并把普通 forget 接到耐久 Tombstone 清理。
- `memoryos/api/sdk/http_client.py` (`M`) — [远程 SDK] 序列化结构化 options，同时保留原 search/assemble 调用形式。
- `memoryos/api/trusted_context.py` (`M`) — [安全边界] 建模可信 Scope grants，校验 Owner/Workspace/applicability scope 只能继承或缩小授权范围。
- `memoryos/api/retrieval_contract.py` (`??`) — [共享契约] 为 HTTP 与 MCP 提供唯一的 RetrievalOptions Schema、解析和序列化实现。
- `memoryos/security/context_projection.py` (`??`) — [安全投影] Fail closed 清洗凭证、绝对路径、超长/Binary Tool 输出、metadata 与 Recall Trace，再允许进入派生层。
- `memoryos/security/workspace_identity.py` (`??`) — [Workspace 安全] 统一规范化逻辑 Workspace ID，并把路径/remote-like identity 转成不泄漏原值的稳定 `project-*` identity。

## 3. Catalog、Store、删除、Retention 与迁移基础设施

- `memoryos/contextdb/context_db.py` (`M`) — [管理/删除/迁移] 暴露耐久 Context/Session 删除、Retention/Compaction/Cold restore，并用共享锁封闭安全 rebuild 与 Current Slot 回填。
- `memoryos/contextdb/catalog.py` (`??`) — [Catalog 模型] 定义 Serving record、record kind/tier/status、受控多路径 taxonomy、结构化时间及安全 Vector metadata。
- `memoryos/contextdb/layers/layer_refresher.py` (`M`) — [安全层投影] 在写入可重建 L0/L1 前执行 fail-closed Sanitization，L2 仍只作为受 SourceStore 权限保护的原始 Evidence，并把刷新动作纳入 migration projection fence。
- `memoryos/contextdb/ordinary_relations.py` (`??`) — [普通 Relation 权威边界] 从非 Canonical Source 对象确定性重建普通关系，统一校验 Tenant/Owner、endpoint lifecycle、Canonical target proof 与 Session delete barrier；RelationStore 始终只是派生索引。
- `memoryos/contextdb/projection_equivalence.py` (`??`) — [迁移证明] 从不可变 Evidence 与精确 Catalog 回读构建有硬上限、payload-free 的 expected/actual projection 等价性证明。
- `memoryos/contextdb/resource/resource_importer.py` (`M`) — [Resource 写链] 用 migration projection fence 串行化 Resource Source/Index 写入与 rebuild/cutover，避免迁移期间丢投影。
- `memoryos/contextdb/retention.py` (`??`) — [生命周期/删除链] 实现 HOT/WARM/COLD/ARCHIVED 迁移、Session/Timeline 压缩、派生层 GC、Tombstone 重放和冷数据恢复。
- `memoryos/contextdb/skill/skill_registry.py` (`M`) — [Skill 写链] 把 Skill Source/Index 注册纳入同一 projection fence，保持 Catalog rebuild 与在线写入原子边界。
- `memoryos/contextdb/store/index_consistency.py` (`M`) — [Repair/Rebuild] 在重建时保留无普通 Source 对象的 Session 等派生记录，并避免把它们误报为孤儿。
- `memoryos/contextdb/store/local_stores.py` (`M`) — [测试 Store/兼容] 让内存 Index/Relation Store 对齐 Catalog record key、Current Slot 唯一性、过滤与有界 Relation 查询，并实现与 SQLite 一致、旧未归属 job 仅在自身 Tenant 内阻断的 Queue health 语义。
- `memoryos/contextdb/store/source_store.py` (`M`) — [Store 契约] 为 RelationStore 查询增加可选 limit，并为 QueueStore 增加按 Tenant/Owner/Workspace 读取健康状态的有界契约。
- `memoryos/contextdb/store/sqlite_index_store.py` (`M`) — [Catalog/FTS/迁移/删除] 将既有 contexts 原位升级到 schema v10，增加 ACL/Path/Validity/FTS rowid 等规范化附属表与索引、数据库前置过滤、VM bound、两阶段 Tombstone CAS、双 Journal 和 scoped Session frontier；pre-v10 升级在事务内记录耐久 provenance，并原子绑定实际 Tenant migration gate。
- `memoryos/contextdb/store/sqlite_queue_store.py` (`M`) — [Projection 健康] 原位迁移 Tenant/Owner/Workspace scope 列与 `queue_jobs_scope_status_idx`，从 job payload 或安全旧 URI 回填；scoped 与 legacy-unresolved 使用两个 Tenant-bound 索引查询，旧未归属任务只在自身 Tenant 内 fail closed。
- `memoryos/contextdb/store/sqlite_relation_store.py` (`M`) — [Relation 候选/删除] 增加确定性有界查询及 Source/Target+Scope 索引，支持在线扩展和 Tombstone 清理。
- `memoryos/contextdb/store/vector_store.py` (`M`) — [Vector 候选/删除] 声明 metadata/namespace/time/delete 能力，提供有界候选接口并阻止本地后端伪装生产 filtered Top-K。
- `memoryos/contextdb/tombstone.py` (`??`) — [删除链] 实现可重放 Projection Tombstone 服务，以 source proof 和 record ownership 清理 Catalog、Vector、Relation 等派生层。
- `memoryos/contextdb/transaction/consistency.py` (`M`) — [Audit/一致性] 让 Source/Index/Relation 全局审计绑定当前 Tenant，防止跨租户对象或关系污染修复结论。
- `memoryos/contextdb/transaction/recovery.py` (`M`) — [Recovery] 在恢复验证 relation effects 时显式绑定 Committer Tenant，避免同 URI 的外租户派生关系被误认作当前事务效果。
- `memoryos/contextdb/unified_migration.py` (`??`) — [迁移] 实现状态机、分批 checkpoint/resume、Session/CurrentSlot backfill、双写、Shadow diff、Cutover 与 Rollback gate；区分真实 greenfield 与旧 Archive，未回填兼容读为空时 fail closed。

## 4. 统一 Retrieval 在线读链

- `memoryos/contextdb/retrieval/__init__.py` (`M`) — [读链边界] 仅导出统一产品检索契约与组件，不再把离线旧 Retriever 暴露为在线入口。
- `memoryos/contextdb/retrieval/context_assembler.py` (`M`) — [兼容读链] 把旧 ContextAssembler 收缩为统一 Orchestrator facade，删除原先的全库 allow-list/多主链组装逻辑。
- `memoryos/contextdb/retrieval/hierarchical_retriever.py` (`M`) — [离线边界] 将旧分层检索显式限制为需确认的 offline admin/audit 工具，防止回接产品在线链。
- `memoryos/contextdb/retrieval/hybrid_search.py` (`M`) — [内部候选] 移除 vector_uris 全枚举，改为候选内向量或硬上限 overfetch，并标记 degraded mode。
- `memoryos/contextdb/retrieval/query_plan.py` (`M`) — [查询契约] 定义 RetrievalOptions/Plan、Intent、时间/时区/路径规范化和可序列化安全边界，同时保留旧 QueryPlan import。
- `memoryos/contextdb/retrieval/service.py` (`M`) — [Recall Trace] 用统一 Sanitizer 清洗并校验 Trace，且无法保护 trace 目录权限时 fail closed。
- `memoryos/contextdb/retrieval/candidate_generator.py` (`??`) — [候选生成] 在 Top-K 前执行 SQL 结构化过滤，生成 exact、FTS、native/bounded vector 与有界 relation 候选和真实分支计数；带文本时只允许 OPEN_RECALL/event range、AS_OF/valid_at、HISTORY/transaction range 三种确定性 structured branch，普通语义查询不枚举 Catalog。
- `memoryos/contextdb/retrieval/canonical_resolver.py` (`??`) — [Canonical Overlay] 只对最终有界候选回源校验 Slot/Claim/Revision/Head/Receipt/Effect；只保留候选中的 serving/proof 白名单；CURRENT 从 proved Source current state 重建业务 metadata，HISTORY/AS_OF/CONFLICTS/OPTIONS 的正文和业务 metadata 完整绑定请求的 immutable Revision；AS_OF 使用派生半开区间且只接受时点 ACTIVE Revision，过期、候选污染、证明篡改或清洗失败均 fail closed。
- `memoryos/contextdb/retrieval/errors.py` (`??`) — [有界失败契约] 定义 Catalog candidate/SQLite VM 超界的 typed failure，供在线入口转换为显式 unavailable。
- `memoryos/contextdb/retrieval/fusion.py` (`??`) — [融合/去重] 用可解释 RRF 融合各分支分数，并按 Intent、Slot、Claim revision、Session 与 Source identity 去重。
- `memoryos/contextdb/retrieval/orchestrator.py` (`??`) — [统一编排] 串联 Plan、Candidate、Fusion/Rerank、Canonical validation、Packing、Trace、降级和迁移 Shadow validation；只在最终有界普通 Resource 候选上精确读取并复核 Source L2，重新清洗后才允许进入 Packer，atomic Message/Tool Result 禁止 raw 回读。
- `memoryos/contextdb/retrieval/packing.py` (`??`) — [上下文组装] 实现 Intent/Agent 优先级、L2→L1→L0→URI 降级、Token 与类型/Slot/Session/Resource 配额及 drop reason；用同一真实 pack quota 预选最多 `max_l2_items` 个 Resource L2 hydration identity。
- `memoryos/contextdb/retrieval/query_planner.py` (`??`) — [Query Planner] 把旧平铺参数、retrieval_views/search_scope 转成统一 options，并按可信约束优先级绑定 Tenant/Owner/Workspace/Adapter/ACL；确定性区分 event-time OPEN_RECALL、valid-time AS_OF 与 transaction-time HISTORY。

## 5. SessionArchive 写入与 Serving 投影

- `memoryos/contextdb/session/session_commit.py` (`M`) — [Session 写链] 在 Archive 提交后按 Migration Feature Gate 执行 Session Catalog 双写并回报投影状态/数量。
- `memoryos/contextdb/session/session_model.py` (`M`) — [Session 写链契约] 在提交结果中增加投影状态和投影记录数，暴露最终一致性进度。
- `memoryos/contextdb/session/context_projector.py` (`??`) — [Session 投影] 将 Root/摘要/segment/message/tool/resource/used context/skill/event 投影到 Catalog/FTS/受控 Vector 和多路径 Tree，且先安全清洗；Root/L0/L1 与引用记录的 Timeline 始终和各自 event_time 同源；事件时间兼容 `occurred_at`/`event_time` 且前者优先，Semantic Segment 按本地 Timeline 日界强制切分。

## 6. Canonical 写链、双投影与运行时装配

- `memoryos/behavior/retrieval/similar_behavior_retriever.py` (`M`) — [隔离的 Behavior 读链] 对有界 behavior case 候选去重并保留完整候选分数解释，不接入通用 Context 检索。
- `memoryos/memory/canonical/__init__.py` (`M`) — [Canonical API] 导出 PromotionPolicy 与离线 Canonical 审计 reader，并保持在线产品检索边界清晰。
- `memoryos/memory/canonical/current_head.py` (`M`) — [Canonical proof] 从不可变 Receipt snapshot 重建历史 head digest，避免历史投影依赖后来变化的 Current Head 指针。
- `memoryos/memory/canonical/episode.py` (`M`) — [Evidence 时间] 在 Session Evidence 边界确定性支持 `occurred_at > event_time > created_at`，使 Canonical 形成和 Unified Timeline 使用同一事件时间语义。
- `memoryos/memory/canonical/final_state.py` (`M`) — [Canonical 最终状态] 在验证 Slot/Claim relation membership 时绑定权威 Tenant，阻止跨租户同 URI 关系参与事务 final-state proof。
- `memoryos/memory/canonical/state.py` (`M`) — [Canonical 时间] 在不改写不可变 Source Revision 的前提下，从下一非 historical Revision 确定性派生 effective `valid_to`，并对重复编号、非法时区和反向区间 fail closed。
- `memoryos/memory/canonical/formation.py` (`M`) — [Canonical 形成] 在 Identity/事务前接入确定性 PromotionPolicy，并为重复证据和跨 Owner 情况保持权威状态边界。
- `memoryos/memory/canonical/prefetch.py` (`M`) — [Canonical Prefetch] 将 Current Slot Catalog serving URI 有界解析到权威 ACTIVE Claim，并复核 Claim/Slot ID、URI、各自 Revision、Tenant 与 active pointer，错绑或过期时 fail closed。
- `memoryos/memory/canonical/projection.py` (`M`) — [Claim/CurrentSlot 写链] 保留 revision-scoped Claim Projection，接入安全 Catalog/Vector metadata，并从耐久 Outbox 驱动 Current Slot 更新或 Tombstone。
- `memoryos/memory/canonical/retrieval.py` (`M`) — [离线 Repair/Audit] 将全量 Canonical snapshot reader 改名并强制 offline_admin，明确禁止作为 SDK/HTTP/MCP 在线链。
- `memoryos/memory/canonical/transition.py` (`M`) — [Canonical 状态转换] 把证据变化纳入 revision 判定，同时保留特殊跨 Owner duplicate evidence 策略和单 ACTIVE Claim 不变量。
- `memoryos/memory/canonical/visibility.py` (`M`) — [Canonical Relation 修复] 按 SourceStore Tenant 对 committed relation snapshot、删除和重建全程隔离，跨租户 Source snapshot 直接 fail closed。
- `memoryos/memory/view.py` (`M`) — [Workspace Scope] 统一从 Archive/connect metadata 生成规范化 Project/Workspace identity，阻止路径泄漏与同仓库 identity 分叉。
- `memoryos/memory/canonical/promotion_policy.py` (`??`) — [Canonical 准入] 以 TransitionProfile、Evidence、Identity、Scope、Authority、Schema 和确定规则决定 Context 是否晋升，LLM 不能单独决定。
- `memoryos/memory/canonical/slot_projection.py` (`??`) — [Current Serving] 实现每 Slot 唯一的 CurrentSlotProjection、完整 proof 校验、原位 active claim 切换、Vector ownership 和耐久 Tombstone。
- `memoryos/operations/commit/effect_marker.py` (`M`) — [Receipt/Recovery proof] 校验普通 relation effect 时绑定 Receipt Tenant，阻止同 URI 外租户行满足当前事务 marker。
- `memoryos/operations/commit/operation_committer.py` (`M`) — [Canonical/删除/普通关系事务] 在 Source DELETE 前绑定耐久 Projection Tombstone，把 redo resume 绑定到唯一完整性校验的 durable entry，并把 Canonical Outbox Queue job 绑定到 Tenant/Owner/Workspace；公共普通 relation 对 retired endpoint fail closed，只有 Schema 声明的 retired ActionPolicy 关系可作为 Source-only Evidence 保存且不发布 RelationStore。
- `memoryos/operations/commit/outbox_envelope.py` (`M`) — [Outbox Queue scope] 从原始或规范化 Canonical operations 确定唯一 Workspace，使首次 enqueue 与 restart dispatch 生成完全一致的 Queue identity。
- `memoryos/operations/commit/redo_log.py` (`M`) — [Recovery] 收紧 redo entry 的 durable identity/effect 校验，为 Tombstone 绑定和幂等 resume 提供可信恢复记录。
- `memoryos/prediction/pipeline/action_context_builder.py` (`M`) — [Embodied 隔离链] 为 ActionPolicy/Behavior/Resource/Skill 的关系和对象读取统一补 Tenant 过滤，仍保持现有 PolicyGate 动作边界，不接入通用 Context API。
- `memoryos/runtime/config.py` (`M`) — [运行时配置] 增加类型安全的 Retention 配置入口并保持安全默认值。
- `memoryos/runtime/container.py` (`M`) — [运行时装配/启动] 统一装配 Session/Claim/CurrentSlot 投影、Vector capability gate、Tombstone、Retention、Migration 与启动期有界回填/全局审计。
- `memoryos/workers/embedding_worker.py` (`M`) — [Vector 派生] 对 Catalog-owned embedding/vector publication 绑定安全 metadata、record ownership 与删除/重建语义。
- `memoryos/workers/memory_proposal_worker.py` (`M`) — [Pending Proposal Worker] 在 lease、恢复与提交期间持有 migration projection fence，防止 cutover/rebuild 与 pending proposal 消费交错。
- `memoryos/workers/reindex_worker.py` (`M`) — [离线 Rebuild Worker] 用耐久 projection fence 和周期 checkpoint 包围允许的 O(N) Source rebuild，明确与在线 bounded 检索隔离。
- `memoryos/workers/runner.py` (`M`) — [Worker 装配] 向 Semantic/Embedding 等 worker 注入同一 MigrationGate，使后台派生写遵守 cutover/rollback fence。
- `memoryos/workers/semantic_worker.py` (`M`) — [L0/L1 Worker] 在队列 lease、层刷新和 ack 周期持有 projection fence，并复用安全 LayerRefresher。
- `memoryos/workers/session_commit_worker.py` (`M`) — [Session Worker] 在 commit-group recovery、Archive/Catalog 投影和 queue ack 期间持有 projection fence，保证迁移可恢复一致性。

## 7. 既有集成与单元测试的兼容/边界更新

- `tests/e2e/test_canonical_memory_formation_flow.py` (`M`) — [E2E/Canonical 形成] 适配 Unified CURRENT/HISTORY 的语义去重与 bounded serving 结果，继续证明 Coding Agent 只形成/召回记忆而不越过动作边界。
- `tests/integration/test_canonical_concurrency_closure.py` (`M`) — [集成/并发] 验证 rebuild 与查询快照隔离、Receipt 历史 proof，以及 head 并发推进时有界最终校验 fail closed。
- `tests/integration/test_canonical_long_history.py` (`M`) — [集成/历史] 适配显式 offline canonical reader，并继续验证长 revision 历史与唯一 Current Head。
- `tests/integration/test_canonical_rebuild_closure.py` (`M`) — [集成/Rebuild] 验证 Canonical rebuild、Projection publication proof、Current Slot 收敛及临时卷路径等价性。
- `tests/integration/test_http_redirect_security.py` (`M`) — [集成/HTTP 安全] 让本地重定向测试服务器在返回 302 前消费请求体，消除完整测试负载下的连接复位夹具竞态，同时保持生产重定向安全断言不变。
- `tests/integration/test_hybrid_retrieval.py` (`M`) — [集成/候选边界] 覆盖 bounded vector overfetch、ACL/Source 权威过滤、异常降级及隔离 Behavior/ActionPolicy 检索兼容。
- `tests/integration/test_memory_artifact_tamper_matrix.py` (`M`) — [集成/Canonical 防篡改] 扩展 Receipt、Current Head、Projection effect 与派生索引篡改矩阵，验证在线 Canonical 解析 fail closed。
- `tests/integration/test_memory_tenant_closure.py` (`M`) — [集成/Tenant] 适配统一 Current Slot 输出并继续证明 callerless SDK 的 Canonical Source、Receipt、Projection 和 Recall 全部 Tenant-local。
- `tests/integration/test_pending_review_concurrency.py` (`M`) — [集成/Pending Crash Recovery] 验证并发 review 单赢家，并证明 crash-resume 后历史 Claim Catalog row 重新绑定原不可变 publication attempt，而非临时 rebuild attempt。
- `tests/integration/test_prediction_loads_policies_from_contextdb.py` (`M`) — [集成/ActionPolicy 边界] 让策略走真实 OperationCommitter；证明 retired Policy 的 schema relation 留在 Source Evidence、RelationStore 不在线服务且 Prediction 仍经 PolicyGate 过滤。
- `tests/integration/test_relation_store_tenant_isolation.py` (`M`) — [集成/Relation 隔离] 证明相同 URI relation 的读取与删除必须绑定 Tenant，禁止跨租户误删。
- `tests/integration/test_sqlite_index_store.py` (`M`) — [集成/Catalog] 扩展 SQLite 索引行为断言以适配多 record_key Catalog 与结构化 serving 语义。
- `tests/integration/test_sqlite_index_store_query_escape.py` (`M`) — [集成/FTS] 验证查询转义、中文匹配、有限分数与 FTS 关闭时不退回在线 contains 全表扫描。
- `tests/integration/test_sqlite_queue_store.py` (`M`) — [集成/Queue] 验证旧 Schema 原位增加 scope 列、subject URI 的 Tenant/Owner/Workspace 隔离、旧未归属 job 不跨 Tenant 阻断，并以真实 trace + EXPLAIN 证明两条健康 SQL 都使用 `queue_jobs_scope_status_idx`。
- `tests/integration/test_tenant_projection_root.py` (`M`) — [集成/租户] 适配离线 canonical reader，继续证明非默认 Tenant 的投影与检索只使用注入 Store。
- `tests/integration/test_workers_production_stores.py` (`M`) — [集成/Worker 安全] 证明 Semantic/Embedding 输入先清洗，L0/L1/Vector 不含 Secret/私有路径，L2 保留 Source Evidence，Sanitizer 失败时 Provider/Vector 均不被调用。
- `tests/unit/contextdb/test_contextdb_final_components.py` (`M`) — [单元/组件] 适配 offline hierarchical reader 的显式 admin 边界并保持 ContextDB 组件契约覆盖。
- `tests/unit/test_canonical_projection.py` (`M`) — [单元/Claim Projection] 覆盖安全层/FTS/Vector metadata、revision record key、派生 valid_to 且 Source 不改写、单次 Vector refresh、迟到写保护、Crash recovery 与投影幂等。
- `tests/unit/test_canonical_retrieval.py` (`M`) — [单元/Canonical 读边界] 适配离线 reader，并验证 CURRENT/HISTORY 分离、bounded Source reads、proof 篡改与权限过滤。
- `tests/unit/test_coding_agent_infrastructure.py` (`M`) — [单元/Coding Agent Packing] 将极小预算断言收紧为合同规定的 URI 层，禁止重新引入 `excerpt` 伪层。
- `tests/unit/test_connect_api_contracts.py` (`M`) — [单元/公开 API] 验证 search/assemble 统一 options 与 connect filters，同时保持 Coding Agent/Embodied PolicyGate 隔离。
- `tests/unit/test_context_db_facade.py` (`M`) — [单元/普通 Relation] 验证 `add_relation` 通过 Source/redo 边界幂等持久化并修复派生行，且未获 Canonical Receipt 证明的 target 必须 fail closed。
- `tests/unit/test_mcp_server_tools.py` (`M`) — [单元/MCP] 覆盖共享 options、Scope 传递、结构化结果与既有工具/动作安全边界。
- `tests/unit/test_planner_isolation_and_commit_group.py` (`M`) — [单元/并发写链] 保持 planner 请求隔离，并验证 Commit Group 只重试失败的 Session Projection consumer。
- `tests/unit/test_release_safety_primitives.py` (`M`) — [单元/安全] 增强统一 API limit、有限分数、时间区间及 Recall Trace 全字段清洗/fail-closed 断言。
- `tests/unit/test_structural_boundaries.py` (`M`) — [单元/架构边界] 静态禁止在线模块使用全量 Source/Vector/Archive/Canonical snapshot API，并限制产品检索导出面。
- `tests/unit/test_target_resolver.py` (`M`) — [单元/Resolver 契约] 让低置信测试桩继承真实 IndexStore endpoint-state 能力，继续验证目标解析 fail closed 且无 Protocol 假实现。

## 8. 新增端到端、集成与单元证明

- `tests/e2e/test_unified_canonical_state_scenarios.py` (`??`) — [E2E/Canonical] 证明重复偏好只产生一条 Current Slot、状态改变原位切换且 CURRENT/HISTORY 分离；真实 `remember()` Claim Revision 在无中文词面重合时仍可由自然 transaction-time HISTORY 与 valid-time AS_OF 公共链召回并完成 proof validation。
- `tests/integration/test_derived_serving_rebuild.py` (`??`) — [集成/全派生重建] 证明普通/Canonical Relation、Session/Catalog、Claim History、Current Slot、Vector 与 Retention 可从事实源重建，并验证 deleted endpoint barrier 防止在线与后续 rebuild 关系复活。
- `tests/integration/test_derived_serving_rebuild_fence.py` (`??`) — [集成/Rebuild 并发] 以跨进程 durable fence 验证 rebuild/cutover 与 Source/Worker 写入互斥、checkpoint 可检测失锁且失败可恢复。
- `tests/integration/test_projection_tombstone_service.py` (`??`) — [集成/删除链] 覆盖两阶段 CLEANING 重试、>1,000 分页、Session/forget 清理及 stale/foreign ownership 不误删新投影。
- `tests/integration/test_retrieval_scope_acl_binding.py` (`??`) — [集成/ACL] 从公开 search 入口证明用户参数不能扩大 Principal、Workspace、Adapter 或 applicability scopes。
- `tests/integration/test_runtime_session_vector_retention.py` (`??`) — [集成/运行时] 验证安全 Session Vector 写入、投影/Retention 失败的耐久重放及生产 Vector capability gate。
- `tests/integration/test_unified_context_catalog_store.py` (`??`) — [集成/Catalog] 覆盖 schema v10 原位迁移、多路径/时间/FTS、FTS rowid map 与 tenant-record 附属更新、Current Slot 唯一键、Tombstone 状态和 EXPLAIN 索引使用；证明三类自然时间计划即使无词面重合也走有界 structured SQL，并反向证明普通 semantic query 不开启 Catalog listing。
- `tests/integration/test_unified_context_migration_retention.py` (`??`) — [集成/迁移生命周期] 覆盖 batch checkpoint/resume、Shadow/Cutover/Rollback、tier/compaction/GC/restore 且保留 Evidence。
- `tests/integration/test_unified_migration_runtime_gate.py` (`??`) — [集成/Feature Gate] 以真实 v2 首启、Archive-only、greenfield/restart、Tenant 隔离证明 legacy/dual-write/shadow/cutover/rollback 路由与既有 Current Slot backfill。
- `tests/integration/test_unified_retrieval_scale_and_bounds.py` (`??`) — [集成/性能边界] 构造 1,000 Sessions、10,000 Tool Results、1,000 Slots，验证候选/Source read 上限和在线零全库扫描调用。
- `tests/integration/test_unified_retrieval_l2_hydration.py` (`??`) — [集成/L2 主链] 通过公开 SDK 证明普通 Resource 的精确、安全、有界 L2 hydration 及 L2→L1→L0→URI 预算降级，并用 forbidden spies 证明 Message/Tool Result 在线召回不读取 raw Source 或 SessionArchive。
- `tests/integration/test_unified_session_retrieval_e2e.py` (`??`) — [集成/Session Recall] 证明 Tool Result 独有 Desktop 文件名、普通 HISTORY 与 transaction-time 候选经统一 Retrieval 按日期/路径/ACL/SQL limit 召回且普通 Context 不创建 Slot/Claim；真实 SessionArchive 的自然 event-time 问句无需词面重合即可返回 Source URI，公共链同时覆盖 `occurred_at`/`event_time` 优先级。
- `tests/unit/test_bounded_canonical_resolver.py` (`??`) — [单元/Canonical Overlay] 验证失败 Claim read 计入 Source read且不超过 candidate_limit，并证明 CURRENT/HISTORY 回源 Secret/路径不会通过正文或 metadata 泄漏。
- `tests/unit/test_canonical_promotion_policy.py` (`??`) — [单元/晋升] 覆盖 AUTHORITATIVE/OBSERVATIONAL/EXPERIENCE、显式 Event、确定门禁与原始日志拒绝规则。
- `tests/unit/test_context_projection_sanitizer.py` (`??`) — [单元/安全] 覆盖 Secret/绝对路径/metadata 清洗、受控 taxonomy、动态路径伪名化及失败关闭。
- `tests/unit/test_current_slot_outbox_projection.py` (`??`) — [单元/Outbox] 验证 active claim 切换的 Tombstone 在 ack 前耐久重试且最终幂等收敛。
- `tests/unit/test_current_slot_projection.py` (`??`) — [单元/Current Slot] 覆盖重复/多值偏好、原位切换、retract、树重分类、SQLite 唯一记录及 proof 篡改 fail closed。
- `tests/unit/test_fusion_ranker.py` (`??`) — [单元/融合] 验证 RRF 不直接比较原始分数、分数组件可解释、Intent 去重和 per-session limit。
- `tests/unit/test_retrieval_failure_semantics.py` (`??`) — [单元/失败与健康边界] 验证 VM/候选超界、Vector 故障、Canonical queue 与 Session frontier lag 显式 fail closed，并证明 Owner/Workspace health 隔离。
- `tests/unit/test_retrieval_query_planner.py` (`??`) — [单元/Planner] 覆盖结构化 round trip、时区跨日、event/valid/transaction 三类自然时间语义、旧参数映射和 trusted constraints 只可收窄。
- `tests/unit/test_session_context_projector.py` (`??`) — [单元/Session 投影] 覆盖 Tool/Resource 分类、Root/L0/L1/Used Context/Used Skill 跨日 event_time 与 Timeline 同源、`occurred_at` 优先和 `event_time` alias、Semantic Segment 本地日界切分、幂等、安全向量策略及同 Session manifest ownership 隔离。
- `tests/unit/test_unified_context_packer.py` (`??`) — [单元/Packing] 覆盖分层降级、Token/类型/Slot/Session/Resource 配额、HISTORY 去重和 Coding Agent 优先级。
- `tests/unit/test_unified_retrieval_api_contract.py` (`??`) — [单元/API 兼容] 证明 HTTP/MCP 共用 Schema、MCP 长短名共用 handler，SDK 调统一 Orchestrator，archive_search 仅为安全兼容包装。
- `tests/unit/test_vector_capabilities.py` (`??`) — [单元/Vector] 验证本地 bounded fallback、生产能力声明及 native metadata-filtered Top-K 可独立召回。
