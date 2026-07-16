# Canonical Memory Change Checklist

本清单是变更门禁，不是“有类名即完成”的功能清单。每一项必须能用源码调用链、真实存储测试或查询计划证明；任一拒绝条件成立时不得 Cutover。

## A. 基线与范围

| 检查项 | 必须证明 | 拒绝条件 |
| --- | --- | --- |
| Source baseline | 分支、HEAD、工作区、未跟踪文件、参考项目实际 SHA | 使用旧基线假设或覆盖用户修改 |
| 约束文件 | 已读 AGENTS 与四份架构文档 | 修改前未核对不变量 |
| 调用者 | SDK、HTTP、MCP、workers、hooks、repair/migration 全部搜索 | 只改一个入口 |
| 语义范围 | Memory、Behavior、ActionPolicy 保持隔离 | 借本变更扩展 Prediction/ActionPolicy |

## B. 九条强制约束

| 约束 | 源码证据 | 验证 |
| --- | --- | --- |
| Slot/Claim 只服务状态型长期记忆 | `CanonicalPromotionPolicy` + canonical formation | ordinary context 不产生 Slot/Claim |
| 普通 Session/Resource/Tool Result 使用统一 Catalog | `SessionContextProjector` | Tool Result/Resource 可搜索且 `canonical_*` 为空 |
| Claim Projection 与 Slot Current Projection 并存 | `MemoryProjectionWorker` + `CurrentSlotProjection` | CURRENT/HISTORY 分离 |
| 所有在线检索经过统一编排 | `RetrievalOptions`、`QueryPlanner`、`UnifiedRetrievalOrchestrator` | SDK/HTTP/MCP/archive wrapper 同一 plan |
| Tree Path 不参与 Canonical Identity | Identity V2 计算与 tree reclassify | Slot/Claim ID/URI 不变 |
| 在线查询禁止全库扫描 | CandidateGenerator/Resolver 的 bounds | spy 阻止 list_objects/snapshot/vector_uris/glob/archive scan |
| Tool Result 必须 Sanitization | `ContextProjectionSanitizer` | FTS/Vector/Trace/metadata 均无 secret/绝对路径 |
| 删除通过 Tombstone 传播 | `ProjectionTombstoneService` + durable table | retry/replay/derived cleanup |
| 迁移支持 Cutover 和 Rollback | `UnifiedContextMigration` + feature gate | checkpoint/shadow threshold/rollback legacy route |

## C. 事实源与 Schema

- `SQLiteIndexStore.contexts` 是唯一主 Serving Catalog；不得增加重复主表。
- SourceStore 仍是普通 Context 事实源，SessionArchive 仍是不可变 session evidence，Canonical Source 仍是 Slot/Claim 权威层。
- 当前 Catalog Schema version 为 v10；增量列、legacy table rebuild、sanitize backfill、FTS/map rebuild 必须幂等。
- `context_paths`、`context_path_closure`、`context_path_acl`、`context_acl_grants`、`context_tenants`、`context_validity_map`/RTree、`context_fts_map`、`context_links`、`context_projection_state`、`context_tombstones`、`migration_state`、双 migration journal 与 `session_projection_frontier` 只能保存可重建候选或耐久控制状态。
- ACL/Path/RTree/FTS token 只能生成候选；必须在 Top-K 前以 `contexts` 精确字段和 trusted Tenant/Owner/Workspace/Scope 再校验。
- `context_fts_map` 必须维持 `record_key -> FTS rowid` 一对一完整性；不一致时启动离线重建，不得恢复按 FTS `UNINDEXED record_key` 的在线逐行删除。
- `context_paths` 必须有 tenant/path、tenant/owner/path、tenant/type/path、tenant/time/path 和 URI 反查索引；一个 record 只有一个 primary path。
- 时间字段 created/updated/event/ingested/transaction/valid_from/valid_to 必须是结构化列并建立适用索引，不能依赖 metadata JSON 模糊查找。
- Session Root/L0/L1、Message、Tool Result、Event、Resource、Semantic Segment、Used Context 与 Used Skill 的 Timeline 必须和各自 `event_time` 同源；`occurred_at` 优先于 `event_time` 别名，无显式引用时间时回退 episode start，不得回退 archive write time；Semantic Segment 不得跨调用方本地 Timeline 日界。
- Canonical Source 旧 Revision 不回写 `valid_to`；Catalog/Vector 与 AS_OF Resolver 必须按下一非 historical Revision 派生同一半开区间。
- Current Slot 唯一键必须是 tenant + canonical_slot_id + current_slot 条件唯一索引。

## D. Canonical 不变量

- Identity V2、Slot/Claim deterministic ID 与 Canonical URI 未改变。
- Tree/日期/文件路径、文件名、标题、L0/L1、Vector/Projection/Session/Tool Result ID 未进入 Hash。
- 每个 Slot 最多一个 ACTIVE Claim；多值使用集合或更细 Slot。
- `active_claim_id`、EvidenceRef、field evidence、Receipt、Current Head、Redo/Recovery、Pending Proposal/Review 保持有效。
- `remember(memory_type="event")` 仍能创建长期 Canonical Event；普通 Event 默认 Catalog-only。
- Canonical Current 篡改、head/receipt/effect mismatch、scope/authority failure 均 fail closed。
- 启动期全局完整性检查仍存在；在线只校验不超过 candidate limit 的候选。

## E. 投影、删除与生命周期

- Claim revision record key 不会因同一 Claim 新 revision 覆盖历史。
- Slot current record key 恒为 `slot:{slot_id}:current`，active Claim 变化原位更新。
- Superseded Claim 不参与 CURRENT，但仍参与显式 HISTORY/AUDIT。
- Unified HISTORY 必须保留普通 Session/Event/Resource/Tool Result 和 Claim Revision，并在 Top-K 前排除 Current Slot。
- Canonical 精确回源后的公开 title/text/metadata 必须再次 Sanitization；清洗失败 fail closed，证明字段必须复核。
- Canonical Outbox 成功后才发布派生；投影失败不 ACK 成功。
- Tombstone 可持久化、幂等、重放；校验 Source 后先以 `begin` 事务摘除 Catalog/FTS/Path/Link/Projection 并进入 CLEANING，再按 record key/revision/effect 做 Vector/Relation CAS 清理，最后 APPLIED；竞态为 STALE，外部失败保留 CLEANING。
- Session 删除必须同时保留稳定 `session_delete_barrier`；APPLIED 抑制未来新增 record kind，CLEANING 阻止 Archive rebuild 前进，且 barrier 未证明可安全遗忘时不得 Tombstone GC。
- ServingTier HOT/WARM/COLD/ARCHIVED 不写入 ClaimState。
- Session/Timeline compaction、stale projection/vector/path/tombstone GC 有 batch 上限；默认不删除 immutable Evidence。
- COLD/ARCHIVED 可显式恢复为 WARM。
- tenant 全量 Derived Serving Rebuild 必须在同一 SQLite 事务写 BACKFILLING gate 并清 Catalog；按 Vector/Source/Session/Relation/Claim history/CurrentSlot/Retention/Verify 分阶段 checkpoint，启动期在 READY 前续跑，失败或回滚状态读链 fail closed。

## F. 检索安全与性能

- Trusted tenant/owner/workspace/session/adapter constraint 先绑定，冲突 fail closed。
- `authorized_scope_keys` 区分缺失 envelope、可信 grants、显式缩小和显式空 grants；Vector native hit 仍须经过 SQL Catalog ACL 复核。
- Path、time、type、ACL structured filter 在 Top-K 前由数据库执行。
- Exact、FTS、Vector、Relation 候选各有硬上限；local vector 仅对 SQL/FTS 有界 URI 计算。
- 不直接比较异构 raw score；RRF/归一化结果保留 exact/lexical/vector/relation/recency/hotness/canonical/rerank/final components。
- CURRENT 按 slot 去重；HISTORY 按 claim+revision；resource 按 URI+digest；session 有 per-session quota。
- ContextPacker 的降级顺序是 L2 -> L1 -> L0 -> URI，并输出 selected layer、source URI、drop reason、token estimate、canonical validation、projection lag、degraded mode。
- Vector 失败显式降级 FTS；reranker 失败保留 fusion 顺序；stale current 不返回；catalog migration 未就绪不静默漏数据。
- SQLite candidate/VM bound 超界必须显式 unavailable，不能捕获后当作空成功。
- 计数至少包含 structured/exact/fts/vector/relation/fusion/rerank/canonical/source_reads/selected/dropped，并证明 `canonical_validated <= candidate_limit`、`source_reads <= candidate_limit + 2` 等 bounds。

## G. 安全投影

- API key、token、cookie、authorization、password、private key、env、DB/SSH credential、binary、超长日志和 tool privacy field 被清洗。
- 路径只保留 basename + controlled location；完整路径只存在于权限保护的 Evidence Source。
- Sanitization 失败 fail closed，且 metadata JSON 与 Recall Trace 不能旁路。
- Forget/delete 后 FTS、Vector、Path、Link、Current/Claim serving projection 都被清理或失效。
- Tenant、Owner、Workspace、Adapter private scope 和 unauthorized tree path 隔离有真实测试。

## H. 迁移和兼容

- 状态覆盖 NOT_STARTED、SCHEMA_READY、BACKFILLING、DUAL_WRITE、SHADOW_VALIDATING、READY_TO_CUTOVER、CUTOVER、ROLLBACK、COMPLETED、FAILED。
- 真实旧 Schema/既有 Archive 首启不得按“迁移行缺失”误判 greenfield；必须耐久 Gate，兼容读为空时明确 unavailable。
- Backfill 分批、checkpoint、resume 幂等，不一次性加载所有 Archive。
- Dual write 的 projection 等价性必须写入 `migration_equivalence_journal`；SHADOW 的独立 LEGACY/Unified 在线结果差异必须写入 `migration_shadow_read_journal`；两者只比较有界稳定 identity，trace/diff 不保留敏感 query/source payload。
- Cutover 必须同时满足 source projection completeness、两类 journal 的 sample/mismatch threshold、queue quiescence 和 source snapshot 稳定；任一失败不能写 READY。
- LEGACY/ROLLBACK 必须是同一 `contexts` 上独立、有界、保守的旧读，不是 Unified reader 别名、第二 Catalog 或文件系统扫描；Rollback 恢复 LEGACY primary route。
- Canonical Outbox job 必须显式绑定 Tenant/Owner/Workspace，Queue health 通过复合索引按该作用域查询，禁止从 subject-hashed Slot URI 猜 Owner；Session frontier health 同样按 Tenant/Owner/Workspace 隔离。旧无作用域且未完成的 job 必须保守 fail closed，无关 Principal/Workspace 的 lag 不得污染当前 caller。
- Source、Receipt、Current Head、Evidence 未被迁移修改或伪造。
- 旧 `retrieval_views`、`search_scope` 和平铺参数转换到 `RetrievalOptions`。
- SDK、HTTP、MCP 共用一个 JSON schema contract；`archive_search()` 是统一检索兼容包装，`archive_read()` 仍精确读取 evidence。

## I. 完成验证

最终必须记录真实的完整 pytest、相关 integration/performance、Ruff、MyPy、Pyright 命令和计数；用 `EXPLAIN QUERY PLAN` 证明 tenant、owner、type、path prefix、event/transaction time 和 current-slot unique key 使用索引。检查 `git status`、`git diff --stat`、`git diff --check` 与本次新增的未完成标记、空方法、临时分支和旧链回退；不得把本任务内未完成项写成“后续限制”。
