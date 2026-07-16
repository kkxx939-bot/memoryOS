# 统一 Context 存储、检索、召回与上下文组装

## 0. 架构结论

MemoryOS 的统一主链采用：

```text
Evidence Plane
    ↓
Unified Context Catalog / Tree / Graph
    ↓
Unified Serving Index
    ↓
Unified Retrieval Orchestration
    ↓
Optional Canonical Resolution
    ↓
L0 / L1 / L2 Context Packing
```

其核心不是“把所有内容都记成 Slot/Claim”，而是：

```text
Context First + Canonical State Overlay
```

所有有检索价值的上下文进入同一个 Catalog 和检索体系；只有 Profile、Preference、Project Rule、Project Decision、稳定实体属性及通过严格门禁的 Agent Experience 等状态型长期记忆，才额外进入 Slot/Claim/Revision 权威层。

本设计的九条硬约束是：

1. Slot/Claim 只服务状态型长期记忆。
2. 普通 Session、Resource、Tool Result 使用统一 Catalog。
3. Claim Revision Projection 与 Slot Current Projection 并存。
4. 所有在线检索经过统一编排。
5. Tree Path 不参与 Canonical Identity。
6. 在线查询禁止全库扫描。
7. Tool Result 必须 Sanitization。
8. 删除必须通过 Tombstone 传播。
9. 迁移必须支持 Cutover 和 Rollback。

实现入口主要位于：

- `memoryos/contextdb/catalog.py`
- `memoryos/contextdb/store/sqlite_index_store.py`
- `memoryos/contextdb/session/context_projector.py`
- `memoryos/memory/canonical/promotion_policy.py`
- `memoryos/memory/canonical/slot_projection.py`
- `memoryos/contextdb/tombstone.py`
- `memoryos/contextdb/retention.py`
- `memoryos/contextdb/unified_migration.py`
- `memoryos/contextdb/retrieval/`
- `memoryos/security/context_projection.py`

## 1. Evidence Plane

Evidence Plane 保存“发生过什么”和“权威状态如何形成”的原始证据：

- 普通 Context 的 SourceStore 对象与内容；
- 不可变 SessionArchive、manifest、commit head、message、tool result、observation 和 action result；
- Canonical Slot、Claim、Revision、字段级 EvidenceRef；
- Canonical Transaction 的 Receipt、Current Head、Redo、Recovery、Pending Proposal/Review；
- Source URI、Digest、Revision、event/ingested/transaction time。

事实源边界如下：

| 数据平面 | 事实源 | 不属于事实源的派生层 |
| --- | --- | --- |
| 普通 Context | SourceStore | Catalog、FTS、Vector、Tree、Projection |
| Session | SessionArchive | Session Catalog records、semantic segment、overview |
| Canonical 状态 | Slot/Claim/Revision Source + Receipt + Current Head | Claim/Current Slot serving projection |

Catalog 可以全部删除后重建，不能反向写成第二套业务状态。Migration、Backfill、Compaction 不得伪造 Receipt、Current Head、Evidence 或修改 Canonical Source。

## 2. Unified Context Catalog

统一 Catalog 是既有 `SQLiteIndexStore.contexts` 的演进，而不是第二张主表。当前 Catalog Schema version 为 10，`record_key` 是 Serving record identity，`uri` 是可共享的逻辑/Source identity；这使同一 Claim 的多个 revision 可以保留不同 record key，同时仍按 Claim URI 查询。v10 迁移仍在原表上幂等增量升级；规范化附属表只加速受控候选、投影健康或迁移证明，不能成为第二套业务状态。

`CatalogRecord`/`contexts` 表结构化保存：

- 租户与主体：`tenant_id`、`owner_user_id`、`project_id/workspace_id`、`session_id`、`adapter_id`；
- 类型与生命周期：`context_type`、`source_kind`、`record_kind`、`lifecycle_state`；
- Tree：`parent_uri`、`primary_tree_path`、`path_depth`；
- 时间：`created_at`、`updated_at`、`event_time`、`ingested_at`、`transaction_time`、`valid_from`、`valid_to`；
- 分层内容：`title`、`l0_text`、`l1_text`、`l2_uri`；
- Source proof：`source_uri`、`source_digest`、`source_revision`；
- 可空 Canonical overlay：Slot/Claim ID/URI、revision/state、head/receipt/effect hash；
- Serving：三类 hotness、`serving_tier`、`projection_status`；
- 兼容列：既有 admission、claim state、scene/action/anchor 和 content digest 字段。

普通 Session、Resource、Tool Result 的 `canonical_*` 为空；Claim revision 和 Current Slot 才填充对应 proof。`metadata_json` 只保存经过安全投影且没有独立列的有界附加信息，不能用来绕过时间、ACL 或 secret 约束。

规范化附属表及控制表为：

- `context_paths`：一对多逻辑路径与唯一 Primary Path；
- `context_path_closure`：受控路径到祖先前缀的物化闭包，用索引生成 path-prefix 候选；
- `context_acl_grants`：按 Tenant、Principal/Public/Tenant/Service/Workspace 与 Scope 规范化的访问候选；
- `context_path_acl`：Path ancestor 与 ACL grant 的组合候选，避免在线先扫路径再做权限过滤；
- `context_tenants`：把 Tenant 映射为 RTree 可用的稳定数值维度；
- `context_validity_map` + `context_validity_rtree`：把 record identity 映射到 `valid_from/valid_to` 区间候选；
- `context_fts_map`：`record_key -> contexts_fts.rowid` 的可重建映射，支持有界单行替换和删除；
- `context_links`：Serving relation/link；
- `context_projection_state`：投影 revision、effect、retry/error；
- `context_tombstones`：耐久删除日志；
- `migration_state`：迁移状态与 checkpoint；
- `migration_equivalence_journal`：不可变 Evidence 与实际 Projection 的 payload-free 等价性证明；
- `migration_shadow_read_journal`：旧读/新读有界结果 identity 的 payload-free 差异证明；
- `session_projection_frontier`：按 Tenant、Owner、Workspace 记录 SessionArchive 到 Catalog 的投影状态。

这些表均不是第二 Catalog 或第二事实源。ACL、Path closure、RTree 与 FTS token 命中只能生成候选；返回前必须用 `contexts` 的精确字段和可信调用约束再次校验。任何附属映射不一致都不能扩大可见范围或伪装成空成功。

## 3. Context Tree

Tree 是逻辑分类与结构化检索，不是文件系统事实源。受控根只有：

```text
timeline
sessions
projects
resources
memories
skills
agents
```

支持的典型路径包括：

```text
timeline/{year}/{month}/{day}
sessions/{session_id}
projects/{project_id}
resources/desktop
resources/repository
resources/uploads
memories/preferences/{subject}/{dimension}
memories/profiles/{attribute}
memories/rules/{topic}
memories/decisions/{topic}
skills/{skill_name}
agents/{adapter_id}
```

路径段必须满足受控字符集，最大深度 12；一个 record 只有一个 primary path，secondary path 最多 7 个。`context_paths` 对 tenant/path、tenant/owner/path、tenant/type/path、tenant/event_time/path 和 tenant/URI 建立索引。Path prefix 使用 SQL 范围条件，在数据库过滤，不递归扫描 View 目录。

Timeline 根据 `event_time` 建路径，不以写入日期替代发生日期。Overview/Abstract 是可重建派生内容。Tree 重分类只修改路径投影；Slot ID、Claim ID、Canonical URI 和 Source URI 不变。LLM 不可无限创建目录，路径来自受控规则、Schema 或有限 taxonomy。

## 4. Session Projection

`SessionContextProjector` 接受已经耐久写入、带 archive/manifest digest 的 SessionArchive，生成幂等 Catalog records，绝不创建 Canonical Slot/Claim：

| Record | Structured/FTS | Vector 默认策略 | L2 |
| --- | --- | --- | --- |
| Session Root | 是 | 是 | Archive URI |
| Session L0/L1 | 是 | 否 | Archive URI |
| Semantic Segment | 是 | 是 | Archive URI |
| Message | 是 | 否 | Archive URI |
| Tool Result | 是 | 否 | Archive URI |
| Resource Reference | 是 | 重要时可选 | Archive URI |
| Used Context/Skill | 是 | 否 | Archive URI |
| Observation/Action Result/Event | 是 | 仅重要且配置允许 | Archive URI |

Semantic segment 有固定最大 chunk size，并在调用方本地 Timeline 日界强制切段，保证单值结构化 `event_time` 与该 Segment 的 Timeline path 同源；Session event 同时提供 `occurred_at` 和 `event_time` 时前者优先，只提供 alias 时使用 `event_time`，两者都缺失才按该记录类型的受控 fallback。Message/Tool Result 的完整内容仍由 SessionArchive L2 持有，不为每条原子事件生成向量。Session Root、L0、L1、segment、event/reference 的 record key 由 archive identity 和 source digest 稳定生成，重复投影是 upsert。

文件读取场景由 Tool Result 独立完成：即使用户 message 不含文件名，Projector 仍从 `resource_uri/file_uri/path/...` 提取 basename 和桌面、仓库、上传区、临时区、用户区或外部来源等受控 location，生成 Tool Result 与 Resource Reference，并加入 timeline、session、project/agent 和 `resources/{location}` 路径。Serving 中只保留 basename/location、Archive source URI 与 digest；完整绝对路径留在权限保护的 Archive Evidence。

## 5. CanonicalPromotionPolicy

`CanonicalPromotionPolicy` 是 ordinary context 与 canonical state 的确定性边界，输入只能是可信结构化事实，输出为 `CATALOG_ONLY`、`PROMOTE` 或 `REJECT`。

- `AUTHORITATIVE_STATE`：Profile、Preference、Project Rule、Project Decision、需维护当前值的实体属性，在门禁满足后允许 PROMOTE。
- `EXPERIENCE`：必须已经提炼、跨会话可复用、Evidence 完整、Identity 稳定、达到 Admission 阈值，并排除原始日志、一次性失败和临时任务状态。
- `OBSERVATIONAL`：普通 message/tool result/file read/used context/skill/observation/action result/browser/command/event/log 默认 `CATALOG_ONLY`。

OBSERVATIONAL 只有认证的显式 `remember()`、Schema 声明 stateful type、确定性 Policy 批准、Evidence/Scope/Authority/Identity 全成立时才晋升。PROMOTE 后仍须通过 Evidence、Schema、Identity、Scope、Authority、Admission、Reconcile、Transition 和 Canonical Transaction。

LLM 可做 query rewrite、synonym 或非安全 intent 辅助；不能决定晋升、ADD/UPDATE/DELETE、Tenant/Owner/Workspace、SQL、URI、Slot ID 或 Claim ID。

## 6. Slot/Claim 适用边界

Slot 表示一个稳定的单值状态维度；Claim 表示该维度某个候选/历史状态；Revision 表示 Claim 的可审计演进。Canonical Event 继续由 `MemoryType.EVENT` 表达，但普通 Session Activity 只是 Catalog Event。

一个 Slot 最多一个 ACTIVE Claim。多值偏好通过一个 Claim 的结构化 collection、更细 Slot identity 或多个独立单值 Slot 表达，不能放宽为同一 Slot 多个 ACTIVE Claim。

Tree Path、日期、文件路径/名、展示标题、L0/L1、Vector/Projection ID、Session ID、Tool Result ID 都不进入 Identity Hash。Tree reclassification 不能改变 canonical identity。

## 7. Claim Revision Projection

Claim Revision Projection 保留而且不被 Current Slot 取代。每个 revision 使用：

```text
record_key = claim:{claim_id}:revision:{source_revision}
record_kind = claim_revision
```

记录绑定 Claim/Slot ID 与 URI、source/canonical revision、transaction、Current Head、Receipt Digest、Projection Effect Hash、Evidence 与 validity。它服务 Claim 精确读、HISTORY、AS_OF、AUDIT、CONFLICTS 和 OPTIONS。

同一 Claim 新 revision 不覆盖旧 record。Canonical Source 的旧 Revision 保持不可变；若其 `valid_to` 为空，Serving Projection 由下一条非 historical Revision 的 `valid_from` 确定性派生半开区间，并同步 Catalog/Vector metadata。HISTORY 按 `claim_id + revision` 去重；AS_OF 使用结构化 `valid_from <= valid_at < valid_to` 语义，只接受该时点 ACTIVE Revision，并再次验证 committed Source/projection proof。

## 8. Slot Current Projection

`CurrentSlotProjection` 是独立的当前态模型：

```text
record_key = slot:{slot_id}:current
record_kind = current_slot
unique = tenant_id + canonical_slot_id for live current_slot
```

它保存 Slot identity fields、active Claim ID/URI/revision、canonical state/value、valid interval、tree paths、L0/L1/L2、transaction ID、Slot/Claim head/receipt、source revision 和组合 effect hash。

Active Claim B 变成 C 时，同一 record key 原位更新，旧 Current 不可继续参与 CURRENT；B 的 Claim revision 保留，C 的 revision 新增/更新。没有 ACTIVE Claim 或 RETRACT 时生成耐久 tombstone。CURRENT 默认只检索 Current Slot；旧 Claim-current 只允许作为明确 degraded 的兼容读并接受同样的精确 proof 校验。

## 9. Tombstone

`context_tombstones` 先耐久记录删除意图，再由 `ProjectionTombstoneService` 幂等处理：

```text
authoritative change
-> enqueue tombstone
-> verify source lifecycle when required
-> begin_tombstone_cleanup()
   -> newer Catalog revision/effect wins: STALE
   -> otherwise atomically remove Catalog/FTS/Path/Link/Projection state
   -> durable state becomes CLEANING and blocks same record_key rewrite
-> compare-and-delete Vector by tenant + record key + revision/effect
-> delete only Relation rows owned by that Catalog record
-> finish_tombstone_cleanup() -> APPLIED
```

`begin` 之前的校验失败进入 FAILED/retry；进入 CLEANING 后任一外部 consumer 失败则保留 CLEANING、增加 retry/error 并可按同一 tombstone id 重放，不能伪装成功。重复处理得到 CLEANING/APPLIED/STALE 等稳定结果；旧 tombstone 绝不删除新 manifest 所拥有的 Vector/Relation。Forget、Retract、Supersede、Slot active switch、Source/Resource/Session delete、owner/visibility/authority 修复、Tree reclassify 和 serving tier 变化均须形成相应投影事件或 Tombstone。

Canonical Transaction 不同步删除所有派生后端，避免扩大事务边界。无明确用户策略时不删除不可变 Source/Session Evidence。

## 10. Retention

Serving lifecycle 与 ClaimState 分离：

| Tier | 默认 Serving |
| --- | --- |
| HOT | FTS、Vector eligible、L0/L1、快速召回 |
| WARM | FTS、L0/L1，Vector 按策略，L2 按需 |
| COLD | 结构化时间/路径与 Source URI，默认无 Vector |
| ARCHIVED | 默认在线查询排除，仅显式 history/archive 恢复或读取 |

`RetentionPolicy` 默认 7 天 HOT、30 天 WARM、180 天 COLD，批次默认 256；这些值可配置且有单调性与上限校验。Current Slot 始终 HOT。`CatalogRetentionManager` 支持 Session segment compaction、Timeline day overview、stale projection GC、Vector GC、orphan path GC、已应用 Tombstone journal GC，以及 COLD/ARCHIVED 到 WARM 的显式恢复。

Compaction 只创建摘要 projection 并调整 Serving tier，不删除 Source/Archive Evidence。缺少可信结构化时间时保持 HOT，避免误归档。

## 11. Query Planner

SDK、HTTP、MCP 使用同一 `RetrievalOptions`/JSON Schema；旧平铺参数、`search_scope` 和 `retrieval_views` 转换到同一结构。`RetrievalQueryPlan` 增加 `semantic_query`，并完整序列化到安全 Recall Trace。

支持：target URI/path、context/source type、tenant/owner/workspace/session/adapter、event/transaction ranges、`valid_at`、timezone、CURRENT/HISTORY/AS_OF/CONFLICTS/OPTIONS/OPEN_RECALL/EXACT、canonical resolution、relation expansion、candidate/final/token bounds。

安全优先级为：

```text
Trusted Caller Constraint
> 用户显式 Filter
> 确定性日期与路径解析
> Schema 默认值
> LLM 非安全辅助
```

`QueryPlanner` 对 trusted scope 做交集/冲突检查，冲突 fail closed。`authorized_scope_keys` 使用三态契约：`None` 仅用于本地旧调用方缺失授权 envelope 的 principal/workspace 兼容；可信非空 grants 且用户没有显式 scope filter 时注入全部 grants；显式 filter 只能取 grants 子集；显式空 grants 只允许 unscoped 记录。日期区间采用半开区间并按调用方 timezone 转为 UTC；普通日期事件由默认 CURRENT 转成 `OPEN_RECALL + event_time`，状态时点问题转成 `AS_OF + valid_at`，系统写入日期问题转成 `HISTORY + transaction_time`；非默认显式 intent 不被覆盖，AS_OF 必须有 `valid_at`。

## 12. Structured Filter

`CandidateGenerator` 首先构造 SQL structured filter，在任何 Top-K 截断前绑定 tenant、owner、workspace、session、adapter、ACL/applicability、record kind、context/source type、lifecycle、serving tier、path、event/transaction time 与 valid interval。

带 `semantic_query` 时，structured Catalog 分支不是通用兜底枚举。只有以下完整、确定性组合可以同时生成 SQL 结构化候选：`OPEN_RECALL + event_time_from + event_time_to`、`AS_OF + valid_at`、`HISTORY + transaction_time_from + transaction_time_to`。这保证“某日发生了什么”“某日时状态是什么”“某日系统写入了什么”不要求自然语言问句原样出现在正文中；Tenant/Owner/Workspace/ACL/Path/Time 仍全部在 SQL 中先于 `candidate_limit` 执行。普通 semantic query 缺少上述组合时 structured 分支为空，只能走 Exact/FTS/Vector/Relation 的有界召回，禁止无条件 `list_catalog()`。

Exact URI 不做前缀扫描。旧 `slot_uris` / `claim_uris` 与新 `target_uris` 统一进入同一个稳定 identity lookup，并以等值 UNION 匹配 `contexts.uri`、`canonical_slot_uri`、`canonical_claim_uri`。每个物理分支都在 branch-local LIMIT 前等价应用 trusted Tenant、Owner/Workspace/ACL、RecordKind/Context/Source、Lifecycle/Admission/ServingTier、Project/Adapter/Connect、Scope、Path、event/transaction/updated time 与 `valid_at`；外层 Catalog 在最终 LIMIT 前再次执行完整约束。CurrentSlot 是通过 ACL/Scope 授权的权威状态覆盖层，不按产生 Evidence 的 Connect provenance 隐藏，因此 Connect 过滤只约束普通 Context/Claim history，不能让 MCP/Coding Adapter 看不到已获授权的 CurrentSlot。明确 Canonical Slot identity 的 CURRENT 只读 `current_slot`，HISTORY/AS_OF/CONFLICTS/OPTIONS 只读 `claim_revision`。因此 CurrentSlot 的公开 Serving URI 可以是 `{slot_uri}/serving/current`，稳定 Slot URI 仍能命中它，而同一 Slot 的大量历史 Revision、较新 archived/时点外/错误路径行都不能在 Top-K 前挤掉较旧的合法结果。

`SQLiteIndexStore` 为 tenant+owner+type、tenant+workspace、tenant+session、tenant+record kind、tenant+event/transaction/updated time、Claim revision 和 Current Slot 建索引。Workspace ID 在进入 Catalog、Scope、Vector metadata 与 Session frontier 前统一规范化；本地仓库路径或 remote-like identity 只保留不可逆 `project-*` identity，不把原始绝对路径复制到索引。

Path prefix 由 `context_path_closure` / `context_path_acl` 的 Tenant、Owner、Workspace、Type 与 ancestor 索引前置生成有限 record keys；`context_paths` 保留完整多路径投影及反查。`valid_at` 先经 `context_tenants`、`context_validity_map`、`context_validity_rtree` 缩小区间候选，再以 `contexts.valid_from/valid_to` 精确复核半开区间，避免 RTree 浮点外扩产生假阳性。ACL grant、Path closure/ACL、RTree 以及 Relation 扩展后的 URI 都必须重新套用 `contexts` 的 trusted tenant/owner/workspace/type/time/path 条件，不能把附属候选表当成权限事实源。

Relation 扩展只处理最终有界 seed。每个 seed 依次尝试 Canonical Claim URI、Canonical Slot URI、公开 Serving URI，identity 数量硬上限为 3；每个 seed 最多读取 5 条 relation。Relation target 不直接成为结果，而是以最多 2 条的 exact identity lookup 重新绑定到当前 Query Plan 的 Tenant/Owner/Workspace/RecordKind/ACL。这样 Relation 中保存稳定 Claim/Slot URI 时仍可返回 CurrentSlot 或 Claim Revision，但不能绕过 Catalog 权限，也不能演变成关系图全量遍历。

结构化结果受 `candidate_limit` 限制，不构建全库 `allowed_uris`。

## 13. FTS

Catalog 使用同一个 `contexts_fts`，包含 `record_key`、URI、title、content、safe metadata text、search terms 与哈希化 `acl_tokens`。运行环境有 FTS5 时用 virtual table；不可用时只创建兼容表以保持 Schema/迁移稳定，在线 lexical 分支关闭，只保留 structured/exact 候选，绝不退回 Python/SQL contains 全表扫描。

FTS5 查询把可信 Tenant/Principal/Workspace/Path/Type/Source/Session/Adapter/Scope 绑定为不可逆 token，并使用 FTS5 隐藏 `rank` 列的固定 BM25 配置先取有界 overfetch；token 只是前置裁剪，不是 ACL 证明。命中行随后按 `record_key` 回连 `contexts`，在 `LIMIT` 前重新执行结构化和权限谓词，再以安全文本做有限 lexical relevance 复核。调用方不能通过伪造 path token、scope token 或 caller 提供的 `matched` 标志扩大结果。

`contexts_fts.record_key` 是 FTS5 的 `UNINDEXED` 列，不能用它做逐行删除。Schema v10 因此增加 `context_fts_map(record_key, fts_rowid)`：正常 upsert 先按主键定位一个 rowid，再精确替换 FTS 行；启动完整性检查发现 orphan/mismatch 时离线重建 FTS 与 map。Path、ACL、Validity 等附属更新和删除同样使用 `tenant_id + record_key` 索引，避免随 Catalog 规模增长的写放大。

只有经过 `ContextProjectionSanitizer` 的 title/L0/L1/metadata 进入 FTS。Secret、完整绝对路径、binary 和超长 Tool Result 不进入索引。Tombstone 应用与 Catalog row 同步清理 FTS record 及其 rowid map。

## 14. Vector

VectorStore 声明四项能力：metadata、namespace、time filtering 和 delete-by-filter。生产后端应支持 filter-before-Top-K。

本地/不具备原生过滤的后端只允许：SQL/FTS 先给出有限 URI，`search_vector_candidates()` 只对该集合计算相似度，overfetch 上限为 200，并在结果/Trace 标记 degraded mode。`vector_uris()` 仍可供 admin/test 使用，但禁止从在线 Search/Assemble 调用。

Session、Claim Revision 与 CurrentSlot 共用 `catalog_vector_metadata()`：Vector row 携带 tenant/owner/workspace/session/adapter、context/source/record kind、受控 path、time/validity、scope keys、Canonical proof，以及 `catalog_record_key`、source revision、effect hash。支持 native filtering 的后端先按这些字段 filter-before-Top-K；返回的 hit 仍由 SQL Catalog 重新套用 trusted scope 与结构化过滤，不能把 Vector metadata 当第二个 ACL 事实源。

Vector 是最终一致层：失败时显式降级 FTS，不得返回“空成功”掩盖错误。Retention、Tombstone 和 Resource/Session delete 负责相应 Vector GC。

## 15. Fusion 与 Rerank

Exact、FTS、Vector、Relation 的原始分数不直接比较。`FusionRanker` 使用 RRF rank fusion，并保留：exact、lexical、vector、relation、recency、hotness、canonical、rerank 和 final score。

Recency/hotness/canonical boost 乘以真实 relevance，不能让无关内容凭时间或 canonical 标签排到前面。Current Slot 只有小幅且相关性依赖的 canonical boost，不无条件第一。Reranker 失败保留 deterministic fusion order，并标记 fallback；所有 score 限制在有限区间。

去重规则：CURRENT 按 slot；HISTORY/AS_OF/CONFLICTS/OPTIONS 按 claim+revision；Resource 按 resource URI+digest；ordinary context 按 source digest/stable key；Session 有 per-session 上限；多 path/branch 命中只返回一次。

## 16. Canonical Validation

`BoundedCanonicalResolver` 在 fusion/rerank 后只验证排序前缀，`canonical_validated <= candidate_limit`，不抓取全量 snapshot。为一次有界 Slot/Claim proof 补读保留固定常数 2，因此整个在线链的硬边界是 `source_reads <= candidate_limit + 2`；超出验证前缀的尾部不会被当成已验证结果，而以 `not_validated_bound` 丢弃并在 Trace 标记 `canonical_validation_bound`。CURRENT 校验 Slot、Tenant/Owner、Scope、Visibility、Authority、`active_claim_id`、ACTIVE Claim、revision、Slot/Claim Current Head、Receipt 和组合 Effect Hash。Catalog/FTS/Vector 候选只允许携带有界 serving/proof 白名单，不能成为业务 payload 权威；CURRENT 的 value、identity、scope、evidence 等业务 metadata 从 receipt-proved Slot/Claim/current Revision 重建。HISTORY/AS_OF/CONFLICTS/OPTIONS 校验 Claim/revision、派生 validity 和 projection record proof；正文以及 `canonical_value`、`value_fields`、revision 列表、evidence、qualifiers、proposal、relation 等业务 metadata 全部来自被请求的 immutable proved Revision，候选中注入的当前值或任意字段不会反射到公开 SDK/HTTP/MCP。任何 Canonical 精确回源的公开 title/text/metadata 都再次经过 `ContextProjectionSanitizer`，清洗失败 fail closed，并复核 ID、Revision、Head、Receipt、Effect Hash 等证明字段未被清洗破坏。

Canonical Current 是强一致语义：stale/tampered/unavailable 时 fail closed，旧状态不返回。健康的 Projection Queue 不能证明某个指定 Slot 一定已有 Serving row；因此 CURRENT 在启用 Canonical validation 且调用方显式指定稳定 Canonical Slot URI 时，如果 exact branch 没有对应 CurrentSlot，必须返回 `missing_canonical_current_projection` unavailable，不能空成功。该规则不改变普通 semantic miss、普通 URI target 或显式禁用 Canonical resolution 的空结果语义。允许有界精确回源或明确 unavailable/degraded。启动、Migration、Repair 和 Audit 继续保留全局 O(N) 完整性检查，与在线 bounded validation 明确分离。

## 17. L0/L1/L2 Context Packing

LayerSelector 的降级顺序固定为：

```text
L2 Full -> L1 Overview -> L0 Abstract -> URI / Reference
```

`ContextPacker` 同时限制 token budget、final limit、context type quota、每个 Slot、每个 Session（默认 5）、每个 resource branch（默认 3）、L2 item（默认 3）和来源多样性。

普通 Resource 的完整 L2 不预先复制进 Catalog，也不在候选生成阶段读取。Fusion、Rerank 和可选 Canonical resolution 完成后，Packer 先用同一套 final/token/type/session/resource quotas 形成有界 hydration 前缀；Orchestrator 只对其中最多 `max_l2_items` 个 `record_kind=context` Resource 精确读取 Source object 与 L2 content，并重新核对 Tenant、Owner、Workspace、Adapter、Lifecycle、Source/L2 URI 和 Source Digest。正文再次经过 `ContextProjectionSanitizer` 后才写入候选；越界、过期、越权、回源或清洗失败只产生明确 degraded mode，并继续使用已清洗的 L1/L0/URI。Session Message 与 Tool Result 的原始内容只留在 SessionArchive Evidence，普通在线召回不得为原子节点读取 raw L2。

CURRENT 优先 current slot、project rule/decision、preference/profile 与 resource；OPEN_RECALL/HISTORY 优先 session/event/resource/tool result/canonical history。Coding Agent 的公开 assemble 仍受 context reduction、workspace 与 adapter scope 约束。

输出包含 selected layer、source URI、drop reason、token estimate、canonical validation status、projection lag、degraded mode 和 score components。

## 18. 时间模型

字段语义严格区分：

- `event_time`：事情实际发生时间，用于“7 月 14 日发生了什么”和 Timeline path；
- `ingested_at`：系统接收时间；
- `transaction_time`：写入 MemoryOS/投影事务时间，用于“当天新增了哪些记忆”；
- `valid_from/valid_to`：现实有效区间，用于 `valid_at`/AS_OF；
- `created_at/updated_at`：派生对象创建与更新。

所有时间标准化为带时区 UTC ISO-8601。Date-only filter 在调用方 timezone 转为当地日界的半开 UTC 区间，正确覆盖跨日。时间过滤依赖结构化列与索引，而非 metadata JSON search。

## 19. Migration

`UnifiedContextMigration` 只迁移 Serving projection，状态机为：

```text
NOT_STARTED -> SCHEMA_READY -> BACKFILLING -> DUAL_WRITE
-> SHADOW_VALIDATING -> READY_TO_CUTOVER -> CUTOVER -> COMPLETED
                         \-> ROLLBACK
any allowed phase -> FAILED -> explicit resume target
```

Backfill 按 tenant、batch（1–1000，默认 256）流式读取经过 manifest/head 完整性校验的 Archive，checkpoint 指向已处理 commit head；中断后从 checkpoint 续跑。每个 Archive 投影幂等，不一次性加载全库。文件系统遍历只存在于明确 offline migration/repair 路径。

SQLite schema 从 pre-v10 原地升级时先写耐久 bootstrap provenance，再以 `ON CONFLICT DO NOTHING` 绑定实际 Tenant 为 `SCHEMA_READY`，不能在升级后因迁移行缺失误判 greenfield。启动期只为 Gate 判断扫描到第一个既有 SessionArchive Evidence 即停止：真正空 Catalog/无 Evidence 记录独立 greenfield origin 并走 Unified；旧 Archive 尚未回填时走 LEGACY/degraded，若兼容读为空则明确 unavailable，不得静默漏数据。已推进的 DUAL_WRITE/ROLLBACK 等状态不会被启动初始化覆盖。

Session 与既有 Canonical CurrentSlot 使用独立 checkpoint；Canonical 回填只从真实 committed Slot/Claim、Receipt 与 Current Head 生成正式 Projection，不把 raw/uncommitted Canonical Source row 复制成 serving state。

`migration_state` 保存 state、checkpoint、batch size、sanitized details/error。`migration_equivalence_journal` 独立记录每个 Session/CurrentSlot 的 immutable Evidence identity、expected/actual count 与 digest；实际行通过精确 projection identity 回读，不能用同一个在线查询自证。`migration_shadow_read_journal` 独立记录旧读与新读有界结果的 count、overlap 和 digest；状态存储会重新计算 `matched`，不信任调用方传入的布尔值。两类 journal 均不保存 query/source payload。

## 20. Cutover 与 Rollback

`MigrationFeatureGate` 根据状态提供 read route：前期与 ROLLBACK 为 LEGACY，DUAL_WRITE/SHADOW/READY 阶段为 SHADOW，CUTOVER/COMPLETED 为 UNIFIED。Dual write gate 在 BACKFILLING 起开启并在 FAILED/ROLLBACK 保持派生兼容写，使恢复或回退期间的新数据不会丢失。

LEGACY 不是另一张表或文件系统扫描：它是同一 v10 `contexts` 上的独立、保守、平面 owner/public reader，不使用 Unified ACL grant、Path closure、Relation、Vector 或 RTree adjunct；跨 Owner 共享在回滚模式 fail closed。SHADOW 以 LEGACY 为主返回，同时运行 Unified candidate chain，把最终有界 identity diff 写入耐久 read journal；ROLLBACK 恢复 LEGACY primary route，而不是偷偷继续返回 Unified 结果。

READY_TO_CUTOVER 必须同时满足：Session 与 CurrentSlot source projection validation 完成；`migration_equivalence_journal` 的 sample/mismatch 门槛达标；`migration_shadow_read_journal` 的在线读 sample/mismatch 门槛达标；projection queues quiescent；shadow source snapshot 未变化。缺少任一 journal、样本不足、mismatch 超阈值或 source set 改变都不得标记 READY。Cutover 仍保留可回滚性；Rollback 记录来源阶段和 sanitized reason，并把主读路径恢复为 LEGACY。COMPLETED 后仍保留显式 rollback 状态转换，不删除事实源。

旧 `archive_search()` 已作为 Unified Retrieval 的 OPEN_RECALL 兼容包装：先查 Catalog，再对最终有界 Archive URI 做精确 evidence read；不恢复递归 Archive 扫描。`archive_read()` 保留精确权限校验读取。

## 21. 性能边界与可观测性

在线 Search/Assemble 不得调用：`source_store.list_objects()`、全量 Canonical Snapshot、全量 Claim/Slot 遍历、全库 `allowed_uris`、`vector_uris()`、`glob views/**`、递归 Tree、Archive 全目录扫描或 Python 全库 lexical match。这些操作只允许 startup/migration/repair/audit/admin/test utility。

硬边界：

```text
canonical_validated <= candidate_limit
source_reads <= candidate_limit + 2
vector_overfetch <= configured max overfetch
per session <= configured per-session limit
CURRENT per slot <= 1
```

`RetrievalMetrics` 记录 structured/exact/fts/vector/relation/fusion candidates、rerank count、canonical candidates/validated、source reads、selected/dropped 和 vector overfetch。SQLite 在线查询安装 VM progress guard；如果结构化/FTS/旧读查询超过固定 VM step bound，抛出 `CatalogCandidateBoundExceeded`，公共入口转换为显式、可重试的 `RetrievalUnavailableError`，不得把超界或后端故障伪装为零结果。`EXPLAIN QUERY PLAN` 用于验证 tenant、owner、context type、path prefix、event/transaction time、FTS rowid map、ACL/path adjunct 和 Current Slot unique key 的索引路径；验收不依赖绝对毫秒阈值。

一致性与降级：Canonical Current 强一致 fail closed；Current Slot、Session Catalog、Vector、Tree Overview 最终一致。Canonical Outbox Queue job 在耐久 payload 与规范化列中绑定 Tenant/Owner/Workspace，并通过 `queue_jobs_scope_status_idx` 查询健康状态；Canonical Slot URI 使用稳定 `subject_*`，不得用 URI 前缀猜 Owner。旧无 Owner 归属且未完成的 Queue job 只在其自身 Tenant 内对所有 Owner/Workspace 保守 fail closed，不能跨 Tenant 阻断；已完成旧 job 不阻断。scoped 与 legacy-unresolved 健康统计必须分成两个 Tenant-bound 索引查询，避免 OR 分支退化成 queue-wide scan。Session projection frontier 同样按 Tenant/Owner/Workspace 检查；另一 Owner 或无关 Workspace 的 pending/failed 记录不能把当前调用者误判为不可用，也不能被当前调用者枚举。Vector failure -> FTS；reranker failure -> fusion order；missing Tree overview -> child/L0；catalog backfill incomplete -> feature-gated LEGACY/SHADOW 或明确 degraded/unavailable。

安全重建与在线查询共用进程内 `serving_lock`，并与 Session 双写共用跨进程 projection fence。重建在修改任何派生层前先验证 Canonical Source、Receipt/Current Head、projection publication、outbox 与 queue；preflight 失败时保持旧 serving 数据并把 runtime 标为 NOT_READY。tenant Catalog 清理和专用 `unified-context-derived-serving-rebuild-v1=BACKFILLING` gate 在一个 SQLite 事务内提交，因此进程崩溃后不会出现“Catalog 已空但状态仍 COMPLETED”。未完成 gate 强制读路由进入保守 LEGACY/fail-closed，runtime 在 READY 之前自动续跑。

完整 Derived Serving Rebuild 按 `Vector tenant cleanup -> Generic Source -> SessionArchive -> receipt relation reconcile -> Claim Revision -> CurrentSlot -> generic vector -> Retention -> Verify` 分阶段执行；Session 和 CurrentSlot 每个 batch 都持久化 checkpoint/counter，FAILED/ROLLBACK 从原 checkpoint 幂等恢复。Claim Revision 由正式 Canonical projector 从不可变 publication/receipt proof 重建全部历史，CurrentSlot 由 committed-state backfill 重建，RelationStore 从 receipt 当前效果校准；最后重新应用 ServingTier/Vector GC 并执行 canonical proof、CurrentSlot equivalence 与 Index consistency 校验。任何阶段失败均保留 durable gate、记录已清洗错误并使 runtime NOT_READY，不伪造 Receipt、Current Head 或 Evidence。

Session 删除除逐 record Tombstone 外还写入稳定 `session_delete_barrier`。APPLIED barrier 会抑制同一 Session 将来新增的投影种类，CLEANING barrier 会阻止 rebuild checkpoint 前进；因此保留的不可变 SessionArchive Evidence 不会在重建时被意外复活。该 barrier 不标记 `gc_safe`，普通 Tombstone GC 不得删除。并发查询在单进程只能看到重建前或重建后的完整快照；另一进程即使尚未获得本地锁，也会被 durable migration gate 明确阻断，而不会把部分结果伪装为空成功。

“统一在线检索”指公共 Context serving：SDK/HTTP/MCP Search/Assemble、`archive_search()` 和 Memory recall 都进入同一个 Orchestrator。Prediction、Behavior、ActionPolicy 的既有域内 policy lookup 保持语义隔离，不是第二个 Context serving API，也不得暴露或回接为公共 Context 检索旁路；Embodied 动作仍经过 PolicyGate。

## 22. 从 OpenViking 吸收的范围

本设计参考 OpenViking 实际 Commit `8fb94e9f4d86f1cfeb1f117f382ad76340e7198a`，吸收的是可迁移的检索思想，而不是其存储事实模型：

- `openviking/retrieve/hierarchical_retriever.py` 的 target URI、层级检索和 L0/L1/L2 渐进加载思想；
- `openviking/session/session.py` 的 Session-as-Context、archive abstract/overview 与 token budget 思想；
- `openviking/session/compressor_v2.py` 的 exact/tree scope、compaction 与受控树路径思想；
- intent-assisted query、rerank 和按目标范围缩小召回空间的模式。

MemoryOS 将这些思想落到 SQL Catalog/Path/FTS、bounded vector、统一 Query Plan 和 ContextPacker，而不是在线递归文件树。

## 23. MemoryOS 不照搬的部分

MemoryOS 不照搬 OpenViking 的 filesystem/URI 事实源模型，也不把 `.abstract.md`、`.overview.md`、目录节点或 tree path 作为 canonical identity。原因是 MemoryOS 已有更强的 Source/Transaction/Canonical 约束：

- SourceStore 与 SessionArchive 是 Evidence；Catalog/Tree 只是 Serving；
- Slot/Claim/Revision、Receipt、Current Head、Redo/Recovery 保持权威状态语义；
- 一个 Slot 最多一个 ACTIVE Claim；
- Tenant/Owner/Workspace/Visibility/Authority 不能交给 intent model 扩大；
- Canonical Current 需要 head/receipt/effect proof，不能把向量或目录命中当真；
- 删除通过 durable outbox/Tombstone，而不是同步删除文件树；
- 在线过滤在数据库完成，不能递归扫描目录；
- ordinary Session/Tool Result/Resource 不因进入统一树而自动成为 Canonical Memory；
- Memory、Behavior、ActionPolicy 和 PolicyGate 语义保持隔离。

因此最终边界是：OpenViking 提供层级上下文组织与渐进加载启发；MemoryOS 保留自己的事实源、Canonical transaction、安全 scope、双投影和可回滚 migration 模型。

## 安全投影与 Recall Trace

虽然安全贯穿以上所有层，仍需单独强调：`ContextProjectionSanitizer` 在 Catalog、FTS、Vector eligibility、Trace 和 L0/L1 之前 fail closed。它清洗 API/access token、cookie、Authorization、password、private/PEM key、env、DB/SSH credential、URL credential、binary、超长输出和 metadata 中的隐私字段；Tool Result L1 默认上限 4000 字符，普通 projection 默认上限 8000 字符，collection 和 metadata depth 同样有界。

`MemoryEgressPolicy` 继续约束进入模型提取/记忆规划的出站内容，`ContextProjectionSanitizer` 约束进入 Serving/Trace 的派生内容；两者职责互补，任何一方都不能被 metadata、Tool Result 或兼容 API 绕开。

绝对路径被替换为 basename + controlled location；完整路径只留在权限保护的 Evidence Source。Sanitization 失败不会写入部分清洗数据。Query Plan/Recall Trace 再次使用同一 sanitizer，防止 query/filter/error metadata 泄漏。

## 公开 API 兼容性

保持：`MemoryOSClient.search_context()`、`assemble_context()`、HTTP `context/search`、MCP `memoryos_search`/`memoryos_assemble`、既有 MCP 别名 `memoryos_search_context`/`memoryos_assemble_context`、`archive_search()`/`archive_read()`、`remember()`/`forget()` 和 Pending Review。MCP 长短名称共享同一 Schema 对象与 handler，不形成两套语义。

新调用传 `options=RetrievalOptions(...)`；旧平铺参数仍接受，并在 `retrieval_options_from_legacy()`/merge 阶段转换，冲突 fail closed。HTTP 与 MCP 从 `memoryos/api/retrieval_contract.py` 读取同一 JSON Schema，不维护两套参数语义。所有在线入口最终构造同一种 `RetrievalQueryPlan` 并进入 `UnifiedRetrievalOrchestrator`。
