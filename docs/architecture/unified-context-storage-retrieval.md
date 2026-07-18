# 统一 Context 存储、检索、召回与上下文组装

## 0. 架构结论

MemoryOS 的公共 Context serving 只有一条在线主链：

```text
Authoritative Sources
    ordinary Context SourceStore
    immutable SessionArchive
    live Markdown memory documents
        ↓
Unified Context Catalog / Tree / Relation
        ↓
FTS / bounded Vector / rebuildable projections
        ↓
Unified Retrieval Orchestration
        ↓
bounded source hydration and validation
        ↓
L0 / L1 / L2 Context Packing
```

其核心约束是：

1. 普通 Context 、Session 和 Markdown memory 使用同一 Catalog 与检索编排。
2. 三类 source 各自保持事实源边界，Catalog 不成为第二份业务正文。
3. Memory document 的当前正文只是 live Markdown exact bytes。
4. 所有在线检索都有候选、Vector overfetch、Relation expansion、source read、per-session、per-document 与 token 上限。
5. Tool Result 与精确回源内容公开前必须经过 `ContextProjectionSanitizer`。
6. 删除先写耐久 barrier/tombstone，再幂等清理派生层。
7. 在线 Search/Assemble 不递归扫描 memory tree、SessionArchive、SourceStore 或全量 Vector URI。

主要实现入口：

- `memoryos/contextdb/catalog.py`
- `memoryos/adapters/persistence/sqlite/`
- `memoryos/application/session/context_projector.py`
- `memoryos/memory/documents/`
- `memoryos/workers/memory_document_projection_worker.py`
- `memoryos/contextdb/tombstone.py`
- `memoryos/contextdb/retention.py`
- `memoryos/application/context/`
- `memoryos/security/context_projection.py`

## 1. 事实源与派生层

| 数据平面 | 业务事实源 | 可重建 serving |
| --- | --- | --- |
| ordinary Context | SourceStore object/content | Catalog、FTS、Path、Vector、Relation |
| Session | immutable SessionArchive + manifest | root、L0/L1、segment、message/tool/resource/event records |
| user memory | live Markdown exact bytes | document/block records、FTS、Path、Vector、Relation、summary |

Memory document 的 revision/after/review blob 是受保护恢复资料，不是当前正文。Control store 仅保存 document identity、path、digest、logical revision、projection generation、status 和 timestamps。如果 control/Catalog 与 live file 不一致，系统必须 rescan、drop stale result 或 fail closed，不得用派生状态覆盖用户文件。

## 2. Unified Context Catalog

`SQLiteIndexStore.contexts` 是唯一主 Serving Catalog。绿地 SQLite schema 从 version 1 建立，不对旧库做原地转换。`record_key` 是 serving row identity，`uri` 是逻辑 source/document identity。

`CatalogRecord` 的结构化字段包含：

- trusted scope：`tenant_id`、`owner_user_id`、`workspace_id`、`session_id`、`adapter_id`；
- record classification：`context_type`、`source_kind`、`record_kind`、`lifecycle_state`；
- tree：`parent_uri`、`primary_tree_path`、`tree_paths`；
- time：`created_at`、`updated_at`、`event_time`、`ingested_at`、`transaction_time`；
- progressive content：`title`、`l0_text`、`l1_text`、`l2_uri`；
- source proof：`source_uri`、`source_digest`、`source_revision`；
- document proof：`document_id`、`block_id`、`document_kind`、`document_revision`、`projection_generation`、`projection_effect_hash`；
- serving：hotness scores、`serving_tier`、`projection_status`。

`metadata` 只保存经安全投影且没有独立列的有界信息，不能绕过 ACL、time、path、identity 或 secret 约束。

规范化附属表/状态包括：

- `context_paths` 与 path-prefix 候选；
- `context_acl_grants` 与受信任 scope 候选；
- `context_fts_map` 的 `record_key -> FTS rowid` 一对一映射；
- `context_links` 的 bounded serving relations；
- `context_projection_state` 的 document generation/digest/deletion state；
- `context_tombstones` 与 Session delete barrier；
- `session_projection_journal` 的 Archive-to-Catalog 恢复进度。

这些附属数据只用于候选、幂等恢复或发布拦截。返回前必须以 `contexts` 精确列和 trusted scope 复核。

## 3. Context Tree

Tree 是逻辑分类与结构化检索，不是 source filesystem。受控根为：

```text
timeline
sessions
projects
resources
memories
skills
agents
```

`projects/{workspace}` 仅对 ordinary Context 表达 workspace 归类；不对应用户 memory 物理路径。Memory document 从受控相对路径投影为：

```text
memories/root
memories/profile
memories/preferences
memories/knowledge
memories/knowledge/open-loops
memories/knowledge/entities/{safe-name}
memories/knowledge/topics/{safe-name}
memories/knowledge/episodes/{safe-name}
memories/experiences/{safe-name}
```

路径深度与 secondary paths 数量有上限。Path prefix 在 SQL 中用索引筛选，不递归扫描 View 或 source 目录。Timeline path 由 `event_time` 构造，不以写入日期替代发生日期。Tree 重分类只更新派生 paths，不改变 source/document identity。

## 4. Session Projection

`SessionContextProjector` 只接收已耐久落盘且 archive/manifest digest 校验通过的 SessionArchive。

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

Semantic segment 有最大 chunk size，并在调用方本地 Timeline 日界切段。Message/Tool Result 原文仍由 Archive 持有，不为每个原子 event 生成 Vector。投影 record key 由 archive identity + source digest 确定性生成，重复 projection 是 upsert。

文件读取场景中，Projector 从已清洗 Tool Result 提取 basename 和 controlled location，生成 Resource Reference。完整绝对路径只留在受保护 Archive。

`SessionProjectionJournal` 记录 tenant + owner + workspace + archive identity 的恢复状态。迟到 projection 不能穿过已应用的 Session delete barrier 复活记录。

## 5. Markdown Memory Source

Memory root 为：

```text
<root>/tenants/<tenant_id>/users/<owner_user_id>/memory/
```

每个 managed file 必须有最小安全 front matter 与不可变 `document_id`。Stable URI 基于 owner + document ID，不基于 path。Tenant/owner 由受信任 root 绑定，不从 front matter 读取。

`FileSystemMemoryDocumentStore` 对 exact bytes 计算 SHA-256，在 strict UTF-8 解析前保留 raw state。它使用 dir-fd、`O_NOFOLLOW`、regular-file/link-count 校验、同目录 temp file、atomic install、file fsync 与 parent fsync。不安全路径、symlink、hardlink、非 UTF-8、过大文件、坏 front matter 与 duplicate identity 都 fail closed。

Raw path state 与 registration state 分离：空文件是 PRESENT 但 UNMANAGED；缺 ID 的安全新文件要显式 adopt；坏 ID 或 duplicate 进 quarantine，不解释为删除。

## 6. Memory Document Projection

`MemoryDocumentProjectionWorker` 从耐久 queue 领取绑定 tenant + owner + document + generation 的 job。它精确读取 live source，校验身份/digest/path，再生成：

- 一条 `memory_document` Catalog record；
- 有界 `memory_block` records；
- `l0_text` / `l1_text`、FTS 与 controlled tree paths；
- 有界 relations 与可选 Vector rows；
- atomic document projection state。

Block 以 heading path + occurrence 分段，同时受 max blocks、max block chars 和 summary chars 限制。每个 block 带 `document_id`、source digest 和 generation。

Worker 对较新 generation 执行 compare-and-swap；低 generation 不发布。Delete/forget job 摘除 document/block Catalog、FTS、Path、Vector 和 document-owned Relation。Publication barrier 拦截已删除 identity 的迟到 job、scan 或 rebuild。

## 7. Scanner 与外部编辑

Watcher 事件仅调度 `MemoryDocumentScanner` 。Scanner 对 bounded full scan generation 比较 `(document_id, relative_path, raw_sha256, size)`，并经稳定窗口确认 create/update/rename/delete。

只有 root identity 不变、scan complete、无 traversal error、无 duplicate identity、无不安全歧义且未命中 mass-delete threshold 时才能发布 delete。Overflow 会强制 full scan。外部确认变更产生新 revision/event/job，不使用 mtime 作为业务版本。

## 8. Tombstone、forget 与 retention

Ordinary Context/Session 删除通过 `context_tombstones`：

```text
persist intent
-> validate authoritative lifecycle
-> atomically remove Catalog/FTS/Path/Link/projection state
-> CLEANING
-> compare-and-delete Vector and owned Relation
-> APPLIED
```

新于 tombstone 的 revision/effect 胜出，旧 tombstone 转 STALE，不删新投影。外部 consumer 失败保持 CLEANING 并可重放。Session delete barrier 会阻止同一 Archive 在 rebuild 中复活。

Memory soft forget 是 document CAS edit/delete，并保留授权 revision 供恢复。Hard erase 使用专用耐久 barrier，清理 live file、revision/after/review blobs 和所有派生层；外部 backend 未确认时保持 `ERASE_PENDING`。

Serving lifecycle 只影响派生可见性：

| Tier | 默认 Serving |
| --- | --- |
| HOT | FTS、Vector eligible、L0/L1、快速召回 |
| WARM | FTS、L0/L1，Vector 按策略 |
| COLD | 结构化 time/path 与 source URI，默认无 Vector |
| ARCHIVED | 默认排除，仅显式 history/archive 读取 |

Compaction 仅创建 summary projection 或调整 serving tier，不改业务事实源。缺少可信结构化时间时保持 HOT，避免误归档。

## 9. Query Planner

SDK、HTTP 和 MCP 共用 `RetrievalOptions` 与同一 JSON Schema。`RetrievalQueryPlan` 支持：

- target URIs/paths；
- context/source/record/document kinds；
- trusted tenant/owner/workspace/session/adapter；
- event/transaction/updated time ranges 与 timezone；
- `CURRENT`、`HISTORY`、`OPEN_RECALL`、`EXACT`；
- relation expansion、candidate/final/token bounds；
- semantic query 和经清洗 metadata filters。

安全优先级：

```text
Trusted Caller Constraint
> explicit user filter
> deterministic date/path parsing
> schema default
> non-security query assistance
```

Trusted scope 与用户 filter 做交集；冲突 fail closed。`authorized_scope_keys` 区分缺失 envelope、可信非空 grants、用户显式缩小和显式空 grants；空 grants 仅允许 unscoped records。

Date-only filter 按调用方 timezone 转 UTC 半开区间。“某日发生了什么”转为 `OPEN_RECALL + event_time`，“某日系统写入了什么”转为 `HISTORY + transaction_time`。

## 10. Structured 与 Exact 候选

`CandidateGenerator` 在任何 Top-K 前将 tenant、owner、workspace、session、adapter、ACL/applicability、record kind、context/source/document type、lifecycle/tier、path 和 time 放入 SQL predicates。

带 semantic query 时，structured branch 不做无条件全库枚举。只有完整受控的时间组合或显式结构化条件才能给出有界 fallback，使“7 月 11 日讨论的 Java 分布式方案”不依赖问句原文出现在正文。

Exact URI 仅做等值 lookup，不做前缀扫描。Document URI 可稳定命中 document 投影，block URI 命中指定 block；物理分支在 branch-local LIMIT 前重复套用 trusted predicates，外层在 final LIMIT 前再次复核。

Workspace identity 在进入 Catalog、Scope 和 Vector metadata 前统一规范化。本地 repo path 或 remote-like identity 仅保留不可逆 workspace identity，不把原始绝对路径复制到索引。

## 11. FTS

Catalog 使用同一 `contexts_fts`，保存 record key、URI、title、safe content、safe metadata text、search terms 和哈希化 ACL tokens。FTS5 可用时用固定 BM25 配置取 bounded overfetch；不可用时关闭 lexical branch，不退化为 Python/SQL contains 全表扫描。

Token 只做前置裁剪，不是 ACL 证明。FTS hit 通过 `record_key` 回连 `contexts`，在 LIMIT 前重新执行 scope/path/time/type predicates。`context_fts_map` 保持 record key 到 FTS rowid 一对一，orphan/mismatch 使启动验证失败并进入离线重建。

只有经 `ContextProjectionSanitizer` 的 title/L0/L1/metadata 进入 FTS。Secret、完整绝对路径、binary 和超长 Tool Result 不进索引。

## 12. Vector

VectorStore 声明 metadata、namespace、time filtering 和 delete-by-filter 能力。支持 native filter 的后端先按 tenant/owner/workspace/session/adapter、record/document kind、path、time、scope、source digest 和 generation 过滤，再 Top-K。

不支持 native filter 的后端只对 SQL/FTS 给出的有界 record keys 计算相似度，overfetch 有硬上限并在 Trace 标记 degraded mode。在线链不枚举 Vector namespace。

Vector hit 始终回 SQL Catalog 以 `(tenant_id, record_key)` 复核 ACL、path、record ownership、digest 和 generation。Vector metadata 不是第二个权限或正文事实源。Vector 失败显式降级 FTS，不伪装成零结果。

## 13. Relation

Relation expansion 只处理融合后的有界 seed，每个 seed 的 identity 和 edge 数量均有上限。Relation target 不直接变成 result，而是经 exact identity lookup 重新绑定到当前 plan 的 tenant、owner、workspace、ACL、record kind、path 和 time 限制。

Memory document relations 必须使用 document/block stable URI，并且由 document projection generation 拥有。Delete/forget 只 compare-and-delete 该 generation/effect 所拥有的 rows，不会删除较新 projection 的 relation。

## 14. Fusion、Rerank 与去重

Exact、FTS、Vector、Relation 的 raw score 不直接比较。`FusionRanker` 使用 RRF，并保留 exact、lexical、vector、relation、recency、hotness、rerank 和 final components。Recency/hotness 只对真实 relevance 调整，不能让无关新数据超过相关旧数据。

Reranker 失败保留 deterministic fusion order 并标记 fallback。去重使用：

- memory document：tenant + owner + document ID；
- memory block：document ID + block ID；
- resource：resource URI + digest；
- ordinary context：source digest/stable key；
- session：stable archive record identity，另受 per-session quota。

同一 record 从多 path/多 branch 命中只返回一次。

## 15. Source Hydration 与记忆校验

候选生成、融合和 rerank 完成后，Orchestrator 只 hydrate 受 final/token/source-read bounds 约束的前缀。

Memory candidate 由 `MemoryDocumentContextOverlay` 精确读 live source，校验：

- trusted tenant 与 owner root；
- document URI 与 front matter ID；
- path-derived document kind；
- exact raw digest；
- logical revision 与 projection generation；
- body size 与 safe projection。

验证失败时 drop candidate 为 `stale_memory_document_projection`，记录 degraded mode 并耐久调度 bounded rescan。不允许直接返回 Catalog L1 作为当前记忆正文。

Ordinary Resource L2 也必须精确读 Source object，复核 tenant/owner/workspace/adapter/lifecycle/source URI/digest，再安全清洗。Session Message/Tool Result 的原文只留在 Archive，普通召回不为每个原子节点读 L2。

## 16. L0/L1/L2 Packing

LayerSelector 降级顺序：

```text
L2 Full -> L1 Overview -> L0 Abstract -> URI / Reference
```

`ContextPacker` 同时限制 token budget、final items、context type quota、per-session、per-document、blocks-per-document、per-resource-branch、L2 items 与来源多样性。Packer 先形成可 hydrate 的有界前缀，避免每个候选触发 source read。

`CURRENT` 优先当前 memory documents 中的 profile/preferences/entity/topic 与相关 resources；`OPEN_RECALL`/`HISTORY` 优先 session/event/resource/tool result/episode 与 document history references。公开 output 包含 selected layer、source URI、drop reason、token estimate、source validation status、projection lag、degraded modes 与 score components。

## 17. 时间模型

- `event_time`：事情实际发生时间，用于 Timeline 与“某日发生了什么”；
- `ingested_at`：系统接收时间；
- `transaction_time`：投影/提交时间，用于“某日新增了什么”；
- `created_at/updated_at`：派生对象创建与更新时间；
- document `occurred_at`：episode 候选的事件时间，不取代 file revision 时间。

所有索引时间使用带时区 UTC ISO-8601。Date-only filter 在 caller timezone 下转为半开 UTC 日界。时间过滤使用结构化列与索引，不对 metadata JSON 做模糊查找。

## 18. Startup、repair、rebuild 与 verification

新 runtime 仅接受 `markdown_memory_v1` layout。不受支持的历史 layout/DB 明确报 reset-required，不自动删库、转换或建并行读写通道。

READY 前恢复顺序：

```text
validate runtime layout
-> recover expired queue leases and ordinary operation groups
-> recover document intents
-> resume Session commit groups and projection journal
-> bounded stable full scans
-> replay/rebuild document projections
-> drain projection jobs and deletion tombstones
-> compare live sources with Catalog generations
-> READY
```

`CallbackDocumentServingMaintenance` 仅通过 bounded owner provider 获得 owner，不递归扫 runtime root。对每个 owner：

1. 执行 complete safe full scan；
2. 验证无 errors、unsafe/unmanaged/quarantined 与 duplicate identities；
3. 重建 document/block/FTS/Path/Relation/Vector 派生；
4. 再次 full scan 并比较 source fingerprint；
5. `CatalogDocumentProjectionVerifier` 逐 document 比较 live ID/path/digest 与 Catalog/projection state。

中途 source 改变、owner/document bound 超限或任一 proof 不一致都使重建失败，runtime 保持 NOT_READY。

## 19. 性能边界与可观测性

在线 Search/Assemble 禁止：

- `source_store.list_objects()`；
- memory tree 或 SessionArchive 递归扫描；
- 全库 `allowed_uris` 或 `vector_uris()`；
- `glob views/**` 或递归 Tree；
- Python 全库 lexical match；
- 无 bound Relation 图遍历。

完整扫描只允许 startup、repair、rebuild、audit 或受限 admin/test utility。

硬边界：

```text
structured/exact/FTS/vector/relation candidates <= configured branch bounds
source_reads <= candidate_limit + fixed allowance
vector_overfetch <= configured maximum
per session <= configured per-session limit
per document and blocks per document <= packing policy
L2 reads <= max_l2_items
```

`RetrievalMetrics` 记录各 branch candidates、fusion/rerank counts、memory candidates/validated、source reads、selected/dropped 和 vector overfetch。SQLite 在线查询安装 VM progress guard；超限转换为显式可重试 unavailable，不捕获后返回空成功。`EXPLAIN QUERY PLAN` 用于验证 tenant、owner、type、path prefix、event/transaction time、FTS map 与 document identity 索引。

## 20. 从 OpenViking 吸收的范围

可迁移的启发是：

- target URI 与受控范围检索；
- Session-as-Context、archive abstract/overview 与 token budget；
- L0/L1/L2 渐进加载；
- intent-assisted query、rerank 与按目标缩小召回空间；
- compaction 与受控 tree organization。

MemoryOS 将这些思想落在 SQL Catalog/Path/FTS、bounded Vector、统一 Query Plan、source validation 和 ContextPacker，而不把在线递归文件树当检索器。

MemoryOS 也不把目录 overview、tree path、Vector hit 或 Catalog text 当作用户 memory 当前正文。对 memory result，公开返回前必须对 live Markdown 做 bounded exact hydration 与 proof validation。

## 21. 安全投影与 Recall Trace

`ContextProjectionSanitizer` 在 Catalog、FTS、Vector eligibility、Trace 和 L0/L1 之前 fail closed。它清洗 API/access token、cookie、Authorization、password、private/PEM key、env、DB/SSH credential、URL credential、binary、超长输出和 privacy metadata。

`MemoryEgressPolicy` 约束进入模型提取/规划的出站内容，`ContextProjectionSanitizer` 约束进入 serving/trace 的派生内容。两者不能被 metadata、Tool Result 或公开 API 绕过。

绝对路径被替换为 basename + controlled location；完整路径仅留在受保护 Source Evidence。Sanitization 失败不写部分清洗数据。Query Plan 和 Recall Trace 使用同一 sanitizer，不保留 raw query/source body、secret 或未清洗 error payload。

## 22. 公开 API 与语义平面

公开 Context 入口包括 SDK/HTTP/MCP Search/Assemble、`archive_search()` 和 memory recall，全部进入 `UnifiedRetrievalOrchestrator`。`archive_read()` 保留精确受权 Archive evidence read，不是另一个 semantic search 通道。

Memory document 命令共用 `memoryos/api/memory_contract.py`：`remember`、`edit_memory_document`、`forget`、history、restore 和 review。所有 mutation 需 trusted caller 并经 runtime readiness gate。

Prediction、Behavior 和 ActionPolicy 的域内 lookup 保持语义隔离，不暴露为公共 Context serving 旁路。Coding Agent 的公开 assemble 仍受 context reduction、workspace 和 adapter scope 约束；Embodied 动作继续经 PolicyGate、authority、visibility 和风险控制链。
