# Unified Context 主链四轮自检记录

实施基线：MemoryOS `main@cbcb6c7d63677f0deff65a69c7aca85a0f1c9850`；OpenViking `main@8fb94e9f4d86f1cfeb1f117f382ad76340e7198a`。

本文件只记录已经从源码、差异或真实命令观察到的事实。A-32/A-33 修复后的定向验收为 `248 passed`；A-37..A-46 后新增 `context_links` 测试，主线随后从零执行最终全量、性能与静态质量门。下面较早阶段的全量结果只作为历史证据，不替代文末最新最终验收。

## 第一轮：需求追踪与实施基线

### 已检查

1. 在修改代码前记录了 MemoryOS HEAD、分支、工作区和未跟踪文件；实施基线为 clean `main@cbcb6c7d63677f0deff65a69c7aca85a0f1c9850`。
2. 记录了本地 OpenViking 实际参考 SHA `8fb94e9f4d86f1cfeb1f117f382ad76340e7198a`，没有把浮动的 `main` 当成可审计基线。
3. 指定的 `AGENTS.md`、三份 canonical 架构约束文档和 `.agents/skills/memoryos-canonical-memory-change/SKILL.md` 在初始工作区不存在。没有伪造“已读取”；当前交付已补齐 `AGENTS.md` 和三份架构约束文档，命名 Skill 仍不存在且不会被假装执行。
4. 编码前创建 `audit/unified_context_requirement_traceability.md`，将任务拆成 UC-001 到 UC-055；每一项均绑定实现文件、测试文件和验收状态。
5. 编码前创建读、Session 写、Canonical 写和删除链调用图，并把已知断点列为实现目标。

### 本轮发现并处理的问题

| 发现 | 处理结果 |
|---|---|
| 仓库约束文件缺失，无法按路径读取 | 如实记录初始状态；补建 `AGENTS.md` 与 canonical 架构文档，不声称不存在的 Skill 已运行 |
| 原在线链存在 Source allow-list、全量 Canonical snapshot、`vector_uris()` 和 Archive glob 风险 | 将公开 SDK / HTTP / MCP / `archive_search()` 汇入统一有界 Orchestrator，并增加禁止调用 spies |
| 初始 `contexts` 无法完整表达统一 Catalog | 在原表上幂等升级到 schema v10，没有创建第二张主 Catalog |
| 需求量大且跨 SDK/HTTP/MCP/worker/migration | 用 UC-001..055 RTM 保持逐项追踪；当前没有遗漏编号 |

第一轮结果：`PASS`。

## 第二轮：架构边界独立自检

独立架构审查及后续独立性能/安全复核累计发现 23 个问题；随后逐项重新核对共享工作区。下表记录的是“发现时问题 -> 当前修复”，不是把曾经存在的断点隐藏掉。

| ID | 独立审查发现 | 当前源码修复与证据 | 结果 |
|---|---|---|---|
| A-01 | Migration / Feature Gate 只定义未接 Runtime | `runtime/container.py` 构造 `RuntimeMigrationCoordinator` 与 `UnifiedContextMigration`；Session 与 CurrentSlot dual-write、Orchestrator shadow diff 均读取耐久 state | 已修复 |
| A-02 | 迁移只回填 Session，不回填既有 Current Slot | `CurrentSlotMigrationBackfill` 分批读取真实 committed Slot/Claim/Head/Receipt；Session 与 Canonical checkpoint 分开；启动后也分批恢复 CurrentSlot | 已修复 |
| A-03 | Fusion 虽按 Claim+Revision 去重，但 Packer 又按 Slot 折叠 HISTORY | `ContextPacker` 只在 CURRENT 使用 `seen_slots`；HISTORY/AS_OF/CONFLICTS/OPTIONS 使用 `(claim_id, revision)` | 已修复 |
| A-04 | 普通 Context `forget()` 绕过耐久 Tombstone | `MemoryOSClient.forget()` 在普通删除提交前记录精确 Tombstone，提交后按 id 清理；失败返回 retryable incomplete 而不伪装成功 | 已修复 |
| A-05 | Session Projector 只有 `vector_eligible` 元数据，没有实际 Vector 写入 | Runtime 注入 Embedding/Vector；Projector 只对已清洗且策略允许的 root/semantic/important 记录写 Vector | 已修复 |
| A-06 | Session/Source 删除一次最多枚举 1,000 条 | Tombstone 使用 `record_key` keyset 分页直至穷尽；Relation 也循环清理；>1,000 集成测试覆盖 | 已修复 |
| A-07 | CONFLICTS / OPTIONS 没有默认 ClaimState | Candidate filters 默认 `CONFLICTED` 或 `PROPOSED, CONFLICTED`，显式更窄 `memory_states` 优先 | 已修复 |
| A-08 | 无 Principal 的公开 Resource/Skill 可能看到带 Scope 的行 | Query plan 标记 `principal_absent`；SQL 和 bounded compatibility filter 要求 Owner/Workspace/applicability scope 全为空 | 已修复 |
| A-09 | RetrievalService Recall Trace 只用旧文本清洗 | Service、SDK Unified Trace、Orchestrator 与 Canonical Resolver 统一使用 `ContextProjectionSanitizer.sanitize_trace()` / `assert_safe()` | 已修复 |
| A-10 | SQLite FTS 无结果时进入 Python 逐对象 contains fallback | 在线 `search_catalog()` 已移除 `_search_contains_bounded` 路径；FTS 不匹配不再做 Python 对象扫描 | 已修复 |
| A-11 | Retention 只有类，没有 Runtime / 管理入口 | Runtime 构造 `CatalogRetentionManager`；ContextDB 暴露 tier cycle、session/timeline compaction、cold restore 和 GC | 已修复 |
| A-12 | Tombstone 没有独立 Tree Current / Cache consumer | 当前没有第二个 Tree Current 或 Cache serving store；Tree Path、FTS、Link 与 Catalog 同 SQLite 删除事务清理，Vector/Relation 是显式 consumer | 按实际所有权闭环，无虚构 consumer |
| A-13 | ContextAssembler 内曾保留旧 O(N) helper，存在被重新接回在线链的风险 | 旧 helper 已删除；`ContextAssembler` 现在只是 Unified Orchestrator 的兼容 facade，结构边界测试禁止在线模块重新引用全量 API | 已修复并固化 |
| A-14 | Packer 只按 context_type 排序，没有 Coding Agent 与 Session 子类型优先级 | `_priority()` 同时考虑 coding-agent、record_kind、source_kind、memory_type 与 intent；专项测试覆盖 | 已修复 |
| A-15 | Fresh Catalog upsert 随全表规模出现 P0 级写放大：FTS5 `record_key` 为 `UNINDEXED` 却按它删除，Path/ACL/Closure 附属删除也没有使用 tenant-leading 索引 | Schema v10 增加 `context_fts_map(record_key, fts_rowid)`，启动校验不一致时离线重建；FTS 精确按 rowid 替换，所有附属更新/删除按 `tenant_id + record_key` 执行 | 已修复；1k/10k fresh upsert 与单行更新 VM probe 证明不再按全表线性放大 |
| A-16 | ACL token、Path token/closure 和 Validity RTree 若被当成最终证明，伪造 token 或 RTree float 外扩可能产生假阳性 | 所有附属表只生成有界候选；在 Top-K 前回连 `contexts` 精确复核 Tenant/Owner/Workspace/Path/Time/Scope；`valid_at` 再核对 ISO 半开区间 | 已修复；伪造 path token 被 exact closure SQL 拒绝 |
| A-17 | 常见词 FTS 或复杂 ACL/Path 计划可能在有 LIMIT 时仍执行过多 SQLite VM steps，并被误报为零结果 | 在线 SQLite progress handler 施加硬 VM bound，`CatalogCandidateBoundExceeded` 在 Orchestrator/MCP 转为显式可重试 unavailable | 已修复；常见词和无 Workspace structured probes 均保持固定上限语义 |
| A-18 | Canonical proof 的失败补读可能使 `source_reads <= candidate_limit` 失真，或把未验证尾部误当已验证 | 固定 Source read allowance 为 2；只接受有序验证前缀，尾部标记 `not_validated_bound` 并进入 degraded/drop trace | 已修复；专项测试证明 `canonical_validated <= candidate_limit` 且 `source_reads <= candidate_limit + 2` |
| A-19 | Canonical queue 与 Session projection health 曾可能按 Tenant 全局聚合；随后反向测试又发现 Canonical Slot URI 使用稳定 `subject_*`，不能用 URI 前缀推断 Owner | Canonical Outbox job 绑定 Tenant/Owner/Workspace，QueueStore 增加规范化 scope 列和复合索引；CURRENT health 按 scope 查询。旧无 scope 的未完成 job 保守阻断，已完成 job 不影响读；Session frontier 同样按 Tenant+Owner+Workspace 查询 | 已修复；真实 subject URI、跨 Owner/Workspace、旧 Queue schema migration 与 EXPLAIN 专项测试覆盖 |
| A-20 | Shadow/ROLLBACK 若复用 Unified reader 或信任调用方 `matched`，就不能证明真实新旧差异 | 同一 `contexts` 上实现独立、保守、平面 LEGACY reader；SHADOW 以 LEGACY 返回并独立跑 Unified；state store 重算 matched，ROLLBACK 恢复 LEGACY primary | 已修复；运行时 route 与 caller-matched distrust 专项测试覆盖 |
| A-21 | Cutover 只看 projection diff 会遗漏真实在线读结果差异 | `migration_equivalence_journal` 与 `migration_shadow_read_journal` 分离；READY 同时要求 source projection、online read、queue quiescence 与 source snapshot 门槛 | 已修复；缺失任一 journal 或样本不足均 fail closed |
| A-22 | 全量测试中的 HTTP 302 夹具未消费 POST body，负载下服务端关闭连接会发 RST，造成只在全量出现的假失败 | 两个测试 Redirect handler 在响应前按 Content-Length 读完请求体；生产 HTTP client、跨域拒绝和鉴权 header 规则未改变 | 已修复；定向 2/2 与最终全量均通过 |
| A-23 | 最终独立源码验收发现旧 subject-hashed、Owner 未归属的 Queue job 会跨 Tenant 阻断 CURRENT；兼容 OR 查询还会使 SQLite 选择 queue-wide claim index | InMemory 未归属判断增加 Tenant 约束；SQLite 将 scoped 与 legacy-unresolved 拆成两个 Tenant-bound 查询，并对真实健康查询使用 `queue_jobs_scope_status_idx` | 已修复；Tenant A/B 反向复现、memory/SQLite 回归、真实 trace + EXPLAIN 和独立复验均通过 |

### 不变量复核

- 没有第二事实源：普通 Context 的 Evidence 仍在 SourceStore / SessionArchive；Canonical Slot/Claim 仍在 Canonical Source。
- 没有第二主 Catalog：SQLite `contexts` 原表升级为 v10；新增 `context_paths`、`context_path_closure`、`context_path_acl`、`context_acl_grants`、`context_tenants`、`context_validity_map`/RTree、`context_fts_map`、Link、Projection State、Tombstone、Migration/双 Journal 与 Session frontier 均为可重建候选或耐久控制状态。
- 普通 Session、Message、Tool Result、Resource、Observation、Action Result 和 ordinary Event 不创建 Slot/Claim。
- CanonicalPromotionPolicy 在 Identity 计算之前运行，使用确定性事实；LLM 不能直接生成 Slot/Claim ID 或扩大晋升范围。
- Tree Path、日期路径、文件路径、Session ID、Tool Result ID 没有进入 Slot/Claim identity hash。
- 一个 Slot 最多一个 ACTIVE Claim；多值状态使用单个结构化集合 Claim 或更细 Slot。
- Claim Revision Projection 与 CurrentSlotProjection 并存，没有互相替代。
- `slot:{slot_id}:current` 稳定主键与 `(tenant_id, canonical_slot_id)` 条件唯一索引同时存在。
- Startup 全局 Canonical 完整性检查保留；在线不执行全局 probe/sentinel，只验证 fusion/rerank 后的有界候选。
- Memory、Behavior、ActionPolicy 仍分属独立 planner；Coding Agent 只做 context reduction，动作 PolicyGate 未被改写。

第二轮结果：`PASS`。A-13 的旧 helper 和在线 probe 已从源码删除，并由 `test_structural_boundaries.py` 与 scale spies 阻止回接；A-15 到 A-23 的性能、安全、健康、迁移、跨 Tenant Queue 和测试稳定性缺口已经在 v10 Schema、typed failure、scoped frontier/queue、双 Journal 和协议正确的 HTTP 夹具中闭环。

## 第三轮：从公开入口重追源码调用链

### 读链

```text
SDK / HTTP / MCP / archive_search
-> shared RetrievalOptions schema
-> legacy parameter conversion + trusted scope binding
-> RetrievalQueryPlan
-> CandidateGenerator structured SQL/FTS/exact/bounded vector/relation
   -> normalized ACL grants / Path closure+ACL / Validity RTree candidates
   -> exact contexts recheck before LIMIT
-> RRF Fusion
-> optional rerank or deterministic fallback
-> intent-aware dedupe
-> bounded CanonicalResolver
-> LayerSelector
-> ContextPacker
-> sanitized Recall Trace
```

检查结果：

- SDK `search_context()` 与 `assemble_context()` 都只构造一次 Query Plan、执行一次 Unified Orchestrator。
- HTTP 与 MCP 从 `memoryos/api/retrieval_contract.py` 取同一 Options JSON Schema；没有两套参数语义。
- `retrieval_views`、`search_scope`、旧 `memory_states` / `memory_types` / URI filters 均在 planner 转成结构化 Options。
- Structured filters 在 SQL `WHERE` / Path JOIN 中完成后才 `LIMIT`；ACL grant、Path closure/ACL、RTree 和 FTS tokens 只生成候选，仍须回连 `contexts` 精确复核。
- FTS 使用哈希 ACL/path/scope token、FTS5 隐藏 rank/BM25 和 `context_fts_map` rowid ownership；伪造 token 不能绕过 SQL recheck，单行替换不扫描所有 FTS/附属行。
- CURRENT 的 Canonical 分支排除 Claim Revision；HISTORY 在 SQL Top-K 前排除 CurrentSlot，同时保留普通 Session/Event/Resource/Tool Result 与 Claim Revision；Packer 不再破坏同 Slot 历史。
- Vector 不支持原生 metadata filtering 时，只在 SQL/FTS 有界 URI 中计算；不调用 `vector_uris()`。
- Canonical resolver 只接受 `candidate_limit` 内的有序验证前缀并校验 Slot/Claim/Head/Receipt/Effect proof；尾部显式 `not_validated_bound`，总 Source reads 不超过 `candidate_limit + 2`。
- SQLite VM guard 超界和 Vector/FTS 运行故障均转换为 explicit unavailable，不被空数组掩盖。
- `archive_search()` 不再扫描 Archive 目录；`archive_read()` 仍是精确 Evidence 读取。

### Session 写链

```text
SessionArchive immutable write
-> SessionCommitService
-> ContextProjectionSanitizer
-> SessionContextProjector
-> contexts + FTS/context_fts_map
-> context_paths + path closure/path ACL + normalized ACL grants
-> validity map/RTree + projection state + scoped Session frontier
-> policy-eligible sanitized Vector rows
```

检查结果：Tool Result 独有 Desktop filename 的 Scenario B 可按 `event_time`、`resources/desktop`、basename 和语义召回；Source URI 与 digest 返回；Catalog、FTS、Vector 和 Trace 中没有 secret / 敏感绝对路径；`canonical_*` 为空。

### Canonical 写链

```text
remember / confirmed pending review
-> Evidence / Schema / Scope / Authority / Admission
-> CanonicalPromotionPolicy
-> Identity / Reconcile / Transition / Transaction
-> Source + Receipt + Current Head + durable outbox
-> Claim Revision Projection
-> CurrentSlotProjection under dual-write gate
-> Queue ACK
```

检查结果：重复偏好保持一个 Slot、一个 ACTIVE Claim、一条 CurrentSlot；不同 Evidence 可增加 committed Revision；状态变化先 Pending Review，确认后旧 Claim `SUPERSEDED`、新 Claim `ACTIVE`，CurrentSlot 同键原位更新；重启后 CurrentSlot 由真实 committed state 恢复。

### 删除链

```text
ordinary forget / session delete / source delete
-> durable Projection Tombstone
-> begin: Catalog/FTS/Path/Link/Projection 事务摘除 + CLEANING
-> Vector/Relation ownership CAS cleanup
-> finish: APPLIED
-> APPLIED only after all consumers succeed

Canonical retract / supersede / active switch
-> durable canonical outbox
-> Claim Projection state
-> CurrentSlot replacement or retirement Tombstone
```

检查结果：begin 前失败保留 FAILED；进入 CLEANING 后外部 consumer 失败保持 CLEANING、增加 retry count、清洗错误并可按指定 id 重放；新 revision/effect 竞态返回 STALE，旧 tombstone 不删除新 manifest 的 Vector/Relation；>1,000 Session rows 分页清理；不可变 Session Evidence 不被 Retention/Tombstone 擅自删除。

### Migration / Retention 链

- Migration 状态覆盖 `NOT_STARTED -> SCHEMA_READY -> BACKFILLING -> DUAL_WRITE -> SHADOW_VALIDATING -> READY_TO_CUTOVER -> CUTOVER -> COMPLETED`，并支持 `FAILED/resume` 与 `ROLLBACK`。
- Session 和 CurrentSlot 分别批处理、checkpoint、resume；`migration_equivalence_journal` 用 immutable Evidence 与精确 Catalog 回读证明 projection 等价，`migration_shadow_read_journal` 证明独立 LEGACY/Unified 在线结果差异；state store 重算 caller `matched`。
- LEGACY/ROLLBACK 是同一 `contexts` 上独立、有界、保守的平面 reader，不是第二 Catalog 或文件系统扫描；SHADOW 以 LEGACY 对外并独立运行 Unified。两类 Journal、projection completeness、queue quiescence 与 source snapshot 门槛未同时满足时不能 Cutover。
- Retention 已接 Runtime/ContextDB，ServingTier 与 ClaimState 分离；compaction、Vector/Path/Tombstone GC 和 Cold restore 不删除 Evidence。
- Safe rebuild 与在线查询共用 serving lock；Canonical preflight 在 mutation 前完成，失败时旧 serving state 不变且 runtime NOT_READY；generic rebuild 保留 Session 派生 row，正式 Claim/CurrentSlot 由 projector/backfill 恢复。
- `authorized_scope_keys` 的缺失 envelope、可信非空 grants、显式缩小和显式空 grants 四种输入语义已复核；Session/Claim/CurrentSlot Vector metadata 统一，native hit 仍经 SQL Catalog ACL/ownership 复核。
- Canonical projection queue health 按 Outbox 绑定的 Tenant/Owner/Workspace 复核，不能从 subject-hashed Slot URI 猜 Owner；Session projection frontier 也按 Tenant/Owner/Workspace 复核，其他 Owner/Workspace 的 pending/failed 状态不会污染当前查询健康。

第三轮结果：`PASS`。当前最终调用图见 `audit/unified_context_source_call_graph.md`。

## 第四轮：最终差异、Schema、API、测试与未完成标记

### 已完成的差异审计

- 初始 MemoryOS 工作区 clean；当前所有修改均在工作区，未执行 commit 或 push。
- `README` 没有修改。
- SQLite Catalog schema version 为 `10`；原 `contexts` 表原位迁移。
- 新增规范化附属/控制表：`context_paths`、`context_path_closure`、`context_path_acl`、`context_acl_grants`、`context_tenants`、`context_validity_map`、`context_validity_rtree`、`context_fts_map`、`context_links`、`context_projection_state`、`context_tombstones`、`migration_state`、`migration_equivalence_journal`、`migration_shadow_read_journal`、`session_projection_frontier`。
- CurrentSlot 唯一索引、Tenant/Owner/Workspace/Session/RecordKind/EventTime/TransactionTime/UpdatedAt、ClaimRevision、Path Prefix、ACL grant、Validity map、FTS rowid map 和 scoped frontier 相关索引均存在。
- SDK、HTTP、MCP 公共 Schema 都暴露 `RetrievalOptions`；旧参数仍作为兼容转换输入。
- 新增统一架构文档和三份 canonical 约束文档；没有改 README。
- 定向搜索本次差异与新增主链文件时，没有本任务相关 TODO、FIXME、NotImplemented、空 `pass`、placeholder 或临时 legacy fallback。搜索命中的 `placeholders` 是 SQL 参数变量；`temporary` 是 Retention location / 测试故障文本；`legacy_fallback_enabled` 是迁移 Rollback 契约，不是未完成代码。

### 已执行的专项验证（不可相加，测试集合有重叠）

| 验证范围 | 真实结果 |
|---|---|
| 早期 Unified Context focused suite | `107 passed` |
| SDK / HTTP / MCP / public compatibility 回归 | `134 passed`，另一个补充批次 `3 passed` |
| Session Tool Result Scenario B | `2 passed` |
| 1k Sessions / 10k Tool Results / 1k Slots 有界检索 | 最终专项运行 `2 passed in 29.59s` |
| Migration 状态机与 Runtime Gate | `4 passed` |
| Canonical Scenario A | `1 passed` |
| 结构化时间与跨时区 | `44 passed` |
| Tombstone / public forget / >1,000 paging | `16 passed` |
| Runtime Session Vector / Retention | `22 passed` |
| Schema v10 focused Catalog/FTS/query/failure | `36 passed` |
| Migration/Tombstone/Outbox v10 回归 | `33 passed` |
| v10 1k/10k scale 与写放大复核 | `2 passed in 29.31s`；fresh upsert 1k=`1.793s`、10k=`22.165s`，单行更新 VM steps 1k=`9582`、10k=`9609` |
| Canonical ordered-prefix Source bound | `2 passed`；8 个 Current 候选中验证前 6、尾 2 为 `not_validated_bound`，`source_reads=12=candidate_limit+2` |
| Owner/Workspace health 与独立 Shadow route | health isolation `2 passed`；SHADOW runtime/caller-matched distrust `2 passed` |
| legacy subject Queue Tenant/索引最终回归 | Queue/health 定向集合 `37 passed`；新增 3 个实例证明 Tenant A 不阻断 Tenant B，真实两条 SQL 均命中 `queue_jobs_scope_status_idx` |

这些专项结果与下面的上一阶段全量结果集合有重叠，不做相加计数；A-24..A-46 后的当前最终质量门见文末最终回填区。

### 上一阶段质量门历史结果（已被后续修复 supersede）

| 命令 | 退出码 | 真实结果 |
|---|---:|---|
| `pytest -q` | 0 | 最终 Queue 修复后 `1639 passed in 1866.41s (0:31:06)`；失败 0、跳过 0 |
| `pytest -q tests/integration/test_unified_retrieval_scale_and_bounds.py` | 0 | 最终重跑 `2 passed in 34.47s`；覆盖 1k Sessions、10k Tool Results、1k Canonical Slots 与硬 bounds |
| `python benchmark/unified_context_100k.py --records 100000` | 0 | 最终重跑 100,000 rows；ingest `273.034153s`；query `0.043541s`；返回 `1/100`；candidate bound=true |
| `python benchmark/smoke/context_assembly_smoke.py` | 0 | `decision=do_nothing`、`candidate_count=0`、`context_uri_count=0` |
| `ruff check memoryos tests` | 0 | `All checks passed!` |
| `mypy memoryos tests --show-error-codes` | 0 | `Success: no issues found in 445 source files` |
| `pyright memoryos tests` | 0 | `0 errors, 0 warnings, 0 informations` |
| `git diff --check` | 0 | 无 whitespace error |

100k `EXPLAIN QUERY PLAN` 实际选择了 `idx_context_path_acl_workspace_event`、`idx_contexts_record_key` 和 `sqlite_autoindex_context_acl_grants_1`，没有递归文件系统或 Archive scan。Queue scope 最终测试追踪 `stats_for_scope()` 实际执行的 scoped 与 legacy-unresolved 两条 SQL，并证明两者都选择 `queue_jobs_scope_status_idx`；旧 Tenant A subject job 对 Tenant B 返回空统计。Canonical ordered-prefix 专项证明 `canonical_validated <= candidate_limit`、`source_reads=12=candidate_limit+2`；scale spies 证明在线不调用 `list_objects()`、`vector_uris()`、全量 Canonical snapshot、Tree glob/rglob 或 Archive 全目录扫描。

项目 CI 的 Ruff 门是 `.github/workflows/ci.yml` 中的 `ruff check memoryos tests`，本次按该命令通过。一次诊断性的 `ruff format --check .` 报告 118 个仓库既有/混合文件会被重排；Formatter 不是仓库 CI gate，且为避免无关大范围格式改写，没有据此批量改动用户代码。

上一阶段第四轮结果：`PASS`。该结论只覆盖 A-23 时点；随后 A-24..A-33 的再次独立复查使其不再是当前最终质量门。README 未修改，未 commit/push。

## 再次独立复查与修复（2026-07-15）

用户要求按最初 Prompt 再次逐项复查后，不能沿用上一轮 PASS 结论。源码、真实旧库启动和公共 API 反向复现先确认 6 个漏项；连续调用链审查又发现 A-30..A-33 四个相邻时间/候选语义漏项。下表记录本轮实际发现和当前闭环，不以原测试全绿代替语义验收。

| ID | 再次复查发现 | 修复与回归证据 | 最终结果 |
|---|---|---|---|
| A-24 | pre-v10 SQLite 原地升级后无 migration row，被误判 greenfield COMPLETED，旧 Archive 静默漏召回 | SQLite 在升级事务内写耐久 provenance，Runtime 原子绑定 Tenant 为 SCHEMA_READY；启动识别旧 Archive，真正 greenfield 使用独立 origin；真实 v2 首启、Archive-only、重启、Tenant/ROLLBACK 回归 | 已修复，独立迁移验收 ACCEPT |
| A-25 | 默认 HISTORY 只保留 Claim Revision，丢弃普通 Session/Event/Resource/Tool Result | HISTORY record kinds 改为 SQL Top-K 前排除 CurrentSlot、保留其他全部；公共 SDK Session HISTORY 与 `candidate_limit=1` SQL 边界测试 | 已修复，独立检索验收 ACCEPT |
| A-26 | 中文状态时点问题只设置 `valid_at`，intent 仍 CURRENT | 默认 CURRENT 确定性转 AS_OF；显式 intent 与 trusted Tenant/Owner/Workspace/path 不覆盖 | 已修复 |
| A-27 | Canonical Resolver 精确回源后把原始 canonical value 放回正文/metadata，且可重建 Catalog 候选中的任意业务字段可能污染 CURRENT/HISTORY 出站结果 | CURRENT/HISTORY 的 title/text/metadata 整体再次 Sanitization；失败 fail closed；只保留候选中的有界 serving/proof envelope；CURRENT 从 receipt-proved Slot/Claim/current Revision 重建业务 metadata，HISTORY/AS_OF/CONFLICTS/OPTIONS 从请求的 proved Revision 重建 `canonical_value`、value/evidence/qualifiers/proposal/relation 等完整镜像；清洗后复核 Slot/Claim/Revision/Head/Receipt/Effect/Transaction 字段 | 已修复；CURRENT/HISTORY secret/path、`current_leak` 和多字段篡改回归通过 |
| A-28 | 旧 ACTIVE Revision 的 Source `valid_to=None` 被当成永久有效，AS_OF 可泄漏旧状态 | 新增不改写 Source 的 effective-validity 派生；Catalog 与 Vector 关闭旧区间；Resolver 只接受时点 ACTIVE Revision并用 timezone-aware 比较 | 已修复；真实状态切换前后 AS_OF E2E 通过 |
| A-29 | Session Root/L0/L1 Timeline 使用 archive write time，不是 episode event time | 三类摘要的结构化 event_time 与 Timeline 都使用 episode start | 已修复 |
| A-30 | Used Context/Used Skill 无自身时间时，event_time 回退 archive write time，但路径继承 episode date | fallback 改为 episode start；显式引用时间同时决定自己的 event_time 与 Timeline | 自检新增并已修复 |
| A-31 | “当天系统新增了哪些记忆”只设置 transaction range，CURRENT 会排除后来失效的 Claim Revision | 默认 CURRENT 确定性转 HISTORY；公共 SDK/SQLite 测试证明 transaction range、Owner/Workspace/ACL 与 SQL limit 前过滤 | 自检新增并已修复 |
| A-32 | Planner 已正确生成 event/transaction/valid 时间过滤，但 `CandidateGenerator` 只要存在 `semantic_query` 就关闭 structured branch；自然问句与正文无词面重合时三类查询均得到 0 candidate | 只对 `OPEN_RECALL + event_time_from/to`、`AS_OF + valid_at`、`HISTORY + transaction_time_from/to` 开启 bounded SQL structured branch；普通 semantic query 继续禁止 Catalog listing；structured metadata 补齐 `catalog_record_key`，计数只反映真实分支 | 已修复；真实 SessionArchive event recall、真实 `remember()` Claim Revision HISTORY/AS_OF、ACL-before-limit 和 no-enumeration 反向测试通过 |
| A-33 | Session event 只读取 `occurred_at` 而忽略 `event_time` alias；Semantic Segment 可跨本地 Timeline 日界，使一个单值 `event_time` 同时挂到多个日期路径 | 事件时间解析采用 `occurred_at > event_time > 受控 fallback`；Semantic Segment 同时按最大 chunk size 与本地 Timeline path 边界切分 | 已修复；Projector 单元测试和公共 Session 检索跨日测试通过 |

### 本轮四轮复验

1. 需求追踪：重新核对 UC-001..UC-055，并更新 UC-013/015/018/020/022/029/032/039/043/052/053 的实现与测试证据；A-24..A-33 十个漏项全部有源码和真实测试对应。
2. 架构边界：确认没有第二事实源/第二 Catalog/专用 Retriever；普通 Context 不创建 Slot/Claim；Identity 与单 ACTIVE 不变量未变；Claim Revision 与 CurrentSlot 并存；startup 扫描未进入在线 Search/Assemble；Tombstone/Retention/Rollback 保留。
3. 调用链：重新追踪 SDK/HTTP/MCP -> Options/Planner -> bounded temporal SQL/FTS/Vector/Relation -> Fusion -> Canonical Resolver -> Packer，并重追 Session `occurred_at/event_time` -> Timeline -> Semantic Segment；A-30..A-33 均已修复。
4. 最终差异与质量门：HISTORY metadata 保持规范化 Claim 值、正文保持 Revision 值；A-32/A-33 后定向验收 `248 passed`。A-46 新测试落地后，主线又从零完成 `1699 passed` 全量、性能、100k、静态检查、差异与未完成实现搜索，第四轮最终为 `PASS`。

### 本轮当前真实命令与最终门状态

| 命令/范围 | 当前结果 |
|---|---|
| A-24..A-33 相关迁移、Session、Canonical、时间、ACL、结构与兼容定向验收集合 | `248 passed` |
| A-32 真实公共链 | SessionArchive 自然 event-time recall、真实 `remember()` Claim Revision transaction-time HISTORY / valid-time AS_OF 均通过；内容不含中文问句、Canonical resolution 未禁用 |
| A-32 反向边界 | 普通 semantic query 不开启 structured Catalog listing；transaction structured candidate 在 SQL limit 前应用 Owner/Workspace/ACL |
| A-33 event/segment | `occurred_at` 优先、`event_time` alias、跨本地日界拆 Segment 的单元与公共链测试通过 |
| 最终 `pytest -q` | `1699 passed in 421.97s`；失败 0、跳过 0；命令在 A-46 新测试落地后从零执行 |
| 最终性能、Ruff、MyPy、Pyright、compileall、pip check、diff check | 规模 `4 passed in 54.18s`；100k bound=true；其余全部 exit 0，详见文末最终回填 |

当前定向测试继续证明 structured time/path/ACL 在 SQL `LIMIT` 前完成，普通 semantic query 不枚举 Catalog；既有结构和规模 spies 继续覆盖在线路径禁止 `list_objects()`、`vector_uris()`、全量 Canonical Snapshot 或 Archive/Tree 递归扫描。最终规模/100k 命令仍由主线质量门重新执行或明确引用真实最新输出。README 未修改，未 commit、未 push。

## 最终独立反向验收与修复（2026-07-16）

在 A-24..A-33 之后再次从公开 SDK 结果倒查到 Source read，新增确认并关闭以下遗漏；本轮不以模型类或单元 Packer 存在代替生产调用链证明。

| ID | 独立审计发现 | 修复与反向证明 | 结果 |
|---|---|---|---|
| A-34 | `CandidateGenerator` 从 Catalog 构造普通 Resource 候选时 `text=""`，原编排只给 Session Root/Semantic Segment 补 L2；Resource 即使有有效 `l2_uri`，公开 SDK 也只能选 L1，UC-040 原测试只证明 Packer 能处理预填充 `text`，未证明生产链能取得 L2 | Fusion/Rerank/Canonical resolve 后先用真实 Packer quota 预选最多 `max_l2_items` 个普通 Resource；精确读取 Source object/content，复核 Tenant/Owner/Workspace/Adapter/Lifecycle/Source URI/L2 URI/Digest/noncanonical，再经 Sanitizer 写入候选。公开 SDK 以 1000/80/10/40 token 分别证明 L2/L1/L0/URI；Source read=2 且保持全局硬界 | 已修复，ACCEPT |
| A-35 | 并行变更曾把全部 Relation behavior cases 额外写入公开 `similarity_scores`，超出本任务禁止修改 Behavior 语义的边界 | 删除新增 score loop 与对应新增断言，仅保留 Tenant filter；当前源码仍只对最终 representative cases 计分 | 已修复，语义边界恢复 |
| A-36 | 全仓 Pyright 在 `LowConfidenceIndex` 测试桩发现新增 `IndexStore.ordinary_relation_endpoint_state` 契约未满足 | 让测试桩继承真实 `InMemoryIndexStore`，保留低置信 `search()` override 并继承生产一致 endpoint-state 行为，无 ignore/空假实现 | 已修复；单文件 34 passed，全仓 Pyright 0 errors |

本轮独立验证的真实结果（集合有重叠，不相加）：

- `pytest -q tests/integration/test_unified_retrieval_l2_hydration.py`：`2 passed`；同时证明 Tool Result/Message 对 SourceStore 与 SessionArchive raw reader 的调用为 0，Secret/绝对路径不进入输出。
- 新 L2 + Packer + Unified Session E2E：`13 passed`。
- ACL/Canonical Resolver/失败语义/Scale/Resource/Rebuild/Fence：`52 passed in 57.96s`。
- Hybrid/Tombstone：`46 passed in 13.77s`。
- `pytest -q tests/unit/test_target_resolver.py`：`34 passed`。
- `ruff check memoryos tests`：`All checks passed!`。
- `mypy memoryos`：`Success: no issues found in 295 source files`。
- `pyright`：`0 errors, 0 warnings, 0 informations`。
- 新增/修改 L2 文件定向 `git diff --check`、未完成标记与 O(N) 禁止调用反向搜索均无命中；在线 Resource L2 只做最终有界 exact reads，没有 `list_objects()`、`vector_uris()`、Tree/Archive glob 或 Python 全库 lexical scan。

独立结论：A-34..A-36 已闭环；未发现新的代码、Schema、迁移、API 或安全阻断。其后 A-37..A-46 继续发现并修复边界缺口，最终全仓与 100k 的最新统一计数以文末回填为准，不能用上述重叠定向集合替代。

## 最终收口复查与修复（2026-07-16）

在 A-34..A-36 后继续按最初 Prompt 从安全派生、Exact ACL、公开 L2、Pending crash、Packing、Relation 删除传播和 Canonical prefetch 七条链反向复查。下列结果均基于当前源码；定向测试集合互有重叠，不相加为“总测试数”。

| ID | 最终收口发现 | 修复与反向证明 | 结果 |
|---|---|---|---|
| A-37 | 通用 `LayerRefresher`、Semantic/Embedding worker 的派生 L0/L1/provider input 仍需证明在任何写入前统一 Sanitization；只清洗 Session Projector 不能覆盖普通 Resource/Skill/Worker 链 | `LayerRefresher` 在 L0/L1 写入前调用 `ContextProjectionSanitizer`，L2 原值只留 Source；Embedding worker 在 Provider/Vector 前清洗 title/content/metadata，失败直接 dead-letter；worker 全周期受 projection fence 保护 | 已修复；`test_workers_production_stores.py` 证明 Secret/私有路径不进 L0/L1/provider/vector，Sanitizer 失败时 Provider/Vector 调用均为 0 |
| A-38 | Exact URI/Slot/Claim 分支若先取 Top-K 再应用 Owner/Workspace/ACL、Lifecycle、Path、Time，会让越权行挤掉授权候选并造成静默漏召回 | SQLite exact identity UNION 的每个分支在 branch `LIMIT` 前应用 Tenant、Principal、Workspace、Lifecycle/Admission/Tier、Scope、Path、event/transaction/valid time；10k same-slot Revision 仍受 VM/candidate hard bound | 已修复；Catalog exact ACL、AS_OF/path/time、transaction ACL-before-limit 与 10k revision EXPLAIN 测试通过 |
| A-39 | A-34 的 Resource L2 修复必须再从公开 SDK 证明：只允许最终有界普通 Resource 精确回源，atomic Message/Tool Result 绝不读取 raw Source/Archive，并继续服从 Packer 配额 | Orchestrator 在 Fusion/Rerank/Canonical resolve 后复用真实 pack quota 预选 L2 identity，逐项复核 Tenant/Owner/Workspace/Adapter/Lifecycle/URI/Digest/noncanonical 后重新 Sanitization；`source_reads` 计入全局硬界 | 独立复验通过；公开 SDK 以不同预算稳定得到 L2/L1/L0/URI，Message/Tool Result raw reader 调用为 0 |
| A-40 | Pending confirm 在 Source 已推进、进程崩溃且 startup rebuild 先生成临时 attempt 后恢复时，历史 Claim Catalog row 可能误绑定 rebuild attempt，而不是原事务不可变 publication proof | Crash-resume 读取原 publication receipt/attempt snapshot，历史 revision row 重新绑定相同 `projection_attempt_id`、artifact digest 与 `publication_record_digest`；临时 rebuild 只可作为可丢弃尝试 | 已修复；`test_pending_review_concurrency.py::test_crashed_confirm_and_apply_command_keeps_exclusive_resolution_ownership` 验证原 attempt/digest 完整一致 |
| A-41 | `LayerSelector` 在 L0 放不下时返回非合同 `excerpt` 层，违反唯一降级顺序 `L2 -> L1 -> L0 -> URI`，也让 Coding Agent 出现第五种隐式层 | 删除 `excerpt` 常量、截断 helper 和特殊分支；Packer 极小预算只能选择 URI；同步收紧 Coding Agent、Packer、API 与 L2 集成断言 | 已修复；定向 Packer/Coding/API/L2 回归 `40 passed`，源码反向搜索不再存在 `selected_layer=excerpt` |
| A-42 | 为保留 retired ActionPolicy 的 Source relation，公共 `ContextDB.add_relation()` 曾放宽到非 serving endpoint 也持久化，导致 delete/tombstone 后普通 Relation 可被重新写入 Source，并在未来 rebuild 复活 | 公共 `add_relation()` 恢复对 DELETED/OBSOLETE/retired endpoint fail closed；OperationCommitter 仅允许 ActionPolicy Schema 声明的关系作为 Source-only Evidence，且 relation manifest 明确不发布 RelationStore；任意普通调用不能使用该内部例外 | 已修复；原两项 relation resurrection 测试、完整 Prediction/Relation/Redo/Recovery 组合 `56 passed`；Ruff、MyPy、Pyright 均通过 |
| A-43 | Canonical formation prefetch 命中 `current_slot` Serving URI 时直接按 hit URI 回源会找不到 Claim；若只信 Catalog 的 claim pointer，又可能把错绑/过期 Current Slot 暴露给模型并接受非法 `related_candidate_refs` | 只把 Catalog 当有界 pointer：精确回源 active Claim 与 Slot，分别校验 Claim revision 和 Slot projection revision，并复核 Claim/Slot ID、URI、Tenant、ACTIVE state、active pointer；任何错绑 fail closed | 已修复；真实 Slot revision=2 / Claim revision=1 可正确 prefetch，篡改 active claim ID 后返回空；相关组合回归通过 |
| A-44 | UC-001..UC-055 覆盖主章节但没有把 EvidenceRef/字段级 Evidence、Redo/Recovery/Pending、MemoryEgressPolicy、`context_links`、受控 Tree、`relation_expansion`、九项分数和完整计数逐项绑定 Requirement ID | 扩展 RTM 到 UC-056..UC-064，逐项写明实现文件、真实测试和硬边界；质量门执行前不提前标记 UC-052/UC-053，执行后再以真实命令回填 | 已修复；RTM 细粒度要求均可反向定位到源码和测试，UC-052/UC-053 已由最终质量门收口为 PASS |
| A-45 | `unified_context_modified_files.md` 声称覆盖当前 `git status`，但集合反查遗漏 27 个变更路径，无法满足最终报告逐文件说明 | 以当前 porcelain status 为左集合补入 24 个 tracked `M` 和 3 个 untracked `??`，逐文件说明安全、迁移 fence、Relation、Canonical prefetch、Worker 和测试主链位置 | 已修复；文档与当前 status 路径集合差为 0，README 仍未修改 |
| A-46 | `context_links` 已有 schema、双向索引和双端 Catalog 删除代码，但原测试没有直接调用 `upsert_context_link()`，RTM 若直接标 PASS 缺少独立证据 | 增加 SQLite 集成测试：写入 link 时清洗 metadata，验证 source/target 两个索引存在，并分别删除 target 和 source Catalog record，确认 link 均在同一 Catalog 删除事务中清零 | 已修复；新增定向测试 `1 passed`；该修改发生在第一次最终全量启动后，因此主线已在其落地后重新从零执行并得到 `1699 passed` |

A-37..A-46 最终组合复验实际执行 Worker Sanitization、Exact ACL/AS_OF/Relation、`context_links`、Resource L2、Pending crash、严格 Packer/Coding Agent、Relation resurrection 和 CurrentSlot prefetch，共 `44 passed in 33.10s`。该集合与前述 `40/56/235/248` 等定向集合有重叠，不做相加。

### 第四轮最终差异与质量门结果

以下项目是宣布完成前的唯一当前门。历史 `1639 passed`、A-32/A-33 的 `248 passed` 以及上述定向集合都保留为诊断证据，最终结论只使用 A-46 新测试落地后的最新结果。

| 检查 | 必须记录的真实命令/证据 | 当前状态 |
|---|---|---|
| 修改后基线 | `git rev-parse HEAD`、`git branch --show-current`、`git -c core.quotepath=false status --short`、未跟踪文件 | PASS：仍为 `main@cbcb6c7d63677f0deff65a69c7aca85a0f1c9850`；86 个 tracked 修改、53 个 untracked，共 139 个；未 commit/push |
| 完整测试 | `pytest -q` 的总数、通过、失败、跳过和耗时 | PASS：A-46 后从零 `1699 passed in 421.97s`；失败 0、跳过 0；HTTP 用例获准绑定 loopback |
| 相关性能/集成 | 1k Sessions / 10k Tool Results / 1k Slots、在线 O(N) spies、硬计数、EXPLAIN | PASS：`pytest -q tests/integration/test_unified_retrieval_scale_and_bounds.py` 为 `4 passed in 54.18s`；spies 与硬界全部通过，SQL 使用受控索引 |
| 100k Benchmark | `python benchmark/unified_context_100k.py --records 100000` 的 records、returned、bound、query plan | PASS：100,000 rows，ingest `311.543022s`、query `0.149481s`、返回 `1/100`、bound=true；使用 `idx_context_path_acl_workspace_event`、`idx_contexts_record_key`、ACL 唯一索引 |
| Smoke | `python benchmark/smoke/context_assembly_smoke.py` | PASS：`decision=do_nothing`、`candidate_count=0`、`context_uri_count=0` |
| 静态检查 | `ruff check memoryos tests`、`mypy memoryos tests --show-error-codes`、`pyright memoryos tests` | PASS：Ruff `All checks passed!`；MyPy `449 source files`；Pyright `0 errors, 0 warnings` |
| 环境检查 | `python -m compileall -q memoryos tests`、`python -m pip check` | PASS：compileall exit 0；`No broken requirements found.` |
| Node 集成 CI | 两个 integration 分别执行 `npm ci`、`npm test`、`npm run build` | PASS：OpenClaw 2 tests、OpenCode 3 tests，构建与依赖审计均 exit 0、0 vulnerabilities |
| 最终差异 | `git diff --check`、`git diff --stat`、`git diff`、README diff、status 与 modified-files 集合差 | PASS：diff check 无输出；README diff 为空；modified-files 与 status 为 139/139、missing/extra/duplicate 均 0 |
| 未完成实现搜索 | 本次主链定向搜索 `TODO/FIXME/pass/NotImplemented/placeholder/temporary/legacy fallback` | PASS：无本任务未完成实现；`pass` 仅异常类型/受控清理与测试分支，placeholder 为 SQL 参数，temporary 为策略/原子临时文件，legacy flag 为 Rollback 合同 |

第四轮结果：`PASS`。所有项目均有真实命令结果，UC-052/UC-053 已更新为 `PASS`，没有失败、任务内未完成项、README 修改、commit 或 push。

## 当前真实限制与阻断判断

- 当前代码不依赖外部 Embedding Provider 才能完成 FTS/structured fallback；但真实生产 Vector pre-filter 集成需要调用方配置具备 metadata/namespace/time/delete-filter capability 的后端。Server mode 对不具备能力的本地别名 fail closed。
- 当前环境未配置真实外部 Embedding Provider / 生产 Vector DB，因此不能声称完成外部服务的联网验收；这是允许报告的基础设施限制。
- A-37..A-46 修复后，最终全量、性能、静态、差异和反向调用链验收未发现剩余代码、Schema、迁移、API 或测试实现阻断。外部 Vector/Embedding 限制保持不变。
