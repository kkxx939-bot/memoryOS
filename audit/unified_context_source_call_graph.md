# Unified Context 主链源码调用图（当前复核版）

本图以当前共享工作区源码为准，记录公开读链、Session 写链、Canonical 写链、删除链、迁移和保留链。它不是设计草图；箭头两端均有当前源码入口。Startup / Repair / Audit 的全局校验与 Online Query 的有界路径明确分离。

## 1. SDK / HTTP / MCP 统一在线读链

```text
HTTP POST /v1/context/search | /v1/context/assemble
  -> memoryos/api/http/app.py
  -> parse_retrieval_options()                    [共享 Schema]
  -> MemoryOSClient.search_context / assemble_context

MCP memoryos_search | memoryos_assemble
    aliases: memoryos_search_context | memoryos_assemble_context
  -> memoryos/api/mcp/tools.py
  -> parse_retrieval_options()                    [同一共享 Schema]
  -> MemoryOSClient.search_context / assemble_context

Local / Remote SDK
  -> RetrievalOptions + 旧平铺参数
  -> retrieval_options_from_legacy()
  -> _merge_public_retrieval_options()
  -> QueryPlanner.build()
     -> bind_trusted_scope()
     -> RetrievalQueryPlan                       [已规范化、可序列化]
  -> UnifiedRetrievalOrchestrator.execute()
     1. Runtime readiness
        -> Canonical queue: indexed Tenant + Owner + Workspace job scope health
        -> Session frontier: Tenant + Owner + Workspace health
     2. CandidateGenerator
        -> SQLite structured filters             [WHERE/JOIN 在 LIMIT 前]
           -> context_acl_grants                 [normalized access candidates]
           -> context_path_closure/context_path_acl
           -> context_validity_map/context_validity_rtree
           -> contexts exact-field recheck       [candidate adjuncts are not authority]
           -> temporal structured branch         [仅确定性 intent + 完整 indexed time]
        -> exact target URI candidates
           -> uri / canonical_slot_uri / canonical_claim_uri equality UNION
           -> each branch: ACL + lifecycle/scope/path/time/project/adapter/connect before bounded LIMIT
        -> SQLite FTS candidates                 [hashed ACL/path/scope tokens + hidden rank]
           -> context_fts_map rowid ownership
           -> contexts structured/ACL recheck before LIMIT
        -> native-filter vector branch，或 SQL/FTS 有界 URI 集内 local vector
        -> bounded relation expansion
           -> seed identities: Claim URI -> Slot URI -> serving URI [max 3]
           -> relation rows [max 5/seed] -> trusted Catalog exact rebind [max 2/target]
     3. FusionRanker                            [RRF + 有限分数组件]
     4. optional reranker / deterministic Fusion fallback
     5. intent-aware semantic dedupe
     6. BoundedCanonicalResolver
        CURRENT     -> CurrentSlotProjection 精确校验
        HISTORY/... -> Claim Revision 精确校验
        -> Slot / Claim / Head / Receipt / Effect proof
        -> explicit Canonical Slot CURRENT miss -> unavailable, never empty success
     7. bounded serving hydration                 [只读最终有界候选]
        -> ordinary Resource L2 exact Source object/content
        -> tenant/owner/workspace/adapter/source digest recheck
        -> ContextProjectionSanitizer; failure degrades without raw egress
        -> atomic Message/Tool Result never reads raw Source/SessionArchive
        -> migration-compatible bounded legacy hydration
     8. LayerSelector + ContextPacker
        -> L2 -> L1 -> L0 -> URI
        -> token/type/session/slot/resource quotas
     9. ContextProjectionSanitizer.sanitize_trace()
     10. durable Recall Trace
```

`RetrievalQueryPlan` 先绑定 Tenant、Owner、Workspace、Adapter 和 ACL，再进入候选生成。Workspace path/remote identity 先规范化为不泄漏本地路径的稳定 `project-*` identity。`CONFLICTS` 默认只看 `CONFLICTED`；`OPTIONS` 默认只看 `PROPOSED / CONFLICTED`，调用方显式的更窄状态过滤优先。无 Principal 的公开检索只允许真正 unscoped 的 Resource / Skill，不能枚举带 Owner、Workspace 或 applicability scope 的行。ACL token、Path closure/ACL 和 RTree 只负责有限候选；每个命中仍以 `contexts` 精确字段与 trusted plan 二次校验。带 `semantic_query` 的 structured branch 只接受 `OPEN_RECALL + event_time_from/to`、`AS_OF + valid_at`、`HISTORY + transaction_time_from/to` 三种完整组合；普通语义查询不调用 `list_catalog()` 枚举 Catalog。

普通 Resource 的 L2 不在候选生成阶段读取。Packer 先用同一 Intent、Token、类型、Session 与 Resource branch quota 预选最终有界前缀，Orchestrator 再对最多 `max_l2_items` 个普通 Resource 执行精确 Source object/content 读取并复核 Source Digest；任何越权、过期、回源或清洗失败都不把 raw L2 写入候选或 Trace，而是带明确 degraded mode 降级到已清洗 L1/L0/URI。Session Message 与 Tool Result 始终只使用 Catalog serving projection。

## 2. Intent 与 Projection 选择

```text
CURRENT
  -> record_kind != claim_revision
  -> Canonical Memory 只接受 current_slot
  -> canonical_slot_id 去重，每 Slot 最多一条

HISTORY
  -> 普通 Session / Event / Resource / Tool Result + claim_revision
  -> 排除可变 current_slot
  -> transaction-time 自然查询由默认 CURRENT 确定性转入 HISTORY
  -> 正文使用请求的 Revision value
  -> metadata.canonical_value/value_fields/evidence/qualifiers/proposal/relation 全部绑定请求的 proved Revision
  -> 丢弃可重建候选中的任意业务 payload，仅保留 serving/proof 白名单

AS_OF / CONFLICTS / OPTIONS
  -> 默认 claim_revision；显式普通 Context 类型仍可按调用方收窄
  -> canonical_claim_id + canonical_revision 去重
  -> 同一 Slot 的不同 Claim / Revision 不在 Packer 中被折叠

OPEN_RECALL
  -> Session / Event / Resource / Tool Result + Canonical History
  -> 完整 event-time 日区间允许 SQL structured 候选

EXACT
  -> target_uris 的有界精确候选
  -> 旧 slot_uris / claim_uris 映射到同一 equality identity lookup
  -> Canonical Slot CURRENT 只读 CurrentSlot；历史 intent 只读 Claim Revision
```

普通 Context 的稳定去重键是 Source Digest / Source Identity；Session 和 Resource 额外执行 per-session 与 per-resource-branch 配额。自然时间问题即使与内容没有词面重合，也由完整结构化时间条件生成有界候选；没有上述确定性 intent/time 组合的文本查询仍由 Exact/FTS/Vector/Relation 召回，不能无条件列举 Catalog。

## 3. Archive 兼容链

```text
MemoryOSClient.archive_search()
  -> RetrievalOptions(
       context_types=session,
       query_intent=OPEN_RECALL,
       canonical_resolution_mode=DISABLED,
       bounded limits
     )
  -> MemoryOSClient.search_context()
  -> UnifiedRetrievalOrchestrator
  -> Catalog / FTS / bounded vector

MemoryOSClient.archive_read(archive_uri)
  -> SessionArchiveStore.read_archive()           [精确 URI Evidence 读取，保留]
```

`archive_search()` 不再 `glob("**/commit_head.json")`，不再逐 Archive 做 Python substring match；只有 `archive_read()` 保留精确的不可变 Evidence 读取能力。

## 4. SessionArchive -> Unified Context Catalog

```text
SDK / HTTP / MCP / agent session input
  -> ingress policy
  -> SessionArchiveStore.write_sync_archive()     [不可变 Evidence Source]
  -> SessionCommitService
     -> SessionContextProjector.project()
        -> build_records()
           Session Root / L0 / L1
           semantic segment / message segment     [按 chunk size + 本地 Timeline 日界切分]
           tool result / resource reference
           used context / used skill
           important observation / action result / ordinary event
        -> ContextProjectionSanitizer              [fail closed]
        -> SQLiteIndexStore.upsert_catalog_batch()
           -> contexts                             [现有表 schema v10 原位演进]
           -> contexts_fts
           -> context_fts_map                      [record_key -> FTS rowid]
           -> context_paths
           -> context_path_closure / context_path_acl
           -> context_acl_grants
           -> context_tenants + validity map/RTree
           -> projection state
           -> session_projection_frontier          [tenant/owner/workspace health]
        -> _prepare_vector_rows()
           root + semantic segment
           configured important event/resource
           atomic message/tool excluded by default
        -> VectorStore.upsert_vector()             [只接收已清洗文本]
```

Session、Tool Result、Resource 和普通 Event 的 `canonical_*` 为空。Timeline 路径由 `event_time` 生成；事件同时携带 `occurred_at` 与 `event_time` 时前者优先，只有 alias 时读取 `event_time`。Semantic Segment 不跨调用方本地 Timeline 日界，因此一个 Segment 的单值结构化 `event_time` 与唯一日期路径同源。完整绝对路径只在受保护的 SessionArchive Evidence 中，Catalog / FTS / Vector / Trace 只保留 basename 与受控 location。每次 FTS 替换先经 `context_fts_map` 精确定位 rowid；Path/ACL/Validity 附属表按 `tenant_id + record_key` 更新，避免新写随全表规模放大。

## 5. Canonical commit -> 双 Projection

```text
remember() / confirmed Pending Review / canonical memory formation
  -> CanonicalMemoryFormationService
     -> ProposalEvidenceValidator
     -> MemorySemanticNormalizer
     -> ProposalAdmissionGate
     -> CanonicalPromotionPolicy                  [Identity 之前的确定性 Gate]
     -> StableMemoryIdentityResolver              [Identity V2 不变]
     -> MemorySemanticReconciler
     -> MemoryTransitionPolicy                    [单 ACTIVE Claim]
     -> MemoryTransactionPlanner
  -> OperationCommitter
     -> FileSystemSourceStore                     [Slot/Claim 权威事实源]
     -> Receipt + Current Head + Redo / Recovery
     -> durable canonical projection outbox / SQLiteQueueStore
  -> MemoryProjectionWorker
     -> CanonicalMemoryProjector                  [Claim Revision Projection 保留]
     -> CurrentSlotProjection                     [Migration dual-write gate]
        -> record_key = slot:{slot_id}:current
        -> tenant_id + canonical_slot_id 唯一
        -> ACTIVE Claim 切换时同一行原位更新
        -> 旧 Claim Revision 仍留 HISTORY
     -> Queue ACK 仅在 Projection 完成后
```

重复写入同一状态不会生成第二个 Slot、第二个 ACTIVE Claim 或第二条 Current Serving Record；不同 Evidence 可以形成新 Revision 并保留可审计证据。状态改变仍经过 Pending Review；确认后旧 Claim `SUPERSEDED`、新 Claim `ACTIVE`，CurrentSlot 原位更新。

## 6. 删除与失效传播链

```text
ordinary forget(uri)
  -> ContextDB durable delete entry
  -> ProjectionTombstoneService.enqueue_uri / enqueue_source_uri

session delete
  -> ProjectionTombstoneService.enqueue_session
  -> record_key keyset pagination until exhausted

Canonical RETRACT / SUPERSEDE / active Claim switch
  -> durable canonical outbox
  -> Claim Projection update
  -> CurrentSlotProjection retirement/replacement tombstone

ProjectionTombstoneService.process_tombstones(ids)
  -> source-boundary and revision validation
  -> begin_tombstone_cleanup()
     -> newer Catalog revision/effect: STALE
     -> SQLite transaction deletes contexts + FTS + paths + links + projection state
     -> durable CLEANING; same record_key rewrite is blocked
  -> Vector compare-and-delete by tenant + record key + revision/effect
  -> owned Relation serving edges drained until empty
  -> finish_tombstone_cleanup() -> APPLIED

consumer failure after begin
  -> remain CLEANING + retry_count + sanitized error
  -> retry/replay exact tombstone id
  -> never report a partial cleanup as success
```

Tombstone 删除 Serving 数据但不擅自删除不可变 Evidence Source。当前实现没有第二个独立 Tree Current 或 Cache 主存储；逻辑 Tree、FTS、Path 和 Link 均随 Catalog 删除事务失效。

## 7. Retention / Compaction / GC / Cold Restore

```text
RuntimeContainer
  -> RetentionPolicy.from_config()
  -> CatalogRetentionManager
  -> ContextDB.run_retention_cycle / compact_session_context /
     compact_timeline_context / restore_cold_context

CatalogRetentionManager
  -> HOT -> WARM -> COLD -> ARCHIVED           [与 ClaimState 分离]
  -> Session segment compaction
  -> Timeline overview compaction
  -> stale projection GC
  -> durable vector tombstones + Vector GC
  -> orphan Path GC
  -> applied Tombstone GC
  -> bounded cold restore from Source URI
```

Retention 只改变 Serving Tier / Projection；未配置明确删除策略时不删除 SessionArchive 或 Canonical Source Evidence。

## 8. Migration / Shadow / Cutover / Rollback

```text
RuntimeContainer
  -> SQLite pre-v10 upgrade provenance          [升级事务内耐久 bootstrap]
  -> startup existing-Archive evidence probe    [首个合法 head 即停止，仅 Gate]
  -> RuntimeMigrationCoordinator                [绑定 Tenant，每次读取当前耐久 state]
  -> UnifiedContextMigration
     NOT_STARTED
       -> SCHEMA_READY
       -> BACKFILLING
          -> bounded SessionArchive batch + checkpoint
          -> bounded existing CurrentSlot batch + checkpoint
       -> DUAL_WRITE
       -> SHADOW_VALIDATING
          -> Session / CurrentSlot immutable-source equivalence proof
          -> migration_equivalence_journal
          -> LEGACY primary read                  [same contexts, conservative flat reader]
          -> Unified shadow read                  [ACL/path/RTree/FTS structured chain]
          -> migration_shadow_read_journal        [bounded identity diff]
       -> READY_TO_CUTOVER                       [projection + read journals threshold gate]
       -> CUTOVER
       -> COMPLETED

failure -> FAILED -> resume_failed()
rollback -> ROLLBACK -> legacy-compatible read route
```

只有真正空 Catalog、无既有 Archive Evidence 的 schema v10 数据库才记录独立 greenfield origin 并按 `COMPLETED` 运行。pre-v10 升级或无 origin 且存在旧 Archive 时耐久进入 `SCHEMA_READY`；LEGACY 有结果时明确 degraded，兼容读为空时 unavailable，不能静默漏数据。初始化与 Tenant 绑定使用冲突不覆盖语义，不会把已推进的 DUAL_WRITE/ROLLBACK 改回初态。LEGACY 是同一个 `contexts` 上独立实现的保守平面 owner/public reader，不使用 Unified ACL grant、Path closure、Relation、Vector 或 RTree adjunct，也不扫描 Source/Archive；SHADOW 以 LEGACY 结果对外，同时独立执行 Unified read 并写入 payload-free diff；ROLLBACK 把主读恢复为 LEGACY。迁移只回填可重建 Projection，不修改 Canonical Source，不伪造 Receipt、Current Head 或 Evidence。Runtime dual-write gate 同时控制 Session Catalog 和 CurrentSlot 写入；启动重建后还会分批恢复既有 CurrentSlot Serving Record，避免重启后 CURRENT 漏读。

进入 `READY_TO_CUTOVER` 前必须同时证明：Session 与 CurrentSlot source projection validation 完成、`migration_equivalence_journal` 达到样本/差异阈值、`migration_shadow_read_journal` 达到在线读样本/差异阈值、projection queue 已静默且 source snapshot 未变化。State store 会重算 shadow read 的 `matched`，不信任调用者布尔值。

## 9. Online 与 Offline 边界

```text
Online Search / Assemble
  -> candidate_limit 有界
  -> canonical_validated <= candidate_limit
  -> source_reads <= candidate_limit + 2
  -> vector_overfetch <= configured maximum
  -> SQLite VM progress steps <= configured online bound
  -> SQL/FTS filters before LIMIT
  -> semantic structured 仅接受三种完整 temporal plan
  -> 普通 semantic query 不枚举 Catalog
  -> 不调用 SourceStore.list_objects()
  -> 不调用 vector_uris()
  -> 不 capture 全量 Canonical snapshot
  -> 候选生成阶段不读取 Canonical Source；仅 fusion/rerank 后由 BoundedCanonicalResolver 精确校验
  -> 不 glob/rglob Tree 或 Archive
  -> 不做 Python 逐对象 lexical fallback

bound exhausted / SQLite VM guard tripped
  -> CatalogCandidateBoundExceeded
  -> RetrievalUnavailableError                  [explicit retryable unavailable]
  -> 不得当作 empty success

Startup / Migration / Repair / Audit
  -> 允许 SourceStore.list_objects()
  -> 允许全局 Canonical integrity validation
  -> 允许离线文件系统遍历和 Projection rebuild
```

在线边界由 `tests/integration/test_unified_retrieval_scale_and_bounds.py` 的 Source、Vector、`Path.glob/rglob` spies 和 1k/10k 真实 SQLite 数据证明；`test_ordinary_semantic_query_does_not_enable_structured_catalog_listing` 反向证明普通文本查询不会开启 structured enumeration；真实 SessionArchive 的自然 event-time 公共检索和真实 `remember()` Claim Revision 的 transaction-time HISTORY / valid-time AS_OF 证明三种 temporal branch 不依赖问句词面。`tests/unit/test_retrieval_failure_semantics.py` 证明 VM/候选超界不是空成功，Canonical ordered prefix 的尾部以 `not_validated_bound` 丢弃且 Source reads 不超过 `candidate_limit + 2`。启动全局完整性检查继续由 canonical rebuild / tamper / concurrency 测试覆盖。

安全 rebuild 与 Online Search 共用 `serving_lock`，并在任何 mutation 前完成 Canonical Source、Receipt/Head、projection publication/outbox/queue preflight。Generic rebuild 保留 SessionArchive 派生 row，排除 raw/uncommitted Canonical row；正式 Claim Revision、CurrentSlot 与 Session Catalog 分别由 projector/backfill 恢复。并发查询只观察重建前或重建后的完整状态。

Trusted `authorized_scope_keys` 的三态在 QueryPlanner 固化：缺失 envelope 仅保留本地 principal/workspace 兼容；非空 grants 自动绑定且显式 filter 只能缩小；显式空 grants 只允许 unscoped。Session/Claim/CurrentSlot 使用同一 Vector metadata contract，native Vector hits 仍须经 SQL Catalog structured/ACL 复核。Canonical Outbox job 显式绑定 Tenant/Owner/Workspace，并由 QueueStore 复合索引查询健康状态；Slot URI 的 `subject_*` 不用于推断 Owner。旧 Owner 未归属 job 通过独立的 Tenant-bound 查询只在自身 Tenant 内保守阻断，不能跨 Tenant 影响 CURRENT；scoped 与 legacy-unresolved 两条真实查询均使用 `queue_jobs_scope_status_idx`。Session frontier 同样按 Tenant/Owner/Workspace 隔离，不因其他 Owner/Workspace 的 lag 误报当前 caller。
