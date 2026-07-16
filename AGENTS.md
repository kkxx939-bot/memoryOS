# MemoryOS Repository Constraints

本文件约束仓库内所有人工和 Agent 变更。其优先级高于局部实现便利；若实现与本文件冲突，必须先修正实现或更新经过评审的架构决策，不能通过兼容分支绕开约束。

## 变更前基线

开始修改前必须记录：

- 当前分支、`HEAD`、`git status --short` 和未跟踪文件；
- 用户已有修改，且不得回退、覆盖或顺手格式化无关文件；
- 本次涉及的公开 SDK、HTTP、MCP、worker、hook 和后台修复入口；
- 相关 Schema、迁移状态、Source/Index/Vector/Relation/Queue 边界；
- 引用外部项目时的实际 Commit SHA，而不是浮动分支名。

不自动提交，不自动 Push，不修改 README，除非用户明确授权。

## 统一 Context 与 Canonical Memory 强制约束

以下九条是不可协商的不变量：

1. Slot/Claim 只服务状态型长期记忆。
2. 普通 Session、Resource、Tool Result 使用统一 Catalog，不得被机械转换成 Slot/Claim。
3. Claim Revision Projection 与 Slot Current Projection 必须并存；后者不能替换前者。
4. 所有公共 Context 在线检索必须经过统一编排链；Prediction、Behavior、ActionPolicy 的隔离域内 policy lookup 不得暴露或回接成第二条 Context Serving API。
5. Tree Path 不参与 Canonical Identity，也不得改变 Slot ID、Claim ID 或 Canonical URI。
6. 在线查询禁止全库扫描；`list_objects()`、全量 canonical snapshot、`vector_uris()`、递归 Tree/Archive 扫描只能用于 startup、migration、repair、audit 或 test utility。
7. Tool Result 必须先经过 `ContextProjectionSanitizer`，才可进入 Catalog、FTS、Vector、Trace 或 L0/L1。
8. 删除必须通过耐久、幂等、可重放的 Tombstone 传播到派生层。
9. 迁移必须支持兼容读、可选双写、Shadow Validation、Feature Gate、Cutover 和 Rollback。

统一模型是：

```text
Evidence Plane
    -> Unified Context Catalog / Tree / Graph
    -> Unified Serving Index
    -> Unified Retrieval Orchestration
    -> Optional Canonical Resolution
    -> L0 / L1 / L2 Context Packing
```

即 `Context First + Canonical State Overlay`。`SQLiteIndexStore.contexts` 是唯一统一 Serving Catalog；不得新增保存同一 Serving 状态的第二张主表。普通 Context 的事实源仍是 SourceStore 或不可变 SessionArchive；Slot/Claim/Revision 的权威状态仍是 Canonical Source。Catalog、FTS、Vector、Tree、Projection 和 Relation Serving Index 都必须可重建。

重建必须与在线查询共用 serving lock，并在修改派生层前完成 Canonical Source/Receipt/Head/Projection preflight。Generic rebuild 不得把 raw/uncommitted Canonical row 当正式 Projection；Session 派生 row、Claim Revision 和 CurrentSlot 分别由其 Evidence backfill/正式 projector 恢复。

## Canonical Memory 不变量

- Identity V2、Slot ID 和 Claim ID 的确定性计算保持单一实现。
- Tree Path、日期路径、文件路径、标题、L0/L1、Vector/Projection ID、Session ID 和 Tool Result ID 不得加入 Slot/Claim Hash 输入。
- 一个 Slot 最多一个 ACTIVE Claim；多值状态使用一个 Claim 的结构化集合、更细粒度 Slot 或多个独立单值 Slot。
- LLM 只能提出语义候选或辅助非安全查询改写，不能决定最终 ADD/UPDATE/DELETE、是否晋升、Tenant/Owner/Workspace、SQL、URI、Slot ID 或 Claim ID。
- `CanonicalPromotionPolicy` 只做确定性路由；PROMOTE 后仍须通过 Evidence、Schema、Identity、Scope、Authority、Admission、Reconcile、Transition 和 Transaction。
- Canonical Current 强一致并 fail closed。启动期全局完整性验证必须保留；在线检索只校验融合后的有界候选。
- Canonical Source 旧 Revision 不得为关闭区间而回写；Serving Projection 和 AS_OF Resolver 必须从下一非 historical Revision 派生同一 `valid_to`，且 AS_OF 只返回时点 ACTIVE Revision。
- `Memory`、`Behavior`、`ActionPolicy` 语义隔离；Coding Agent 默认只走 context reduction；Embodied Agent 动作继续经过 PolicyGate。

## 写、读和删除主链

Session 写链：

```text
SessionArchive -> ContextProjectionSanitizer -> SessionContextProjector
-> contexts / FTS / context_paths / optional bounded Vector
```

Canonical 写链：

```text
Evidence -> CanonicalPromotionPolicy -> Canonical formation/transaction
-> durable Outbox -> Claim Revision Projection + Current Slot Projection
-> contexts / FTS / context_paths / optional Vector
```

在线读链：

```text
SDK / HTTP / MCP / archive_search compatibility wrapper
-> RetrievalOptions -> QueryPlanner -> trusted scope binding
-> structured filters -> exact / FTS / bounded vector / relation
-> fusion / rerank -> semantic dedup -> bounded canonical validation
-> L0/L1/L2 selection -> token packing -> sanitized Recall Trace
```

统一 HISTORY 同时服务普通 Session/Event/Resource/Tool Result 与 Claim Revision，并在 SQL Top-K 前排除 CurrentSlot；状态时点问题使用 AS_OF，系统写入日期问题使用 transaction-time HISTORY。Session Timeline 与每条记录的 `event_time` 必须同源，不能用 Archive 写入日期代替发生日期。

删除链：

```text
Forget / Retract / Supersede / Source or Session delete / reclassify
-> durable projection event or Tombstone
-> Catalog / FTS / Vector / Path / Link / Projection cleanup
```

Canonical 事务不得同步跨越所有派生后端做删除。Tombstone 先以 SQLite 事务摘除 Catalog/FTS/Path/Link/Projection 并进入 CLEANING，再按 record key/revision/effect compare-and-delete 外部 Vector/Relation，最后 APPLIED；外部失败保持 CLEANING 可重放，不能 ACK 或伪装成功。

## 安全边界

- Trusted Caller Constraint 优先于用户显式 Filter；用户 Filter 优先于确定性日期/路径解析；Schema 默认值优先于 LLM 辅助。
- `authorized_scope_keys` 必须区分授权 envelope 缺失、可信非空 grants、用户显式缩小和显式空 grants；空 grants 只允许 unscoped 数据。
- Tenant、Owner、Workspace、Adapter Private Scope 和 Visibility/Authority 必须在候选生成前绑定并在 Canonical Resolution 再验证。
- Session、Claim Revision、CurrentSlot 的 Vector metadata 必须使用统一安全契约；native Vector 命中后仍由 SQL Catalog 复核 ACL、path、time 与 proof ownership。
- 凭证、Authorization、Cookie、密码、私钥、环境变量、数据库/SSH 凭证、敏感绝对路径、Binary 和超长日志不得通过 metadata JSON 绕过清洗。
- 完整路径仅留在受保护 Source Evidence；Serving 层默认只保留 basename 与受控 location。
- Sanitization 失败必须 fail closed；Recall Trace 使用相同安全投影。
- Canonical 精确回源后的公开 title、text 和 metadata 也必须再次经过同一安全投影，并在清洗后复核 Slot/Claim/Revision/Head/Receipt/Effect 等证明字段。

## 验证门槛

涉及本主链的变更必须同时覆盖真实存储和公开入口，而不是只写 Mock。至少验证：

- FileSystemSourceStore、SQLiteIndexStore、RelationStore、QueueStore、SessionArchiveStore；
- Canonical Claim Revision Projection、Current Slot Projection、Outbox/Recovery；
- Session Projection、Tombstone、Retention、Migration/Checkpoint/Shadow/Cutover/Rollback；
- SDK、HTTP、MCP、`archive_search()` 和 Context Packing；
- Tenant/Owner/Workspace/Adapter 隔离、secret/path/trace 清洗和篡改 fail closed；
- 在线查询不会调用禁止的 O(N) API，且候选、source read、vector overfetch、per-session 和 per-slot 均有硬上限；
- `EXPLAIN QUERY PLAN` 证明结构化过滤使用索引。

完成前运行项目实际配置支持的完整 pytest、相关集成/性能测试、Ruff、MyPy 和 Pyright。工具缺失或外部服务不可用时记录真实命令与真实错误，不得伪造通过，也不得为跑检查随意修改依赖。

详细设计见：

- `docs/architecture/canonical-memory-pipeline.md`
- `docs/architecture/MemoryOS第一主链-记忆形成与更新设计.md`
- `docs/architecture/canonical-memory-change-checklist.md`
- `docs/architecture/unified-context-storage-retrieval.md`
