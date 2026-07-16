# MemoryOS 第一主链：记忆形成与更新设计

## 1. 目标与边界

第一主链负责把 Evidence 转成可审计、可恢复的状态型长期记忆，并把普通上下文保留在统一检索体系中。它不把“可搜索”错误等同于“必须成为 Canonical Memory”。

```text
所有 Context -> Unified Context Catalog
状态型长期记忆 -> Canonical Slot/Claim/Revision Overlay
```

必须同时遵守：Slot/Claim 只服务状态型长期记忆；普通 Session/Resource/Tool Result 使用统一 Catalog；Claim Projection 与 Slot Current Projection 并存；所有在线检索经过统一编排；Tree Path 不参与 Canonical Identity；在线查询禁止全库扫描；Tool Result 必须 Sanitization；删除必须通过 Tombstone 传播；迁移必须支持 Cutover 和 Rollback。

## 2. Evidence Plane

Evidence Plane 包含：

- 用户显式 `remember()`/`forget()` 命令及认证调用上下文；
- SessionArchive 的 message、tool result、observation、action result、used context/skill；
- 字段级 EvidenceRef、source URI、digest、发生时间和摄取时间；
- Canonical Transaction 的 Receipt、Current Head、Redo、Recovery 和 Pending Review 证据。

普通 SessionArchive 保持不可变证据源。SourceStore 是普通 Context 的事实源；Canonical Source 是 Slot/Claim/Revision 的权威状态源。任何 Catalog、FTS、Vector 或 Tree 行都不能反向成为业务真相。

## 3. Context First 路由

每个输入先接受安全投影并可进入 Catalog，然后才由 `CanonicalPromotionPolicy` 判断是否允许进入状态层：

```text
input/evidence
-> ContextProjectionSanitizer
-> Context Catalog projection
-> CanonicalPromotionPolicy
   -> CATALOG_ONLY
   -> PROMOTE -> canonical gates
   -> REJECT
```

`AUTHORITATIVE_STATE` 默认覆盖 Profile、Preference、Project Rule、Project Decision 和明确的单值实体属性。`EXPERIENCE` 需要提炼、跨会话价值、稳定 Identity、完整 Evidence 与 Admission 阈值。`OBSERVATIONAL` 默认只进 Catalog；认证的显式 remember、stateful Schema 和全部安全/身份门禁同时成立时才可晋升。

LLM 不决定晋升，也不生成最终 storage operation、URI、Slot ID 或 Claim ID。

## 4. Canonical 形成链

PROMOTE 后必须完整通过：

```text
Evidence
-> Schema validation
-> Identity V2
-> Scope / Visibility / Authority
-> Admission
-> Reconcile
-> TransitionProfile
-> Canonical Transaction
-> Receipt / Current Head / Redo
```

Identity Hash 只使用稳定业务身份字段。Tree Path、日期、文件路径/名、标题、L0/L1、Vector/Projection/Session/Tool Result ID 禁止参与。

一个 Slot 最多一个 ACTIVE Claim。重复偏好应复用 Slot/Claim 或增加 Evidence；状态改变应保持 Slot ID，旧 Claim 进入 SUPERSEDED，新 Claim 成为 ACTIVE；多值偏好存为结构化集合或拆分更细 Slot。

## 5. 事务后投影

Canonical commit 成功后写入耐久 Outbox，再由 `MemoryProjectionWorker` 发布两个模型：

- Claim Revision Projection：历史、AS_OF、AUDIT、CONFLICTS、Evidence 与 revision 精确读取；
- Current Slot Projection：每个 ACTIVE Slot 唯一一条 `slot:{slot_id}:current`，服务 CURRENT。

Current Slot 绑定 Slot 与 active Claim 的 Current Head、Receipt 和组合 Projection Effect Hash。Active Claim 切换时原位更新；旧 Claim history 保留。投影失败不回滚 Canonical 事务，也不 ACK 成功，而是持久记录并重试。

## 6. Session 与普通 Context 形成链

```text
SessionArchive
-> SessionContextProjector
-> sanitized Catalog records
   root / L0 / L1 / semantic segment
   message / tool result / resource reference
   used context / used skill / observation / action result / event
```

Session Root 与 semantic segment 可进入 FTS、Vector 和 structured filter；重要 event/resource 按配置向量化；原子 message/tool result 默认仅 structured + FTS，L2 留在 Archive。

文件读取时，即使用户消息没有文件名，只要 Tool Result 有文件 URI，Projector 也会提取 basename、受控 location、Archive source URI/digest，并挂到 timeline、session 和 `resources/{location}` 路径。它不会创建 Slot/Claim。

## 7. 更新与状态转换

状态更新流程以 Slot 为稳定身份：

```text
old ACTIVE claim B
-> reconcile new evidence/value
-> transition B to SUPERSEDED
-> create/activate claim C
-> update slot.active_claim_id
-> atomic canonical transaction
-> outbox
-> preserve B revision projection
-> overwrite one current-slot serving row with C
```

CURRENT 去重键是 `slot_id`；HISTORY 去重键是 `claim_id + revision`。Tree 重分类只更新 `context_paths`，不触发 Identity 变化。

## 8. 删除与撤销

Forget、Retract、Supersede、Source/Resource/Session Delete 等先形成权威变化，再写耐久 Tombstone 或投影事件。Projection Tombstone 可重放并清理 Catalog、FTS、Vector、Path、Link、Current/Claim Projection 和 Relation Serving 数据；失败保留 error/retry 状态。

Canonical Transaction 不同步跨所有派生后端删除，避免扩大事务边界。不可变 Evidence Source 无明确策略时不删除。

## 9. 召回闭环

写入后的所有在线召回统一进入：

```text
RetrievalOptions
-> QueryPlanner / Trusted Scope
-> SQL structured filters
-> exact / FTS / bounded vector / relation
-> RRF fusion / optional rerank / dedup
-> bounded Canonical Resolver
-> L0/L1/L2 selector / ContextPacker
-> sanitized Recall Trace
```

CURRENT 只把通过强校验的 Current Slot 作为 canonical current；统一 HISTORY 同时包含普通 Session/Event/Resource/Tool Result 和 Claim Revision，并排除 Current Slot；AS_OF/CONFLICTS 使用 Claim Revision。Canonical Source 回源后的 title/text/metadata 必须再次安全清洗；AS_OF 使用不改写 Source 的派生半开有效区间且只接受时点 ACTIVE Revision。Vector 或 reranker 失败有明确 degraded/fallback；Catalog backfill 未完成由 migration feature gate 选择 legacy/shadow/unified route，不能静默漏数据。

## 10. 生命周期与迁移

ServingTier 与 ClaimState 分离：HOT/WARM/COLD/ARCHIVED 只控制检索派生层。Current Slot 保持 HOT；普通记录按 event/created/transaction time 分层。Compaction 和 GC 不改变事实源，冷数据可恢复到 WARM。

统一 Catalog 迁移依次经过 NOT_STARTED、SCHEMA_READY、BACKFILLING、DUAL_WRITE、SHADOW_VALIDATING、READY_TO_CUTOVER、CUTOVER、COMPLETED；任何允许阶段可进入 FAILED 或 ROLLBACK，并从 checkpoint 恢复。pre-v10 原地升级必须先耐久记录 provenance；既有 Archive 但未回填时不得把缺失迁移行解释为 greenfield。回填只重建 Serving，不伪造 Receipt、Current Head 或 Evidence。
