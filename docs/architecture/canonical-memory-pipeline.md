# Canonical Memory Pipeline

## 定位

Canonical Memory 是状态型长期记忆的权威覆盖层，不是通用上下文容器。MemoryOS 先把所有可检索上下文投影到统一 Context Catalog；只有需要跨会话维护“当前值”的对象，才额外进入 Slot/Claim/Revision。

```text
Context First
    ordinary context -> Unified Context Catalog
    stateful long-term context -> Catalog + Canonical State Overlay
```

## 九条主链约束

1. Slot/Claim 只服务状态型长期记忆。
2. 普通 Session、Resource、Tool Result 使用统一 Catalog，不创建 Slot/Claim。
3. Claim Revision Projection 与 Slot Current Projection 并存。
4. 所有在线检索经过统一编排。
5. Tree Path 不参与 Canonical Identity。
6. 在线查询禁止全库扫描。
7. Tool Result 必须 Sanitization。
8. 删除必须通过 Tombstone 传播。
9. 迁移必须支持 Cutover 和 Rollback。

## 事实源与派生层

| 数据 | 权威事实源 | Serving 派生 |
| --- | --- | --- |
| 普通 Context | SourceStore | `SQLiteIndexStore.contexts`、FTS、Path、Vector、Relation |
| Session 活动 | 不可变 SessionArchive | Session Root/L0/L1/Segment/Message/Tool Result/Resource 等 Catalog Record |
| 状态型长期记忆 | Canonical Slot/Claim/Revision Source、Receipt、Current Head | Claim Revision Projection、Current Slot Projection、FTS、Path、Vector |
| Tree Overview/Compaction | 上述 Source 的确定性摘要 | Catalog 内可删除重建的 overview/compaction record |

Catalog 不是第二事实源。删除 Catalog、FTS、Vector、Tree、Projection 后，可以从 SourceStore、Canonical Source 和 SessionArchive 重建。Receipt、Current Head、Evidence 与不可变 Archive 不得由回填伪造。

## 晋升边界

`CanonicalPromotionPolicy` 使用结构化事实和既有 `TransitionProfile`：

- `AUTHORITATIVE_STATE`：Profile、Preference、Project Rule、Project Decision、明确维护当前值的实体属性，满足完整门禁后允许晋升。
- `EXPERIENCE`：只有已提炼、可跨会话复用、Evidence 完整、Identity 稳定、达到 Admission 阈值，且不是原始日志、一次性失败或临时状态时允许晋升。
- `OBSERVATIONAL`：Session Message、Tool Result、文件读取、Used Context/Skill、Observation、Action Result、浏览/命令/Event/原始 Agent Log 默认 `CATALOG_ONLY`。

OBSERVATIONAL 只有在认证的显式 `remember()`、Schema 明确声明 stateful canonical type、确定性 Policy 批准且 Evidence/Scope/Authority/Identity 全部成立时，才进入 Canonical pipeline。LLM 输出不是晋升授权。

`MemoryType.EVENT` 继续存在：普通 Session Activity 是 Catalog Event；显式 `remember(memory_type="event")` 或确定规则确认长期价值的事件可以形成 Canonical Event。

Policy 的 `PROMOTE` 只允许进入既有形成链，不代表存储成功：

```text
Evidence -> Schema -> Identity -> Scope -> Authority
-> Admission -> Reconcile -> Transition -> Canonical Transaction
```

## Identity 和单值状态

Identity V2、Slot ID 和 Claim ID 的确定性算法保持不变。以下信息绝不进入 Hash：Tree/日期/文件路径、文件名、展示标题、L0/L1、Vector/Projection ID、Session ID、Tool Result ID。

一个 Slot 最多一个 ACTIVE Claim。多值状态使用：

- 同一个 Claim 的结构化集合；
- 更细粒度的 Slot Identity；
- 多个独立单值 Slot。

例如 `ice_cream_flavors` 的 `canonical_value` 可以是 `['vanilla', 'chocolate', 'strawberry']`；不能创建三个同时 ACTIVE 的口味 Claim。

## Canonical 写链

```text
Evidence Episode
-> CanonicalPromotionPolicy
-> proposal/admission/reconcile/transition
-> Canonical Transaction
-> Source + Relation + Receipt + Current Head + Redo
-> durable canonical Outbox
-> MemoryProjectionWorker
   -> Claim Revision Projection
   -> CurrentSlotProjection
   -> stale-current Tombstone when active head changes
```

Canonical Transaction 保持 Source 原子边界；Projection 是最终一致的派生发布。Worker 只有在发布或耐久失败状态记录完成后才处理队列结果，不能把投影失败伪装成成功。

## 双投影

### Claim Revision Projection

Claim Projection 保留每个 Claim revision，record key 为 `claim:{claim_id}:revision:{source_revision}`。它绑定 Claim/Slot URI、ID、source revision、transaction、receipt/head 和 projection effect proof，服务：

- Claim 精确读取；
- HISTORY；
- AS_OF；
- AUDIT；
- CONFLICTS；
- Evidence 和 Revision 检查。

### Current Slot Projection

`CurrentSlotProjection` 为每个有 ACTIVE Claim 的 Slot 生成一条 Serving Record：

```text
record_key = slot:{slot_id}:current
record_kind = current_slot
unique = tenant_id + canonical_slot_id where serving-current
```

它包含 identity fields、active Claim ID/URI/revision、canonical state/value、valid interval、L0/L1/L2 URI、tree paths、transaction/receipt/head/effect proof。Active Claim 从 B 切到 C 时同一 record key 原位更新；B 的 Claim Revision Projection 保留，但旧 Current 不再参与 CURRENT。

Canonical 分支中，CURRENT 默认只搜索 `current_slot`；AS_OF、AUDIT、CONFLICTS/OPTIONS 搜索 Claim revision。统一 HISTORY 还同时包含普通 Session/Event/Resource/Tool Result 历史，但排除可变 `current_slot`。普通 Context 的 `canonical_*` 为空。

## 有界 Canonical Resolution

候选生成、融合和去重先完成；`BoundedCanonicalResolver` 只检查不超过 `candidate_limit` 的最终候选。校验包括 Tenant、Owner、Scope、Visibility、Authority、Slot/Claim 关系、`active_claim_id`、ACTIVE 状态、revision、Current Head、Receipt Digest 和 Projection Effect Hash。

Canonical Current 错误时 fail closed；过期 Current 不可返回。允许有界精确回源或明确 unavailable/degraded，禁止生成候选前扫描所有 Slot/Claim。启动、Repair 和 Audit 仍保留全局 O(N) 完整性验证。

## Tree 与 Canonical Identity

Canonical Projection 可属于 `memories/preferences/...`、`memories/profiles/...`、`memories/rules/...` 或 `memories/decisions/...` 等受控路径，但 Tree 只是 Serving 分类。重分类只更新 `context_paths` 和派生 Overview，不改变 Slot ID、Claim ID 或 URI。

## 删除、Supersede 与 Tombstone

- Supersede：旧 Claim revision 保留为历史；Current Slot 原位指向新 ACTIVE Claim。
- Retract/Forget：权威 Source 先形成合法状态变化，再写耐久投影事件/Tombstone。
- Source/Resource/Session 删除、Visibility/Authority/Owner 修复、Tree 重分类或 Serving tier 变化同样通过可重放事件更新派生层。
- Tombstone 负责 Catalog、FTS、Vector、Path、Link、Projection 和 Relation Serving cleanup；失败记录 retry/error，不能伪装成功。

Projection cleanup 不物理删除不可变 Evidence Source，除非已有明确用户/retention policy 授权。

## 公开接口兼容

`remember()`、`forget()`、Pending Proposal/Review、SDK/HTTP/MCP 的既有语义继续存在。`search_context()`、`assemble_context()` 与兼容 `archive_search()` 统一转换为 `RetrievalOptions`，不再各自拥有独立 canonical/session 文件扫描检索器。

## 与其他语义平面的边界

Memory、Behavior 和 ActionPolicy 继续隔离。Coding Agent 的召回只做 context reduction；动作能力不从 Memory 直接产生。Embodied Agent 的动作仍由现有 PolicyGate、Authority、Visibility 和风险控制链决定。
