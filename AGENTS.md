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

## 统一 Context 与 Markdown File Memory 强制约束

以下约束不可协商：

1. 用户长期记忆的当前正文只以受信任 runtime root 下的 live Markdown exact bytes 为事实源。
2. 文档稳定身份来自 front matter 中由系统创建、在固定内部命名空间与 owner 范围内唯一且不可变的 `document_id`；路径、标题、摘要和 tree path 均不参与身份。
3. Catalog、FTS、Vector、Tree、Relation 和 L0/L1 投影只是可删除、可从 Markdown 与 SessionArchive 重建的 serving 层，不得成为第二份当前记忆正文。
4. SessionArchive 是历史对话 evidence 的事实源；普通 Session、Resource 和 Tool Result 继续使用统一 Catalog。
5. 所有系统文档写入都必须比较 exact raw bytes 的 expected digest。live state 是 expected、after、unsafe 之外的第三状态时必须停止，不得覆盖或回滚用户外部编辑。
6. 文件 watcher 事件只是扫描提示，不能直接授权 create、rename、forget 或 delete；只有完整、稳定且通过安全校验的 scan generation 才能确认外部变化。
7. soft forget 与 hard erase 必须区分。两者都要以耐久、幂等、可重放的 barrier/tombstone 清理派生层；hard erase 还必须清除文档子系统中所有可枚举的正文副本。
8. 在线查询禁止递归扫描 memory tree、SessionArchive、全量 Vector URI 或 `list_objects()`；完整扫描只允许用于 startup、repair、rebuild、audit 或受限后台任务。
9. 当前产品只支持本地单用户。HTTP 只能监听回环地址，不实现 Token、多租户或 Capability 授权；用户和 Workspace 由本地进程或插件配置绑定，不能由模型输出或 front matter 覆盖。持久化层使用固定内部命名空间，但不得把它重新暴露成可选择的 tenant。
10. Tool Result 必须先经过 `ContextProjectionSanitizer`，才可进入 Catalog、FTS、Vector、Trace 或 L0/L1。
11. `Memory`、`Behavior`、`ActionPolicy` 语义隔离；Behavior/Action 的 support object 不得伪装成用户 Markdown memory。Coding Agent 默认只走 context reduction，Embodied Agent 动作继续经过 PolicyGate。
    Behavior 支撑锚点归 `behavior/core/support`，其 Context 写入归 `behavior/projection`；用户明确规则归 `policy/action_policy/model`，其写入归 `policy/action_policy/update`。禁止重新建立跨领域的 `memoryos.support` 聚合模型。
12. `infrastructure/store` 只承载文件、SQLite、向量、锁、轨迹等持久化实现及其查询原语；`infrastructure/context` 只承载上下文规划、召回、分层、组装、精确读取和轨迹语义。Context 可以依赖 Store，Store 禁止反向依赖 Context；持久化实现不得放在 Context 目录。
13. Prediction 和动作执行都是 ActionPolicy 的内部阶段，不再作为独立领域存在；请求、候选上下文、排序、门控、决策结果、工具注册和最终执行统一归 `policy/action_policy`。Resource 与 Skill 继续作为统一 Context 数据类型，由事实源和 Catalog 提供，不建立无人装配的根级 `capability` 包。ActionPolicy 必须最终执行风险、确认要求、生命周期、冷却期和支撑证据校验，外部注入候选不得绕过这些规则。
14. 后台任务不建立跨领域的 `memoryos.workers` 聚合包。统一轮询和窄运行时协议归 `runtime/worker`，事务恢复归 `runtime/recovery`，记忆提交、编辑、扫描与投影归 `memory/worker`，Context 派生层维护归 `infrastructure/context/maintenance`，行为冷却执行归 `behavior/execute`；具体 Worker 只能调用其所属领域的公开服务或端口。
15. 公共进程配置只归根级 `config.py`，并且只定义数据根目录、运行模式和日志级别；模型连接配置归 `infrastructure/model/config.py`，HTTP、MCP、Agent Hook 和 Runtime 专属配置归各自模块。禁止恢复 `memoryos/config.py` 或通过兼容模块转发配置。
16. `memoryos` 只作为发行名称、CLI 命令前缀和 `memoryos://` URI scheme 保留，不再对应 Python 聚合包。对外 Python 能力统一从 `openApi` 及其 `sdk`、`http`、`mcp`、`cli` 子模块导入；禁止恢复根级 `memoryos/` 目录或兼容导出包。
17. 禁止自动记忆形成。SessionArchive、LiveSession、Tool Result 和普通 Context 只能作为历史事实源与召回输入，不能自动生成、修改、合并、遗忘或删除用户长期记忆。长期记忆耐久写入只允许由本地用户显式发起的 remember、edit、rename、merge、forget、restore 等命令触发。
18. 显式记忆命令必须经过确定性身份与目标解析、当前 Markdown exact-byte 读取、DocumentEditPlan 编译和 CAS commit；LLM 可以辅助显式命令的内容整理，但不能根据会话自行决定是否写入、目标路径、document ID、Owner/Workspace、删除动作或最终 authority。
19. `EvidenceSlice`、Session evidence window、Formation scheduler、MemoryEditor 自动形成链、自动演化计划/产物和自动提交消费者必须从源码、配置、Schema 与公开契约中完整删除，不保留兼容别名、反序列化分支、禁用状态、空实现或新旧双轨。
20. HOT/WARM/COLD/ARCHIVED 只改变召回优先级，不改变事实是否存在。active 冷层仍保留 FTS；CURRENT 先查 HOT/WARM，仅在已验证结果不足时用剩余预算做一次 COLD/ARCHIVED 词法回退，冷层不得进入 Vector。Consolidation 不得用删除源 Markdown 实现降温。

统一模型是：

```text
Explicit local-user memory command
    -> deterministic identity, target and authority validation
    -> read current Markdown exact bytes
    -> deterministic DocumentEditPlan compilation
    -> Markdown document CAS commit
    -> durable projection job
    -> Unified Context Catalog / Tree / Graph
    -> Unified Retrieval Orchestration
    -> bounded L0 / L1 / L2 Context Selection and Assembly

SessionArchive / LiveSession / Tool Result
    -> sanitized immutable Context projection
    -> bounded retrieval only; never an implicit memory write trigger
```

`SQLiteIndexStore.contexts` 是统一 Serving Catalog；不得新增保存同一 Serving 状态的第二张主表。普通 Context 的事实源仍是 SourceStore 或不可变 SessionArchive。Memory document 的 live Markdown 是其正文事实源，受保护的 revision/after blob 只服务恢复并必须可由 hard erase 枚举删除。控制数据只能保存身份、路径、digest、generation、状态和时间，不得成为语义 current pointer。

## Markdown Memory 不变量

- 每个 Store、Planner、Committer、Scanner、Revision 和 Erase 调用都必须携带固定内部存储命名空间与本地 `owner_user_id`，并与绑定 root 再校验；公开入口不得接收或切换 tenant。
- LLM 只能提出语义候选或辅助非安全查询改写，不能决定文件路径、document ID、Owner/Workspace、SQL、delete/hard erase、projection generation 或最终 authority。
- 文档 kind 由受控相对路径推导；`projects/` 不得成为用户长期记忆的物理目录。项目只作为 entity、topic、episode 或 workspace/applicability 元数据。
- 单文档 create/update/delete/rename 使用 exact raw-state CAS、同目录原子安装和 file/parent fsync。跨文档整理是只向前恢复的 saga，不承诺全局原子性。
- Cooperative lock 只约束合作进程。恢复时 `live == after` 可补齐 event/outbox，`live == before` 可 roll-forward，第三状态或 unsafe 必须进入 conflict/quarantine，绝不恢复 before image。
- 外部新建但缺少系统 ID 的安全文件是 unmanaged，必须显式 adopt；非法 UTF-8、坏 front matter、重复或被修改的 ID 必须 quarantine；symlink、hardlink 和非 regular file 必须 fail closed。
- 删除 publication barrier 位于受保护、耐久的 control store。低 generation 的 queue、scan、vector retry 或 projection 不得复活 soft-forgotten/hard-erased 文档；hard-erased document ID 永不复用。
- Revision restore 是一次新的 CAS 更新并产生更高 revision/generation；不得通过切换数据库 pointer 改变当前正文。

## 写、读和删除主链

Session 写链：

```text
SessionArchive -> ContextProjectionSanitizer -> SessionContextProjector
-> contexts / FTS / context_paths / optional bounded Vector
```

Memory 写链：

```text
Explicit local-user remember / edit / rename / merge / forget / restore
-> deterministic target resolution and current Markdown read
-> deterministic DocumentEditPlan
-> MemoryDocumentCommitter
-> live Markdown -> durable projection job
-> contexts / FTS / context_paths / optional bounded Vector
```

在线读链：

```text
SDK / HTTP / MCP / archive_search convenience entry
-> RetrievalOptions -> QueryPlanner -> local user/workspace binding
-> structured filters -> exact / FTS / bounded vector / relation
-> fusion / rerank -> document/session semantic dedup
-> source digest and projection generation validation
-> L0/L1/L2 selection -> count-bounded assembly -> sanitized Recall Trace
```

`CURRENT` 表示当前 Markdown 或普通 Context 投影；`HISTORY` 服务普通 Session/Event/Resource/Tool Result 与文档历史入口；`OPEN_RECALL` 服务统一开放召回；`EXACT` 服务本地用户的精确读取。Session Timeline 与每条记录的 `event_time` 必须同源，不能用 Archive 写入日期代替发生日期。

删除链：

```text
Document CAS edit / soft forget / hard erase / Source or Session delete
-> durable change event, publication barrier or Tombstone
-> Catalog / FTS / Vector / Path / Link / Projection cleanup
```

Memory document 提交成功定义为 live Markdown 已耐久安装且 projection job 已耐久入队，不要求外部 Vector 同步完成。删除不得同步跨越所有派生后端；Tombstone 先以 SQLite 事务摘除 Catalog/FTS/Path/Link/Projection 并进入 CLEANING，再按内部命名空间限定的 record key、generation 和 digest compare-and-delete 外部 Vector/Relation，最后 APPLIED。外部失败保持 CLEANING 可重放，不能 ACK 或伪装成功。

## 本地数据保护边界

- 本地用户和插件 Workspace 优先于普通请求字段；用户 Filter 优先于确定性日期/路径解析；Schema 默认值优先于 LLM 辅助。
- Scope 只用于适用范围和检索过滤，不承担多用户授权；公开入口不得接受 tenant、授权 grant 或调用者 Capability。
- Owner、Workspace 和 Adapter Scope 必须在候选生成前转成过滤条件，并在 source hydration 前再次验证；不得恢复 Visibility/Authority 多用户授权模型。
- Session 与 Memory Document/Block 的 Vector metadata 必须使用统一数据保护契约；native Vector 命中后仍由 SQL Catalog 以固定命名空间和 `record_key` 复核 path、generation、digest 和 source ownership。
- 凭证、Authorization、Cookie、密码、私钥、环境变量、数据库/SSH 凭证、敏感绝对路径、Binary 和超长日志不得通过 metadata JSON 绕过清洗。
- 完整路径仅留在受保护 Source Evidence；Serving 层默认只保留 basename 与受控 location。
- Sanitization 失败必须 fail closed；Recall Trace 使用相同安全投影。
- Memory document 精确回源后的公开 title、text 和 metadata 也必须再次经过同一安全投影，并在清洗后复核内部命名空间、owner、document ID、revision、generation 和 digest。

## 验证门槛

涉及本主链的变更必须同时覆盖真实存储和公开入口，而不是只写 Mock。至少验证：

- FileSystemSourceStore、SQLiteIndexStore、RelationStore、QueueStore、SessionArchiveStore；
- MemoryDocumentStore、CAS commit、revision restore、scanner、projection、soft forget/hard erase 与 conflict recovery；
- Session Projection、SessionProjectionJournal、Tombstone、Retention、commit-group recovery；
- SDK、HTTP、MCP、CLI、`archive_search()` 和按条目数量限制的 Context Assembly；
- 本地 Owner/Workspace/Adapter 一致性、HTTP 回环限制、secret/path/trace 清洗和篡改 fail closed；
- 在线查询不会调用禁止的 O(N) API，且候选、source read、vector overfetch、per-session 和 per-document 均有硬上限；
- `EXPLAIN QUERY PLAN` 证明结构化过滤使用索引。

旧测试资产已经退役，等待按新架构重新设计。在新测试体系建立前，完成前必须运行生产源码 Compile、Ruff、MyPy、Pyright，以及现存集成包的构建检查；恢复测试体系后，必须重新启用相关测试和完整测试门禁。工具缺失或外部服务不可用时记录真实命令与真实错误，不得伪造通过，也不得为跑检查随意修改依赖。
