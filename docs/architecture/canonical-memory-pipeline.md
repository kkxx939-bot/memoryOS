# Markdown Memory Document Pipeline

## 1. 定位

用户长期记忆是一组可直接查看、编辑、创建、移动和删除的 Markdown 文件。受信任 runtime root 下的 live file exact bytes 是当前记忆正文的唯一业务事实源；SQLite Catalog、FTS、Vector、Tree、Relation、L0/L1 和 block record 都只是可删除、可重建的 serving projection。

```text
Evidence
-> sealed semantic proposal
-> deterministic routing and read-before-write planning
-> exact-byte document CAS
-> durable change event and projection job
-> unified Catalog serving
-> bounded retrieval and source validation
```

SessionArchive 继续作为历史对话 evidence 的事实源。普通 Context 继续由 SourceStore 或 SessionArchive 持有，不能借记忆重构改成文件记忆。Memory、Behavior 与 ActionPolicy 仍是三个隔离的语义平面。

## 2. 强制不变量

1. live Markdown exact bytes 决定文档当前正文。
2. 系统生成且不可变的 `document_id` 决定稳定身份，路径、标题和摘要均不参与身份。
3. tenant、owner、workspace、ACL、authority 和 visibility 只来自 trusted caller、受信任根与服务端策略。
4. 所有系统写入都比较 exact raw state；第三状态、unsafe state 或身份不一致时停止写入。
5. watcher 事件只触发 scan，不能直接授权 create、rename、forget 或 delete。
6. 在线 Search/Assemble 不递归扫描 memory tree、SessionArchive、全量 Vector URI 或 SourceStore。
7. soft forget 可恢复；hard erase 必须枚举并清除文档子系统内所有正文副本。
8. Tool Result 必须先经过 `ContextProjectionSanitizer`，才可进入任何 serving 或 trace。
9. 单文档提交是原子边界；跨文档整理是可恢复、只向前推进的 saga。

## 3. 目录与身份

物理根由 runtime 推导，不信任文件内容自报：

```text
<root>/tenants/<tenant_id>/users/<owner_user_id>/memory/
├── MEMORY.md
├── profile.md
├── preferences.md
├── knowledge/
│   ├── MEMORY.md
│   ├── entities/
│   ├── topics/
│   ├── episodes/
│   └── open-loops.md
└── experiences/
```

项目只作为 entity、topic、episode 或 workspace/applicability metadata，不建立用户记忆的 `projects/` 物理目录。文档 kind 由受控相对路径推导。

每个 managed document 的最小 front matter 为：

```yaml
---
memoryos_schema: 1
document_id: memdoc_01J...
---
```

稳定 URI 为：

```text
memoryos://user/<owner_user_id>/memory/documents/<document_id>
memoryos://user/<owner_user_id>/memory/documents/<document_id>/blocks/<block_id>
```

rename 不改变 `document_id` 或 document URI。`block_id` 仅在指定 source digest 和 projection generation 下标识 serving block，不是正文身份。

## 4. Raw state 与 registration state

Raw path state 只回答路径的物理状态：

- `ABSENT`；
- `PRESENT(relative_path, raw_sha256, size)`；
- `UNSAFE(relative_path, reason)`。

Registration state 单独回答文件是否可由系统解释：

- `MANAGED`：安全 front matter、唯一且合法的 `document_id`；
- `UNMANAGED`：路径安全，但尚未显式 adopt；
- `QUARANTINED`：非法 UTF-8、坏 front matter、重复或被修改的身份。

空文件在 raw 层是 PRESENT，在 registration 层是 UNMANAGED。Symlink、hardlink、目录替代文件、越界路径或权限异常是 UNSAFE。UNMANAGED、QUARANTINED 和 UNSAFE 都不能解释成删除。

## 5. 形成与规划

Session 形成链从已校验的 immutable archive 提取语义候选；显式 `remember()` 保存真实命令 evidence。模型只可输出 title、subject、body、entity/topic hints、occurred time、relations、evidence references 与 confidence，不能输出路径、文档身份、tenant、owner、workspace、ACL、SQL、删除动作或 projection generation。

`MemoryDocumentRouter` 确定性路由：

| 候选 | 目标 |
| --- | --- |
| profile fact | `profile.md` |
| preference | `preferences.md` |
| entity note | `knowledge/entities/<safe-slug>.md` |
| topic note | `knowledge/topics/<safe-slug>.md` |
| episode | `knowledge/episodes/<date>-<safe-slug>.md` |
| open loop | `knowledge/open-loops.md` |
| experience | `experiences/<date>-<safe-slug>.md` |

路由后必须从 Catalog 找到有界相关文档，再读取 live raw state，完成去重、补充、修正或合并，最后生成绑定 `(tenant_id, owner_user_id, document_id)` 的 `DocumentEditPlan`。CAS 冲突后的重规划只使用已 sealed proposal 和最新 live bytes，不再次调用模型。

## 6. 单文档提交与恢复

`MemoryDocumentCommitter` 独立于 ordinary operation committer。Create/Update 的共同顺序是：

```text
validate trusted scope and expected raw vector
-> fsync body-bearing revision/after blob
-> write content-free PREPARED intent
-> acquire cooperative document identity lock
-> re-read every affected path
-> atomic install and parent fsync
-> append content-free DocumentChangeEvent
-> enqueue durable projection job
-> mark COMPLETED
```

不同 edit kind 的 effect vector：

| Edit | Before | After |
| --- | --- | --- |
| CREATE | target ABSENT | target PRESENT(new digest) |
| UPDATE | target PRESENT(old digest) | target PRESENT(new digest) |
| DELETE | target PRESENT(old digest) | target ABSENT |
| RENAME | old PRESENT(d), new ABSENT | old ABSENT, new PRESENT(d) |

Path lock 只协调合作进程，不能阻止外部编辑器。恢复时只允许：

- live 等于 after：补齐 event、queue 与完成标记；
- live 等于 before：roll forward 安装 after；
- live 为第三状态或 UNSAFE：进入 conflict/quarantine，不写、不删、不恢复 before image。

提交成功的业务边界是 live Markdown 已耐久安装且 projection job 已耐久入队，不要求 Vector 同步完成。

## 7. 外部编辑闭环

`MemoryDocumentScanner` 在启动、watcher hint、watch overflow 与 repair 时执行受限 full scan。只有 scan 完整、root identity 稳定、无遍历错误、同一身份不在多个路径、并跨过稳定观察窗口，才能确认外部 create/update/rename/delete。

以下情况暂停整批删除：root 不可访问、scan 不完整、unsafe/unmanaged/quarantined 路径造成歧义、root identity 改变或 mass-delete threshold 命中。常见编辑器的 temp-write + rename 必须在稳定 scan 后被解释为一个文档更新，而不是 forget。

显式 adopt 在 raw digest CAS 下加入最小 front matter；scanner 不静默改写用户新文件。

## 8. Revision、review 与跨文档整理

Revision blob 位于受保护控制区，只用于 history/restore。Restore 读取指定旧 blob，但执行一次新的 CAS update，产生更高 logical revision 与 projection generation。控制记录只保存身份、路径、digest、generation、状态与时间，不能改变 live 文件正文。

Review record 可以封存有界 diff 和 after blob；接受时仍须重新校验 tenant、owner、document identity 和 expected raw digest。拒绝后不得再应用；为审计保留的 body-bearing artifact 必须可枚举，hard erase 必须立即清除与目标或来源文档相关的副本。

跨文档 merge/consolidation 先提交并投影目标，再逐个 soft-forget 来源；中途失败允许暂时重复，不允许丢内容或用多个文件替换冒充全局事务。

## 9. Serving 投影与统一召回

`MemoryDocumentProjectionWorker` 读取 projection job 与 live source，发布 document record、bounded block records、FTS、paths、relations 和可选 vector。每条 projection 必须携带 trusted tenant/owner、`document_id`、source digest、logical revision 与 projection generation。低 generation job、旧 scan 或旧 retry 不得覆盖较新状态，也不得复活已删除文档。

在线召回统一经过：

```text
RetrievalOptions
-> QueryPlanner and trusted scope binding
-> structured / exact / FTS / bounded vector / relation candidates
-> RRF fusion / optional rerank / document-session dedup
-> bounded hydration of live Markdown or SessionArchive
-> digest, generation and ownership validation
-> L0/L1/L2 selection and token packing
-> sanitized Recall Trace
```

Catalog 文本可帮助召回，但 hydrate memory result 时必须精确读取 live document，复核 tenant、owner、document ID、digest 与 generation，并再次安全清洗。发现 source drift 时丢弃 stale candidate、标记 degraded mode，并安排受限 rescan；不能返回 Catalog 中的旧正文冒充当前记忆。

## 10. Forget 与 erase

Soft forget 是一次 document CAS edit/delete，清除 live 内容和 serving projection，但保留可恢复 revision。它必须明确返回“可恢复”。

Hard erase 只接受整文档 URI，并使用耐久 publication barrier 防止旧 queue、scan、vector retry 或 rebuild 复活该 document ID。它枚举清理：

- live file；
- revision 与 prepared after blobs；
- review artifacts、redo/cache/trace 中的正文副本；
- Catalog、FTS、Vector、Relation、Path、Link 与 projection state。

外部 backend 未确认时状态保持 `ERASE_PENDING` 并可重放；只有全部配置后端确认才完成。允许保留不含正文的最小 digest audit。底层介质不承诺安全擦除。

独立 SessionArchive 默认不随文档删除；若 archive 仍保存原始对话，结果必须列出 `independent_evidence_retained`。删除 Session 需要另一条精确授权链。

## 11. Runtime 与公开接口

新 runtime 使用 `markdown_memory_v1` layout marker。空 root 可初始化；发现不受支持的旧 layout 或旧数据库时 fail closed，并要求显式 reset，不自动删除或转换。首次初始化某 owner 时创建五个最小 managed template，之后用户删除的文件不会在每次启动中被静默重建。

Runtime 在 READY 前依次恢复 document intents、Session commit groups、稳定扫描、文档投影、erase tombstones 和 serving verification。公开 SDK、HTTP、MCP 共享 document-native contract：`remember`、`edit_memory_document`、`forget`、`list_memory_history`、`restore_memory_revision` 与 review；未 READY 时 mutation fail closed。
