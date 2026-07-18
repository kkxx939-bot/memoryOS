# MemoryOS 第一主链：Markdown 记忆形成与更新

## 1. 目标与边界

第一主链把不可变 SessionArchive 或受信任显式命令转换为用户可编辑的 Markdown 记忆，并保证写入、恢复、投影、检索和删除闭环。它不把“可搜索”等同于“应写入长期记忆”；普通 Session、Resource、Tool Result 继续进入统一 Catalog。

```text
Session evidence or trusted command
-> semantic extraction and salience gate
-> sealed MemoryEditProposal
-> deterministic document router
-> read-before-write DocumentEditPlan
-> exact-byte CAS commit
-> durable projection job
-> unified retrieval
```

live Markdown exact bytes 是文档当前正文的唯一业务事实源。Catalog、FTS、Vector、Tree、Relation、L0/L1 及 block records 都是 serving 派生。

## 2. Evidence Plane

Evidence Plane 包含：

- immutable SessionArchive、manifest digest、event IDs 和发生时间；
- 受信任用户的显式 `remember`/edit/forget 命令引用；
- field-level evidence references、actor binding 和 durability/atomicity 约束；
- document change event 中不含正文的 lineage digest 与逻辑引用。

SessionArchive 的原始消息和 tool result 不复制进 control/event/queue payload。`MemoryEgressPolicy` 控制进入模型的内容，`ContextProjectionSanitizer` 控制进入 serving 与 trace 的内容。

## 3. Session 同步提交

```text
commit_entry.commit_session
-> SessionCommitService.sync_archive
-> immutable SessionArchive + manifest
-> idempotent SessionContextProjector
-> SessionProjectionJournal
```

Archive 写入前失败不创建异步任务。Archive 已耐久落盘后，同步投影可幂等重试。`SessionProjectionJournal` 保存 archive/manifest identity 到 Catalog projection 的耐久进度，解决 archive 已写但 projection 尚未完成的崩溃窗口。

`SessionContextProjector` 生成 root、L0/L1、semantic segment、message、tool result、resource reference、event、observation 和 action result 等 ordinary Catalog records。Timeline path 与 `event_time` 来自同一发生时间；不用 archive write time 伪造事件日期。

## 4. Session 异步提交

```text
inline async_commit
-> CommitGroupStore(memory, behavior, action_policy, context)
-> independent consumer leases
-> document changes and ordinary operation effects
-> async outputs
```

保留 inline-or-queued 产品语义：

1. inline 成功时直接返回。
2. Archive 已落盘但 inline 失败时，以相同 task/commit-group identity 耐久入队。
3. `SessionCommitWorker` 按同一 identity lease/retry/ack，每个 consumer 独立幂等恢复。

Memory consumer 的完成定义是所有计划的 live Markdown change 已耐久提交，且对应 projection jobs 已耐久入队。Behavior、ActionPolicy 和 ordinary Context 继续由 operation pipeline 处理，不得写入 document URI 或伪装成文件记忆。

## 5. 语义提取与 sealing

提取器对 archive/manifest digest、event ordering、actor、evidence span 和 egress policy 先做确定性校验，再输出 `MemoryEditProposal`：

```text
candidate_kind
title / subject / body
entity_hints / topic_hints
occurred_at / temporal_status
relation_hints
evidence_refs / field_evidence_refs
confidence
```

模型不能决定路径、文档身份、tenant/owner/workspace/ACL、最终 authority、SQL、删除或 generation。Proposal set 封存后不可替换；CAS 冲突重计划必须使用同一 sealed bytes，不二次调用模型。

Salience gate 排除原始 tool log、一次性失败、短期执行状态和无 evidence 结论。用户显式命令可绕过自动 salience 阈值，但仍不能绕过 trusted scope、文档 CAS 与安全边界。

## 6. 确定性路由与 read-before-write

`MemoryDocumentRouter` 仅按 candidate kind、safe slug 和受控日期选择目标。项目名是 entity/topic/workspace applicability，不产生项目物理目录。

Planner 在生成 `DocumentEditPlan` 前必须：

1. 以 trusted tenant/owner 查询 Catalog 的有界相关文档；
2. 按 document URI 或 controlled relative path 精确读 live raw bytes；
3. 校验 front matter 身份、path-derived kind 和 raw SHA-256；
4. 做确定性 dedup/append/correction/merge；
5. 限制 patch 范围、文件大小、front matter 与单次 edit 数量；
6. 绑定 exact expected raw effect vector 与 idempotency key。

Planner 不依赖 serving 文本作为最新正文。Catalog 候选只帮助定位，最终 plan 以 live read 为准。

## 7. 文档 CAS 提交

`MemoryDocumentCommitter` 的提交边界为单文档。Plan、intent、revision blob、event、queue payload 与 idempotency identity 全部绑定 `(tenant_id, owner_user_id, document_id)`。

```text
validate exact before vector
-> fsync body-bearing blob when applicable
-> persist content-free PREPARED intent
-> lock document identity
-> re-read all affected paths
-> atomic create/replace/unlink/rename + parent fsync
-> append content-free DocumentChangeEvent
-> enqueue projection job
-> COMPLETED
```

DELETE 的 after 是 ABSENT，不伪造空 after blob。RENAME 同时校验 old/new 两条路径，且目标必须 ABSENT。不合作外部编辑可能穿过 cooperative lock，因此恢复只根据 before/after/第三状态决策，不回写 before image。

## 8. Revision 与 review

每次变更产生 monotonic logical revision 和 projection generation。Revision blob 是受保护的恢复资料，不是当前正文 pointer。`restore_memory_revision()` 把旧 blob 作为新 after bytes，仍须通过当前 digest CAS，并产生新 revision。

审核流封存 plan identity、expected effects、bounded diff 和 body-bearing blob。Approve 时执行同一 committer；Reject 只改审核状态。任何 hard erase 都要删除该文档的 review blobs/records。

## 9. Scanner 与外部编辑

Watcher 只调用 `notify()`。Scanner 使用完整、有上限的 scan generation 对比 `(document_id, path, raw digest)`，并经稳定窗口确认外部 create/update/rename/delete。

以下条件 fail closed：

- root identity 变化、不可访问或遍历不完整；
- symlink、hardlink、non-regular file 或权限错误；
- unmanaged/quarantined 文件使删除判定产生歧义；
- duplicate document identity；
- mass-delete threshold 命中。

外部确认的变更也产生 revision、change event 和 projection job，不直接依赖 mtime 作为版本事实。

## 10. Projection 与读链

```text
MemoryDocumentProjectionWorker
-> document and bounded block Catalog records
-> FTS / paths / relations / optional vector
-> atomic document projection state
```

Projection row 必须带 source digest、logical revision 和 generation。更低 generation 不可覆盖更新投影。Soft-forgotten 或 hard-erased identity 的 publication barrier 会拒绝迟到 job。

在线读链使用 SQL structured filter、exact、FTS、bounded vector 和 relation 生成候选，融合后只 hydrate 有界前缀。Memory document hydrate 必须回读 live bytes 并重新验证 tenant、owner、document identity、digest 与 generation。不一致时丢弃 stale candidate 并触发 rescan，不从 Catalog 返回旧正文。

## 11. Forget、erase 与恢复保障

Soft forget 是可恢复的 CAS edit/delete，并以耐久删除状态清理 serving。Hard erase 仅允许整文档 target，先建立不可逆 publication barrier，再以幂等 backend 清理 live file、revision/after/review blobs、Catalog、FTS、Vector、Relation、Path、Link 和其他可枚举正文副本。

外部 backend 失败时保持 `ERASE_PENDING`，旧任务仍被 barrier 拦截。全部 backend 确认后才能返回 completed。独立 SessionArchive 不默认删除；如果其仍包含原始 evidence，返回精确的 `independent_evidence_retained` references。

## 12. Startup 恢复顺序

Runtime 在 READY 前执行：

```text
validate markdown_memory_v1 layout
-> recover expired queue leases and ordinary operations
-> recover incomplete document intents
-> resume Session commit groups
-> complete bounded stable scans
-> rebuild/replay document projections
-> replay deletion tombstones
-> verify live source against Catalog generations
-> READY
```

任一恢复或验证失败都保持 NOT_READY，公开 mutation fail closed。新 layout 是绿地格式；发现不受支持的历史 layout 或数据库时要求显式 reset，不自动删除、转换或建立并行读写路线。

## 13. 公开合同

SDK、HTTP 与 MCP 共用 `memoryos/api/memory_contract.py` 的 document-native schema：

- `remember(content, occurred_at, target_hint, expected_document_digest)`；
- `edit_memory_document(document_uri, edit, expected_digest)`；
- `forget(document_uri, section_anchor, mode, expected_digest)`；
- `list_memory_history(document_uri)`；
- `restore_memory_revision(document_uri, revision, expected_digest)`；
- review list/approve/reject。

结果字段只暴露 document URI、digest、revision/generation、change event、projection job 与 erase status，不暴露受保护 control path 或未清洗 evidence payload。
