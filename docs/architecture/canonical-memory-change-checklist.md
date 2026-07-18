# Markdown Memory Change Checklist

本清单是源码、真实存储与恢复门禁，不是“存在类名就完成”的功能清单。任一拒绝条件成立时，不得将 runtime 标记为 READY。

## A. 基线与范围

| 检查项 | 必须证明 | 拒绝条件 |
| --- | --- | --- |
| 源码基线 | 分支、HEAD、工作区、未跟踪文件 | 使用旧快照或覆盖用户改动 |
| 约束 | 已读 `AGENTS.md` 和四份架构文档 | 修改前未核对不变量 |
| 调用者 | SDK、HTTP、MCP、workers、hooks、startup/repair 全量搜索 | 只改一个入口 |
| 范围 | Memory、Behavior、ActionPolicy 保持隔离 | 借本变更扩展 prediction/action 语义 |
| 历史 layout | 非新 layout 明确 fail closed 并要求 reset | 自动删数据、转换或建立并行路线 |

## B. 事实源与身份

- [ ] live Markdown exact bytes 是用户记忆当前正文的唯一业务事实源。
- [ ] SourceStore 只保持 ordinary Context 事实；SessionArchive 是历史对话 evidence 事实源。
- [ ] Catalog、FTS、Vector、Tree、Relation、L0/L1 和 block records 可全部删除后重建。
- [ ] `document_id` 由系统创建，在 tenant + owner 内唯一、不可变。
- [ ] URI 以 `document_id` 稳定；rename 和 tree reclassification 不改 URI。
- [ ] kind 从受控相对路径推导，front matter 不提供 tenant、owner、workspace、ACL、authority 或 generation。
- [ ] 项目只作为 entity/topic/episode/workspace metadata，不建立项目物理记忆目录。
- [ ] `block_id` 只是特定 source digest 下的派生定位符，不是用户手写身份。

## C. Runtime layout 与 bootstrap

- [ ] `<root>/tenants/<tenant>/users/<owner>/memory/` 仅从 trusted runtime root 推导。
- [ ] `markdown_memory_v1` marker 在打开 SQLite 和任何写入前被校验。
- [ ] 空 root 可初始化；非空且不受支持的 root 不被自动清理。
- [ ] 某 owner 首次初始化时创建五个最小 managed template，每个有独立身份。
- [ ] 已初始化 owner 主动删除的文件不在每次启动中重建。
- [ ] reset helper 拒绝 `/`、HOME、repo/workspace root、`.git`、symlink、glob、未展开变量与 tracked files。
- [ ] 外部 Vector namespace 与 hook queue 不随本地 root 被暗中清除。

## D. File store 与 front matter

- [ ] digest 对 exact bytes 做 SHA-256，不对换行、YAML 或文本做规范化后再计算。
- [ ] BOM、非 UTF-8、文件大小、header 大小和 YAML depth 都有硬限制。
- [ ] safe YAML 拒绝 duplicate keys、custom tags、aliases/anchors 和多文档。
- [ ] 受控 path 拒绝绝对路径、`..`、NUL、反斜杠、Unicode/casefold collision 和 root escape。
- [ ] dir-fd、`O_NOFOLLOW`、regular-file 与 link-count 校验真实执行。
- [ ] create/replace 使用同目录 temp file、file fsync、atomic install 和 parent fsync；权限为 file 0600 / dir 0700。
- [ ] rename 目标必须 ABSENT，不覆盖现有用户文件；delete/rename 后 fsync 所有相关 parent。
- [ ] 空文件是 raw PRESENT + registration UNMANAGED，不是 delete。
- [ ] 安全但无 ID 的用户文件需显式 adopt；坏身份或 duplicate 进 quarantine。

## E. Planner 与模型边界

- [ ] `MemoryEditProposal` 只包含语义候选、relation hints 与 evidence references。
- [ ] 模型输出不包含 path、document ID、tenant/owner/workspace/ACL、SQL、delete/erase、generation 或 authority。
- [ ] proposal set 与 archive/manifest digest 封存，冲突 replan 不二次调用模型。
- [ ] router 确定性选择 profile/preferences/entity/topic/episode/open-loop/experience 路径。
- [ ] planner 先有界定位，再读 live bytes，以最新 raw state 做 dedup/merge。
- [ ] 文件大小、patch 范围、单次 edit 数量、front matter 与身份都由确定性代码校验。
- [ ] LLM/provider 错误分类、egress policy、salience ledger 和 evidence span 校验仍有真实测试。

## F. Document commit 与 crash recovery

- [ ] `MemoryDocumentCommitter` 不调用 ordinary operation committer。
- [ ] Plan、intent、blob key、event、queue payload 与 idempotency identity 都绑定 tenant + owner + document。
- [ ] Body-bearing blob 先 fsync，再写 content-free PREPARED intent。
- [ ] 获取 cooperative identity lock 后重读每个 affected path。
- [ ] CREATE/UPDATE/DELETE/RENAME 使用正确 effect vector；DELETE 不伪造空 body blob，RENAME 同时验证 old/new。
- [ ] install 后才 append content-free event，然后耐久 enqueue projection job。
- [ ] live == after 可补齐 event/queue；live == before 只 roll forward；第三状态进 conflict/quarantine。
- [ ] 恢复不写回 before image，也不因 before 为 ABSENT 就盲删 live path。
- [ ] 重复 idempotency key 仅在完整 identity/effect/digest 一致时返回已有结果。
- [ ] logical revision 与 projection generation 单调上升。

## G. Session commit group

- [ ] Archive 写入前失败不入队；Archive 写入后 inline 失败使用同一 task identity 入队。
- [ ] `SessionProjectionJournal` 幂等解决 archive 已写但 Catalog 未投影的窗口。
- [ ] Commit group 只有 memory、behavior、action_policy、context consumers，每个有独立 lease/retry/result。
- [ ] Memory effect 只保存 document change-event ID/digest，不保存 Markdown 正文。
- [ ] Memory consumer 在 document source 已耐久落盘且 projection job 已耐久入队时完成。
- [ ] Behavior/ActionPolicy/ordinary Context 只提交其原有 domain operation，文档 URI 作为 operation endpoint 被拒绝。

## H. Scanner 与外部编辑

- [ ] watcher 事件只是 scan hint；overflow 后必须 full scan。
- [ ] scan 有文件数上限、root identity、完整性、错误和 unsafe path 记录。
- [ ] create/update/rename/delete 都经稳定观察窗口。
- [ ] rename 仅在 old 缺失且恰好一个 new path 有同一 document ID 时确认。
- [ ] 不完整 scan、root change、permission error、unsafe registration 或 mass delete 时暂停删除。
- [ ] 外部变更产生 revision/event/job，mtime 不作为版本事实。

## I. Projection、rebuild 与验证

- [ ] Document 与 block records 带 tenant、owner、document ID、source digest、revision 和 generation。
- [ ] Catalog document row 与 projection state 按 tenant + owner + document 唯一绑定。
- [ ] 迟到的低 generation job/scan/vector retry 不能覆盖新状态或复活删除 identity。
- [ ] 发布或外部 backend 失败保留 retry/error，不 ACK 成功。
- [ ] rebuild 要求 bounded owner provider、安全 full scan 与 max-documents 上限，不递归发现用户。
- [ ] rebuild 前后 scan fingerprint 必须一致；中途 source 变化则失败。
- [ ] verifier 逐文档比较 live ID/path/digest 与 Catalog/projection state，证明无 missing/stale/duplicate row。
- [ ] Startup 在 intents、commit groups、scan、projection、erase tombstones 恢复且 verification 成功前保持 NOT_READY。

## J. Retrieval 安全与性能

- [ ] SDK、HTTP、MCP、`archive_search()` 和 Memory recall 全部构造同一 `RetrievalQueryPlan`。
- [ ] Trusted tenant/owner/workspace/adapter constraint 先绑定，与用户 filter 冲突时 fail closed。
- [ ] Path、time、type、ACL 在 Top-K 前由 SQL 执行；Vector native hit 仍回 Catalog 复核。
- [ ] structured、exact、FTS、Vector、Relation、fusion、rerank 和 hydration 都有硬上限。
- [ ] 在线链不调用 `list_objects()`、递归 memory/archive scan、全量 Vector URI 或 Python 全库 lexical match。
- [ ] Memory candidate hydrate 精确回读 live raw bytes，重验 tenant/owner/ID/digest/generation 和 document URI。
- [ ] source drift 丢弃 stale candidate、标记 degraded mode 并调度 rescan，不把 Catalog 旧文本返回为当前记忆。
- [ ] Vector 失败显式降级 FTS，reranker 失败保留 deterministic fusion order。
- [ ] `ContextPacker` 按 L2 -> L1 -> L0 -> URI 降级，限制 token、final items、per-session、per-document 与 L2 reads。
- [ ] Recall Trace 记录 score components、selected layer、drop reason、source reads、generation lag 和 degraded mode，不包含 secret/source body。
- [ ] `EXPLAIN QUERY PLAN` 证明 tenant、owner、type、path prefix、event/transaction time、FTS map 使用索引。

## K. Revision、forget 与 hard erase

- [ ] Restore 是一次新 CAS update，不通过切换数据库 pointer 改变当前正文。
- [ ] Soft forget 返回可恢复语义，清理 live projection 但保留授权的 revision。
- [ ] Hard erase 仅接受整文档 target；section request 不宣称删除历史副本。
- [ ] Hard erase 在删除前写耐久 publication barrier，document ID 永不复用。
- [ ] Live file、revision/after/review blobs、body-bearing redo/cache/trace 和所有 serving 派生都被可枚举清理。
- [ ] Catalog/FTS/Path/Link/projection state 以 SQLite 事务摘除，Vector/Relation 按 tenant-qualified identity + generation/digest compare-and-delete。
- [ ] 外部 backend 失败保持 `ERASE_PENDING` 且可重放，全部确认后才 completed。
- [ ] 只留不含正文的最小 digest audit，不宣称底层介质安全擦除。
- [ ] 独立 SessionArchive 默认保留；结果精确列出 `independent_evidence_retained`。

## L. 安全投影

- [ ] API key、token、cookie、authorization、password、private key、env、DB/SSH credential、binary 与超长 log 被清洗。
- [ ] 完整绝对路径只在受保护 evidence；serving 只保留 basename + controlled location。
- [ ] Sanitization 失败 fail closed，metadata JSON、Vector metadata 和 Trace 均不能旁路。
- [ ] Memory document 回源文本在公开返回前再次清洗，清洗后重验 proof 字段。
- [ ] Tenant、Owner、Workspace、Adapter private scope 与 unauthorized path 隔离有真实存储测试。

## M. 公开合同与完成验证

- [ ] SDK、HTTP、MCP 共用 `memoryos/api/memory_contract.py` 的 JSON schema 与 result fields。
- [ ] `remember`、`edit_memory_document`、`forget`、history、restore 与 review 全部绑定 trusted caller，mutation 在 NOT_READY 时 fail closed。
- [ ] `archive_search()` 是统一 retrieval 包装，`archive_read()` 仅做精确受权 evidence read。
- [ ] 已运行实际配置支持的 unit、integration、recovery、E2E、完整 pytest、Ruff、MyPy 和 Pyright。
- [ ] 已运行 `git diff --check`、未完成标记搜索、旧路径/字段搜索与 write-bypass 搜索。
- [ ] 失败、工具缺失或环境限制记录真实命令和真实输出，不伪造通过。
- [ ] 最终检查 `git status --short`、`git diff --stat`、新增/删除文件与未提交 diff，不 commit、不 push、不修改 README。
