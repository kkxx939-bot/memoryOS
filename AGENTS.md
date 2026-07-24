# m2bOS Repository Constraints

本文件约束仓库内所有人工和 Agent 变更。当前用户决策优先于历史实现，不得通过兼容层恢复已经删除的架构。

## 当前记忆重构阶段

- 旧长期记忆树、写入链、文档身份、编辑/删除命令、投影和公开接口已经退出；不保留别名、迁移读取、禁用分支、空实现或新旧双轨。
- 新记忆树的 L2 权威目录已经确认：`profile.md`、`preferences/{topic}.md`、`entities/{category}/{name}.md`、`tools/{tool_name}.md`、`events/YYYY/MM/DD/{event_name}.md`、`intentions/{intent_name}.md`。集合目录采用 OpenViking 的目录分层思想，允许保存可重建的 `.abstract.md`（L0）和 `.overview.md`（L1）；`profile.md` 仍是直接读取的单文件 L2。当前实现记忆地址、目录地址、L0/L1/L2 路径、Markdown 原文安全操作、有界枚举和自下向上刷新；不得加入旧实现或兼容入口。
- 新记忆地址协议固定为 `memory://`，其后路径与上述记忆树根目录一一对应，例如 `memory://profile.md`、`memory://preferences/{topic}.md` 和 `memory://events/YYYY/MM/DD/{event_name}.md`；根、合法目录以及目录下 `.abstract.md`/`.overview.md` 也可以被确定性寻址。URI 不包含 owner、用户空间、稳定文档 ID 或旧 Context scope，只接受完整协议形式，不兼容短路径、`viking://` 或 `memoryos://`。URI 解析必须复用 `MemoryAddress`、`MemoryDirectory` 和 `MemoryLevel` 的树约束，禁止产生树外路径或通过静默清洗改名。
- L2 是唯一权威记忆，只能由未来 Memory Editor 的确定性写入阶段修改。L0/L1 是从直接 L2 文件和子目录 L0 派生的 Serving 内容，必须能够删除并重建；生成失败不得回滚或改写已经成功写入的 L2。Memory Candidate、Memory Schema 和 LLM 不得把 L0/L1 当作新的长期记忆字段。
- L2 物理格式统一为 Schema 渲染的 Markdown 正文加文件末尾唯一 `M2BOS_MEMORY_FIELDS` JSON 注释；正文与结构字段属于同一个原子 Markdown 文件。注释顶层只保存可信系统字段 `memory_type`、`revision`、`created_at`、`updated_at` 和业务 `fields`，其中版本和时间不得由 LLM 或内容 Schema 生成。结构字段负责确定性更新，正文负责人类阅读和 L0/L1 派生。读取必须校验 Schema、真实路径、规范序列化和正文重渲染完全一致；旧裸 Markdown、缺少系统字段、重复注释、宽松 JSON 和双轨写入均不兼容。内部注释不得进入 L0/L1 提示词。
- L2 正文字符上限和完整物理文件字节上限统一由 `MemoryDocumentConfig` 管理，记忆树在写入前和读取后执行同一组检查；语义层不得维护另一套相互冲突的单文件限制。当前 `MemoryTree.write()` 只落盘已经构造完成的 `MemoryDocument`，不读取旧记忆、不执行 YAML 合并策略、不推进版本或生成时间；这些职责只属于后续 Memory Editor。
- 六类长期记忆内容 Schema 使用独立声明，限定字段角色、字段类型、路径模板、Markdown 模板、操作模式和未来合并策略。Schema 不得包含 owner、稳定文档 ID、tag、置信度、EvidenceSlice、链接或反向链接。
- `tools/{tool_name}.md` 只保存真实工具调用形成的稳定工具知识，不保存调用流水、次数、成功率或 Skill。`tool_name` 必须逐字来自 Conversation 中真实 `tool_call.name`，不得依赖工具注册表或由 LLM 翻译、概括、改名；其余知识字段允许按来源部分出现，但至少一个必须非空。真实调用、结果、失败和恢复的来源绑定属于未来 Memory Editor，不得在树或语义层伪造。
- `intentions/{intent_name}.md` 只保存 ConversationSegment 结束时仍未完成、未来仍需推进或等待触发的明确事项，并始终覆盖为最新状态快照，不追加执行历史。文件不重复保存 objective；`intent_name` 本身必须清楚表达事项。状态只允许 `open`、`waiting`、`blocked`、`completed`，其中 `completed` 是保留文件但不再有效的终止状态。默认召回排除完成事项、长期未完成事项的 stale 检查和重新概括均等待后续检索与生命周期配置，不得提前写死时间阈值。
- `memory/editor` 是新长期记忆编辑领域包，但旧数据契约仍然退出，不保留 `MemoryEditSource`、旧 `MemoryEditBatch` 或兼容入口。旧记忆读取已经确认为两层：`infrastructure/editor/snapshot` 提供领域无关的版本快照机制，`memory/editor/reader.py` 只接受严格 L2 `memory://` URI 并读取完整 `MemoryDocument`。相关旧记忆检索采用 OpenViking 的编排思路：从完整 `ConversationSegment` 生成有界查询，按 Memory Schema 限定目录，固定文件直接读取，动态文件通过显式语义搜索契约返回 L2 URI，再预取有界 Top-N 完整快照；搜索失败、越界结果和完整性错误不得静默伪装为“没有旧记忆”。LLM 结构化候选、合并和写入编排继续逐项确认，不得恢复旧 Context/Catalog 检索或提前实现未确认职责。
- `memory/semantic` 独立负责目录 L0/L1：使用受控目录快照生成覆盖全部直接子项的 L1，再从 L1 的简介确定性提取 L0，并按受影响目录链自下向上刷新。它不得读取 Conversation、决定 L2 内容或执行长期记忆合并。`memory/editor/retrieval` 采用参考 OpenViking、但严格适配当前记忆树的稳定分阶段检索架构：可配置 Embedder 是语义召回的必需依赖，每次搜索只生成一次 query embedding；索引层只接受向量和受限 `memory://` 根目录，按内容摘要复用有界文档向量缓存；检索层支持直接 L2 向量召回，以及利用目录 L0/L1 的可配置层级召回；Reranker 是独立、可选的第二阶段，不得与 Embedding 混为一体。无 Embedder 时必须明确失败，不能以关键词匹配冒充语义搜索。阶段之间使用稳定领域契约，使以后替换持久向量索引或调整层级策略时不改写 Memory Editor。不得复活旧通用 Context、Catalog、FTS、Vector 或 Query 链，也不得引入租户、热度、关系、后台队列或旧数据兼容。
- Conversation 使用 `pre/conversation/messages` 与 `pre/conversation/summaries` 两层：messages 保存完整、严格区分 prompt、completion、tool_call、tool_result 的原始会话事实；summaries 保存绑定不可变消息片段的宽语义历史过程，不能替代原文参与记忆解析。
- Conversation Schema 继续只放在 `pre/conversation`；旧 SessionArchive projector、旧本地文件读写和单会话覆盖式摘要已经退出。正式路径、live 追加、读取和 history 封存统一由 `memory/conversation` 实现，并复用通用 PathLock 与耐久原子文件能力；不得恢复旧入口或在 `pre` 中写存储逻辑。LLM 摘要生成、周期压缩、保留期删除和长期记忆写入仍未实现。
- Conversation 生命周期主链为 `live.jsonl -> history/{segment_id}.jsonl -> summaries/{segment_id}.json`：消息先进入有界 live 窗口，完整片段封存为不可变 history，再从该 history 派生 summary。history 是有保留期的原文层，不是永久层；只有对应 summary 成功、通过来源绑定与结构校验并满足后续保留策略后，history 才可删除。history 与 segment summary 在原文保留期间通过 `conversation_id`、`segment_id` 和 `source_message_digest` 一一绑定，但 summary 在生成期间允许暂时不存在或处于失败状态。
- segment summary 也不是无限累积的最终层；后续必须按可配置时间周期或容量条件，将多个较旧 summary 继续压缩为覆盖更长时间范围的语义摘要，并在新摘要耐久、校验成功且满足保留策略后清理被覆盖的旧 summary。时间桶、层级数量、保留期、容量阈值和失败回退尚未确定。
- live 归档阈值、最近消息保留窗口、完整轮次和工具调用边界、主动 commit、超大单条消息处理、history 保留期、summary 压缩周期以及各阶段失败重试都属于后续生命周期配置；这些策略必须由显式 Config 驱动，当前只记录设计边界，不提前写死配置值或实现行为。
- SessionArchive、Behavior、ActionPolicy、Resource、Skill、通用 Context/ContextURI 及 `memoryos://` 协议已经整体退出，等待后续分别重构；必须同步删除其领域模型、存储、运行时、事务、SDK、HTTP、MCP、CLI 和集成调用者，不得保留别名、迁移器、禁用分支、占位实现或兼容入口。
- `pre.session`、旧 SessionArchive 文件布局、SourceStore、Context Catalog/FTS/Vector/Relation/Queue 等仅服务上述旧领域的实现一并退出。原子文件、路径安全和锁等确实被新 `memory/` 或 Conversation 独立使用的底层原语可以保留，但不得继续暴露旧领域语义。
- 新长期记忆只能以后续 ConversationSegment 为事实来源；在 Memory Editor 写入机制明确前，不得自动生成或修改长期记忆。
- 当前阶段不新增测试；新架构明确后重新设计测试体系。

## 稳定边界

- 通用大模型调用统一由仓库一级包 `LLMClient/` 承载，公开主入口为 `LLMClient`、`LLMClientFactory` 和 `build_llm_client`；配置、请求响应契约和 Provider 也归属该包。旧 `infrastructure/model` 路径已经退出，不保留兼容导入或双轨实现。
- 通用 Context、ContextURI、ContextProjectionSanitizer 以及旧 Catalog/FTS/Vector/Trace/L0/L1 链路已经退出，不得被新记忆树复用或恢复。
- Behavior、ActionPolicy、SessionArchive、Resource 和 Skill 当前没有公开入口或运行时主链；后续重构必须作为新设计另行确认。
- `memoryos` 暂时只保留为 Python 发行名称；旧 CLI、环境变量协议和 `memoryos://` URI scheme 不再构成稳定契约。

## 工作区安全

- 修改前记录分支、HEAD、工作区状态、未跟踪文件和相关公开入口。
- 用户已有修改不得回退、覆盖或顺手格式化。
- 不自动提交、不自动 Push；README 仅在用户明确授权时修改。
- 源码中的说明性注释和 docstring 统一使用中文；许可证声明、协议名称、公开标识符以及 `noqa`、`type: ignore` 等机器指令保持原样。
- 调整契约时检查所有调用者、运行时装配、Schema、Worker、SDK、HTTP、MCP、CLI 和外部集成。

## 当前验证门槛

旧测试资产已经退役。新测试体系建立前，完成生产源码变更至少运行：

- Python compile；
- Ruff；
- MyPy；
- Pyright；
- 与改动相关的真实运行时或存储 smoke check。

工具缺失或外部服务不可用时记录真实错误，不得伪造通过，也不得为运行检查随意修改依赖。
