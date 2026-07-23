# m2bOS Repository Constraints

本文件约束仓库内所有人工和 Agent 变更。当前用户决策优先于历史实现，不得通过兼容层恢复已经删除的架构。

## 当前记忆重构阶段

- 旧长期记忆树、写入链、文档身份、编辑/删除命令、投影和公开接口已经退出；不保留别名、迁移读取、禁用分支、空实现或新旧双轨。
- 新记忆树的基础目录已经确认：`profile.md`、`preferences/{topic}.md`、`entities/{category}/{name}.md`、`tools/{tool_name}.md`、`events/YYYY/MM/DD/{event_name}.md`、`intentions/{intent_name}.md`。当前实现记忆地址、路径、Markdown 原文读写删除和有界枚举；不得加入旧实现或兼容入口。
- 六类长期记忆内容 Schema 使用独立声明，限定字段角色、字段类型、路径模板、Markdown 模板、操作模式和未来合并策略。Schema 不得包含 owner、稳定文档 ID、tag、置信度、EvidenceSlice、链接或反向链接。
- Memory Editor 结构化操作 Schema、实际 merge 执行、索引、链接和压缩机制尚未设计完成；当前不得提前实现或用占位逻辑推断。
- Conversation 使用 `pre/conversation/messages` 与 `pre/conversation/summaries` 两层：messages 保存完整、严格区分 prompt、completion、tool_call、tool_result 的原始会话事实；summaries 保存绑定不可变消息片段的宽语义历史过程，不能替代原文参与记忆解析。
- Conversation Schema 继续只放在 `pre/conversation`；旧 SessionArchive projector、旧本地文件读写和单会话覆盖式摘要已经退出。正式路径、live 追加、读取和 history 封存统一由 `memory/conversation` 实现，并复用通用 PathLock 与耐久原子文件能力；不得恢复旧入口或在 `pre` 中写存储逻辑。LLM 摘要生成、周期压缩、保留期删除和长期记忆写入仍未实现。
- Conversation 生命周期主链为 `live.jsonl -> history/{segment_id}.jsonl -> summaries/{segment_id}.json`：消息先进入有界 live 窗口，完整片段封存为不可变 history，再从该 history 派生 summary。history 是有保留期的原文层，不是永久层；只有对应 summary 成功、通过来源绑定与结构校验并满足后续保留策略后，history 才可删除。history 与 segment summary 在原文保留期间通过 `conversation_id`、`segment_id` 和 `source_message_digest` 一一绑定，但 summary 在生成期间允许暂时不存在或处于失败状态。
- segment summary 也不是无限累积的最终层；后续必须按可配置时间周期或容量条件，将多个较旧 summary 继续压缩为覆盖更长时间范围的语义摘要，并在新摘要耐久、校验成功且满足保留策略后清理被覆盖的旧 summary。时间桶、层级数量、保留期、容量阈值和失败回退尚未确定。
- live 归档阈值、最近消息保留窗口、完整轮次和工具调用边界、主动 commit、超大单条消息处理、history 保留期、summary 压缩周期以及各阶段失败重试都属于后续生命周期配置；当前只记录设计边界，不提前写死配置值或实现行为。
- 旧 Evidence 改名得到的 SessionEvent、SessionEpisode、Scope 适配器和 ArchiveEpisodeAdapter 已退出；`pre.session` 只保留原始 SessionArchive，不得恢复中间证据模型或兼容别名。
- 原子文件、路径安全、SQLite、Queue、Vector、Relation、SourceStore 等通用基础设施继续保留，但不能包含只服务旧长期记忆的接口或字段。
- SessionArchive、Tool Result 和普通 Context 可以作为历史事实源与召回输入；在新写入机制明确前，不得自动生成或修改长期记忆。
- 当前阶段不新增测试；新架构明确后重新设计测试体系。

## 稳定边界

- `infrastructure/store` 只承载通用持久化实现及查询原语；`infrastructure/context` 承载上下文规划、召回、分层、组装、精确读取和轨迹语义。Store 不得反向依赖 Context。
- Tool Result 必须经过 `ContextProjectionSanitizer`，才能进入 Catalog、FTS、Vector、Trace 或 L0/L1。
- Prediction 和动作执行属于 ActionPolicy；Behavior、ActionPolicy 与未来长期记忆保持语义隔离。
- `memoryos` 继续作为发行名称、CLI 前缀、环境变量前缀和 `memoryos://` URI scheme；不得恢复根级兼容聚合包。
- 旧运行时布局不迁移、不兼容；发现旧状态时必须要求显式重置。
- 本地用户、Workspace 和 Adapter 必须由可信运行时绑定，不能由模型输出覆盖。
- 在线查询必须有界，不得递归扫描事实源、归档目录或全量向量 URI。

## 工作区安全

- 修改前记录分支、HEAD、工作区状态、未跟踪文件和相关公开入口。
- 用户已有修改不得回退、覆盖或顺手格式化。
- 不自动提交、不自动 Push；README 仅在用户明确授权时修改。
- 调整契约时检查所有调用者、运行时装配、Schema、Worker、SDK、HTTP、MCP、CLI 和外部集成。

## 当前验证门槛

旧测试资产已经退役。新测试体系建立前，完成生产源码变更至少运行：

- Python compile；
- Ruff；
- MyPy；
- Pyright；
- 现存 TypeScript 集成构建；
- 与改动相关的真实运行时或存储 smoke check。

工具缺失或外部服务不可用时记录真实错误，不得伪造通过，也不得为运行检查随意修改依赖。
