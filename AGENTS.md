# m2bOS Repository Constraints

本文件约束仓库内所有人工和 Agent 变更。当前用户决策优先于历史实现，不得通过兼容层恢复已经删除的架构。

## 当前记忆重构阶段

- 旧长期记忆树、写入链、文档身份、编辑/删除命令、投影和公开接口已经退出；不保留别名、迁移读取、禁用分支、空实现或新旧双轨。
- 新记忆树的基础目录已经确认：`profile.md`、`preferences/{topic}.md`、`entities/{category}/{name}.md`、`tools/{tool_name}.md`、`events/YYYY/MM/DD/{event_name}.md`、`intentions/{intent_name}.md`。当前实现记忆地址、路径、Markdown 原文读写删除和有界枚举；不得加入旧实现或兼容入口。
- 六类长期记忆内容 Schema 使用独立声明，限定字段角色、字段类型、路径模板、Markdown 模板、操作模式和未来合并策略。Schema 不得包含 owner、稳定文档 ID、tag、置信度、EvidenceSlice、链接或反向链接。
- Memory Editor 结构化操作 Schema、实际 merge 执行、索引、链接和压缩机制尚未设计完成；当前不得提前实现或用占位逻辑推断。
- Conversation 使用 `pre/conversation/messages` 与 `pre/conversation/summaries` 两层：messages 保存完整、严格区分 prompt、completion、tool_call、tool_result 的原始会话事实；summaries 保存绑定 messages 摘要的语义文本，不能替代原文参与记忆解析。
- 当前 Conversation 只实现纯模型、SessionArchive 到 ConversationBatch 的确定性转换和本地文件读写；不得调用 LLM、自动生成摘要或写入长期记忆。
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
