# m2bOS

m2bOS 正在进行完整的记忆架构重构。

## 当前状态

旧长期记忆树、文档式写入链、编辑命令、投影、Worker 和公开写入接口已经删除，不保留兼容层。

新的记忆树、对话解析器、Memory Editor、长期记忆 Schema、索引策略和会话压缩策略尚未实现。

当前仓库保留：

- Conversation 与 SessionArchive 持久化；
- message、tool call、tool result 等严格会话角色；
- 通用 Context 投影、检索和精确读取；
- 通用 SourceStore、SQLite、Queue、Vector、Relation、锁和原子文件基础设施；
- 不写入用户长期记忆的 Behavior 与 ActionPolicy 能力。

当前阶段故意不提供长期记忆的写入、编辑、重命名、合并、遗忘、恢复和复核接口。

## 开发检查

在新测试体系完成设计前，生产源码变更通过 Python 编译、Ruff、MyPy、Pyright、运行时/存储 smoke check 和现有 TypeScript 集成构建进行验证。
