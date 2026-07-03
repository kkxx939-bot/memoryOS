# Storage Model

这份文档只回答三个问题：

1. 记忆怎么存。
2. 当前存了哪些东西。
3. 权重应该建立在哪些存储语义上。

## 存储分层

当前系统不是只有一个数据库，分三层：

```text
Markdown tree      = 事实源，可读、可审计、可手工修正
SQLite memories    = 结构化索引，支持列表、过滤、FTS、排序
SQLite embeddings  = 向量索引，只用于召回，可重建
```

所以权重应该首先写在 Markdown front matter 和 `memories` 表里。embedding 表不应该成为事实源。

## 目录结构

```text
memory-root/
├── user/{user_id}/
│   ├── profile/
│   ├── preferences/
│   ├── habits/
│   ├── triggers/
│   ├── interventions/
│   ├── feedback/
│   ├── policies/
│   ├── events/
│   ├── cases/
│   ├── daily/
│   ├── sessions/
│   └── episodes/
└── index/
    └── memory.sqlite3
```

## 单条记忆格式

每条长期记忆是 Markdown 文件：

```markdown
---
{
  "id": "...",
  "user_id": "gulf",
  "type": "habit",
  "title": "computer smoking trigger",
  "path": "user/gulf/habits/computer-smoking-trigger-xxxx.md",
  "tags": ["smoking", "computer"],
  "source": "episode:ep-1",
  "confidence": 0.7,
  "created_at": "...",
  "updated_at": "...",
  "last_accessed_at": null,
  "active_count": 0,
  "hotness": 0.2,
  "lifecycle_state": "warm",
  "temporal_scope": "rolling_30d",
  "base_weight": 0.82,
  "evidence_count": 1,
  "positive_count": 1,
  "negative_count": 0,
  "effective_weight": 0.4,
  "abstract": "..."
}
---
# computer smoking trigger

User may want to smoke after sitting at the computer for a long time.
```

## 当前 memory type

| Type | Directory | Storage role | Update mode | Default time scope | Prediction role |
| --- | --- | --- | --- | --- | --- |
| `profile` | `profile/` | 稳定用户画像 | replace/patch single file | `stable` | 背景先验 |
| `preference` | `preferences/` | 显式偏好 | topic patch/split | `stable` | 默认选择先验 |
| `habit` | `habits/` | 行为规律 | rolling pattern update | `rolling_30d` | 候选行为先验 |
| `trigger` | `triggers/` | 场景触发信号 | rolling pattern update | `rolling_7d` | 候选激活信号 |
| `intervention` | `interventions/` | agent 行为历史 | append/aggregate | `rolling_30d` | 过去怎么干预 |
| `feedback` | `feedback/` | 用户反馈 | append/aggregate | `rolling_30d` | reward 信号 |
| `policy` | `policies/` | 权限和边界 | strict patch/replace | `stable` | 硬约束/软约束 |
| `event` | `events/` | 可审计事实 | append only | `episodic` | 近期证据，低先验 |
| `case` | `cases/` | 相似案例 | replace/version | `rolling_30d` | 相似 episode 先验 |

## 权重应该基于什么

权重不是单独一套外部配置，而应该来自存储语义：

- `type` 决定基础可信层级。
- `temporal_scope` 决定时间衰减窗口。
- `evidence_count / positive_count / negative_count` 决定规律是否被足够样本支持。
- `confidence` 决定模型/用户/观察来源的可信度。
- `hotness / active_count` 决定最近是否经常被使用。
- `source` 决定来源，例如 manual、episode、session、event。

因此“最近 7 天天气热出汗后需要开空调”不应该被存成普通 event 后直接强影响预测。它应该逐步从多个 `event` 或 `feedback` 聚合成一个 `trigger` 或 `habit`，并带有：

```json
{
  "type": "trigger",
  "temporal_scope": "rolling_7d",
  "evidence_count": 6,
  "positive_count": 5,
  "negative_count": 1
}
```

这类聚合后的规律才应该在预测中拥有较高 `effective_weight`。

## 当前缺口

目前已经有单条记忆的权重字段，但还缺少“从事件样本聚合成规律记忆”的机制。

## Update Policy

借鉴 OpenViking 的方向，记忆不能只有 `add/update/delete` 三种粗操作。每种类型应该有自己的更新策略：

| Type | Policy | 说明 |
| --- | --- | --- |
| `profile` | `patch_or_replace` | 单文件紧凑画像，持续重写/修正，不无限 append |
| `preference` | `topic_patch_or_split` | 按主题 patch，超过阈值拆分 |
| `habit` | `aggregate_from_evidence` | 不能由一次观察直接形成强习惯，先存 evidence |
| `trigger` | `aggregate_from_evidence` | 由多次 context-action 样本聚合，通常 rolling 7d |
| `intervention` | `append_then_aggregate` | 先记录 agent 做过什么，再按动作/结果聚合 |
| `feedback` | `append_then_aggregate` | 先记录 reward/反馈，再汇总给策略层 |
| `policy` | `strict_patch` | 权限/安全边界，必须有明确用户意图 |
| `event` | `append_only` | 事件是证据，不改写 |
| `case` | `replace_or_version` | 相似案例可替换或版本化 |
 
当前代码里的 `memoryos/domain/memory/update_policy.py` 固化了这些规则。后续 extractor 产生 memory operation 后，应该先经过 update policy，再决定是否写入、转换为 event evidence、忽略或要求用户确认。

下一步应该做：

```text
episode/event/feedback logs
  -> pattern miner
  -> habit/trigger update
  -> evidence_count update
  -> effective_weight refresh
```

也就是说，权重设计要落在“先存事件样本，再聚合成规律”的机制上，而不是直接让一次观察形成高权重习惯。
