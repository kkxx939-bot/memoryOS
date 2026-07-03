# Memory Architecture

这个项目的目标不是给某一个机器人做记忆，而是做一个可复用的个人记忆底座。机器人、桌面 agent、车机、家居设备、可穿戴设备都只是输入源。

## 核心边界

记忆系统分为三条链路：

1. 写入链路：把原始输入变成结构化长期记忆。
2. 召回链路：根据当前场景找到相关记忆。
3. 注入链路：把召回结果压缩成主模型可用的上下文。

不要把这三件事混成一个模型。混在一起后，后面会很难调试、评估和替换。

在应用侧，一个同步请求应该封装成 `Episode`。输入不要只依赖一段自然语言 `scene`，更稳定的入口是结构化 `ObservationContext`：

```text
ObservationContext/messages
  -> scene_text + retrieval_query + context_tags
     -> retrieval orchestrator
     -> memory context retrieval
     -> behavior pattern retrieval
     -> behavior feedback retrieval
  -> candidate generation
  -> ranking
  -> top prediction
  -> intervention recommendation
  -> memory extraction operations
  -> memory write/update/ignore or deferred pending write
  -> feedback/reward
```

所以“是否存储”和“预测下一步”不是两个割裂流程，而是同一个 episode 的两个必要输出。时序上预测优先，因为观察发生后需要立即给出下一步判断；记忆写入可以在预测之后同步完成，也可以延迟到后台。

`ObservationContext` 记录的是当前观察的可计算字段：

- `raw_text`：VLM、桌面 agent、传感器或人工输入生成的原始摘要。
- `location`：位置，例如 `computer_desk`、`room`、`car_cabin`。
- `activity`：当前活动，例如 `computer_work`、`arrive_home`。
- `started_at` / `observed_at` / `duration_minutes`：用于表达“从什么时候开始坐到电脑桌”和持续多久。
- `signals`：显式信号，例如 `sweating`、`says_hot`、`cigarette_pack_visible`。
- `environment`：温度、湿度、设备可用状态等上下文。

系统会从它派生三类字段：

- `scene`：可读文本，进入 episode 审计和预测模型。
- `retrieval_query`：稳定召回 query，进入 retrieval layer，驱动 memory context、behavior pattern 和 behavior feedback 三类召回。
- `context_tags`：位置、活动、时间段、持续时间、环境桶等标签，用于后续聚合和权重。

预测错了默认不代表记忆错。反馈首先进入 prediction/policy 校准数据，例如 `behavior_stats.json`、`policy_stats.json` 和 episode feedback log。只有用户明确纠正事实或记忆内容时，才写 `memory_correction`。

反馈会拆成两类信号：

- 行为预测信号：`predicted_action` 是否等于 `actual_action`，写入 `behavior_stats.json`，下次同类场景进入 `behavior_reward`。
- 系统动作信号：用户是否接受本次提醒、询问或执行，写入 `policy_stats.json`，下次选择 `intervention` 时使用。

behavior pattern 和 `behavior_stats.json` 都属于 retrieval layer。前者召回“过去类似场景聚合后稳定发生了什么”，后者召回“过去同类场景预测准不准”。它们不应该放在 prediction 层里直接读文件。

behavior pattern 和 `behavior_stats.json` 都不只做精确 hash 匹配。场景签名分三层：

- `exact`：完整标签，适合完全相同的观察。
- `semantic`：去掉温度、湿度、精确分钟数等高波动值，保留语义桶，例如 `hot_environment`。
- `coarse`：只保留位置、活动和核心信号，作为冷启动和弱泛化。

召回时会合并 exact、semantic、coarse。behavior pattern 会带 `match_level`、`match_weight`、`evidence_confidence`、`prediction_coefficient`；行为反馈会带 `match_level`、`match_weight`、`weighted_behavior_reward`。排序使用加权后的反馈，避免“温度 30 度学到的规律”在“温度 31 度”时完全失效，也避免把相似场景当成同一场景过度放大。

预测层按推荐/搜索排序系统设计：

```text
candidate generation -> feature scoring -> ranking -> top prediction
```

候选可以是 `smoke`、`seek_cooling`、`continue_current_activity` 等行为/需求。排序特征包括 behavior pattern、记忆匹配、行为反馈得分和记忆热度。`prediction` 是 top1 行为候选的简化摘要，`ranked_candidates` 保留完整候选和分数，便于调试和后续训练。

系统动作不属于行为候选。`intervention` 由单独的 selector 根据 top 行为候选、可执行动作、策略统计和打扰成本选择，例如 `remind_no_smoking`、`turn_on_ac`、`ask_user`、`do_nothing`。

候选生成和排序是两个模块：

```text
memoryos/services/prediction/candidate_generator.py         -> 生成用户行为候选
memoryos/services/prediction/candidate_ranker.py            -> 对用户行为候选提特征并排序
memoryos/usecases/intervention/select_intervention.py       -> 根据 top 行为候选选择系统动作
```

候选必须尽量全，排序才能工作。候选生成采用 pattern-first：

```text
1. feedback 里的 actual_action 先聚合成 behavior pattern
2. 新观察先召回相似 behavior pattern
3. pattern 证据足够时，从历史真实行为生成候选
4. 规则/记忆/LLM 只做补充和冷启动
```

例如连续 3 天或 5 天的相似场景都是“用户回房间 + 说热 + 出汗 -> open_ac”，新场景就优先生成 `open_ac` 候选，而不是只因为文本里有“热”就生成泛化的 `seek_cooling`。

这里不用“最近三次”作为强信号，因为同一天内重复出现的样本置信度太低。系统会计算 `prediction_coefficient`：

```text
prediction_coefficient =
  action_consistency
  * observation_day_factor
  * reward_factor
```

预测命中会逐步提高系数；预测错误不会立刻清零，避免短期噪声把稳定规律打掉。

记忆权重分两层：

- `retrieval_weight`：这条记忆为什么被召回，来自 FTS/embedding/hotness/confidence。
- `usage_weight`：这条记忆对当前候选行为是否真正有用，取决于 memory type、内容是否支持该候选、置信度和上下文。

存储层还会给每条记忆计算 `effective_weight`：

```text
effective_weight =
  base_weight
  * temporal_weight
  * evidence_weight
  * confidence
```

`temporal_scope` 用来表达规律的时间窗口：

- `stable`：稳定画像、长期偏好、长期策略。
- `rolling_7d`：短期触发规律，例如最近天气热导致的空调需求。
- `rolling_30d`：中期习惯、干预反馈、案例规律。
- `episodic`：一次性事件，默认快速衰减。
- `seasonal`：季节性规律，后续可结合日期/天气增强。

候选里的 `memory_evidence` 会记录每条参与判断的记忆：

```json
{
  "path": "user/gulf/habits/smoking.md",
  "type": "habit",
  "retrieval_weight": 0.71,
  "usage_weight": 1.0,
  "combined_weight": 0.71,
  "support": "positive"
}
```

这样可以区分“被召回”与“被用于预测”，也方便预测错误后调整算法权重，而不是误改长期记忆。

## 写入链路

输入可以来自：

- `SessionEvent`：用户和 agent 的对话。
- `ObservationEvent`：视觉、音频、传感器或桌面行为摘要。
- `ActionEvent`：agent 已执行的提醒、询问、设备控制或任务动作。
- `FeedbackEvent`：用户接受、拒绝、纠正、忽略或表达不满。

LLM Memory Extractor 的职责是把这些输入整理成记忆操作：

```text
raw events
  -> extractor prompt
  -> memory operations
  -> validate schema
  -> write Markdown tree
  -> update SQLite index
  -> write memory_diff.json
```

Extractor 不应该直接决定下一步动作。它只负责把事实、偏好、习惯、触发条件和反馈写清楚。

当前已经有两个 extractor 边界：

- `RuleBasedExtractor`：只支持 `记住：...` 这类显式写入，用于无模型情况下的本地测试。
- `JsonLLMMemoryExtractor`：面向真实 LLM 的 JSON 协议层，provider 可以是云 API、本地 vLLM 服务或其他模型适配器。

`JsonLLMMemoryExtractor` 要求模型返回 memory operations。真实模型通过 OpenAI-compatible chat/completions API 接入，默认环境变量前缀是 `MEMORYOS_LLM_*`。

```json
{
  "operations": [
    {
      "action": "add",
      "memory_type": "habit",
      "title": "computer smoking trigger",
      "text": "User may want to smoke after sitting at the computer for a long time.",
      "tags": ["smoking", "computer", "prediction"],
      "confidence": 0.72,
      "target": null,
      "rationale": "Repeated behavioral pattern useful for future prediction."
    }
  ]
}
```

支持的 `action`：

- `add`：新增一条长期记忆。
- `update`：修改已有记忆，必须提供 `target`。
- `ignore`：进入 `memory_diff.json` 审计，但不写入长期记忆。

Extractor 仍然不决定下一步动作。它只决定哪些内容应该进入记忆层，以及如何更新。

## 记忆类型

当前 memory type：

- `profile`：稳定用户画像。
- `preference`：显式偏好和默认选择。
- `habit`：重复行为模式。
- `trigger`：某个行为或需求之前常见的条件。
- `intervention`：过去 agent 做过的提醒、询问或动作。
- `feedback`：用户对 agent 行为的接受、拒绝、纠正。
- `policy`：权限、风险边界和自动化规则。
- `event`：可追溯事件证据。
- `case`：可复用的情境、行动、结果样例。

这些类型是为了支持预测和决策，不只是为了搜索。

## 存储层

主存储是 Markdown tree：

```text
user/{user_id}/profile/
user/{user_id}/preferences/
user/{user_id}/habits/
user/{user_id}/triggers/
user/{user_id}/interventions/
user/{user_id}/feedback/
user/{user_id}/policies/
user/{user_id}/events/
user/{user_id}/cases/
user/{user_id}/daily/
user/{user_id}/sessions/
```

SQLite 是索引，不是唯一事实源。这样做的好处是：

- 记忆可读、可审计、可手工修正。
- 文件树天然支持 profile、daily、event、case 分层。
- SQLite 可以重建，不把系统锁死在某一个向量库里。

## 召回链路

当前版本：

```text
query -> SQLite FTS -> Chinese LIKE fallback -> hotness update -> digest
```

当前也已经有 embedding index 和 hybrid retrieval 骨架：

```text
memory write/update
  -> Markdown tree
  -> SQLite FTS
  -> SQLite memory_embeddings

query
  -> keyword candidates
  -> embedding candidates
  -> hybrid score
  -> hotness update
  -> digest
```

默认 `HashingEmbeddingProvider` 只用于离线开发和测试。它不是语义模型，不能代表真实 embedding 效果。

真实 embedding 通过 OpenAI-compatible embeddings API 接入，默认环境变量前缀是 `MEMORYOS_EMBEDDING_*`。Markdown 和 SQLite 仍然是事实源，`memory_embeddings` 只是可重建索引。

后续版本：

```text
current scene
  -> query rewrite
  -> keyword retrieval
  -> embedding retrieval
  -> type/time/hotness rerank
  -> compact digest
```

Embedding Index 只负责召回候选，不负责存储真相。真相仍然在 Markdown 和 SQLite metadata 中。

## Hybrid Retrieval 排序

最终排序应该至少考虑：

- 关键词匹配：明确词命中。
- 向量相似度：语义接近。
- `hotness`：最近经常被用到的记忆优先。
- `memory_type`：不同任务偏向不同类型。
- 时间衰减：过旧、未复用的事件权重降低。
- 置信度：低置信记忆只能辅助，不能强决策。

示例：

```text
final_score =
  keyword_score * 0.25
  + embedding_score * 0.35
  + hotness * 0.15
  + type_boost * 0.15
  + confidence * 0.10
```

权重后续要根据真实使用数据调。

## 注入链路

Hook 注入只给主模型 compact digest：

```text
<personal-memory source="memoryos" format="digest">
Relevant long-term memory:
- [habit] ...
- [trigger] ...
- [policy] ...
</personal-memory>
```

注入内容不能被 extractor 当作新事实再写回，否则会造成记忆污染和重复放大。

## 预测层

预测层不属于存储层，但依赖记忆层。当前通过 `EpisodeProcessor` 把记忆写入和预测放在同一次同步请求里。

它的输入：

- 当前结构化观察和场景摘要。
- 召回记忆 digest。
- 可执行动作列表。
- 安全策略和权限边界。

它先输出行为预测：

- `predicted_need`：用户可能需要什么。
- `predicted_action`：用户下一步可能做什么。
- `confidence`：预测置信度。
- `ranked_candidates`：用户行为候选排序，不包含系统动作。
- `reason`：简短可审计理由。

然后由 `InterventionSelector` 输出系统动作：

- `intervention.action`：提醒、询问、执行或不动作。
- `intervention.features.policy_reward`：这个系统动作在类似预测下的历史反馈。
- `intervention.alternatives`：其他可选系统动作及分数。

示例：

```text
场景：用户在电脑前久坐，检测到烟盒动作。
记忆：用户有室内抽烟倾向，过去接受过提醒，不喜欢强硬语气。
预测：用户可能准备抽烟。
系统动作：温和提醒不要在房间抽烟。
反馈：用户接受或拒绝，写入 feedback/intervention/event。
```

## 当前阶段

当前应该优先完成：

1. schema 稳定。
2. memory type 的写入/更新策略稳定。
3. 热/冷生命周期稳定。
4. session commit 和 memory_diff 稳定。
5. hook 注入边界稳定。

下一步不是直接把所有逻辑塞进模型，而是继续补两块：

1. 根据实际 API 服务配置模型名和 base URL，开始真实 session 与 observation 抽取。
2. 用真实 embedding provider 重建 `memory_embeddings`。
3. 增加预测层接口：`ObservationContext + retrieved memories -> predicted behavior -> intervention`。
