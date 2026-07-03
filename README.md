# Personal Memory OS

一个面向具身智能体、桌面智能体和环境智能体的本地记忆底座。Reachy Mini 只是第一类接入载体；后续输入也可以来自机器人、桌面 agent、车机、家居设备、可穿戴设备或传感器事件。

第一阶段只解决记忆，不绑定具体观察源；后续所有观察统一转成 `ObservationContext` / `SessionEvent` 后接入记忆系统。

## 设计目标

- 用树状结构组织用户记忆，而不是只堆向量碎片。
- 支持 session 记录和 commit，把短期事件沉淀为长期记忆。
- 每次写入生成 `memory_diff.json`，方便审计和回滚。
- 检索时生成 compact memory digest，通过 hook 注入给主模型。
- 第一版不依赖外部模型，后续可替换为 LLM 抽取和 embedding 检索。

## 模型边界

完整系统分三类模型职责：

- `Memory Extractor`：LLM，用于把 session、观察事件、行为日志整理成结构化记忆。
- `Embedding Model`：向量模型，用于把记忆和当前场景转成向量，做语义召回。
- `Reasoning Model`：主模型，用于读取 memory digest，理解当前状态，预测下一步行为，并决定提醒、询问或执行动作。

当前版本已经实现工程骨架：规则抽取 + JSON LLM extractor 协议 + SQLite FTS/LIKE 检索 + embedding index + hybrid retrieval + hook 注入。后续会继续接真实模型 provider。

## 记忆树

```text
memory-root/
├── user/
│   └── gulf/
│       ├── profile/
│       ├── preferences/
│       ├── habits/
│       ├── triggers/
│       ├── interventions/
│       ├── feedback/
│       ├── policies/
│       ├── events/
│       ├── cases/
│       ├── daily/
│       ├── episodes/
│       └── sessions/
└── index/
    └── memory.sqlite3
```

每条长期记忆是一个 Markdown 文件，带 JSON front matter：

```markdown
---
{"id":"...","type":"preference","path":"user/gulf/preferences/...","tags":["temperature"],"active_count":0,"hotness":0.2,"lifecycle_state":"warm"}
---
# 用户偏好

用户怕热，偏好空调 25 度。
```

## 快速开始

```bash
cd /Users/gulf/PycharmProjects/memoryOS
python3 main.py init --root ./memory-root --user gulf
python3 main.py add-memory --root ./memory-root --user gulf --type preference --title "空调偏好" --text "用户怕热，通常希望空调设置为 25 度。"
python3 main.py search --root ./memory-root --user gulf --query "天气热的时候用户喜欢什么"
python3 main.py hybrid-search --root ./memory-root --user gulf --query "天气热的时候用户喜欢什么"
python3 main.py digest --root ./memory-root --user gulf --query "室温 30 度，用户在电脑前并出汗，空调可控"
```

维护记忆：

```bash
python3 main.py update-memory --root ./memory-root --user gulf --id user/gulf/preferences/xxx.md --text "用户怕热，偏好空调 24-26 度。"
python3 main.py merge-memory --root ./memory-root --user gulf --target user/gulf/preferences/xxx.md --source user/gulf/events/yyy.md
python3 main.py delete-memory --root ./memory-root --user gulf --id user/gulf/events/obsolete.md
python3 main.py lifecycle-report --root ./memory-root --user gulf --limit 20
```

可变用户状态：

```bash
python3 main.py update-profile --root ./memory-root --user gulf --mode replace --text "用户怕热，工作时不喜欢频繁打扰。"
python3 main.py update-daily --root ./memory-root --user gulf --date 2026-07-01 --text "上午在电脑前工作，室温较高。"
python3 main.py record-event --root ./memory-root --user gulf --date 2026-07-01 --event-type ac_acceptance --text "用户出汗后接受了开空调。"
```

这里的生命周期不同：

- `profile/user-profile.md`：长期用户画像，持续修正。
- `daily/YYYY-MM-DD/behavior.md`：当天行为滚动摘要，用于短期预测。
- `events/*.md` 与 `daily/YYYY-MM-DD/events.jsonl`：可追溯事件证据，不轻易覆盖。

## 热/冷记忆生命周期

每条记忆都有生命周期元数据：

- `active_count`：被检索命中的次数。
- `last_accessed_at`：最近一次被检索命中的时间。
- `hotness`：根据命中次数和更新时间计算的热度分数。
- `lifecycle_state`：`hot`、`warm` 或 `cold`。

`search` 和 `digest` 命中记忆时会自动升温，并同步更新 Markdown front matter 与 SQLite 索引。`lifecycle-report` 会优先列出 cold 记忆，后续可以在这个入口上继续做冷记忆压缩、摘要、归档或删除候选审计。

## Embedding 与混合召回

当前已经有 embedding provider 边界和本地 `HashingEmbeddingProvider`。本地 provider 只用于开发和测试，不是语义模型；真实使用时应替换成 OpenAI-compatible embedding API。

`hybrid-search` 会合并：

- 关键词 FTS/LIKE 分数。
- embedding 相似度。
- `hotness` 热度。
- `memory_type` 权重。
- `confidence` 置信度。

Markdown/SQLite 仍然是事实源，embedding 表只是召回索引，可随时重建。

## API Provider

LLM 和 embedding 都按 OpenAI-compatible API 接入，不绑定具体厂商。

环境变量：

```bash
export MEMORYOS_LLM_BASE_URL="https://api.openai.com/v1"
export MEMORYOS_LLM_API_KEY="..."
export MEMORYOS_LLM_MODEL="..."

export MEMORYOS_EMBEDDING_BASE_URL="https://api.openai.com/v1"
export MEMORYOS_EMBEDDING_API_KEY="..."
export MEMORYOS_EMBEDDING_MODEL="..."
```

本地 OpenAI-compatible 服务也可以用同一套变量，例如 `http://localhost:8000/v1`。

使用 API extractor 写入记忆：

```bash
python3 main.py commit-session --root ./memory-root --user gulf --session demo --extractor api
```

使用 API embedding 做混合召回：

```bash
python3 main.py hybrid-search --root ./memory-root --user gulf --query "天气热的时候用户喜欢什么" --embedding-provider api
```

默认 `auto` 模式会在环境变量存在时使用 API，否则回落到本地规则/哈希实现，便于离线开发。

## Session 流程

```bash
python3 main.py add-message --root ./memory-root --user gulf --session demo --role user --text "记住：我怕热，空调一般开 25 度。"
python3 main.py add-message --root ./memory-root --user gulf --session demo --role assistant --text "我记下来了。"
python3 main.py commit-session --root ./memory-root --user gulf --session demo
```

第一版规则抽取支持：

- `记住：...`
- `记住: ...`
- `remember: ...`

后续可以把 `memoryos/services/memory/extractor.py` 里的 `RuleBasedExtractor` 替换成 `JsonLLMMemoryExtractor`。它不绑定具体模型 SDK，只要求 provider 返回 JSON memory operations。

## 代码结构

```text
memoryos/
├── domain/           # 业务对象和稳定规则：action、scene、memory、behavior、feedback、policy
├── usecases/         # 业务用例编排：episode、session、feedback、intervention
├── services/         # 领域服务：memory、retrieval、prediction、learning、policy
├── ports/            # 抽象端口：repository、index、provider、event
├── adapters/         # 技术实现：SQLite、filesystem、provider、本地 outbox
├── interfaces/       # CLI、API、agent hook 等外部入口
├── workers/          # 后台任务入口：feedback、memory、reindex、replay
├── observability/    # audit、trace、metrics、explanation
├── security/         # path safety、id validation 等安全工具
├── config/           # 配置和依赖装配
└── shared/           # 低层通用工具
```

当前仍然是本地 Markdown + SQLite 实现，但代码边界已经按生产生命周期拆开：

```text
Observation
  -> usecases/episode
  -> services/retrieval
  -> services/prediction
  -> usecases/intervention + services/policy
  -> usecases/feedback
  -> services/learning
  -> services/memory consolidation
```

## Hook 注入边界

`digest` 命令生成的内容用于注入主模型上下文：

```text
<personal-memory source="memoryos" format="digest">
- ...
</personal-memory>
```

这类注入内容不能再被当作新事实写回长期记忆，避免记忆污染和重复滚雪球。

当前规则抽取器会在 commit 时忽略 `<personal-memory ...>...</personal-memory>` 块。

## Episode 同步请求

最终应用不需要走 CLI，可以直接调用 `EpisodeProcessor`。一次 episode 的核心要求是：**观察一到，先预测；记忆写入可以在预测之后或后台完成**。

默认时序：

1. 基于当前场景和已有记忆预测用户下一步行为或需求。
2. 判断当前输入是否需要写入/更新记忆。
3. 记录 episode 结果，等待反馈。
4. 如果预测或干预不对，通过 feedback 写回修正信号。

```python
from memoryos import EpisodeProcessor, MemoryStore
from memoryos.workers.feedback_worker import FeedbackWorker

store = MemoryStore("./memory-root")
episode = EpisodeProcessor(store)

result = episode.process(
    user_id="gulf",
    episode_id="ep-1",
    scene="用户坐在电脑前很久，手边有烟盒。",
    messages=[{"role": "observation", "text": "记住：用户在电脑前久坐后可能想抽烟。"}],
    available_actions=["remind_no_smoking", "ask_user", "do_nothing"],
    memory_write_timing="after_prediction",
)

feedback = episode.record_feedback(
    user_id="gulf",
    episode_id="ep-1",
    feedback="prediction_wrong",
    reward=-1,
    actual_action="organize_desk",
    correction="用户只是整理桌面，不是准备抽烟。",
)

# record_feedback 只可靠记录反馈事件和 outbox，不直接做学习。
# 本地开发可以显式跑 worker；生产环境后续换成后台 worker 消费 outbox。
FeedbackWorker(store).process_pending(user_id="gulf")
```

CLI 中也可以显式消费 feedback outbox：

```bash
python3 main.py process-feedback --root ./memory-root --user gulf --limit 20
```

结构化观察可以直接传 `ObservationContext`。系统会从中生成：

- `scene`：给模型和审计看的可读观察文本。
- `retrieval_query`：给记忆召回和 behavior pattern 匹配用的稳定 query。
- `context_tags`：位置、活动、持续时间、时间段、信号、环境等标签。

```python
from memoryos import EpisodeProcessor, MemoryStore, ObservationContext

result = episode.process(
    user_id="gulf",
    episode_id="hot-room-1",
    observation=ObservationContext(
        raw_text="用户回到房间，说热，额头出汗。",
        location="room",
        activity="arrive_home",
        started_at="2026-07-01T18:25:00+00:00",
        observed_at="2026-07-01T19:05:00+00:00",
        signals=["hot", "sweating", "says_hot"],
        environment={"temperature": 30},
    ),
    available_actions=["turn_on_ac", "ask_user", "do_nothing"],
)
```

`result["ranked_candidates"]` 会返回多个用户行为候选及分数构成，`result["prediction"]` 是 top1 行为候选的简化结果，`result["intervention"]` 是系统动作选择结果。

这两层不要混在一起：

- `ranked_candidates`：预测用户下一步可能做什么。
- `intervention`：系统应该提醒、询问、执行还是不动作。

每个候选还会包含 `memory_evidence`，区分：

- `retrieval_weight`：记忆被召回的权重。
- `usage_weight`：记忆对当前候选预测的使用权重。

也就是说，召回到某条记忆不代表它一定强影响预测。

候选行为生成采用 pattern-first：优先从跨天聚合后的 `behavior_pattern` 生成候选；规则、记忆和 LLM 只作为样本不足时的补充。同一天重复几次不会直接形成高置信候选。干预动作由 `InterventionSelector` 单独选择，负反馈会先影响对应系统动作的选择，而不是默认改写记忆或污染行为候选。

`record_feedback()` 会先写入：

- `user/{user_id}/events/feedback_events.jsonl`
- `user/{user_id}/events/outbox_events.jsonl`
- 当前 episode 的 `feedback.jsonl`

学习更新由 `FeedbackWorker` 消费 outbox 后执行。这样用户请求线程只负责可靠记录事实，行为模式、策略统计、RL 校准和长期记忆沉淀都在学习阶段完成。

feedback 会拆成两条校准链路：

- `behavior_stats.json`：记录 `predicted_action -> actual_action`，用于下次同类场景调整行为候选的 `behavior_reward`。
- `policy_stats.json`：记录 `predicted_action -> recommended_intervention -> reward`，用于调整系统动作选择。

`behavior_stats.json` 使用三层场景签名做泛化：

- `exact`：完整 `context_tags`，完全相同场景权重最高。
- `semantic`：去掉 `temperature_30`、`duration_45_minutes` 这类精确值，保留 `hot_environment`、`duration_30m_plus` 等语义桶。
- `coarse`：只保留位置、活动和核心信号，作为弱兜底。

召回行为反馈时会返回 `match_level` 和 `match_weight`，排序使用 `weighted_behavior_reward`，避免相似场景被当成完全相同场景。

## 记忆权重层

每条记忆存储时就会带权重语义：

- `base_weight`：由 memory type 决定，画像/策略/偏好通常更高，一次性事件更低。
- `temporal_scope`：时间作用域，例如 `stable`、`rolling_7d`、`rolling_30d`、`episodic`。
- `evidence_count`、`positive_count`、`negative_count`：规律被多少样本支持。
- `effective_weight`：结合基础权重、时间衰减、证据强度、置信度后的当前有效权重。

例如最近天气热导致“出汗后需要开空调”，更像 `rolling_7d` 或 `rolling_30d` 规律；稳定身份画像则是 `stable`。预测时会优先使用当前有效权重高的记忆，而不是所有召回记忆同权。

这里的 `reward` 是后续 bandit/RL 的训练信号。预测错默认只调整预测/策略统计，不改记忆。只有 `corrects_memory=True` 时，才会把反馈当作事实纠正写入记忆。

`memory_write_timing` 支持：

- `after_prediction`：默认，先预测，再同步写入记忆。
- `deferred`：先返回预测，把记忆操作保存成 pending，之后后台调用 `commit_pending_memory()`。
- `before_prediction`：先写记忆再预测，只适合明确需要让本次事实立刻参与预测的特殊场景。

## 路线图

1. 稳定 schema、memory type、事件更新和热/冷生命周期。
2. 接入 LLM Memory Extractor，从 session 和 observation event 中抽取 `profile`、`preference`、`habit`、`trigger`、`event`、`feedback`、`policy`。
3. 接入 Embedding Index，SQLite/Markdown 继续作为主存储，向量索引只负责召回加速。
4. 做 hybrid retrieval：关键词 FTS + embedding + hotness + memory_type 权重 + 时间衰减。
5. 做预测层：当前场景 + 召回记忆 -> 行为倾向 -> 干预策略 -> 用户反馈 -> 记忆更新。

更完整的架构说明见 [docs/memory-architecture.md](docs/memory-architecture.md)。

存储机制和 memory type 划分见 [docs/storage-model.md](docs/storage-model.md)。权重设计必须基于“记忆怎么存、存了什么、来源是什么、是否由多次样本聚合而来”，不能只在召回后临时打分。

## 完整链路

下面是一条从观察到预测、干预、反馈、记忆更新的完整链路。它描述的是最终运行时的主流程，不绑定 Reachy Mini，也不绑定具体 VLM、传感器或桌面 agent。

### 1. 观察输入

所有外部输入先统一成 `ObservationContext` 或 session messages。

典型观察：

```python
ObservationContext(
    raw_text="用户回到房间，说热，额头出汗。",
    location="room",
    activity="arrive_home",
    started_at="2026-07-01T18:25:00+00:00",
    observed_at="2026-07-01T19:05:00+00:00",
    signals=["hot", "sweating", "says_hot"],
    environment={"temperature": 30, "humidity": 72},
)
```

系统会派生三类字段：

- `scene`：可读审计文本，给预测模型和 episode 记录使用。
- `retrieval_query`：给记忆召回和 behavior pattern 匹配使用。
- `context_tags`：位置、活动、时间段、持续时间、信号、环境桶，例如 `room`、`arrive_home`、`hot_environment`。

这里的关键点是：不要只保存一段自然语言。结构化字段决定后面能不能做稳定召回、统计和泛化。

### 2. Retrieval Layer

一次 episode 进入后，系统先进入 **Retrieval Layer**。这里不直接做“行为预测”，而是从不同历史来源取证据。

当前 retrieval 分三类：

```text
RetrievalOrchestrator
├── MemoryContextBuilder      # 召回长期/近期记忆上下文
├── BehaviorPatternStore      # 召回聚合后的行为模式和 actual_action 分布
└── BehaviorStats             # 召回过去预测是否命中的反馈统计
```

这三类都属于召回，只是召回对象不同：

- `memory_context`：召回用户画像、偏好、习惯、trigger、case、event、近期状态。
- `behavior_patterns`：召回过去相似场景下聚合形成的行为模式。
- `behavior_distribution`：召回同类场景下过去 `predicted_action -> actual_action` 的统计反馈。

后面的 prediction 层只消费这些 retrieval 结果，不直接读文件树。

#### 2.1 Memory context retrieval

`MemoryContextBuilder` 先构建一个 **memory context**。

这里更接近 OpenViking 的思路：长期记忆不是一堆向量碎片，而是树状结构里的不同层级上下文。召回时应该先按记忆类型和目录语义取上下文，再用 FTS / embedding / hotness 做补充排序。

推荐的召回顺序：

1. 固定读取稳定上下文：
   - `profile/user-profile.md`
   - 高优先级 `policy`
   - 明确偏好 `preference`
2. 读取短期状态：
   - 当天 `daily/YYYY-MM-DD/behavior.md`
   - 最近 episode 摘要
   - 最近 feedback / intervention 统计
3. 根据 `retrieval_query` 做相关召回：
   - `habit`
   - `trigger`
   - `case`
   - 相关 `event`
4. 对召回结果压缩成 digest：
   - 去重
   - 合并同主题记忆
   - 控制 token 长度
   - 保留路径和类型，方便审计

当前工程里的 `hybrid_search` 是候选召回子步骤，负责从 SQLite FTS / embedding index 中拿相关记忆；完整上下文由 `MemoryContextBuilder` 负责组装。

当前 `hybrid_search` 用到的信号包括：

- SQLite FTS / LIKE 关键词召回。
- embedding 相似召回。
- `hotness` 热度。
- memory type 权重。
- `confidence` 和 `effective_weight`。

更合理的边界是：

```text
memory tree
  -> stable context
  -> recent context
  -> relevant memory candidates
  -> compact memory digest
  -> behavior candidate generation / ranking
```

也就是说，记忆召回层只回答：

```text
当前场景下，哪些用户画像、偏好、习惯、触发条件、历史案例和近期状态应该被拿出来？
```

后面的行为预测层才回答：

```text
基于这些记忆，用户下一步可能做什么？
```

召回结果只是候选证据。被召回不等于一定参与预测，所以行为候选里会记录 `memory_evidence`：

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

`retrieval_weight` 表示为什么被召回；`usage_weight` 是行为预测层后续计算出来的，表示这条记忆是否真正支持当前候选行为。

当前代码已经通过 `MemoryContextBuilder` 做这一层组装：

- profile / policy / preference 这类稳定上下文不应该完全依赖 query 命中。
- daily / recent episode 这类近期上下文应该有独立入口。
- habit / trigger / case / event 才主要走相关召回。
- 输出是结构化 `memory_context` 和 compact digest，而不是单纯 top-k memory list。

`MemoryContextBuilder` 还会输出 `route_trace`，记录每类记忆为什么被读取、使用了什么策略、选中了多少条：

```json
{
  "memory_type": "habit",
  "strategy": "query_routed_relevant_memory",
  "reason": "behavior patterns related to the current observation",
  "selected_count": 2,
  "directory_abstract": "- hot room ac habit: ..."
}
```

这部分的目标是让召回可审计：

- `fixed_stable_context`：固定读稳定画像、策略和偏好。
- `recent_state_context`：读取近期反馈、干预和事件。
- `query_routed_relevant_memory`：对 habit、trigger、case、event 做相关召回。

digest 会先输出 route trace，再输出 Stable / Recent / Relevant 三段，并控制总长度，避免把整棵记忆树原样塞进模型上下文。

#### 2.2 Behavior pattern retrieval

系统不会在预测时直接扫描 raw episode。episode feedback 里的 `actual_action` 会先聚合成 behavior pattern：

```text
user/{user_id}/behavior/.overview.md
user/{user_id}/behavior/{domain}/.overview.md
user/{user_id}/behavior/{domain}/patterns/{semantic_signature}-{action}.json
```

匹配时使用 pattern 的 `retrieval_query`、`context_tags` 和分层 token，聚合相似场景下用户真实发生的 `actual_action`。

这部分参考 OpenViking 的分层召回思路，不只做一段文本相似度。behavior pattern 召回分三层：

- `exact`：完整 `context_tags` 完全一致。
- `semantic`：去掉 `temperature_30`、精确分钟数等高波动值，保留语义桶。
- `coarse`：只保留位置、活动和核心信号。

每个 behavior pattern 会带：

```json
{
  "action": "open_ac",
  "source": "behavior_pattern",
  "pattern_uri": "user/gulf/behavior/room/patterns/xxx-open_ac.json",
  "sample_count": 5,
  "distinct_days": 4,
  "evidence_confidence": 0.73,
  "similarity": 1.0,
  "match_level": "semantic",
  "match_weight": 0.7,
  "prediction_coefficient": 0.51
}
```

同一场景下的多个 action 会进入同一个 group，group 记录 `action_distribution`、`group_entropy`、`top_action_margin`，用于表达“这个场景到底是稳定行为，还是多个行为在竞争”。pattern metadata 会写入 `user/{user_id}/behavior/.pattern_index.sqlite`，召回先查索引再读 pattern detail。单个 pattern 的旧 evidence 会压缩进 `old_evidence_summary`，保留近期明细和 lifetime 统计。

这一步解决的是：

```text
相似场景下，用户稳定倾向于实际做什么？
```

例如连续多天都是：

```text
用户回房间 + 说热 + 出汗 -> actual_action=open_ac
```

那么新场景里会优先生成 `open_ac` 行为候选。

注意：raw episode 只作为审计和 pattern evidence 来源，不再作为预测召回输入。系统不会因为同一天重复出现几次就形成强规律，pattern 会结合样本量、跨天分布和 reward 计算 `evidence_confidence`。

#### 2.3 Behavior feedback retrieval

`behavior_stats.json` 记录预测行为和真实行为的关系：

```text
predicted_action -> actual_action
```

它只回答一个问题：

```text
这个场景下，过去预测这个用户行为准不准？
```

它不负责系统动作，也不负责修改长期记忆。

场景签名分三层：

- `exact`：完整 `context_tags`，完全相同场景权重最高。
- `semantic`：去掉温度、湿度、精确分钟数等高波动值，保留 `hot_environment` 等语义桶。
- `coarse`：只保留位置、活动和核心信号，作为弱兜底。

召回行为反馈时会返回：

```json
{
  "action": "open_ac",
  "match_level": "semantic",
  "match_weight": 0.7,
  "behavior_reward_score": 0.7,
  "weighted_behavior_reward": 0.49
}
```

这样 `temperature_30` 学到的规律可以泛化到 `temperature_31`，但不会被当成完全相同场景。

### 3. 生成用户行为候选

`CandidateGenerator` 只生成用户下一步行为候选，不生成系统动作。

候选来源包括：

- `behavior_pattern`：跨天聚合后的真实 `actual_action` 行为模式。
- `memory_*`：从长期记忆中召回的画像、偏好、习惯、trigger、case 等。
- `baseline`：没有强信号时默认继续当前活动。

`behavior_feedback` 不生成候选，只在 ranking 阶段校准已有候选的 `behavior_reward`。

候选示例：

```json
{
  "action": "open_ac",
  "need": "cooling",
  "prior": 0.72,
  "sources": ["behavior_pattern"],
  "score": 0.81
}
```

这里的 `action` 是用户行为或需求，不是 agent 的动作。

### 4. 行为候选排序

`CandidateRanker` 对用户行为候选打分。

当前特征包括：

- `candidate_prior`：候选自身先验。
- `scene_match`：当前场景是否直接支持该行为。
- `memory_support`：召回记忆是否支持该行为。
- `behavior_reward`：同类场景下该行为预测是否靠谱。
- `memory_hotness`：相关记忆近期是否经常被使用。

排序只回答：

```text
用户下一步最可能做什么？
```

它不考虑系统动作的打扰成本，也不考虑某个提醒是否礼貌。这些属于下一层。

### 5. 选择系统干预动作

`InterventionSelector` 在 top 行为候选出来后，再选择系统应该怎么做。

输入：

- top behavior candidate。
- `available_actions`。
- `policy_stats.json`。
- 打扰成本。

输出：

```json
{
  "action": "turn_on_ac",
  "predicted_action": "open_ac",
  "predicted_need": "cooling",
  "score": 0.82,
  "features": {
    "policy_reward": 0.8,
    "behavior_confidence": 0.85,
    "interruption_cost": 0.14
  }
}
```

例如：

- 预测用户要抽烟：系统动作可能是 `remind_no_smoking`。
- 预测用户需要降温：系统动作可能是 `turn_on_ac` 或 `ask_before_turning_on_ac`。
- 预测不确定：系统动作可能是 `ask_user` 或 `do_nothing`。

行为预测和系统动作必须分开。否则预测错了会污染干预策略，干预不合适也会污染行为预测。

### 6. 返回 episode 结果

一次 `EpisodeProcessor.process()` 会返回：

```json
{
  "scene": "...",
  "retrieval_query": "...",
  "context_tags": ["room", "arrive_home", "hot_environment"],
  "memory_context": {
    "stable_context": [],
    "recent_context": [],
    "relevant_memories": [],
    "digest": "..."
  },
  "retrieved_memories": [],
  "behavior_patterns": [],
  "behavior_distribution": [],
  "ranked_candidates": [],
  "intervention": {},
  "prediction": {},
  "memory_diff": {},
  "pending_memory_operations": []
}
```

核心读取方式：

- 看 `ranked_candidates` 判断行为预测是否合理。
- 看 `intervention` 判断系统动作选择是否合理。
- 看 `memory_evidence` 判断召回记忆是否被正确使用。
- 看 `behavior_patterns` 判断稳定行为模式是否参与了候选生成。
- 看 `behavior_distribution` 判断历史反馈是否真的参与了排序。
- 看 `memory_diff` 判断本次有没有写入或更新记忆。

### 7. 记忆写入

记忆写入和预测不是同一件事。

`memory_write_timing` 控制写入时序：

- `after_prediction`：默认。先预测，再同步写记忆。
- `deferred`：先返回预测，把写入操作存成 pending，后台再提交。
- `before_prediction`：先写记忆再预测，只适合明确要让本次事实立即参与预测的特殊情况。

长期记忆按类型进入不同目录：

- `profile`：稳定用户画像，持续修正。
- `preference`：显式偏好。
- `habit`：行为规律。
- `trigger`：场景触发信号。
- `intervention`：agent 做过什么。
- `feedback`：用户反馈。
- `policy`：权限和边界。
- `event`：一次性可审计事实。
- `case`：相似 episode 案例，作为可审计事实和 pattern evidence 的补充来源。

写入会生成 `memory_diff.json`，方便审计本次新增、更新、忽略了什么。

当前写入由 `MemoryUpdateService` 统一处理，`SessionManager` 和 `EpisodeProcessor` 都走同一套逻辑：

```text
MemoryOperation
    -> update_policy 归一化
    -> 按 memory_type 路由
    -> MemoryStore 写 Markdown + SQLite
    -> memory_diff.json
```

关键策略：

- `event` 永远按可审计证据追加。
- `profile` 写入固定用户画像文件，持续 patch/append。
- `policy` 默认需要显式用户意图，不允许模型随便新增权限边界。
- `habit` / `trigger` 先写成 `event` 证据，跨天证据达到阈值后才聚合成稳定规律。
- 聚合后的 `habit` / `trigger` 会更新 `evidence_count`、`positive_count` 和 `effective_weight`。

### 8. 反馈闭环

episode 结束后，调用 `record_feedback()`。

反馈里最重要的是两类字段：

```python
feedback = episode.record_feedback(
    user_id="gulf",
    episode_id="hot-room-1",
    feedback="accepted",
    reward=1,
    actual_action="open_ac",
)

worker_result = FeedbackWorker(store).process_pending(user_id="gulf")
```

`record_feedback()` 只做可靠入队：

- `feedback.jsonl`：episode 级别审计日志。
- `feedback_events.jsonl`：不可丢的反馈事实。
- `outbox_events.jsonl`：等待 worker 消费的学习事件。

`FeedbackWorker.process_pending()` 消费 outbox 后才会更新：

- `behavior_stats.json`：行为预测是否命中。
- `behavior/{domain}/patterns/*.json`：相似行为 pattern 和 action distribution。
- `policy_stats.json`：系统动作是否合适。
- `rl/policy_ledger.json`：低风险行为预测校准。
- `case` / `habit` 等长期记忆沉淀。

如果预测错了：

```text
predicted_action=smoke
actual_action=organize_desk
```

系统会降低同类场景下 `smoke` 的 `behavior_reward`，并把 `organize_desk` 作为行为反馈候选纳入后续排序。

如果系统动作不合适：

```text
predicted_action=smoke
recommended_intervention=remind_no_smoking
reward=-1
```

系统会降低 `smoke::remind_no_smoking` 的策略收益。下次仍然可能预测用户要抽烟，但不一定继续选择同样的提醒方式。

预测错默认不改长期记忆。只有 `corrects_memory=True` 且用户明确纠正事实时，才写 `memory_correction` 事件。

### 9. 当前已实现

- Markdown tree 作为主存储。
- SQLite 作为 FTS / metadata / embedding 索引。
- memory type schema 和默认权重。
- OpenViking 风格的 profile / preference / event / case 更新思路。
- hook digest 注入边界。
- LLM extractor API 边界。
- OpenAI-compatible LLM / embedding provider 边界。
- `RetrievalOrchestrator` 统一召回层。
- `MemoryContextBuilder` 树状记忆上下文召回。
- behavior pattern 分层召回和 actual_action 分布。
- behavior feedback 分层召回。
- `ObservationContext` 结构化观察。
- behavior pattern-first 候选生成。
- behavior / intervention 两层拆分。
- `behavior_stats.json` 行为反馈闭环。
- `policy_stats.json` 系统动作反馈闭环。
- exact / semantic / coarse 场景泛化。

### 10. 还需要完善

当前工程还处在记忆和预测骨架阶段，后续重点包括：

- 接真实 LLM Memory Extractor：从 observation/session 中抽取 profile、habit、trigger、event、case。
- 接真实 embedding model：替换本地 hash embedding，提高语义召回质量。
- 增加 query rewrite：把观察转成更适合召回的多路 query。
- 增强 memory consolidation：当前已有 habit/trigger 的 evidence-first 聚合入口，后续要增加更强的语义归并、冲突处理和 case 聚合。
- 增加冷记忆压缩：对低热度、低使用记忆做摘要、归档或降权。
- 增加权限策略：哪些动作能自动执行，哪些必须询问。
- 增加设备能力层：把 `turn_on_ac` 映射到真实空调、机器人或桌面 agent 动作。
- 增加离线评估集：固定 episode 样本，评估召回、候选、排序、干预是否变好。
- 增加可视化调试：查看某次预测用了哪些记忆、哪些历史 episode、哪些 feedback 统计。

最终目标不是“把所有事情交给一个大模型猜”，而是让模型、记忆、统计反馈和策略层各自负责清楚的部分：

```text
观察 -> 结构化上下文 -> retrieval layer -> 行为候选 -> 行为排序 -> 系统干预 -> 反馈 -> 行为/策略校准 -> 记忆更新
```
