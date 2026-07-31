# Question Quality 评审 Rubric(M4 eval 周 / M6.1 T3)

> 状态:底座就绪(rubric + judge 校准集);**人工评分与 LLM-judge 跑分待用户执行**(需 live LLM key 或人工时间)。
> 适用对象:问题发现引擎(`tools/question_discovery.py` 模板路 + `agents/question_agent.py` LLM 路)产出的 `QuestionCandidate`。
> 配套文件:`judge_calibration.json`(≥10 条校准题,含期望分与理由)、`test_question_quality_assets.py`(结构守卫)。

## 1. 评分维度(5 分制,四维)

每个候选问题按四个维度各打 1–5 分(整数),总分 = 四维算术平均(保留两位小数)。

### D1 可回答性 Answerability——"当前数据 + 平台能力能否直接回答?"

| 分 | 锚点 |
|---|---|
| 5 | 一条只读 SQL / 一次确定性分析即可完整回答;所需列、表、join 全部存在且已验证 |
| 4 | 可回答,但需要正确处理一个已知陷阱(去重、NULL、类型转换)或一次中置信 join |
| 3 | 大体可回答,但答案会带显著限定(样本太小、缺失率高、需要近似) |
| 2 | 只能回答问题的一部分;关键口径缺失或问题前提有方法论错误(如直接 join 已知重复键) |
| 1 | 现有数据无法回答(所需字段/事实不存在),或问题本身无明确可判定答案 |

### D2 业务价值 Business Value——"答案能改变什么决策?"

| 分 | 锚点 |
|---|---|
| 5 | 直接支撑一个明确决策(定价、预算分配、渠道取舍),且结论可能改变现状 |
| 4 | 对理解业务驱动因素有实质贡献,间接影响决策 |
| 3 | 常规监控类信息,有用但不改变行动 |
| 2 | 描述性 trivia,业务上"知道了也就知道了" |
| 1 | 同义反复 / 构造性恒等(如线性换算列之间的相关性)、或与业务完全无关 |

### D3 具体性 Specificity——"问题是否指明了对象、口径与范围?"

| 分 | 锚点 |
|---|---|
| 5 | 明确到列/实体/时间窗/输出格式,两个分析师看完写出的 SQL 语义一致 |
| 4 | 对象与口径明确,个别边界(如时间窗)留白但有惯例默认 |
| 3 | 对象明确但口径含糊("表现最好"未定义指标) |
| 2 | 只有方向没有对象("看看销售有什么规律") |
| 1 | 完全开放式("数据里有什么有趣的?"),或一题打包多个不相关子问题 |

### D4 数据支撑 Data Support——"数据的质量与结构撑得起这个答案吗?"

| 分 | 锚点 |
|---|---|
| 5 | 所需列完整、类型正确、join 已验证(高置信/multiplier≈1)、无质量红旗 |
| 4 | 有轻度质量问题(个别 NULL、需 try_cast)但不影响结论方向 |
| 3 | 中度风险:中置信 join、明显缺失、或样本量勉强 |
| 2 | 高风险:未验证 join、关键列缺失率高、或已知膨胀陷阱未在问题中处理 |
| 1 | 所需数据不存在,或问题建立在错误的数据前提上 |

## 2. 汇总与门槛

- **overall = mean(D1..D4)**,两位小数。
- **硬性否决**:D1 ≤ 2 或 D4 ≤ 2 时,无论 overall 多高,候选判为 `reject`(不进 top 榜、不自动执行)。
- 建议解释顺序:先给四维分,再给 overall,再给一句话理由(校准集同构)。
- 与确定性分的关系:rubric 是**人工/LLM 评审尺**,不替代 `QuestionScore` 里的确定性门(`data_availability`、`quality_risk`、`join_risk` 仍由引擎硬算,LLM 分只影响展示序——v2-plan §4.10 原则不变)。D4 与确定性分冲突时,以确定性分为准并记录分歧。

## 3. Judge 校准协议(LLM-as-judge 启用前置)

1. **校准集**:`judge_calibration.json`,12 条,覆盖 4 条真实模板路产出 + 8 条手工构造(含高分锚点与低分锚点;每条含 `expected_scores`、`expected_overall`、`rationale`)。
2. **通过线**:judge 在校准集上须满足——
   - ≥ 80% 条目 |judge_overall − expected_overall| ≤ 0.5;
   - 全部硬性否决项(expected 里 D1≤2 或 D4≤2 的条目)被 judge 同样判为 reject;
   - 高低锚点不倒挂(expected_overall ≥ 4.0 的条目,judge 均分须高于 expected_overall ≤ 2.5 的条目)。
3. **judge prompt 模板**(建议,live 时接 `core/llm.py` structured 输出):

   ```text
   你是数据分析问题的质量评审。给定数据集 schema 摘要与一个候选分析问题,
   按 rubric(四维 1-5 整数分:可回答性/业务价值/具体性/数据支撑)打分,
   输出 JSON: {"answerability": n, "business_value": n, "specificity": n,
   "data_support": n, "reject": bool, "reason": "一句话"}。
   不要臆造数据中不存在的列;数据支撑维度必须以给出的 schema 与质量摘要为准。
   ```

4. **校准未过**:调 prompt(加锚点例子)重跑;连续两轮未过 → 该模型不作 judge,回退纯人工。

## 4. 人工评分流程(**待用户执行**)

1. 取一次 live run 的 top-10 `QuestionCandidate`(auto-executed 3 条 + 榜单前 7 条)。
2. 单人过一遍四维打分(每条 ≤ 1 分钟,先 D1/D4 后 D2/D3),记录在 eval report 的评分表模板里(见 `docs/archive/2026-07/base/eda-agent-platform-m4-eval-report.md` §deferred)。
3. 与 judge 分对照:|Δoverall| > 1.0 的条目写一句分歧原因。
4. 验收参考线(M4 计划 §7 eval 周):top-10 人工均分 ≥ 3.5,且无 reject 项进入 auto-executed 三席。

## 5. 维度与既有信号的映射(评审时可参考)

| rubric 维度 | 引擎确定性信号 |
|---|---|
| 可回答性 | `data_availability`、模板是否带 `sql_template` |
| 业务价值 | (无确定性信号,评审主判;LLM 路 `llm_business_relevance` 仅参考) |
| 具体性 | 问题文本是否落到列名/分组/时间窗(模板路天然较高) |
| 数据支撑 | `quality_risk`、`join_risk`、所引 relation 的 `confidence` 与 validation 数字 |
