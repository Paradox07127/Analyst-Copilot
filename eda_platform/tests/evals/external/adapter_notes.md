# 外部基准抽样与适配说明(M4 eval 周 / M6.1 T3,D7 决策:抽样冒烟,只记分不追分)

> 状态:**任务样本已本地化**(DABStep hard 7 条 + KramaBench hard 5 条,均含标准答案);
> **上下文数据与跑分推迟**——两个基准的上下文数据体量大(DABStep 支付数据湖、KramaBench 各领域 data/ 树),
> 且跑分需要 live LLM,本环境无 key。推迟项与手动步骤见下文及 `docs/archive/2026-07/base/eda-agent-platform-m4-eval-report.md`。

## 1. 已落地样本

| 文件 | 来源 | 条数 | 含答案 | 选取规则 |
|---|---|---|---|---|
| `dabstep_hard_sample.json` | HF `adyen/DABstep` tasks/dev | 7(全部 hard) | 是 | dev split 全部 hard 任务(default split 答案对榜单隐藏,不可本地判分) |
| `kramabench_sample.json` | GitHub `mitdbg/Kramabench` workload/ | 5(全部 hard) | 是 | 跨 4 领域各取 1–2 条 hard,保留 subtasks(可作分步判分点) |

两个文件头部 `_meta` 均记录:来源 URL、抓取日期、选取规则、license 提示、未附带的上下文数据说明。
结构由 `test_external_benchmark_assets.py` 守卫(纯离线)。

## 2. 手动补齐上下文数据(用户执行)

DABStep(约 300MB,支付数据湖:`payments.csv`、`fees.json`、`merchant_data.json`、手册 markdown):

```bash
# 方式一:huggingface_hub(推荐,不进本仓库依赖;临时环境安装即可)
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('adyen/DABstep', repo_type='dataset', \
  allow_patterns=['data/context/*'], local_dir='./dabstep_context')"

# 方式二:datasets-server 只取任务不取上下文(样本文件即来源于此,无需重复)
curl 'https://datasets-server.huggingface.co/first-rows?dataset=adyen%2FDABstep&config=tasks&split=dev'
```

KramaBench(每个领域一棵 data/ 子树,见任务的 `data_sources` 字段):

```bash
git clone --depth 1 https://github.com/mitdbg/Kramabench
# 数据在 Kramabench/data/<workload>/ 下;评分器为 Kramabench/evaluate.py
```

## 3. 适配到本平台的映射(跑分时用)

- **DABStep** 任务形态 = 数据湖 + 自然语言问题 + 精确答案(格式由每题 `guidelines` 规定)。
  映射:上下文 CSV/JSON → 平台上传(J1 多文件入口);问题走 chat NL2SQL 链(`drivers/chat.run_chat_turn`)或
  M5 CodeAgent 开放式分析;答案判分 = 按 guidelines 格式化后精确匹配(与 NL2SQL harness 的结果集等价不同,
  DABStep 是单值/短文本 exact-match,建议直接字符串比对 + 数值容差 1e-6)。
- **KramaBench** 任务形态 = 多文件 data lake + 主问题 + 分步 subtasks(每步含期望答案)。
  映射:`data_sources` 全量上传 → 关系发现(J3)→ 主问题走 CodeAgent;subtasks 可作中间断言,
  非数值型答案按其 `answer_type`(`string_approximate` 等)用仓库自带 `evaluate.py` 的比较器。
- 两者都超出纯 SQL 范畴(需要文档阅读/多步推理/建模),预期得分口径是"记分观测",
  不纳入 M4 DoD(D7:只记分不追分)。

## 4. InsightEval(Eval-0 追加,2026-08-01)

**License 结论:不可 vendor。** 仓库 `zhenghaozhu23/InsightEval`(检查于 commit
`7bf3e160`,2026-07-16)无 LICENSE 文件,README §License 原文:
"No license has been selected for this repository yet. Add an explicit `LICENSE`
file before distributing the code or dataset for reuse." —— 明确禁止再分发,
因此本仓库不落任何实例(哪怕 10 条),只写 adapter + 下载说明。

下载(用户执行,~几 MB):

```bash
git clone --depth 1 https://github.com/zhenghaozhu23/InsightEval
# 数据:data/csvs/data_<id>.csv(+ 可选 data_<id>_sysuser.csv)
#       data/jsons/data_<id>.json(metadata/goal/insights_detail/insights/summary)
# 官方 loader:src/insighteval/dataset.py::load_instance(instance_id, data_dir)
```

接入 `exploration_baseline` harness 的映射:

- 每个 instance = 一个 item(`item_id = insighteval_<id>`,bucket `external`,
  capability suite;占位条目已在 `../exploration_baseline/suites/capability.json`,
  status `blocked_no_license`,拿到本地 clone 后改为指向 clone 路径即可)。
- 输入:`data/csvs/data_<id>.csv` 上传 + `goal` 作为 business_context;
  agent 走 auto-EDA/question 基线(与 planted 同一 adapter)。
- 判分:`insights_detail` 的 10 条参照 insight 为 ground truth;
  `data_type` 字面(Descriptive/Diagnostic/Predictive/Prescriptive/Evaluative/
  Exploratory)与 harness 的 `SIX_INSIGHT_FAMILIES` 逐字一致(六类覆盖直接可算)。
  逐条命中判定语义模糊,确定性 checker 只算:六类覆盖率、条数、grounding;
  insight 相似度匹配属 LLM-as-judge 范畴,按 R7 规则与确定性指标分开报告,不进硬门。
- 首轮建议子集:每个 category 取 1–2 条、难度覆盖 1–4,共 ≤10 instance,
  id 列表写在运行命令里(`--items insighteval_3,insighteval_17,...`),不落库。

## 5. BLADE(Eval-0 追加,只记 adapter notes,不 vendor)

- 来源:`behavioral-data/BLADE`(GitHub)。任务形态 = 研究问题 + 数据集 +
  专家标注的分析决策空间(conceptual variables / transforms / statistical model),
  评的是**分析决策轨迹**,不是单值答案。
- 判分器:仓库自带 LLM-assisted 匹配(把 agent 的分析决策与标注决策空间对齐),
  含 judge 成分 → 按 R7 只进 capability suite 的"记分观测",不进确定性硬门。
- 映射:数据文件走平台上传;research question 作为 business_context;
  agent 的 transform/model 选择从 receipt/trace 里抽取后喂 BLADE 的 matcher。
  在 E2 journal(工具调用账本)落地前,决策轨迹抽取不完整,故 Eval-0 只登记
  adapter,不跑分。
- DABStep 见 §1(已本地化 7 条 hard 样本);本轮不再动。

## 6. 推迟原因记录

- 无 live LLM key → agent 侧无法产生候选答案(与 NL2SQL live 跑分同因)。
- 上下文数据数百 MB,不宜进 git 仓库;按上文命令在用户机器上按需拉取。
- KramaBench 部分任务(astronomy 训模型类)超出平台当前 J 路径能力,首轮跑分建议先跑
  environment/legal/wildfire 三条 SQL-可达任务,astronomy 留给 CodeAgent 成熟后。
