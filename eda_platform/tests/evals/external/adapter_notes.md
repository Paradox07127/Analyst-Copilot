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

## 4. 推迟原因记录

- 无 live LLM key → agent 侧无法产生候选答案(与 NL2SQL live 跑分同因)。
- 上下文数据数百 MB,不宜进 git 仓库;按上文命令在用户机器上按需拉取。
- KramaBench 部分任务(astronomy 训模型类)超出平台当前 J 路径能力,首轮跑分建议先跑
  environment/legal/wildfire 三条 SQL-可达任务,astronomy 留给 CodeAgent 成熟后。
