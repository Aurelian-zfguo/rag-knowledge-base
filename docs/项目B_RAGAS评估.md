# 项目B RAGAS 评估记录

> 本文档记录 RAG 系统的量化评估闭环：发现路由 bug → 修复 → 验证指标提升。
> 供后续重复评估、排查回归时查阅。

---

## 1. 评估架构

| 项 | 说明 |
|---|---|
| 评估入口 | `scripts/evaluate_ragas.py`（支持 `--limit N` 冒烟测试） |
| 评估数据集 | `tests/eval_dataset.json`（15 条，覆盖 制度定义/管控措施/编码规则/工艺类型 四类） |
| 评估指标 | Faithfulness、AnswerRelevancy、ContextPrecision |
| 评判模型（judge） | DeepSeek（`deepseek-chat`，走 OpenAI 兼容接口 `https://api.deepseek.com/v1`） |
| 嵌入模型 | 本地 `BAAI/bge-small-zh-v1.5` |
| 输出 | `evaluation_report.md`（总体得分 + 分项明细） |

评估流程：读数据集 → 每条问题走真实 RAG 链路（`qa_chain.answer_question()`）→ 组装 RAGAS 样本 → `evaluate()` 打分 → 写报告。

---

## 2. 依赖与环境坑（重要）

RAGAS 与 LangChain 1.x / Python 3.14 存在兼容性问题，已通过以下方式解决：

| 问题 | 解决方式 |
|---|---|
| `scikit-network` 在 Python 3.14 无预编译 wheel，源码编译失败 | `requirements.txt` 固定 `scikit-network==0.12.1`（有纯 Python wheel） |
| ragas 0.4.3 依赖 `langchain_community.chat_models.vertexai`，但新版 langchain-community 已移除该模块 | `scripts/evaluate_ragas.py` 内置 `_ensure_compat_shim()` 自动补占位 `ChatVertexAI`（评估只用 DeepSeek，不实际调用它） |
| `ragas.metrics.collections` 的新指标不兼容 `evaluate()` 的 `Metric` 校验 | 回退使用旧版 `ragas.metrics`（deprecated 但可用），脚本已注释说明 |

> 若换新 venv 跑评估，`_ensure_compat_shim()` 会在导入 ragas 前自动补齐 shim，无需手动处理。

---

## 3. 发现的 Bug 与修复

### 3.1 现象

评估时 #10 号问题三个指标全为 0：

```
Q10: 物料编码 100000008H0899A 中，第 10 位 H 和第 11–12 位 08 分别表示什么？
实际答案：未找到生产型号「100000008H0899A」的数据。
```

### 3.2 根因

问题被**意图路由误判为数据查询**：`100000008H0899A` 命中了 `data_tool.query_excel()` 里的「具体型号」正则，走 Excel 数据查询去找型号趋势，查不到就返回「未找到生产型号」，**根本没进 RAG 检索**。

这是「编码含义解释」和「型号数据趋势」两类意图在路由边界上的混淆，不是知识库或 Prompt 问题。

### 3.3 修复

在 `src/data_tool.py` 新增 `_is_definition_query()`，并在 `query_excel()` 入口加排除：

```python
def _is_definition_query(query: str) -> bool:
    has_code_context = ("编码" in query) or \
        (re.search(r"第\s*\d+[\s–-]*\d*\s*位", query) is not None)
    has_interpretation = any(k in query for k in ("什么", "表示", "代表", "含义", "定义"))
    return has_code_context and has_interpretation
```

判定逻辑：**「编码」或「第 N 位」上下文 + 解释措辞** 才触发，命中即返回 `None` 交给 RAG。

关键设计：纯数据查询「金州厂报废率是多少」因无「编码/第几位」上下文，**不会被误伤**（有单元测试锁定）。

---

## 4. 修复前后指标对比

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| Faithfulness（忠实度） | 0.9014 | **1.0000** | ↑ +0.0986 |
| AnswerRelevancy（答案相关性） | 0.7872 | **0.8718** | ↑ +0.0846 |
| ContextPrecision（上下文精度） | 0.8667 | **0.9333** | ↑ +0.0666 |

#10 号问题三个指标：`0.0000 → 1.0000 / 0.9061 / 0.9167`。

---

## 5. 结论与后续

- **结论**：RAG 系统的生成忠实度（Faithfulness 1.0）和上下文精度（0.93）均健康；本次瓶颈在于「意图路由边界」，已修复。
- **ContextPrecision 未到 1.0** 的原因：RAGAS 对「召回片段里混入相邻但略偏的块」敏感，属**指标特性，不是系统 bug**，暂不继续优化。
- 若后续再次评估：先确保 `.env` 配置好 `DEEPSEEK_API_KEY`，然后运行 `python scripts/evaluate_ragas.py` 即可复现。
