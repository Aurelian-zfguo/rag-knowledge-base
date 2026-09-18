"""RAGAS 评估入口。

对 ``tests/eval_dataset.json`` 的每个问题，走 B 项目的 RAG 链路（retriever + qa_chain），
生成答案后，用 RAGAS 量化评估 Faithfulness / AnswerRelevancy / ContextPrecision。

评判模型（judge）用 DeepSeek（OpenAI 兼容接口）；嵌入用本地 BGE 模型。

用法::

    python scripts/evaluate_ragas.py              # 全量评估
    python scripts/evaluate_ragas.py --limit 3    # 只跑前 3 条（冒烟测试）

输出：``evaluation_report.md``（总体得分 + 分项明细）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入以加载 .env


def _ensure_compat_shim() -> None:
    """兼容 shim：ragas 0.4.3 导入 langchain_community.chat_models.vertexai，但新版
    langchain-community 已移除该模块。评估只用到 DeepSeek(ChatOpenAI)，这里补一个
    占位 ChatVertexAI 满足 isinstance 检查，不参与实际调用。"""
    import langchain_community
    from pathlib import Path

    pkg_dir = Path(langchain_community.__file__).parent
    target = pkg_dir / "chat_models" / "vertexai.py"
    if target.exists():
        return
    target.write_text(
        'from langchain_core.language_models.chat_models import BaseChatModel\n\n'
        'class ChatVertexAI(BaseChatModel):\n    pass\n',
        encoding="utf-8",
    )


def _build_judge_llm():
    """DeepSeek 评判模型（OpenAI 兼容接口），并绕过 DeepSeek 不支持的 n 参数。"""
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

    llm_cfg = config.get_llm_config()
    chat = ChatOpenAI(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        temperature=0,
    )
    return LangchainLLMWrapper(chat, bypass_n=True)


def _build_embeddings():
    """本地 BGE 嵌入模型，包装为 RAGAS 可用。"""
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from src.embeddings import build_embeddings
    return LangchainEmbeddingsWrapper(build_embeddings())


def _load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_samples(dataset: list[dict]) -> list:
    """跑真实 RAG 链路，构造 RAGAS 评测样本。"""
    from ragas.dataset_schema import SingleTurnSample
    from src.qa_chain import answer_question

    samples = []
    for item in dataset:
        result = answer_question(item["question"])
        retrieved = [s["content"] for s in result.get("sources", [])]
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result.get("answer", ""),
                retrieved_contexts=retrieved,
                reference=item["reference"],
                reference_contexts=item["reference_contexts"],
            )
        )
    return samples


def _run_evaluation(samples: list):
    # 注：ragas 0.4.3 中，ragas.metrics.collections 的新指标不兼容 evaluate() 的
    # Metric 校验，只能用旧版 ragas.metrics（deprecated 但可用）。
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    llm = _build_judge_llm()
    embeddings = _build_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
    ]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    return result


def _write_report(dataset: list[dict], result, path: Path) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    # 总体分在 result._repr_dict；分项在 result.to_pandas()
    overall = getattr(result, "_repr_dict", {})

    lines = [
        "# RAG 评估报告",
        "",
        f"- 评估样本数：{len(dataset)}",
        f"- 评判模型：DeepSeek（{config.get_llm_config()['model']}）",
        f"- 嵌入模型：{config.get_retrieval_config()['embedding_model']}",
        "",
        "## 总体得分",
        "",
        "| 指标 | 得分 |",
        "|---|---|",
    ]
    metric_names = {
        "faithfulness": "Faithfulness（忠实度）",
        "answer_relevancy": "Answer Relevancy（答案相关性）",
        "context_precision": "Context Precision（上下文精度）",
    }
    for name in metrics:
        value = overall.get(name)
        lines.append(f"| {metric_names[name]} | {value:.4f} |" if value is not None
                     else f"| {metric_names[name]} | — |")
    lines.append("")
    lines.append("## 分项明细")
    lines.append("")
    lines.append("| # | 问题 | Faithfulness | AnswerRelevancy | ContextPrecision |")
    lines.append("|---|---|---|---|---|")
    per_sample = result.to_pandas()
    for i, item in enumerate(dataset):
        row = [str(i + 1), item["question"]]
        for name in metrics:
            try:
                value = per_sample.iloc[i][name]
                row.append(f"{value:.4f}" if isinstance(value, (int, float)) else "—")
            except Exception:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条")
    parser.add_argument("--dataset", type=Path,
                        default=_PROJECT_ROOT / "tests" / "eval_dataset.json")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    _ensure_compat_shim()  # 必须先于 ragas 导入
    print(f"评估 {len(dataset)} 条样本 ...")
    samples = _build_samples(dataset)
    result = _run_evaluation(samples)

    report_path = _PROJECT_ROOT / "evaluation_report.md"
    _write_report(dataset, result, report_path)
    print(f"报告已写入：{report_path}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
