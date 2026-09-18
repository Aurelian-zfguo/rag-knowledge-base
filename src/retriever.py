"""两阶段检索：向量召回 + Rerank 精排。

第一阶段用 BGE 嵌入模型从 Chroma 召回 Top-N 候选；第二阶段用
``src.reranker`` 对候选重排序，输出最终 Top-k。

关于 BGE 检索指令前缀
---------------------
本模块**不手工拼接**前缀字符串。前缀由 ``src.embeddings`` 通过
``HuggingFaceEmbeddings.query_encode_kwargs`` 配置，在 ``embed_query()`` 内部
由 sentence-transformers 自动加到查询文本前；文档侧走 ``encode_kwargs``，不加前缀。

Rerank 降级
-----------
若 ``config.ENABLE_RERANK`` 为 False，或 Rerank 模型加载失败，自动回退到纯向量
检索（直接返回召回阶段的前 Top-k），不影响现有功能。

用法::

    from src.retriever import search
    for doc, score in search("补投需要审批吗"):
        print(doc.metadata["source"], score)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须早于 huggingface 相关库导入
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src.embeddings import build_embeddings  # noqa: E402
from src.reranker import is_available as _rerank_available, rerank  # noqa: E402


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """打开已持久化的 Chroma 集合（结果缓存，避免重复加载嵌入模型）。

    Returns:
        指向本地持久化集合的 ``Chroma`` 实例。

    Raises:
        FileNotFoundError: 持久化目录不存在，说明尚未灌库。
    """
    if not config.CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"未找到向量库目录 {config.CHROMA_PERSIST_DIR}。\n"
            "请先运行 python ingest.py 完成灌库。"
        )

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )


def search(query: str, top_k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """两阶段检索：向量召回 Top-N → Rerank 精排 Top-k。

    Args:
        query: 用户的自然语言问题，无需自行添加 BGE 指令前缀。
        top_k: 最终返回的片段数量。

    Returns:
        ``(Document, score)`` 元组列表，按相关度从高到低排列。
        ``Document.metadata`` 含 ``source`` / ``header`` / ``title``。
        score 是**距离**，数值越小越相关。

    Raises:
        ValueError: query 为空。
        RuntimeError: 向量库为空，尚未灌库。
    """
    if not query or not query.strip():
        raise ValueError("检索内容不能为空。")

    store = load_vectorstore()
    if len(store.get(limit=1)["ids"]) == 0:
        raise RuntimeError(
            f"向量库集合 {config.COLLECTION_NAME} 为空。\n"
            "请先运行 python ingest.py 完成灌库。"
        )

    # 第一阶段：向量召回 Top-N（至少 10 个候选，供精排）
    recall_k = max(10, top_k * 2)
    candidates = store.similarity_search_with_score(query, k=recall_k)

    # 第二阶段：Rerank 精排（可降级）
    if config.get_rerank_config()["enabled"] and _rerank_available():
        return rerank(query, candidates, top_k)

    return candidates[:top_k]


if __name__ == "__main__":
    # 覆盖不同知识文档的抽样问题，用于人工核对召回是否命中预期来源。
    SAMPLE_QUERIES = [
        ("P 代表哪种表面处理工艺", "04-表面处理码对照表.md"),
        ("补投超过投料标准多少需要审批", "07-补投管控措施.md"),
        ("订单不齐数可以入余数吗", "08-余数净入库管控措施.md"),
        ("1 个 pnl 等于多少 pcs", "06-计量单位换算.md"),
        ("补投工单的编码第一位数字是什么", "02-工单编码规则.md"),
    ]

    print(f"向量库: {config.CHROMA_PERSIST_DIR}")
    print(f"集合名: {config.COLLECTION_NAME}\n")

    hit = 0
    for question, expected_source in SAMPLE_QUERIES:
        results = search(question)
        top_doc, top_score = results[0]
        ok = top_doc.metadata["source"] == expected_source
        hit += ok

        print(f"{'✓' if ok else '✗'} 问题: {question}")
        print(f"   预期来源: {expected_source}")
        print(f"   实际 Top1: {top_doc.metadata['source']}  "
              f"[{top_doc.metadata['header']}]  距离={top_score:.4f}")
        if not ok:
            print("   其余召回:")
            for doc, score in results[1:]:
                print(f"     - {doc.metadata['source']} [{doc.metadata['header']}] "
                      f"距离={score:.4f}")
        print()

    print(f"Top1 命中率: {hit}/{len(SAMPLE_QUERIES)}")
