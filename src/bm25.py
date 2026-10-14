"""BM25 关键词检索（混合检索的稀疏召回一路）。

设计
----
- 索引来源：直接从**已灌库的 Chroma 集合**取全部 documents + metadatas（过滤
  deprecated），与向量检索用同一批 chunk；绝不从原始 Markdown 重建，否则两路会
  对不齐。
- 中文分词：用 jieba 把文本切成词，BM25 对词项打分，专补向量召回对「物料编码
  第 10 位」这类精确词/编码的召回不足。
- 返回 ``(Document, rank)``：rank 为 0 起始名次，供 rrf_fusion 按名次计分，避免
  两路分数量纲不一致无法直接相加。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入以加载 .env
from langchain_core.documents import Document  # noqa: E402


def _tokenize(text: str) -> list[str]:
    """jieba 分词，去掉空白词。延迟导入，避免未安装时拖累其它模块。"""
    import jieba

    return [t for t in jieba.lcut(text) if t.strip()]


@lru_cache(maxsize=1)
def build_bm25_index() -> tuple:
    """从已灌库的 Chroma 集合建 BM25 索引（与向量检索同一批活跃 chunk）。

    Returns:
        ``(bm25, documents)`` —— ``BM25Okapi`` 实例 + 与之一一对应的
        ``Document`` 列表（顺序一致，便于按名次取回 Document）。

    Raises:
        FileNotFoundError / RuntimeError: 尚未灌库（由 load_vectorstore 抛出）。
    """
    from src.retriever import load_vectorstore  # 延迟导入，避免与 retriever 循环依赖

    store = load_vectorstore()
    result = store.get(include=["metadatas", "documents"])
    metas = result.get("metadatas") or []
    texts = result.get("documents") or []

    documents: list[Document] = []
    corpus: list[list[str]] = []
    for meta, text in zip(metas, texts):
        if meta and meta.get("deprecated"):
            continue  # 与向量检索一致：失效块不进索引
        documents.append(Document(page_content=text, metadata=meta or {}))
        corpus.append(_tokenize(text))

    from rank_bm25 import BM25Okapi  # 延迟导入，依赖缺失时此处报错，上层降级

    return BM25Okapi(corpus), documents


def bm25_search(query: str, top_k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """BM25 关键词召回 Top-k，返回 ``(Document, rank)``（rank 从 0 起）。

    用排名而非原始 BM25 分作为返回值：rrf_fusion 只关心名次。
    """
    bm25, documents = build_bm25_index()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(documents[i], float(rank)) for rank, i in enumerate(order)]


if __name__ == "__main__":
    # 独立跑一遍：打印索引规模 + 抽样查询的 Top-k，人工核对 BM25 召回质量。
    bm25, docs = build_bm25_index()
    print(f"BM25 索引规模：{len(docs)} 个 chunk（与向量库同一批）")

    for q in ("物料编码第 10 位 H 代表什么", "补投超过投料标准多少需要审批"):
        print(f"\n问：{q}")
        for doc, rank in bm25_search(q, top_k=3):
            src = doc.metadata.get("source", "?")
            header = doc.metadata.get("header", "")
            print(f"  #{int(rank) + 1} [{src}] {header}")
