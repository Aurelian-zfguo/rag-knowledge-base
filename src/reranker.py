"""Rerank 重排序封装：BAAI/bge-reranker-v2-m3。

设计
----
- **懒加载 + 单例**：模型约 1.2GB，首次调用 :func:`rerank` 或 :func:`preload` 时才加载，
  之后复用同一实例。api.py 通过 lifespan 钩子在启动时后台预加载，避免首个请求卡顿。
- **降级容错**：模型加载失败（内存不足/文件损坏/网络失败）不抛异常，置
  ``_available=False``，之后所有调用直接原样返回候选，不影响纯向量检索。
- **分数口径**：CrossEncoder 输出的是相似度（越大越相关），本模块统一转成**负数**
  （越小越相关），与 retriever 的 L2 距离口径一致，下游无需区分两套比较方向。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入以加载 .env

RERANK_MODEL_NAME = config.get_rerank_config()["model_name"]

_model = None           # 单例：CrossEncoder 实例
_available = None       # None=未尝试加载；True=可用；False=加载失败


def _ensure_loaded() -> bool:
    """懒加载模型，返回是否可用（只尝试一次，失败后不再重试）。"""
    global _model, _available
    if _available is not None:
        return _available
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL_NAME)
        _available = True
    except Exception:
        _model = None
        _available = False
    return _available


def preload() -> bool:
    """供 api.py 在服务启动时预加载，返回是否加载成功。"""
    return _ensure_loaded()


def is_available() -> bool:
    """模型是否可用（未加载时先尝试加载）。"""
    return _ensure_loaded()


def rerank(query: str, candidates: list[tuple], top_k: int) -> list[tuple]:
    """对候选片段重排序。

    Args:
        query: 用户问题。
        candidates: 向量召回阶段返回的 ``(Document, score)`` 列表。
        top_k: 精排后返回的条数。

    Returns:
        ``(Document, score)`` 列表，score 为负数（越小越相关）。
        模型不可用或重排过程异常时，原样返回 candidates 前 ``top_k`` 条。
    """
    if not candidates:
        return candidates
    if not _ensure_loaded():
        return candidates[:top_k]
    try:
        passages = [doc.page_content for doc, _score in candidates]
        pairs = [(query, passage) for passage in passages]
        # 相似度，越大越相关
        scores = _model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        # 转成负数，保持「越小越相关」的口径
        return [(doc, -float(score)) for (doc, _), score in ranked[:top_k]]
    except Exception:
        return candidates[:top_k]
