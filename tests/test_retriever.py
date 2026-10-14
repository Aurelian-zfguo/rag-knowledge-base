"""retriever 两阶段检索的单元测试：用假 reranker 覆盖重排与降级容错。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import config  # noqa: E402
from src import retriever  # noqa: E402


def _make_docs(n: int) -> list[tuple[Document, float]]:
    return [
        (Document(page_content=f"片段{i}", metadata={"source": f"doc{i}.md"}), float(i))
        for i in range(n)
    ]


class FakeStore:
    def __init__(self, candidates):
        self.candidates = candidates
        self.called_with = None
        self.last_filter = None

    def get(self, **kwargs):
        return {"ids": ["x"]}  # 非空

    def similarity_search_with_score(self, query, k, filter=None,
                                     where_document=None, **kwargs):
        self.called_with = (query, k)
        self.last_filter = filter
        return self.candidates[:k]


class TestSearchTwoStage(unittest.TestCase):
    def setUp(self):
        self.candidates = _make_docs(12)
        self.store = FakeStore(self.candidates)

    def test_rerank_path_recalls_then_reranks(self):
        def fake_rerank(query, candidates, top_k):
            # 精排：只保留偶数下标，证明走的是 rerank 结果
            return [(doc, -1.0) for doc, _ in candidates if int(doc.page_content[-1]) % 2 == 0][:top_k]

        with patch.object(retriever, "load_vectorstore", return_value=self.store), \
             patch.object(retriever, "_rerank_available", return_value=True), \
             patch.object(retriever, "rerank", side_effect=fake_rerank):
            config.ENABLE_RERANK = True
            result = retriever.search("测试问题", top_k=4)

        # 召回阶段应至少取 10 个候选
        self.assertEqual(self.store.called_with[1], max(10, 4 * 2))
        # 检索应带 deprecated=False 过滤（软删除）
        self.assertEqual(self.store.last_filter, {"deprecated": False})
        # 结果应来自 rerank
        self.assertEqual(len(result), 4)
        self.assertTrue(all(score == -1.0 for _, score in result))

    def test_rerank_unavailable_falls_back_to_vector_topk(self):
        with patch.object(retriever, "load_vectorstore", return_value=self.store), \
             patch.object(retriever, "_rerank_available", return_value=False):
            config.ENABLE_RERANK = True
            result = retriever.search("测试问题", top_k=4)

        # 降级：返回向量召回前 4 条，不调用 rerank
        self.assertEqual(len(result), 4)
        self.assertEqual([s for _, s in result], [0.0, 1.0, 2.0, 3.0])

    def test_rerank_disabled_by_config(self):
        with patch.object(retriever, "load_vectorstore", return_value=self.store), \
             patch.object(retriever, "_rerank_available", return_value=True):
            config.ENABLE_RERANK = False
            result = retriever.search("测试问题", top_k=4)

        self.assertEqual(len(result), 4)
        self.assertEqual([s for _, s in result], [0.0, 1.0, 2.0, 3.0])


class TestSearchValidation(unittest.TestCase):
    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            retriever.search("")


class _FakeEmbeddings:
    """确定性假嵌入：按长度派生向量，供真实 Chroma 集成测试使用。"""

    def embed_documents(self, texts):
        return [[float(len(t) % 7)] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7)]


def test_search_filters_deprecated_chunks(tmp_path):
    """端到端：deprecated=True 的失效块不参与检索（软删除生效）。"""
    store = Chroma(
        collection_name="test_filt",
        embedding_function=_FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents([
        Document(page_content="补投红线是 ±5%", metadata={
            "source": "a.md", "header": "h", "deprecated": False}),
        Document(page_content="补投红线是 ±10%", metadata={
            "source": "a.md", "header": "h", "deprecated": True}),
    ])

    with patch.object(retriever, "load_vectorstore", return_value=store), \
         patch.object(retriever, "_rerank_available", return_value=False):
        results = retriever.search("补投红线")

    assert len(results) == 1
    assert "5%" in results[0][0].page_content  # 只返回 active，deprecated 被过滤


def test_rrf_fusion_accumulates_sorts_dedupes():
    a = Document(page_content="A", metadata={})
    b = Document(page_content="B", metadata={})
    c = Document(page_content="C", metadata={})
    ranked = [
        [(a, 0.0), (b, 1.0)],   # 路1：A 第1名，B 第2名
        [(b, 0.0), (c, 1.0)],   # 路2：B 第1名，C 第2名
    ]
    fused = retriever.rrf_fusion(ranked, k=60)

    scores = {d.page_content: s for d, s in fused}
    assert len(fused) == 3                        # 去重：B 只出现一次
    assert fused[0][0].page_content == "B"        # B 两路命中 → 分最高
    assert abs(scores["B"] - (1 / 61 + 1 / 60)) < 1e-9  # 路1 rank1 + 路2 rank0
    assert abs(scores["A"] - 1 / 60) < 1e-9
    assert abs(scores["C"] - 1 / 61) < 1e-9


def test_hybrid_disabled_by_default(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_HYBRID", False)
    sentinel = [(Document(page_content="sentinel", metadata={}), 0.0)]

    def fake_search(query, top_k=config.TOP_K):
        return sentinel

    monkeypatch.setattr(retriever, "search", fake_search)
    assert retriever.hybrid_search("任意问题") is sentinel  # 未开启 → 委托纯向量 search


def test_hybrid_fallback_to_vector_when_bm25_raises(monkeypatch):
    from src import bm25 as bm25_module

    monkeypatch.setattr(config, "ENABLE_HYBRID", True)
    candidates = _make_docs(12)
    store = FakeStore(candidates)
    with patch.object(retriever, "load_vectorstore", return_value=store), \
         patch.object(retriever, "_rerank_available", return_value=False), \
         patch.object(bm25_module, "bm25_search", side_effect=ImportError("no rank_bm25")):
        result = retriever.hybrid_search("测试问题", top_k=4)

    # BM25 不可用 → 降级纯向量，返回 FakeStore 前 4 条
    assert len(result) == 4
    assert [s for _, s in result] == [0.0, 1.0, 2.0, 3.0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
