"""retriever 两阶段检索的单元测试：用假 reranker 覆盖重排与降级容错。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

    def get(self, **kwargs):
        return {"ids": ["x"]}  # 非空

    def similarity_search_with_score(self, query, k):
        self.called_with = (query, k)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
