"""loader 切分的单测：锁无标题降级 / 标题优先 / 中文句读不硬切。"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src import loader  # noqa: E402


def test_no_title_long_doc(tmp_path):
    """无 #/## 的长文档（> chunk_size）：不报错、无空块、块长 ≤ chunk_size、header 为空。"""
    paragraph = "这是没有标题的长文档里的一段说明，用于触发无标题降级的二次切分。"
    long_text = "\n\n".join(paragraph for _ in range(40))  # 约 1278 字，超过 1024
    f = tmp_path / "no_title.md"
    f.write_text(long_text, encoding="utf-8")

    docs = loader.split_files([f])

    assert docs, "无标题长文档不应切出空列表"
    assert all(doc.page_content.strip() for doc in docs), "不应产生空块"
    assert all(len(doc.page_content) <= config.CHUNK_SIZE for doc in docs), "块长应 ≤ chunk_size"
    assert all(doc.metadata.get("header") == "" for doc in docs), "无标题文档 header 应为空"


def test_with_title_split_by_h2(tmp_path):
    """有 ## 的文档：按二级标题切，header 记录标题。"""
    f = tmp_path / "titled.md"
    f.write_text("# 标题\n\n## 第一条\n\n内容甲\n\n## 第二条\n\n内容乙\n", encoding="utf-8")

    docs = loader.split_files([f])

    headers = {doc.metadata.get("header") for doc in docs}
    assert "第一条" in headers
    assert "第二条" in headers


def test_chinese_separators_not_hard_cut(tmp_path):
    """无标题、纯中文长文本：按句号切块，不按单字硬切。"""
    sentence = "这是用于验证中文句读切分的一句话，确保不会退化成按单字硬切。"
    long_text = sentence * 60  # 约 1680 字，超过 1024，触发二次切分
    f = tmp_path / "zh.md"
    f.write_text(long_text, encoding="utf-8")

    docs = loader.split_files([f])

    assert len(docs) > 1, "长文本应被二次切分"
    assert len(docs) < 10, "应按句号切成少数几块，而非按单字硬切上千块"
    assert all(len(doc.page_content.strip()) > 1 for doc in docs), "不应出现单字块"
