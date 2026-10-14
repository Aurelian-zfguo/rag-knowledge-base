"""ingest 增量灌库的单测：锁「未变文件不重切 / 变化文件被 upsert / 旧块标 deprecated」。

用确定性假嵌入（FakeEmbeddings）避免下载真实模型；只测 ingest 的辅助函数与
公开入口的增量/全量分派，不真连 DeepSeek。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_chroma import Chroma  # noqa: E402

import config  # noqa: E402
import ingest as ingest_module  # noqa: E402
from ingest import (  # noqa: E402
    _existing_file_hash,
    _ingest_full,
    _ingest_incremental,
    _mark_deprecated,
)


class FakeEmbeddings:
    """确定性假嵌入：按文本长度派生向量，避免下载 95MB 真实模型。"""

    def embed_documents(self, texts):
        return [[float(len(t) % 7)] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7)]


def _write(knowledge_dir: Path, name: str, content: str) -> None:
    (knowledge_dir / name).write_text(content, encoding="utf-8")


def _make_store(tmp_path: Path) -> Chroma:
    return Chroma(
        collection_name="test_coll",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )


def _metas(store: Chroma, source: str) -> dict:
    return store.get(where={"source": source}, include=["metadatas"])


def test_full_ingest_assigns_ids_and_active_flags(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge_dir)
    _write(knowledge_dir, "a.md", "# A\n\n## a1\n\n内容甲\n")

    store = _make_store(tmp_path)
    count = _ingest_full(store, sorted(knowledge_dir.glob("*.md")))

    result = _metas(store, "a.md")
    assert count == len(result["ids"]) >= 1
    for m in result["metadatas"]:
        assert m["chunk_id"] and m["file_hash"]
        assert m["deprecated"] is False
        assert m["version"] == "1.0"


def test_incremental_skips_unchanged(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge_dir)
    _write(knowledge_dir, "a.md", "# A\n\n## a1\n\n内容甲\n")

    store = _make_store(tmp_path)
    files = sorted(knowledge_dir.glob("*.md"))
    _ingest_full(store, files)

    calls = []
    real_split = ingest_module.split_files

    def record_split(files_arg):
        calls.append([f.name for f in files_arg])
        return real_split(files_arg)

    monkeypatch.setattr(ingest_module, "split_files", record_split)

    before = _metas(store, "a.md")["ids"]
    count = _ingest_incremental(store, files)
    after = _metas(store, "a.md")["ids"]

    assert calls == []          # 未变文件不重切
    assert before == after      # 原有块未被改写
    assert count == len(before)


def test_incremental_reembeds_changed_and_deprecates_old(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge_dir)
    _write(knowledge_dir, "a.md", "# A\n\n## a1\n\n内容甲\n")
    _write(knowledge_dir, "b.md", "# B\n\n## b1\n\n内容乙\n")

    store = _make_store(tmp_path)
    _ingest_full(store, sorted(knowledge_dir.glob("*.md")))

    old_b_ids = set(_metas(store, "b.md")["ids"])
    _write(knowledge_dir, "b.md", "# B\n\n## b1\n\n内容乙改\n")  # 改 b.md
    files = sorted(knowledge_dir.glob("*.md"))

    calls = []
    real_split = ingest_module.split_files

    def record_split(files_arg):
        calls.append([f.name for f in files_arg])
        return real_split(files_arg)

    monkeypatch.setattr(ingest_module, "split_files", record_split)

    _ingest_incremental(store, files)

    assert calls == [["b.md"]]          # 只重切变化文件

    result = _metas(store, "b.md")
    deprecated_ids = {i for i, m in zip(result["ids"], result["metadatas"])
                      if m["deprecated"]}
    active_ids = {i for i, m in zip(result["ids"], result["metadatas"])
                  if not m["deprecated"]}
    assert deprecated_ids == old_b_ids  # 旧块标 deprecated
    assert active_ids and active_ids.isdisjoint(deprecated_ids)  # 新块 active

    a_metas = _metas(store, "a.md")["metadatas"]
    assert all(m["deprecated"] is False for m in a_metas)  # 未变文件不受影响


def test_deleted_file_marks_deprecated(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge_dir)
    _write(knowledge_dir, "a.md", "# A\n\n## a1\n\n内容甲\n")
    _write(knowledge_dir, "b.md", "# B\n\n## b1\n\n内容乙\n")

    store = _make_store(tmp_path)
    _ingest_full(store, sorted(knowledge_dir.glob("*.md")))

    (knowledge_dir / "b.md").unlink()  # 删除 b.md
    _ingest_incremental(store, sorted(knowledge_dir.glob("*.md")))

    b_metas = _metas(store, "b.md")["metadatas"]
    assert b_metas and all(m["deprecated"] is True for m in b_metas)


def test_ingest_public_api_incremental_default(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(config, "validate_config", lambda *a, **k: None)
    monkeypatch.setattr(ingest_module, "build_embeddings", lambda: FakeEmbeddings())
    _write(knowledge_dir, "a.md", "# A\n\n## a1\n\n内容甲\n")

    n_full = ingest_module.ingest(incremental=False)
    n_incr = ingest_module.ingest(incremental=True)  # 无变化 → 跳过
    assert n_full == n_incr >= 1
