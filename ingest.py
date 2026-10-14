"""向量化灌库入口脚本。

把 ``knowledge/`` 下的 Markdown 切块、向量化，写入本地 Chroma 持久化库。

用法::

    python ingest.py            # 增量：只重新向量化变化的文件（默认）
    python ingest.py --full     # 全量：清空集合重建

增量模式按源文件内容哈希（loader 写入每个 chunk 的 ``file_hash``）比对，只重新
切分+向量化变化的文件，用 ``upsert`` 就地更新（内容寻址的确定性 chunk id 保证
同内容幂等）；旧块标 ``deprecated=True`` 而非物理删除，检索时过滤，可回滚。

首次运行会从 HuggingFace 下载嵌入模型 BAAI/bge-small-zh-v1.5（约 95MB）。
国内网络请在 ``.env`` 中启用 ``HF_ENDPOINT=https://hf-mirror.com``。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入，使 .env 中的 HF_ENDPOINT 在下载模型前生效
from langchain_chroma import Chroma  # noqa: E402

from src.embeddings import build_embeddings  # noqa: E402
from src.loader import (  # noqa: E402
    _compute_file_hash,
    find_markdown_files,
    load_and_split,
    print_statistics,
    split_files,
)


def _get_store(embeddings) -> Chroma:
    """打开（必要时创建）Chroma 持久化集合。"""
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )


def _existing_file_hash(store: Chroma, source_name: str) -> str | None:
    """返回该 source 当前「活跃」块的 file_hash；无活跃块返回 None。

    只认 deprecated=False 的块，避免把已失效旧版本的哈希当成当前版本。
    """
    result = store.get(where={"source": source_name}, include=["metadatas"])
    for meta in result.get("metadatas") or []:
        if meta and not meta.get("deprecated") and meta.get("file_hash"):
            return meta["file_hash"]
    return None


def _mark_deprecated(store: Chroma, source_name: str) -> None:
    """把该 source 的全部旧块标记 deprecated（软删除：不物理删，检索过滤，可回滚）。

    langchain 的 Chroma 封装未暴露 update，这里直接调 chromadb 原生集合的
    ``update``（仅更新 metadatas，不动 embeddings/documents）。
    """
    result = store.get(where={"source": source_name}, include=["metadatas"])
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []
    if not ids:
        return
    store._collection.update(
        ids=ids,
        metadatas=[{**(m or {}), "deprecated": True} for m in metas],
    )


def _dedupe_by_chunk_id(documents: list) -> tuple[list, list[str]]:
    """按 chunk_id 去重，返回 (文档, ids)；避免重复内容产生重复 id 使 upsert 报错。"""
    seen: set[str] = set()
    docs, ids = [], []
    for doc in documents:
        cid = doc.metadata["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        docs.append(doc)
        ids.append(cid)
    return docs, ids


def _ingest_incremental(store: Chroma, files: list[Path]) -> int:
    """增量灌库：只重新切分+向量化「变化/新增」的文件，旧块标 deprecated。"""
    changed = [f for f in files
               if _existing_file_hash(store, f.name) != _compute_file_hash(f)]

    # 磁盘上已删除的文件：把库里残留块标 deprecated（软删除）
    existing_sources = {
        m.get("source")
        for m in (store.get(include=["metadatas"]).get("metadatas") or [])
        if isinstance(m, dict) and m.get("source")
    }
    current_sources = {f.name for f in files}
    for source in existing_sources - current_sources:
        _mark_deprecated(store, source)

    documents: list = split_files(changed) if changed else []

    # 先标旧块 deprecated，再 upsert 新块；同内容同 id 会被 upsert 翻回 active。
    for f in changed:
        _mark_deprecated(store, f.name)
    if documents:
        docs, ids = _dedupe_by_chunk_id(documents)
        store.add_documents(docs, ids=ids)
        print(f"[增量] 重新向量化 {len(changed)} 个变化文件，写入 {len(docs)} 个切块")
    else:
        print("[增量] 无文件变化，跳过灌库")

    return len(store.get(include=["metadatas"])["ids"])


def _ingest_full(store: Chroma, files: list[Path]) -> int:
    """全量重建：清空集合，重新切分全部文件并写入。"""
    documents = load_and_split()
    print_statistics(documents, files)
    store.reset_collection()
    docs, ids = _dedupe_by_chunk_id(documents)
    store.add_documents(docs, ids=ids)
    return len(store.get(include=["metadatas"])["ids"])


def ingest(incremental: bool = True) -> int:
    """执行灌库流程。

    Args:
        incremental: True 增量（默认），只处理变化文件；False 全量重建。

    Returns:
        入库后集合中的实际向量条数。
    """
    # 注：灌库本身不调用 DeepSeek，此处校验是为了尽早暴露 .env 未配置的问题，
    # 避免灌完库到问答环节才发现缺 key。
    config.validate_config()

    files = find_markdown_files()
    if not files:
        raise FileNotFoundError(f"{config.KNOWLEDGE_DIR} 下没有 .md 文件。")

    print(f"\n正在加载嵌入模型 {config.EMBEDDING_MODEL_NAME} ...")
    print("（首次运行需下载约 95MB，请耐心等待）")
    embeddings = build_embeddings()
    store = _get_store(embeddings)

    if incremental:
        return _ingest_incremental(store, files)
    return _ingest_full(store, files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="向量化灌库（默认增量，--full 全量重建）")
    parser.add_argument("--full", action="store_true",
                        help="全量重建：清空集合后重新灌入全部文档")
    args = parser.parse_args()

    count = ingest(incremental=not args.full)
    print(f"\n灌库成功，共 {count} 条向量")
    print(f"持久化目录: {config.CHROMA_PERSIST_DIR}")
    print("\n下一步：运行 python src/retriever.py 测试检索效果")
