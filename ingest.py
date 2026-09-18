"""向量化灌库入口脚本。

把 ``knowledge/`` 下的 Markdown 切块、向量化，写入本地 Chroma 持久化库。

用法::

    python ingest.py

本脚本可重复运行：每次都会先清空旧集合再写入，因此修改知识文档后直接重跑即可，
不会产生重复向量。集合内容完全由 ``knowledge/*.md`` 决定，随时可重建。

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
from src.loader import find_markdown_files, load_and_split, print_statistics  # noqa: E402


def ingest() -> int:
    """执行灌库流程。

    Returns:
        入库后集合中的实际向量条数。
    """
    # 注：灌库本身不调用 DeepSeek，此处校验是为了尽早暴露 .env 未配置的问题，
    # 避免灌完库到问答环节才发现缺 key。
    config.validate_config()

    files = find_markdown_files()
    documents = load_and_split()
    print_statistics(documents, files)

    print(f"\n正在加载嵌入模型 {config.EMBEDDING_MODEL_NAME} ...")
    print("（首次运行需下载约 95MB，请耐心等待）")
    embeddings = build_embeddings()

    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )

    # 先清空再写入，保证脚本可重复运行而不累积重复向量。
    store.reset_collection()
    print(f"\n正在向量化 {len(documents)} 个切块并写入 Chroma ...")
    store.add_documents(documents)

    # 从库里读回真实条数，而不是直接用 len(documents)，确保写入确实成功。
    return len(store.get()["ids"])


if __name__ == "__main__":
    count = ingest()
    print(f"\n灌库成功，共 {count} 条向量")
    print(f"持久化目录: {config.CHROMA_PERSIST_DIR}")
    print("\n下一步：运行 python src/retriever.py 测试检索效果")
