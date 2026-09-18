"""嵌入模型封装：BAAI/bge-small-zh-v1.5。

为什么单独成一个模块
--------------------
BGE 中文模型使用**非对称编码**：检索指令前缀只加在 query 侧，文档侧不加。
灌库（ingest.py，走文档侧）与检索（retriever.py，走查询侧）必须共用同一份配置，
否则两侧编码方式一旦不一致，向量落在不同分布上，召回率会显著下降。
把这份配置收敛到本模块，就不会出现两边各写一遍、改了一处漏了另一处的情况。

实现要点
--------
``HuggingFaceEmbeddings`` 提供两个独立入口：

- ``encode_kwargs``       -> ``embed_documents()`` 使用，**不加**前缀
- ``query_encode_kwargs`` -> ``embed_query()`` 使用，**加**前缀

注意 ``embed_query()`` 只在 ``query_encode_kwargs`` 非空时才用它，为空会回退到
``encode_kwargs``。因此前缀绝不能写进 ``encode_kwargs``，否则文档侧也会被加上前缀。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须早于 huggingface 相关库导入，使 HF_ENDPOINT 生效
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402


@lru_cache(maxsize=1)
def build_embeddings() -> HuggingFaceEmbeddings:
    """构建（并缓存）BGE 中文嵌入模型。

    首次调用会从 HuggingFace 下载约 95MB 的模型权重，耗时较长；
    ``lru_cache`` 保证同一进程内只加载一次。

    Returns:
        配置好非对称编码的 ``HuggingFaceEmbeddings`` 实例。
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        # 文档侧：仅做 L2 归一化，不加任何前缀
        encode_kwargs={"normalize_embeddings": True},
        # 查询侧：归一化 + BGE 检索指令前缀
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": config.BGE_QUERY_INSTRUCTION,
        },
    )


if __name__ == "__main__":
    print(f"正在加载嵌入模型 {config.EMBEDDING_MODEL_NAME} ...")
    model = build_embeddings()

    probe = "补投工单的编码规则"
    as_document = model.embed_documents([probe])[0]
    as_query = model.embed_query(probe)

    print(f"向量维度: {len(as_query)}")
    if as_document == as_query:
        print("✗ 异常：同一段文本的文档向量与查询向量完全相同，")
        print("  说明 BGE 检索指令前缀未生效，请检查 query_encode_kwargs 配置。")
        sys.exit(1)
    print("✓ 文档向量与查询向量不同，BGE 检索指令前缀已生效（非对称编码正常）")
