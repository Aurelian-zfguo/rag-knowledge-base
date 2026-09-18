"""知识文档加载与切片（方案 B：标题优先切块）。

处理流程
--------
1. 用 ``MarkdownHeaderTextSplitter`` 按 ``##`` 二级标题切分 ``knowledge/`` 下的所有
   Markdown，使每张对照表、每条管控措施天然成为一个完整切块，不会被字符数从中间硬切断；
2. 仅当某个二级标题下的内容超过 ``chunk_size`` 时，才用
   ``RecursiveCharacterTextSplitter`` 做二次切分。

每个切块携带的 metadata::

    {
        "source": "07-补投管控措施.md",         # 来源文件名
        "header": "第 1 条：投料标准与超标审批",  # 所属二级标题
        "title":  "补投管控措施",                # 所属一级标题
    }

单独运行本模块可打印切片统计，用于验证切块质量::

    python src/loader.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# 支持 `python src/loader.py` 直接运行：把项目根目录加入模块搜索路径。
# 否则 sys.path[0] 会是 src/ 而不是项目根，下面的 `import config` 将失败。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须早于 langchain 导入，使 .env 中的 HF_ENDPOINT 生效
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 切分依据的标题层级：一级标题仅作为 metadata 记录，实际切块发生在二级标题。
HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [("#", "title"), ("##", "header")]

# 中文优先的分隔符链。RecursiveCharacterTextSplitter 的默认分隔符只有
# ["\n\n", "\n", " ", ""]，对中文长句会直接退化到按单字硬切；补入中文句读可让
# 二次切分尽量落在句子边界上。
CHINESE_SEPARATORS: list[str] = [
    "\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", "",
]


def find_markdown_files(knowledge_dir: Path | None = None) -> list[Path]:
    """列出知识目录下的所有 Markdown 文件，按文件名升序。

    Args:
        knowledge_dir: 知识目录，默认取 ``config.KNOWLEDGE_DIR``。

    Returns:
        .md 文件路径列表。
    """
    directory = knowledge_dir or config.KNOWLEDGE_DIR
    return sorted(directory.glob("*.md"))


def _split_one_file(
    path: Path,
    header_splitter: MarkdownHeaderTextSplitter,
    fallback_splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[Document]:
    """切分单个 Markdown 文件，返回带完整 metadata 的切块列表。"""
    header_chunks = header_splitter.split_text(path.read_text(encoding="utf-8"))

    documents: list[Document] = []
    for chunk in header_chunks:
        # MarkdownHeaderTextSplitter 只回填 title/header，source 需自行补上。
        chunk.metadata["source"] = path.name
        # 一级标题下、首个二级标题之前的内容不带 header，回退用 title 兜底，
        # 避免下游 metadata["header"] 取值时 KeyError。
        chunk.metadata.setdefault("header", chunk.metadata.get("title", ""))

        if len(chunk.page_content) <= chunk_size:
            documents.append(chunk)
        else:
            # 超长块才二次切分；metadata 由 split_documents 自动复制到每个子块。
            documents.extend(fallback_splitter.split_documents([chunk]))

    return documents


def load_and_split(
    knowledge_dir: Path | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """加载知识目录下的全部 Markdown 并切分为文档块。

    Args:
        knowledge_dir: 知识目录，默认取 ``config.KNOWLEDGE_DIR``。
        chunk_size: 单块字符上限，超过则触发二次切分。
        chunk_overlap: 二次切分时相邻块的重叠字符数。

    Returns:
        切分后的 ``Document`` 列表。

    Raises:
        FileNotFoundError: 知识目录下没有任何 .md 文件。
    """
    files = find_markdown_files(knowledge_dir)
    if not files:
        directory = knowledge_dir or config.KNOWLEDGE_DIR
        raise FileNotFoundError(
            f"{directory} 下没有找到任何 .md 文件，无法切分。\n"
            "请先在该目录放入知识文档，再重新运行。"
        )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        # 保留标题行在正文中：标题往往含「表面处理码」「补投」这类关键检索词，
        # 剥离后会削弱该块的向量表征。
        strip_headers=False,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )

    documents: list[Document] = []
    for path in files:
        documents.extend(
            _split_one_file(path, header_splitter, fallback_splitter, chunk_size)
        )
    return documents


def print_statistics(documents: list[Document], files: list[Path]) -> None:
    """打印切片统计，用于人工核对切块质量。"""
    per_file = Counter(doc.metadata["source"] for doc in documents)
    # 同一 (source, header) 出现多次，说明该二级标题触发了二次切分。
    section_counts = Counter(
        (doc.metadata["source"], doc.metadata["header"]) for doc in documents
    )
    resplit = sum(1 for count in section_counts.values() if count > 1)
    lengths = [len(doc.page_content) for doc in documents]

    print(f"{'文件名':<34}{'切块数':>8}")
    print("-" * 44)
    for path in files:
        print(f"{path.name:<34}{per_file.get(path.name, 0):>8}")
    print("-" * 44)
    print(f"共处理文件数: {len(files)}")
    print(f"总切块数     : {len(documents)}")
    print(f"最长块字符数 : {max(lengths)}")
    print(f"平均块字符数 : {sum(lengths) // len(lengths)}")
    print(f"触发二次切分 : {resplit} 个二级标题（阈值 {config.CHUNK_SIZE} 字符）")


if __name__ == "__main__":
    md_files = find_markdown_files()
    chunks = load_and_split()
    print_statistics(chunks, md_files)

    print("\n===== 前 3 个切块预览 =====")
    for doc in chunks[:3]:
        meta = doc.metadata
        print(f"\n[source] {meta['source']}   [header] {meta['header']}")
        preview = doc.page_content[:120].replace("\n", " ⏎ ")
        print(f"[content] {preview}{'...' if len(doc.page_content) > 120 else ''}")
