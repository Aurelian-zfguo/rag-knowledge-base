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
        "version": "1.0",                        # 文档版本（frontmatter version:）
        "deprecated": False,                     # 失效标记（旧版置 True）
        "file_hash": "a1b2...",                  # 源文件内容哈希（增量比对）
        "chunk_id":  "c3d4...",                  # 内容寻址确定性 id（upsert 幂等）
    }

单独运行本模块可打印切片统计，用于验证切块质量::

    python src/loader.py
"""

from __future__ import annotations

import hashlib
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


def _compute_file_hash(path: Path) -> str:
    """返回文件的 MD5 内容哈希（十六进制小写），供增量灌库比对文件是否变化。"""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _read_version(text: str) -> str:
    """从 Markdown 头部 YAML frontmatter 读取 version，缺省返回 "1.0"。

    只识别文档最开头的 ``---`` ... ``---`` 块；没有 frontmatter 或没有
    ``version:`` 字段时返回默认值，不抛异常。当前 knowledge/ 下 9 篇文档
    均无 frontmatter，故版本全部缺省为 "1.0"。
    """
    if not text.startswith("---"):
        return "1.0"
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return "1.0"
    for line in lines[1:end]:
        if line.startswith("version:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            return value or "1.0"
    return "1.0"


def _make_chunk_id(file_hash: str, header: str, content: str) -> str:
    """生成内容寻址的确定性 chunk id：同一份内容永远得到同一个 id。

    这样 ingest 用 ``upsert`` 时，内容没变的块会被原地覆盖（幂等），内容变了的
    块得到新 id，旧的再被标记 deprecated 过滤掉。
    """
    return hashlib.md5(f"{file_hash}\x00{header}\x00{content}".encode("utf-8")).hexdigest()


def _is_headerless(chunk: Document) -> bool:
    """判断切块是否来自「无标题文档」：header 与 title 均为空。

    MarkdownHeaderTextSplitter 对没有 #/## 的文档会返回单个 header/title 均为空的
    整块；此时没有标题可当语义边界，需走「段落 → 句子 → 字符」降级阶梯。
    """
    return not chunk.metadata.get("header") and not chunk.metadata.get("title")


def _split_one_file(
    path: Path,
    header_splitter: MarkdownHeaderTextSplitter,
    fallback_splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[Document]:
    """切分单个 Markdown 文件，返回带完整 metadata 的切块列表。

    每个切块额外带 4 个 metadata（供增量灌库与检索过滤使用）：
        version     —— 文档版本，来自 frontmatter ``version:``，缺省 "1.0"
        deprecated  —— 失效标记，旧版本块置 True，检索时过滤
        file_hash   —— 源文件内容哈希，用于增量比对
        chunk_id    —— 内容寻址的确定性 id（同内容同 id），供 upsert 幂等
    """
    raw_text = path.read_text(encoding="utf-8")
    header_chunks = header_splitter.split_text(raw_text)

    documents: list[Document] = []
    for chunk in header_chunks:
        # MarkdownHeaderTextSplitter 只回填 title/header，source 需自行补上。
        chunk.metadata["source"] = path.name
        # 一级标题下、首个二级标题之前的内容不带 header，回退用 title 兜底，
        # 避免下游 metadata["header"] 取值时 KeyError。
        chunk.metadata.setdefault("header", chunk.metadata.get("title", ""))

        if len(chunk.page_content) <= chunk_size:
            documents.append(chunk)
            continue

        # 超长块二次切分。fallback_splitter 的 CHINESE_SEPARATORS 顺序即成本阶梯
        # 「标题 → 段落 → 句子 → 字符」的后三级：\n\n(段落) → \n → 中文句读
        # （。；！？，、）→ 固定长度 + CHUNK_OVERLAP 滑动窗口，尽量落在语义边界、
        # 不按单字硬切。
        if _is_headerless(chunk):
            # 无标题降级：原文无 #/##，整篇是一块、header 与 title 均为空，没有
            # 标题当语义边界，只能靠段落/句读/固定长度这条阶梯。
            # 更贵的语义切分/LLM 切分不实现，只作话术（能用便宜规则就不上模型）。
            documents.extend(fallback_splitter.split_documents([chunk]))
        else:
            # 有标题但某标题下超长：同一 fallback_splitter 二次切分。
            documents.extend(fallback_splitter.split_documents([chunk]))

    file_hash = _compute_file_hash(path)
    version = _read_version(raw_text)
    for doc in documents:
        doc.metadata["version"] = version
        doc.metadata["deprecated"] = False
        doc.metadata["file_hash"] = file_hash
        doc.metadata["chunk_id"] = _make_chunk_id(
            file_hash, doc.metadata.get("header", ""), doc.page_content)
    return documents


def split_files(
    files: list[Path],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """只切分给定文件列表（供增量灌库：只重新切分变化的文件）。"""
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

    return split_files(files, chunk_size, chunk_overlap)


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
