"""RAG 问答链（含结构化数据查询路由）。

把「检索 → 拼装 Prompt → DeepSeek 生成 → 结构化返回」串成一条链路。

流程
----
1. 若问题命中数据类关键词（趋势/报废率/型号/表面处理/层数/面积等），
   优先走 :func:`src.data_tool.query_excel` 查询结构化 Excel 数据并生成图表；
2. 否则调用 :func:`src.retriever.search` 召回 Top-k 知识片段；
3. 拼装 Prompt，调用 DeepSeek（对齐项目 A 的 client.py 调用约定）生成答案；
4. 返回 ``{"answer", "sources", "image_path"}`` 结构。

防幻觉约束
----------
系统提示词中强制要求：检索内容无法回答问题时，必须明确回复「未找到相关信息」，
不得编造。当检索结果为**空**时，直接短路返回该固定话术，连 LLM 都不调用——
空上下文交给模型反而容易诱发它自行发挥。

用法::

    from src.qa_chain import answer_question
    result = answer_question("补投超过投料标准多少需要审批？")
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入以加载 .env
from src.data_tool import extract_factory, is_data_query, query_excel  # noqa: E402
from src.llm_client import DeepSeekClient  # noqa: E402
from src.retriever import search  # noqa: E402

# 固定话术，与系统提示词保持一致，作为检索为空时的短路返回。
NOT_FOUND_REPLY = "未找到相关信息"

# 意图分类提示词：关键词未命中时，由大模型做三分类兜底。
_INTENT_SYSTEM_PROMPT = (
    "你是意图分类器。判断用户问题属于以下哪一类，只输出一个词：\n"
    "- data：需要查询结构化数据/报表做统计（报废率、趋势、型号、表面处理、层数、"
    "工艺类型、面积、对比、各工厂的整体情况/表现/怎么样等）；\n"
    "- rag：询问规则、定义、编码含义、工单/物料编码规则等制度知识；\n"
    "- other：其他无法归类的问题。\n"
    "注意：提到工厂名并询问其整体情况/表现/怎么样时，属于 data。\n"
    "只输出 data / rag / other 三者之一，不要输出解释或标点。"
)

_SYSTEM_PROMPT = (
    "你是一名 PCB 生产投料与余数管理领域的知识问答助手。\n"
    "请严格依据下方【知识片段】回答用户问题，规则如下：\n"
    "1. 只使用【知识片段】中提供的信息，不要引入片段之外的知识或猜测。\n"
    "2. 回答使用简体中文，简洁、直接、条理清晰，尽量分点说明。\n"
    "3. 涉及编码、数字、百分比、工艺代码时，必须与片段原文一致，不得改写或估算。\n"
    "4. 如果检索到的内容不足以回答用户问题，请明确回答「未找到相关信息」，"
    "不要编造、不要转移话题、不要给出模棱两可的猜测。"
)


def _trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """只保留最近 max_turns 轮（1 轮 = 1 问 1 答 = 2 条），多余从最旧丢弃。

    同时只保留 ``role`` / ``content`` 字段，丢弃 UI 展示用的 sources/image_path 等。
    """
    if not history:
        return []
    cleaned = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return cleaned[-(max_turns * 2):]


def _build_messages(query: str, retrieved: list[tuple],
                    history: list[dict] | None = None) -> list[dict]:
    """把召回片段 + 多轮历史拼装成 messages，返回可交给 DeepSeekClient.chat 的列表。

    结构：``[system, ...历史消息, 当前 user]``。历史只取最近 MAX_HISTORY_TURNS 轮。
    """
    context_blocks = []
    for index, (doc, _score) in enumerate(retrieved, start=1):
        source = doc.metadata.get("source", "未知文件")
        header = doc.metadata.get("header", "")
        context_blocks.append(
            f"[{index}] 来源：{source}（{header}）\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_blocks)
    user_prompt = (
        f"【知识片段】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        "请根据上述知识片段回答问题。"
    )

    history_msgs = _trim_history(history or [], config.MAX_HISTORY_TURNS)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *history_msgs,
        {"role": "user", "content": user_prompt},
    ]


def _extract_sources(retrieved: list[tuple]) -> list[dict]:
    """把召回的 (Document, score) 列表转换为返回结构所需的 sources 列表。"""
    sources = []
    for doc, _score in retrieved:
        sources.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知文件"),
                "header": doc.metadata.get("header", ""),
            }
        )
    return sources


@lru_cache(maxsize=1)
def _get_client() -> DeepSeekClient:
    """惰性构建 DeepSeek 客户端（进程内缓存，避免每次提问都新建）。"""
    llm = config.get_llm_config()
    return DeepSeekClient(api_key=llm["api_key"] or "")


def _classify_intent_llm(query: str) -> str:
    """关键词未命中时，用大模型做三分类兜底。

    Returns:
        ``"data"`` / ``"rag"`` / ``"other"``。任何异常或无法解析时返回 ``"rag"``，
        保证服务不中断。
    """
    try:
        client = _get_client()
        response = client.chat([
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ])
        match = re.search(r"\b(data|rag|other)\b", response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "rag"
    except Exception:
        return "rag"


def _route_data(query: str) -> dict | None:
    """尝试走 data_tool；返回 None 表示 data_tool 无法处理（需回退 RAG）。"""
    result = query_excel(query, factory=extract_factory(query))
    if result.get("text"):
        return {
            "answer": result["text"],
            "sources": [],
            "image_path": result.get("image_path"),
        }
    return None


# Query Rewriting 提示词：把指代性 query 补全成独立 query，并强制保留业务实体。
_REWRITE_PROMPT = (
    "你是检索 query 改写器。以下是对话历史和用户的最新问题。\n\n"
    "要求：\n"
    "1. 把最新问题改写成一个独立、完整、不含指代的检索 query\n"
    "2. 必须保留历史中的业务实体（如「补投红线」、「金州厂」、「报废率」等）\n"
    "3. 只输出改写后的 query，不要任何解释、不要引号\n\n"
    "对话历史：\n{history}\n\n"
    "最新问题：{query}\n\n"
    "改写后的 query："
)


def _format_history_for_rewrite(history: list[dict], max_turns: int) -> str:
    """把历史格式化成「用户：… / 助手：…」文本，供改写 prompt 使用。"""
    trimmed = _trim_history(history, max_turns)
    lines = []
    for msg in trimmed:
        speaker = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{speaker}：{msg['content']}")
    return "\n".join(lines)


def _rewrite_query(query: str, history: list[dict],
                   client: DeepSeekClient | None = None) -> str:
    """用历史把指代性 query 补全成独立 query；任何异常/异常返回都降级用原始 query。

    降级条件（满足任一即返回原始 query）：
    - LLM 抛异常
    - 返回为空或只有空白
    - 返回长度超过原始 query 的 5 倍（防止失控）

    Args:
        query: 原始 query。
        history: 对话历史（非空才会被调用）。
        client: 可选，测试时注入；默认用 _get_client()。

    Returns:
        改写后的 query，或降级时的原始 query。
    """
    llm = client or _get_client()
    try:
        history_text = _format_history_for_rewrite(history, config.MAX_HISTORY_TURNS)
        prompt = _REWRITE_PROMPT.format(history=history_text, query=query)
        response = llm.chat([
            {"role": "system", "content": "你是检索 query 改写器。"},
            {"role": "user", "content": prompt},
        ])
        rewritten = (response or "").strip().strip('"').strip("'").strip()
        # 降级：空 / 纯空白 / 异常长
        if not rewritten:
            return query
        if len(rewritten) > len(query) * 5:
            return query
        return rewritten
    except Exception:
        return query


def _route_rag(query: str, top_k: int, history: list[dict] | None = None) -> dict:
    """走 RAG 流程（向量检索 + DeepSeek 生成）。

    多轮时：检索前先用 Query Rewriting 把指代性 query 补全成独立 query，
    提升召回准确率；拼 prompt 时仍用原始 query（保持对话自然）。
    单轮（history 为空）不触发改写。
    """
    search_query = query
    if history and config.ENABLE_QUERY_REWRITE:
        search_query = _rewrite_query(query, history)

    retrieved = search(search_query, top_k=top_k)
    if not retrieved:
        return {"answer": NOT_FOUND_REPLY, "sources": [], "image_path": None}
    messages = _build_messages(query, retrieved, history)
    answer = _get_client().chat(messages)
    return {"answer": answer, "sources": _extract_sources(retrieved),
            "image_path": None}


def answer_question(query: str, top_k: int = config.TOP_K,
                    history: list[dict] | None = None) -> dict:
    """回答单个问题，返回带来源（可含图片）的结构化结果。

    意图路由（关键词优先，LLM 兜底）：
    1. 命中数据关键词 / 分类对比意图 → 走 data_tool；
    2. 未命中 → 大模型三分类：data 走 data_tool，rag/other 走 RAG；
    3. 数据路由无法识别（text 为 None）、LLM 异常 → 均回退 RAG。

    Args:
        query: 用户问题。
        top_k: RAG 召回片段数，默认取 ``config.TOP_K``（4）。
        history: 多轮对话历史，``[{"role": "user"|"assistant", "content": "..."}]``。
            仅影响 RAG 生成路径；数据查询路径不拼接历史。默认 None 表示单轮。

    Returns:
        ``{"answer": str, "sources": [...], "image_path": str | None}``。

    Raises:
        ValueError: query 为空，或未配置 DEEPSEEK_API_KEY。
        FileNotFoundError: 尚未灌库（由 retriever.search 抛出）。
        requests.exceptions.RequestException: DeepSeek 调用失败。
    """
    if not query or not query.strip():
        raise ValueError("问题不能为空。")

    # 第一步：关键词路由（从 dimensions.yaml 聚合，新增维度自动同步）
    if is_data_query(query):
        data_result = _route_data(query)
        if data_result is not None:
            return data_result
        # 关键词命中但 data_tool 无法识别 → 回退 RAG

    # 第二步：LLM 兜底三分类
    intent = _classify_intent_llm(query)
    if intent == "data":
        data_result = _route_data(query)
        if data_result is not None:
            return data_result

    # 第三步：RAG（含 other 与异常兜底）
    if not config.get_llm_config()["api_key"]:
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY，无法调用大模型。\n"
            "请复制 .env.example 为 .env 并填入密钥。"
        )
    return _route_rag(query, top_k, history)


if __name__ == "__main__":
    demo_questions = [
        "补投超过投料标准多少需要审批？",
        "订单不齐数可以入余数吗？",
        "物料编码第 10 位是 H 代表什么表面处理？",
    ]
    for question in demo_questions:
        print(f"问：{question}")
        result = answer_question(question)
        print(f"答：{result['answer']}\n")
        print(f"   引用来源 {len(result['sources'])} 条:")
        for source in result["sources"]:
            print(f"   - {source['source']} [{source['header']}]")
        print("-" * 60)
