"""RAG 知识库问答 —— Streamlit 前端。

启动方式::

    streamlit run app.py

依赖前置条件：需先运行 ``python ingest.py`` 完成向量化灌库，否则问答会提示未找到
向量库。首次提问会加载嵌入模型并调用 DeepSeek，稍慢属正常现象。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入以加载 .env
import streamlit as st  # noqa: E402

from src.qa_chain import answer_question  # noqa: E402

# 聊天历史存放在 session_state 中；每条消息形如
# {"role": "user" | "assistant", "content": str, "sources": list[dict], "image_path": str|None}
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(sources: list[dict]) -> None:
    """以可展开列表展示召回片段及其所属文件 / 标题。"""
    with st.expander("📚 来源引用", expanded=False):
        if not sources:
            st.caption("本次回答未召回任何知识片段。")
            return
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}. {source['source']}** · `{source['header']}`"
            )
            st.text(source["content"])


def _render_image(image_path: str | None) -> None:
    """展示数据查询生成的图表；路径为相对项目根的 static/...png。"""
    if not image_path:
        return
    full = _PROJECT_ROOT / image_path
    if full.exists():
        st.image(str(full))
    else:
        st.caption(f"图表文件未找到：{image_path}")


st.set_page_config(page_title="投料与余数知识问答", page_icon="🔍", layout="wide")
st.title("🔍 投料与余数管理知识问答")
st.caption(
    "基于本地知识库检索增强生成（RAG）。"
    f"嵌入模型 {config.EMBEDDING_MODEL_NAME}，对话模型 {config.DEEPSEEK_MODEL}。"
)

# 输入区
with st.container():
    query = st.text_input(
        "输入你的问题",
        placeholder="例如：补投超过投料标准多少需要审批？",
        key="query_input",
    )
    submitted = st.button("提问", type="primary", use_container_width=True)

def _history_for_prompt() -> list[dict]:
    """从 session_state 提取 role + content，丢弃 sources/image_path 等 UI 字段。"""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("role") in ("user", "assistant")
    ]


if submitted and query and query.strip():
    with st.spinner("检索并生成中…"):
        try:
            result = answer_question(query.strip(), history=_history_for_prompt())
        except FileNotFoundError:
            result = {
                "answer": "尚未灌库，找不到向量数据库。请先在终端运行 `python ingest.py` 完成灌库。",
                "sources": [],
                "image_path": None,
            }
        except Exception as exc:  # noqa: BLE001  界面兜底，避免裸异常堆栈吓到用户
            result = {
                "answer": f"处理出错：{exc}",
                "sources": [],
                "image_path": None,
            }

    st.session_state.messages.append(
        {"role": "user", "content": query.strip()}
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "image_path": result.get("image_path"),
        }
    )

# 历史与答案展示区
if not st.session_state.messages:
    st.info("上方输入问题开始提问。示例：补投超过投料标准多少需要审批？")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            _render_image(message.get("image_path"))
            _render_sources(message.get("sources", []))
