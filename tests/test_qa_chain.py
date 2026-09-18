"""qa_chain 的单元测试：验证防幻觉约束、意图路由（关键词命中 + LLM 兜底）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from src import qa_chain  # noqa: E402


def _make_doc(content: str, source: str = "07-补投管控措施.md",
              header: str = "第 1 条") -> Document:
    return Document(page_content=content, metadata={"source": source, "header": header})


class TestBuildMessages(unittest.TestCase):
    def test_system_prompt_contains_no_hallucination_instruction(self):
        messages = qa_chain._build_messages(
            "补投需要审批吗",
            [(_make_doc("偏离投料标准±5%需申请审批。"), 0.1)],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("未找到相关信息", messages[0]["content"])
        self.assertIn("不要编造", messages[0]["content"])

    def test_user_prompt_contains_context_and_question(self):
        doc = _make_doc("偏离投料标准±5%需申请审批。")
        messages = qa_chain._build_messages("补投需要审批吗", [(doc, 0.1)])
        user = messages[1]["content"]
        self.assertIn("07-补投管控措施.md", user)
        self.assertIn("偏离投料标准±5%", user)
        self.assertIn("补投需要审批吗", user)


class TestExtractSources(unittest.TestCase):
    def test_sources_structure(self):
        doc = _make_doc("内容A", source="04-表面处理码对照表.md", header="表面处理码")
        sources = qa_chain._extract_sources([(doc, 0.2)])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "04-表面处理码对照表.md")
        self.assertEqual(sources[0]["header"], "表面处理码")
        self.assertEqual(sources[0]["content"], "内容A")


class TestHistoryTrim(unittest.TestCase):
    """多轮历史裁剪：保留最近 N 轮、丢弃 UI 字段、超窗口从最旧丢弃。"""

    def _history(self, n_pairs):
        history = []
        for i in range(n_pairs):
            history.append({"role": "user", "content": f"问{i}"})
            history.append({"role": "assistant", "content": f"答{i}", "sources": [], "image_path": None})
        return history

    def test_keeps_recent_turns(self):
        # 3 轮（6 条）→ 全部保留，且 assistant 的 sources/image_path 被丢弃
        history = self._history(3)
        trimmed = qa_chain._trim_history(history, max_turns=3)
        self.assertEqual(len(trimmed), 6)
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(trimmed[0]["content"], "问0")
        # 丢弃 UI 字段
        self.assertNotIn("sources", trimmed[1])
        self.assertNotIn("image_path", trimmed[1])

    def test_trims_oldest_beyond_window(self):
        # 5 轮（10 条）→ 只保留最近 3 轮（6 条），最旧的 4 条被丢弃
        history = self._history(5)
        trimmed = qa_chain._trim_history(history, max_turns=3)
        self.assertEqual(len(trimmed), 6)
        self.assertEqual(trimmed[0]["content"], "问2")  # 最旧保留的是第 3 轮
        self.assertEqual(trimmed[-1]["content"], "答4")  # 最新是第 5 轮

    def test_empty_history(self):
        self.assertEqual(qa_chain._trim_history([], max_turns=3), [])
        self.assertEqual(qa_chain._trim_history(None, max_turns=3), [])


class TestBuildMessagesWithHistory(unittest.TestCase):
    def test_history_inserted_between_system_and_user(self):
        history = [
            {"role": "user", "content": "上一问"},
            {"role": "assistant", "content": "上一答"},
        ]
        messages = qa_chain._build_messages("当前问", [(_make_doc("片段"), 0.1)], history)
        # 结构：system, user(历史), assistant(历史), user(当前)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "上一问")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "上一答")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("当前问", messages[-1]["content"])

    def test_no_history_unchanged(self):
        messages = qa_chain._build_messages("当前问", [(_make_doc("片段"), 0.1)], None)
        # 无历史时仍是 system + user 两条
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")


class TestIntentClassification(unittest.TestCase):
    """LLM 兜底三分类的解析与降级。"""

    def test_llm_classifier_returns_data(self):
        class DataClient:
            def chat(self, messages):
                return "data"
        with patch.object(qa_chain, "_get_client", return_value=DataClient()):
            self.assertEqual(qa_chain._classify_intent_llm("帮我看看报废情况"), "data")

    def test_llm_classifier_returns_rag(self):
        class RagClient:
            def chat(self, messages):
                return "rag"
        with patch.object(qa_chain, "_get_client", return_value=RagClient()):
            self.assertEqual(qa_chain._classify_intent_llm("补投需要审批吗"), "rag")

    def test_llm_classifier_exception_falls_back_to_rag(self):
        class BoomClient:
            def chat(self, messages):
                raise RuntimeError("boom")
        with patch.object(qa_chain, "_get_client", return_value=BoomClient()):
            self.assertEqual(qa_chain._classify_intent_llm("任意问题"), "rag")


class TestAnswerQuestionRouting(unittest.TestCase):
    """意图路由：关键词命中 vs LLM 兜底。"""

    def _data_result(self):
        return {"answer": "数据查询结果", "sources": [], "image_path": "static/plot.png"}

    def _rag_client(self, answer="模拟回答"):
        class Client:
            def __init__(self):
                self.called_intent = False
                self.called_answer = False
            def chat(self, messages):
                system = messages[0]["content"] if messages else ""
                if "意图分类器" in system:
                    self.called_intent = True
                    return "rag"
                self.called_answer = True
                return answer
        return Client()

    def test_keyword_match_routes_to_data_without_llm(self):
        with patch.object(qa_chain, "_route_data", return_value=self._data_result()) as rd, \
             patch.object(qa_chain, "_route_rag") as rr:
            result = qa_chain.answer_question("帮我看看报废情况")
            self.assertEqual(result["answer"], "数据查询结果")
            rd.assert_called_once()
            rr.assert_not_called()

    def test_llm_fallback_routes_to_data(self):
        # 「金州厂表现怎么样」命中关键词「金州」「表现」→ 其实走关键词路径。
        # 这里用无数据关键词的问法，验证 LLM 兜底分类为 data 后走 data_tool。
        with patch.object(qa_chain, "is_data_query", return_value=False), \
             patch.object(qa_chain, "_classify_intent_llm", return_value="data"), \
             patch.object(qa_chain, "_route_data", return_value=self._data_result()) as rd, \
             patch.object(qa_chain, "_route_rag") as rr:
            result = qa_chain.answer_question("金州厂情况如何")
            self.assertEqual(result["answer"], "数据查询结果")
            rd.assert_called_once()
            rr.assert_not_called()

    def test_llm_fallback_routes_to_rag(self):
        with patch.object(qa_chain, "is_data_query", return_value=False), \
             patch.object(qa_chain, "_classify_intent_llm", return_value="other"), \
             patch.object(qa_chain, "_route_rag", return_value={
                 "answer": "RAG 答案", "sources": [], "image_path": None}) as rr:
            result = qa_chain.answer_question("今天天气怎么样")
            self.assertEqual(result["answer"], "RAG 答案")
            rr.assert_called_once()

    def test_keyword_match_but_data_tool_unhandled_falls_back_rag(self):
        with patch.object(qa_chain, "_route_data", return_value=None), \
             patch.object(qa_chain, "_route_rag", return_value={
                 "answer": "RAG 答案", "sources": [], "image_path": None}) as rr:
            # 命中数据关键词，但 _route_data 返回 None → 回退 RAG
            result = qa_chain.answer_question("帮我看看报废情况")
            self.assertEqual(result["answer"], "RAG 答案")
            rr.assert_called_once()


class TestEmptyContextShortCircuit(unittest.TestCase):
    def test_empty_context_short_circuits(self):
        # 检索为空时返回固定话术（通过 _route_rag 内部短路）
        with patch.object(qa_chain, "is_data_query", return_value=False), \
             patch.object(qa_chain, "_classify_intent_llm", return_value="rag"), \
             patch.object(qa_chain, "search", return_value=[]):
            result = qa_chain.answer_question("无关问题")
            self.assertEqual(result["answer"], "未找到相关信息")
            self.assertEqual(result["sources"], [])


class TestQueryRewrite(unittest.TestCase):
    """Query Rewriting：有历史改写、降级、保留实体。"""

    def _history(self):
        return [
            {"role": "user", "content": "补投的红线是什么"},
            {"role": "assistant", "content": "补投有后批在线等三条红线"},
        ]

    def _client(self, response):
        class FakeClient:
            def chat(self, messages):
                self.messages = messages
                return response
        return FakeClient()

    def test_rewrites_with_history_preserving_entity(self):
        client = self._client("补投红线的第二条具体是什么")
        result = qa_chain._rewrite_query("那第二条具体是什么", self._history(), client)
        self.assertEqual(result, "补投红线的第二条具体是什么")
        # prompt 里应包含历史中的实体
        user_prompt = client.messages[-1]["content"]
        self.assertIn("补投的红线是什么", user_prompt)
        self.assertIn("那第二条具体是什么", user_prompt)

    def test_empty_response_falls_back_to_original(self):
        result = qa_chain._rewrite_query("那第二条", self._history(), self._client("   "))
        self.assertEqual(result, "那第二条")

    def test_exception_falls_back_to_original(self):
        class BoomClient:
            def chat(self, messages):
                raise RuntimeError("boom")
        result = qa_chain._rewrite_query("那第二条", self._history(), BoomClient())
        self.assertEqual(result, "那第二条")

    def test_overlong_response_falls_back_to_original(self):
        # 超过原始 query 5 倍长度的返回 → 降级
        result = qa_chain._rewrite_query("那第二条", self._history(),
                                         self._client("长" * 100))
        self.assertEqual(result, "那第二条")


class TestRouteRagWithRewrite(unittest.TestCase):
    def test_rewrite_called_when_history_present(self):
        with patch.object(qa_chain, "_rewrite_query", return_value="改写后的query") as rw, \
             patch.object(qa_chain, "search", return_value=[(_make_doc("片段"), 0.1)]) as s, \
             patch.object(qa_chain, "_get_client") as gc:
            gc.return_value.chat.return_value = "答案"
            qa_chain._route_rag("那第二条", 4, history=[
                {"role": "user", "content": "补投红线"},
                {"role": "assistant", "content": "三条红线"},
            ])
            rw.assert_called_once()
            # 检索用的是改写后的 query
            s.assert_called_once()
            self.assertEqual(s.call_args[0][0], "改写后的query")

    def test_no_history_no_rewrite(self):
        with patch.object(qa_chain, "_rewrite_query") as rw, \
             patch.object(qa_chain, "search", return_value=[(_make_doc("片段"), 0.1)]) as s, \
             patch.object(qa_chain, "_get_client") as gc:
            gc.return_value.chat.return_value = "答案"
            qa_chain._route_rag("当前问", 4, history=None)
            rw.assert_not_called()
            self.assertEqual(s.call_args[0][0], "当前问")


if __name__ == "__main__":
    unittest.main(verbosity=2)
