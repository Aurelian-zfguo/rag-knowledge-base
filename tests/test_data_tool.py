"""data_tool 纯函数与意图路由的单元测试（不触发重型数据加载）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_tool import (  # noqa: E402
    _detect_compare_dimension,
    _extract_model_attrs,
    _extract_surface_and_layer,
    _is_definition_query,
    _parse_factory,
    extract_factory,
    is_data_query,
    query_excel,
)


class TestSurfaceAndLayer(unittest.TestCase):
    def test_standard_model(self):
        self.assertEqual(_extract_surface_and_layer("100356985C0299A"), ("C", "02"))

    def test_layer_two_digit(self):
        self.assertEqual(_extract_surface_and_layer("100496918T1099A"), ("T", "10"))

    def test_other_surface_codes(self):
        self.assertEqual(_extract_surface_and_layer("100384132H0299A"), ("H", "02"))
        self.assertEqual(_extract_surface_and_layer("100466563E0899F"), ("E", "08"))

    def test_invalid_model_returns_empty(self):
        self.assertEqual(_extract_surface_and_layer(""), ("", ""))
        self.assertEqual(_extract_surface_and_layer("12345"), ("", ""))


class TestExtractModelAttrs(unittest.TestCase):
    def test_extracts_surface_layer_process_type(self):
        self.assertEqual(_extract_model_attrs("100000008H0899A"), ("H", "08", "99"))

    def test_extracts_other_process_types(self):
        self.assertEqual(_extract_model_attrs("100356985C0299A"), ("C", "02", "99"))
        self.assertEqual(_extract_model_attrs("100496918T1099A"), ("T", "10", "99"))

    def test_invalid_returns_empty(self):
        self.assertEqual(_extract_model_attrs(""), ("", "", ""))
        self.assertEqual(_extract_model_attrs("12345"), ("", "", ""))


class TestFactoryParsing(unittest.TestCase):
    def test_strips_suntak_prefix(self):
        self.assertEqual(_parse_factory("SUNTAK_金州厂"), "金州厂")

    def test_non_string(self):
        self.assertEqual(_parse_factory(None), "")


class TestRouting(unittest.TestCase):
    def test_data_keyword_detection(self):
        self.assertTrue(is_data_query("金州厂 C 表面处理的平均报废率是多少"))
        self.assertTrue(is_data_query("整体报废率趋势"))
        self.assertFalse(is_data_query("补投超过投料标准多少需要审批"))

    def test_definition_questions_not_routed_to_data(self):
        # 「层数是什么」「表面处理是什么」是定义类问题，应走 RAG，而非 Excel 查询
        self.assertFalse(is_data_query("层数是什么"))
        self.assertFalse(is_data_query("表面处理是什么"))


class TestQueryExcelFactoryFallback(unittest.TestCase):
    """「工厂整体情况」模糊问法应兜底到工厂整体统计，而非返回 None。"""

    def test_factory_overview_query_returns_overall(self):
        from unittest.mock import patch
        import pandas as pd

        fake_long = pd.DataFrame({
            "批次类型": ["本批", "本批"],
            "报废率": [10.0, 20.0],
        })
        with patch("src.data_tool.preprocess_history_data", return_value=fake_long), \
             patch("src.data_tool._query_factory_overall", return_value={
                 "text": "金州厂 整体平均报废率 15.00%", "image_path": "static/x.png"}):
            result = query_excel("金州厂情况如何", factory="金州厂")
            self.assertIn("15.00%", result["text"])
            self.assertEqual(result["image_path"], "static/x.png")


class TestDefinitionQueryDetection(unittest.TestCase):
    """含义解释类问题应识别为定义类，交给 RAG，而不是数据查询。"""

    def test_definition_queries_detected(self):
        self.assertTrue(_is_definition_query("物料编码 100000008H0899A 第 10 位 H 代表什么"))
        self.assertTrue(_is_definition_query("第 11–12 位 08 表示什么"))
        self.assertTrue(_is_definition_query("工单编码规则怎么定义"))

    def test_data_queries_not_misdetected(self):
        # 纯数据查询「是多少」无「编码/第几位」上下文，不应误判为定义类
        self.assertFalse(_is_definition_query("金州厂报废率是多少"))
        self.assertFalse(_is_definition_query("金州厂 C 表面处理的平均报废率"))
        self.assertFalse(_is_definition_query("金州厂 10 层板的报废率"))

    def test_factory_extraction(self):
        self.assertEqual(extract_factory("金州厂 C 表面处理的报废率"), "金州厂")
        self.assertEqual(extract_factory("珠海一厂 10 层板的报废率"), "珠海一厂")
        self.assertEqual(extract_factory("整体报废率"), None)


class TestCompareDimension(unittest.TestCase):
    """分类对比 vs 总体平均 的意图区分。"""

    def test_surface_compare_via_all_various_different_by(self):
        self.assertEqual(_detect_compare_dimension("所有表面处理方式的平均报废率"), "surface")
        self.assertEqual(_detect_compare_dimension("各种表面处理码的报废率对比"), "surface")
        self.assertEqual(_detect_compare_dimension("不同处理方式的报废率"), "surface")
        self.assertEqual(_detect_compare_dimension("按表面处理方式分组统计报废率"), "surface")

    def test_surface_compare_via_gege_and_bianma(self):
        # 「各个」「编码」等口语化表达也应触发表面处理分类对比
        self.assertEqual(_detect_compare_dimension("各个表面处理编码的报废率"), "surface")
        self.assertEqual(_detect_compare_dimension("金州厂列出各个表面处理编码的报废率"), "surface")
        self.assertEqual(_detect_compare_dimension("各表面处理的报废率"), "surface")

    def test_layer_compare(self):
        self.assertEqual(_detect_compare_dimension("不同层数的报废率对比"), "layer")
        self.assertEqual(_detect_compare_dimension("所有层数的平均报废率"), "layer")
        self.assertEqual(_detect_compare_dimension("比较各层数的报废率"), "layer")
        # 层数相关变体关键词
        self.assertEqual(_detect_compare_dimension("按层数分"), "layer")
        self.assertEqual(_detect_compare_dimension("按层统计报废率"), "layer")
        self.assertEqual(_detect_compare_dimension("分层统计报废率"), "layer")
        self.assertEqual(_detect_compare_dimension("各层的平均报废率"), "layer")
        self.assertEqual(_detect_compare_dimension("不同层的报废率"), "layer")

    def test_single_value_not_compare(self):
        # 单值查询不应触发分类对比
        self.assertIsNone(_detect_compare_dimension("金州厂 C 表面处理的平均报废率是多少"))
        self.assertIsNone(_detect_compare_dimension("金州厂 10 层板的报废率"))
        self.assertIsNone(_detect_compare_dimension("金州厂整体报废率趋势"))
        self.assertIsNone(_detect_compare_dimension("补投超过投料标准多少需要审批"))

    def test_compare_keyword_without_dimension(self):
        # 只有「对比/不同」等关键词、但没有表面处理/层数维度，不触发分类对比
        self.assertIsNone(_detect_compare_dimension("不同型号的报废率对比"))

    def test_layer_not_overridden_by_surface(self):
        # 「按层数分」含「按」这一对比关键词，但应命中层数而非表面处理
        self.assertEqual(_detect_compare_dimension("按层数分"), "layer")
        self.assertEqual(_detect_compare_dimension("按层数分组统计报废率"), "layer")

    def test_earlier_keyword_wins_when_both_dimensions_present(self):
        # 两类关键词同时出现时，以先出现的一类为准
        self.assertEqual(_detect_compare_dimension("按层数和表面处理统计报废率"), "layer")
        self.assertEqual(_detect_compare_dimension("按表面处理和层数统计报废率"), "surface")

    def test_process_type_compare(self):
        # 工艺类型维度（配置驱动）
        self.assertEqual(_detect_compare_dimension("各工艺类型的报废率对比"), "process_type")
        self.assertEqual(_detect_compare_dimension("按工艺类型统计报废率"), "process_type")
        self.assertEqual(_detect_compare_dimension("不同板件类型的报废率"), "process_type")
        # 具体工艺名称单独出现也应触发
        self.assertEqual(_detect_compare_dimension("金州厂 HDI 的报废率"), "process_type")
        self.assertEqual(_detect_compare_dimension("高频板的报废率"), "process_type")
        self.assertEqual(_detect_compare_dimension("镀金手指的报废率"), "process_type")

    def test_customer_level_compare(self):
        # 示例维度：客户等级（配置驱动，数据暂无该列）
        self.assertEqual(_detect_compare_dimension("各客户等级的报废率对比"), "customer_level")
        self.assertEqual(_detect_compare_dimension("按等级统计报废率"), "customer_level")


if __name__ == "__main__":
    unittest.main(verbosity=2)
