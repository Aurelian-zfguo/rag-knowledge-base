"""结构化 Excel 数据查询与图表生成。

数据源
------
``data/CUX_历史报废率查询报表.xlsx`` —— 历史报废率宽表：每行 1 条当前批次 +
9 条历史批次（``历史订单号-1`` … ``历史订单号-9``），共 10 个批次点。

预处理（``preprocess_history_data``）把宽表展开为长表：每行一个批次，含标准列
``工厂 / 生产型号 / 订单号 / 订单面积 / 报废面积 / 入库面积 / 报废率 / 工单发放时间 /
批次类型 / 父订单号 / 原始行索引 / 表面处理码 / 层数``。

报废率口径
----------
- 单批报废率 = 报废面积 / 发放面积 × 100，其中发放面积 = 报废面积 + 入库面积。
- 近3批 / 近6批平均报废率 = 对应历史批次报废面积之和 / 发放面积之和 × 100。
- 近3批剔除最高报废率：去掉报废率最高的 1 个批次后，用剩余批次面积之和计算；
  近6批剔除最高报废率：去掉报废率最高的 2 个批次后计算；批次不足则不去除。

缓存
----
预处理结果缓存到 ``data/history_long.parquet``。源 Excel 更新后（mtime 更晚）自动
重新展开，避免重复处理。
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd
import yaml

import config  # noqa: E402

# 无界面环境使用非交互后端，避免在服务端弹窗
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 数据与输出目录
DATA_DIR = _PROJECT_ROOT / "data"
STATIC_DIR = _PROJECT_ROOT / "static"
CONFIG_DIR = _PROJECT_ROOT / "config"
HISTORY_FILE = DATA_DIR / "CUX_历史报废率查询报表.xlsx"
CACHE_FILE = DATA_DIR / "history_long.parquet"
DIMENSIONS_FILE = CONFIG_DIR / "dimensions.yaml"

# 历史批次数量
HISTORY_BATCHES = 9

# 表面处理码合法取值（物料编码第 9 位）
SURFACE_CODES = "CHLFTNPXJWESM"

# matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 长表标准列
LONG_COLUMNS = [
    "工厂", "生产型号", "订单号", "订单面积", "报废面积", "入库面积",
    "发放面积", "报废率", "工单发放时间", "批次类型", "父订单号",
    "原始行索引", "表面处理码", "层数", "工艺类型码", "客户等级",
]


# --------------------------------------------------------------------------- #
# 维度配置（config/dimensions.yaml 驱动）
# --------------------------------------------------------------------------- #
def _parse_markdown_label_mapping(md_path: Path) -> dict[str, str]:
    """解析 Markdown 对照表，建立「代码 → 中文名称」映射。

    支持 ``| 代码 | 名称 | 代码 | 名称 |`` 双列表格、单列表格，以及靠右对齐导致的
    空单元格（如 ``| | 88 | 任意阶互联 |``）。
    """
    mapping: dict[str, str] = {}
    if not md_path.exists():
        return mapping
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # 分隔行（|--|--|）跳过
        if any(cell and set(cell) <= {"-", ":"} for cell in cells):
            continue
        # 表头行跳过（cells 中同时含「代码」与「工艺类型」这类表头词）
        if "代码" in cells:
            continue
        # 顺次扫描：代码格（非空且非表头词）后紧跟名称格
        i = 0
        while i < len(cells) - 1:
            code, name = cells[i], cells[i + 1]
            if code and name and code != name and "工艺类型" not in code:
                mapping[code] = name
                i += 2
            else:
                i += 1
    return mapping


@lru_cache(maxsize=1)
def _load_dimensions() -> list[dict]:
    """加载维度配置，并预解析 label_mapping 为 ``label_map`` 字典。

    缓存结果，避免每次查询重复读盘、重复解析 markdown。
    """
    with DIMENSIONS_FILE.open(encoding="utf-8") as f:
        dims_config = yaml.safe_load(f) or {}
    dims = dims_config.get("dimensions", [])

    resolved: list[dict] = []
    for dim in dims:
        entry = dict(dim)
        entry["keywords"] = list(dim.get("keywords") or [])
        entry["exclusive_keywords"] = list(dim.get("exclusive_keywords") or [])
        mapping_file = dim.get("label_mapping")
        entry["label_map"] = {}
        if mapping_file:
            # 文件名由 config.resolve_knowledge_path 统一补全为知识库绝对路径
            md_path = config.resolve_knowledge_path(mapping_file)
            entry["label_map"] = _parse_markdown_label_mapping(md_path)
        resolved.append(entry)
    return resolved


# --------------------------------------------------------------------------- #
# 预处理：宽表 → 长表
# --------------------------------------------------------------------------- #
def _find_history_file() -> Path:
    """优先读 data/CUX_历史报废率查询报表.xlsx，找不到再兼容历史报废率汇总.xlsx。"""
    if HISTORY_FILE.exists():
        return HISTORY_FILE
    alt = DATA_DIR / "历史报废率汇总.xlsx"
    if alt.exists():
        return alt
    raise FileNotFoundError(
        "数据文件未找到，请确认 data/ 目录下存在对应文件"
        "（CUX_历史报废率查询报表.xlsx 或 历史报废率汇总.xlsx）。"
    )


def _extract_model_attrs(model: str) -> tuple[str, str, str]:
    """从生产型号提取（表面处理码, 层数, 工艺类型码）。

    料号形如 ``100356985C0299A``：

    - 表面处理码：第 10 位单个字母（C）
    - 层数：紧接其后的两位数字（02）
    - 工艺类型码：层数后的两位数字（99）

    解析失败时对应字段返回空字符串，由调用方转为「未知」。
    """
    if not model or not isinstance(model, str):
        return "", "", ""
    m = re.search(rf"([{SURFACE_CODES}])(\d{{2}})(\d{{2}})", model)
    if not m:
        return "", "", ""
    return m.group(1), m.group(2), m.group(3)


def _extract_surface_and_layer(model: str) -> tuple[str, str]:
    """兼容旧接口：仅返回表面处理码与层数。"""
    surface, layer, _ = _extract_model_attrs(model)
    return surface, layer


def _parse_factory(org) -> str:
    """从组织列提取工厂名：SUNTAK_金州厂 -> 金州厂。"""
    if not isinstance(org, str):
        return ""
    return org.replace("SUNTAK_", "").strip()


def _to_datetime(value):
    """当前批次工单发放时间转 datetime；历史批次返回 NaT。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        return pd.NaT


def _batch_scrap_rate(scrap_area, issue_area) -> float:
    """报废率百分比数值 = 报废面积 / 发放面积 × 100。"""
    if scrap_area is None or issue_area is None or issue_area == 0:
        return float("nan")
    return float(scrap_area) / float(issue_area) * 100.0


def _compute_metrics(group: pd.DataFrame) -> pd.Series:
    """对一个父订单（本批行 + 历史批次行）计算 4 项聚合报废率指标。"""
    hist = group[group["批次类型"] != "本批"].sort_values("批次类型")
    # 只取有面积的批次
    hist = hist[hist["报废面积"].notna() & hist["发放面积"].notna()]

    def aggregate(sub: pd.DataFrame):
        scrap = sub["报废面积"].sum()
        issue = sub["发放面积"].sum()
        return scrap / issue * 100.0 if issue and issue > 0 else float("nan")

    def trimmed(sub: pd.DataFrame, n_remove: int):
        if len(sub) <= n_remove:  # 批次不足则不去除
            return aggregate(sub)
        # 按报废率降序，剔除最高的 n_remove 个批次后，用剩余批次面积之和计算
        kept = sub.sort_values("报废率", ascending=False).iloc[n_remove:]
        return aggregate(kept)

    first3 = hist.head(3)
    first6 = hist.head(6)
    return pd.Series({
        "近3批平均报废率": aggregate(first3),
        "近3批剔除最高报废率": trimmed(first3, 1),
        "近6批平均报废率": aggregate(first6),
        "近6批剔除最高报废率": trimmed(first6, 2),
    })


def preprocess_history_data(factory: Optional[str] = None) -> pd.DataFrame:
    """读取宽表并按工厂展开为长表（带缓存）。

    Args:
        factory: 工厂名（如 ``金州厂``）。传入则仅保留该工厂；None 返回全部。

    Returns:
        长表 DataFrame，含 ``LONG_COLUMNS`` 全部列 + 4 项聚合报废率指标列
        （指标仅在本批行上有值，历史行置 NaN）。

    Raises:
        FileNotFoundError: 数据文件不存在。
    """
    if CACHE_FILE.exists():
        src_mtime = _find_history_file().stat().st_mtime
        cache_mtime = CACHE_FILE.stat().st_mtime
        if cache_mtime >= src_mtime:
            df = pd.read_parquet(CACHE_FILE)
            # 缓存 schema 校验：缺列（如新增了维度列）时失效重建，避免读到旧缓存
            if all(col in df.columns for col in LONG_COLUMNS):
                if factory:
                    df = df[df["工厂"] == factory]
                return df.reset_index(drop=True)

    # 读宽表
    wide = pd.read_excel(_find_history_file(), sheet_name=0)
    if "组织" not in wide.columns:
        raise ValueError("数据文件中缺少「组织」列，无法按工厂筛选。")

    rows: list[dict] = []
    for idx, row in wide.iterrows():
        factory_name = _parse_factory(row.get("组织"))
        model = str(row.get("生产型号", "") or "")
        surface_code, layer, process_type = _extract_model_attrs(model)
        # 第 13–14 位不是数字或为空 → 工艺类型码置为「未知」
        if not process_type:
            process_type = "未知"
        parent_order = row.get("本批订单号")
        release_time = _to_datetime(row.get("工单发放时间"))

        # 当前批次（batch_num = 0）
        issue_area = float(row.get("发放面积")) if pd.notna(row.get("发放面积")) else \
            float(row.get("报废面积") or 0) + float(row.get("入库面积") or 0)
        rows.append({
            "工厂": factory_name,
            "生产型号": model,
            "订单号": parent_order,
            "订单面积": row.get("订单面积"),
            "报废面积": row.get("报废面积"),
            "入库面积": row.get("入库面积"),
            "发放面积": issue_area,
            "报废率": _batch_scrap_rate(row.get("报废面积"), issue_area),
            "工单发放时间": release_time,
            "批次类型": "本批",
            "父订单号": parent_order,
            "原始行索引": idx,
            "表面处理码": surface_code,
            "层数": layer,
            "工艺类型码": process_type,
            "客户等级": None,  # 示例维度，当前数据暂无该字段
        })

        # 历史批次 1..9
        for n in range(1, HISTORY_BATCHES + 1):
            order_col = f"历史订单号-{n}"
            if order_col not in wide.columns:
                continue
            scrap = row.get(f"报废面积.{n}")
            inbound = row.get(f"入库面积.{n}")
            issue = (float(scrap) if pd.notna(scrap) else 0.0) + \
                    (float(inbound) if pd.notna(inbound) else 0.0)
            rows.append({
                "工厂": factory_name,
                "生产型号": model,
                "订单号": row.get(order_col),
                "订单面积": row.get(f"订单面积.{n}"),
                "报废面积": scrap,
                "入库面积": inbound,
                "发放面积": issue,
                "报废率": _batch_scrap_rate(scrap, issue),
                "工单发放时间": pd.NaT,  # 历史批次无工单发放时间
                "批次类型": f"历史-{n}",
                "父订单号": parent_order,
                "原始行索引": idx,
                "表面处理码": surface_code,
                "层数": layer,
                "工艺类型码": process_type,
                "客户等级": None,
            })

    long = pd.DataFrame(rows, columns=LONG_COLUMNS)

    # 按父订单号 + 原始行索引 计算聚合指标，回填到本批行
    metric_cols = ["近3批平均报废率", "近3批剔除最高报废率",
                   "近6批平均报废率", "近6批剔除最高报废率"]
    for col in metric_cols:
        long[col] = float("nan")
    for (parent, ridx), group in long.groupby(["父订单号", "原始行索引"]):
        metrics = _compute_metrics(group)
        mask = (long["父订单号"] == parent) & \
               (long["原始行索引"] == ridx) & (long["批次类型"] == "本批")
        for col in metric_cols:
            long.loc[mask, col] = metrics.get(col, float("nan"))

    # 写缓存
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(CACHE_FILE, index=False)

    if factory:
        long = long[long["工厂"] == factory]
    return long.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 查询：工厂 + 多维度过滤
# --------------------------------------------------------------------------- #
def _norm_factory(factory: Optional[str]) -> Optional[str]:
    if not factory:
        return None
    return str(factory).replace("SUNTAK_", "").strip()


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.2f}%"


def _save_plot(fig) -> str:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    # 毫秒级时间戳，避免同一秒内多张图互相覆盖
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = STATIC_DIR / f"plot_{ts}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(_PROJECT_ROOT))


def _bar_plot(categories: list[str], values: list[float], title: str,
              ylabel: str, show_labels: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(categories, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("分类")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if show_labels:
        for i, v in enumerate(values):
            if v is not None and not pd.isna(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        # 分组过多时不逐柱标注，避免标签互相重叠
        ax.tick_params(axis="x", labelsize=8, rotation=45)
    return _save_plot(fig)


def _line_plot(x: list[str], y: list[float], title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("批次类型")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)
    return _save_plot(fig)


def _facet_label(factory: Optional[str]) -> str:
    return factory if factory else "全工厂"


# 通用对比词：命中这些词 + 某维度的 keywords，才算分类对比意图
_COMPARE_KEYWORDS = ("所有", "各种", "各个", "各", "不同", "按", "对比", "比较")


def _first_keyword_position(query: str, keywords: list[str]) -> int:
    """返回关键词在 query 中最早出现的位置；都不存在时返回 -1。"""
    positions = [query.find(k) for k in keywords if k in query]
    return min(positions) if positions else -1


def _detect_compare_dimension(query: str) -> Optional[str]:
    """识别「分类对比」意图，返回维度 key（由 config/dimensions.yaml 定义）。

    Returns:
        维度 key（如 ``"surface"`` / ``"layer"`` / ``"process_type"`` /
        ``"customer_level"``）；不是分类对比意图时返回 None。

    匹配顺序（配置驱动）
    ------------------
    1. 先查各维度的 ``exclusive_keywords``：出现即直接返回该维度，无需通用对比词；
    2. 否则需命中通用对比词，再遍历各维度 ``keywords``，返回**最先出现**的维度。
    """
    dims = _load_dimensions()

    # 1) 专属关键词：出现即触发
    for dim in dims:
        if any(k in query for k in dim["exclusive_keywords"]):
            return dim["key"]

    # 2) 通用对比词 + 维度关键词，取最先出现者
    if not any(k in query for k in _COMPARE_KEYWORDS):
        return None

    best_key, best_pos = None, -1
    for dim in dims:
        pos = _first_keyword_position(query, dim["keywords"])
        if pos != -1 and (best_key is None or pos < best_pos):
            best_key, best_pos = dim["key"], pos
    return best_key


def _query_grouped(long: pd.DataFrame, dimension: str,
                   factory: Optional[str]) -> dict:
    """按维度 key 分组，计算各组平均报废率并生成分组柱状图。

    列名、展示名、label 映射均来自 ``config/dimensions.yaml``，新增维度无需改代码。
    """
    dim_cfg = next((d for d in _load_dimensions() if d["key"] == dimension), None)
    if dim_cfg is None:
        return {"text": f"未知维度：{dimension}", "image_path": None}

    column = dim_cfg["column"]
    dim_name = dim_cfg["display_name"]
    label_suffix = dim_cfg.get("label_suffix", "")
    label_map = dim_cfg.get("label_map", {})

    sub = long[long["批次类型"] == "本批"]
    if sub.empty or column not in sub.columns:
        return {"text": f"未找到 {_facet_label(factory)} 的{dim_name}数据。", "image_path": None}

    grouped = sub.groupby(column)["报废率"].mean().dropna().sort_values(ascending=False)
    if grouped.empty:
        return {"text": f"未找到可对比的{dim_name}数据。", "image_path": None}

    def _label(key) -> str:
        key_str = str(key)
        # 有映射用中文名，无映射回退原始码
        return label_map.get(key_str, key_str) + label_suffix

    labels = [_label(k) for k in grouped.index]
    values = [float(v) for v in grouped.values]

    title = f"{_facet_label(factory)}各{dim_name}平均报废率对比"
    # 分组较多时不逐柱标注，避免标签重叠
    image_path = _bar_plot(labels, values, title, "平均报废率(%)",
                           show_labels=len(labels) <= 12)

    max_label, max_val = labels[0], values[0]
    min_label, min_val = labels[-1], values[-1]
    n_cat = len(labels)
    text = (f"{_facet_label(factory)} 共涉及 {n_cat} 种{dim_name}，"
            f"最高为「{max_label}」{max_val:.2f}%，"
            f"最低为「{min_label}」{min_val:.2f}%。")

    # 分组过多（>20）：在图表下方用表格展示前 10 与后 10，避免图表拥挤
    if n_cat > 20:
        top = list(zip(labels[:10], values[:10]))
        bottom = list(zip(labels[-10:], values[-10:]))
        lines = ["\n分组过多，仅列出报废率前 10 与后 10：",
                 "| 排名 | 分组 | 平均报废率 |",
                 "|---|---|---|"]
        for rank, (lab, val) in enumerate(top, 1):
            lines.append(f"| {rank} | {lab} | {val:.2f}% |")
        for rank, (lab, val) in enumerate(bottom, n_cat - 9):
            lines.append(f"| {rank} | {lab} | {val:.2f}% |")
        text += "\n".join(lines)

    return {"text": text, "image_path": image_path}


def _query_surface_treatment(long: pd.DataFrame, code: str,
                             factory: Optional[str]) -> dict:
    sub = long[(long["表面处理码"] == code) & (long["批次类型"] == "本批")]
    if sub.empty:
        return {"text": f"未找到表面处理码为「{code}」的数据。", "image_path": None}
    mean_rate = sub["报废率"].mean()
    n = len(sub)
    path = _bar_plot([code], [mean_rate],
                     f"{_facet_label(factory)} {code} 表面处理平均报废率", "平均报废率(%)")
    text = (f"{_facet_label(factory)} 表面处理码「{code}」共 {n} 个批次，"
            f"平均报废率 {mean_rate:.2f}%。")
    return {"text": text, "image_path": path}


def _query_layer(long: pd.DataFrame, layer: str, factory: Optional[str]) -> dict:
    sub = long[(long["层数"] == layer) & (long["批次类型"] == "本批")]
    if sub.empty:
        return {"text": f"未找到层数为「{layer}」的数据。", "image_path": None}
    mean_rate = sub["报废率"].mean()
    path = _bar_plot([f"{layer} 层"], [mean_rate],
                     f"{_facet_label(factory)} {layer} 层板平均报废率", "平均报废率(%)")
    text = (f"{_facet_label(factory)} {layer} 层板共 {len(sub)} 个批次，"
            f"平均报废率 {mean_rate:.2f}%。")
    return {"text": text, "image_path": path}


def _query_area_threshold(long: pd.DataFrame, threshold: float,
                          factory: Optional[str]) -> dict:
    sub = long[(long["订单面积"] > threshold) & (long["批次类型"] == "本批")]
    if sub.empty:
        return {"text": f"未找到订单面积大于 {threshold} 平的数据。", "image_path": None}
    mean_rate = sub["报废率"].mean()
    n = len(sub)
    path = _bar_plot([f">{threshold}平"], [mean_rate],
                     f"{_facet_label(factory)} 面积>{threshold} 平均报废率", "平均报废率(%)")
    text = (f"{_facet_label(factory)} 订单面积大于 {threshold} 平共 {n} 个批次，"
            f"平均报废率 {mean_rate:.2f}%。")
    return {"text": text, "image_path": path}


def _query_model_trend(long: pd.DataFrame, model: str, factory: Optional[str]) -> dict:
    sub = long[long["生产型号"] == model]
    if sub.empty:
        return {"text": f"未找到生产型号「{model}」的数据。", "image_path": None}
    order = ["本批"] + [f"历史-{n}" for n in range(1, HISTORY_BATCHES + 1)]
    sub = sub.sort_values("批次类型", key=lambda s: s.map(
        lambda x: order.index(x) if x in order else len(order)))
    # 每个批次类型取均值（同型号可能有多行）
    grouped = sub.groupby("批次类型")["报废率"].mean().reindex(order).dropna()
    if grouped.empty:
        return {"text": f"生产型号「{model}」无有效报废率数据。", "image_path": None}
    path = _line_plot(list(grouped.index), [float(v) for v in grouped.values],
                      f"{_facet_label(factory)} 型号 {model} 报废率趋势", "平均报废率(%)")
    latest = grouped.get("本批", None)
    text = (f"{_facet_label(factory)} 型号「{model}」按批次类型趋势见上图。"
            f"本批平均报废率 {_pct(latest)}。")
    return {"text": text, "image_path": path}


def _query_factory_overall(long: pd.DataFrame, factory: Optional[str]) -> dict:
    sub = long[long["批次类型"] == "本批"]
    if sub.empty:
        return {"text": f"未找到 {_facet_label(factory)} 的数据。", "image_path": None}
    mean_rate = sub["报废率"].mean()
    n = len(sub)
    path = _bar_plot([_facet_label(factory)], [mean_rate],
                     f"{_facet_label(factory)} 整体平均报废率", "平均报废率(%)")
    text = (f"{_facet_label(factory)} 本批共 {n} 个批次，"
            f"整体平均报废率 {mean_rate:.2f}%。")
    return {"text": text, "image_path": path}


def _query_factory_trend(long: pd.DataFrame, factory: Optional[str]) -> dict:
    order = ["本批"] + [f"历史-{n}" for n in range(1, HISTORY_BATCHES + 1)]
    grouped = long.groupby("批次类型")["报废率"].mean().reindex(order).dropna()
    if grouped.empty:
        return {"text": f"未找到 {_facet_label(factory)} 的数据。", "image_path": None}
    path = _line_plot(list(grouped.index), [float(v) for v in grouped.values],
                      f"{_facet_label(factory)} 整体报废率趋势", "平均报废率(%)")
    text = (f"{_facet_label(factory)} 各批次类型平均报废率趋势见上图。")
    return {"text": text, "image_path": path}


def _is_definition_query(query: str) -> bool:
    """判断是否为「编码/字段含义解释类」问题，应交给 RAG 而非数据查询。

    触发条件：出现「编码」或「第 N 位」这类编码/字段上下文，且同时出现
    「什么/表示/代表/含义/定义」等解释措辞。纯数据查询如「金州厂报废率是多少」
    因不含「编码/第几位」上下文，不会被误判。
    """
    has_code_context = ("编码" in query) or \
        (re.search(r"第\s*\d+[\s–-]*\d*\s*位", query) is not None)
    has_interpretation = any(k in query for k in ("什么", "表示", "代表", "含义", "定义"))
    return has_code_context and has_interpretation


def query_excel(query: str, factory: Optional[str] = None) -> dict:
    """理解查询意图并返回结构化结果。

    Args:
        query: 用户问题。
        factory: 可选工厂名；未指定时查询全部工厂并注明「全工厂汇总」。

    Returns:
        ``{"text": ..., "image_path": ...}``。查询意图无法识别时返回
        ``{"text": None, "image_path": None}``，由调用方回退到 RAG 流程。
    """
    factory = _norm_factory(factory)

    # 含义解释类问题（如「物料编码 XXX 第 10 位 H 代表什么」）应走 RAG，
    # 不走 Excel 数据查询，避免被「具体型号」正则误判。
    if _is_definition_query(query):
        return {"text": None, "image_path": None}

    try:
        long = preprocess_history_data(factory)
    except FileNotFoundError as e:
        return {"text": str(e), "image_path": None}

    if long.empty:
        return {"text": f"未找到 {factory} 的数据。", "image_path": None}

    q = query.strip()

    # 0) 分类对比（所有/各种/不同/按 + 表面处理或层数；对比/比较 + 表面处理或层数）
    compare_dim = _detect_compare_dimension(q)
    if compare_dim:
        return _query_grouped(long, compare_dim, factory)

    # 1) 具体型号（15位料号，含表面处理码+两位层数）
    model_match = re.search(r"(\d{8,}[A-Z]\d{3,}[A-Z]?)", q)
    if model_match:
        return _query_model_trend(long, model_match.group(1), factory)

    # 2) 表面处理码
    surface_match = re.search(rf"([{SURFACE_CODES}])\s*表面处理", q) or \
        re.search(rf"表面处理\s*([{SURFACE_CODES}])", q) or \
        re.search(rf"([{SURFACE_CODES}])\s*处理", q)
    if surface_match:
        code = surface_match.group(1)
        return _query_surface_treatment(long, code, factory)

    # 3) 层数
    layer_match = re.search(r"(\d{1,2})\s*层", q)
    if layer_match:
        return _query_layer(long, layer_match.group(1), factory)

    # 4) 面积阈值
    area_match = re.search(r"(\d+(?:\.\d+)?)\s*平", q)
    if area_match:
        return _query_area_threshold(long, float(area_match.group(1)), factory)

    # 5) 工厂整体趋势
    if "趋势" in q:
        return _query_factory_trend(long, factory)

    # 6) 工厂整体 / 全工厂汇总
    if any(k in q for k in ("整体", "汇总", "平均报废率", "报废率")):
        return _query_factory_overall(long, factory)

    # 7) 工厂整体情况的模糊问法（如「金州厂情况如何」「金州厂表现怎么样」）
    #    兜底为工厂整体统计，避免落到 RAG 返回「未找到相关信息」。
    if any(k in q for k in ("情况", "表现", "怎么样", "如何")):
        return _query_factory_overall(long, factory)

    return {"text": None, "image_path": None}


# --------------------------------------------------------------------------- #
# 意图路由（供 qa_chain 调用）
# --------------------------------------------------------------------------- #
# 通用数据查询关键词（与维度无关，保留在代码中）。
# 各维度的触发关键词（表面处理/层数/工艺类型等）不再手动维护在这里，
# 而是由 build_data_keywords() 从 config/dimensions.yaml 自动聚合。
DATA_GENERIC_KEYWORDS = (
    "趋势", "统计", "报废率", "报废情况", "型号", "面积",
    "最近", "表现", "工厂", "金州", "珠海", "平均", "汇总", "整体",
    "对比", "分组", "最高", "最低",
)
# 注：刻意不含裸的「表面处理」「层数」。它们单独出现时通常是定义类问题
# （如「层数是什么」），应交给 RAG 向量检索，而不是 Excel 查询。


@lru_cache(maxsize=1)
def build_data_keywords() -> tuple[str, ...]:
    """聚合数据查询关键词：通用词 + 各维度 keywords / exclusive_keywords。

    新增维度时，其关键词会自动进入本函数，无需改动路由代码。

    排除歧义关键词：像「层数」「表面处理」这类词单独出现时通常是**定义类问题**
    （如「层数是什么」），应交给 RAG；分类对比意图已由 _detect_compare_dimension
    单独处理，因此这里不把它们当作数据查询触发词。
    """
    ambiguous = {"层数", "表面处理"}
    words = list(DATA_GENERIC_KEYWORDS)
    for dim in _load_dimensions():
        for kw in dim.get("keywords", []):
            if kw not in ambiguous:
                words.append(kw)
        words.extend(dim.get("exclusive_keywords", []))
    return tuple(dict.fromkeys(words))  # 去重且保序


# 向后兼容：DATA_KEYWORDS 现由配置动态生成
DATA_KEYWORDS = build_data_keywords()


def is_data_query(query: str) -> bool:
    """判断问题是否更可能属于结构化数据查询。"""
    if any(k in query for k in build_data_keywords()):
        return True
    # 分类对比意图（如「按层数分」）也应路由到数据查询，即便没有其他统计关键词
    return _detect_compare_dimension(query) is not None


def extract_factory(query: str) -> Optional[str]:
    """从问题中提取工厂名（金州厂 / 珠海一厂 等）。"""
    for token in ("金州厂", "珠海一厂", "金州", "珠海"):
        if token in query:
            return token if token.endswith("厂") else f"{token}厂"
    return None


if __name__ == "__main__":
    demo = [
        ("金州厂 C 表面处理的平均报废率是多少", "金州厂"),
        ("金州厂 10 层板的报废率", "金州厂"),
        ("金州厂 100356985C0299A 最近表现", "金州厂"),
        ("金州厂整体报废率趋势", "金州厂"),
        ("整体报废率", None),
    ]
    for question, factory in demo:
        print(f"\n问：{question}")
        result = query_excel(question, factory)
        print("  text:", result["text"])
        print("  image:", result["image_path"])
