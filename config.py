"""配置中心。

集中管理项目路径、模型名称、切片与检索参数，并在应用启动时校验运行环境。

导入顺序约束
------------
本模块在**导入时**立即执行 ``load_dotenv()``。HF_ENDPOINT 镜像地址必须在
``sentence_transformers`` / ``huggingface_hub`` 被导入之前写入 ``os.environ``
才会生效，因此各入口脚本请始终把 ``import config`` 放在这些库之前。

统一配置入口
------------
外部模块请通过 :func:`get_llm_config` / :func:`get_rerank_config` /
:func:`get_retrieval_config` 读取配置。当前它们仍读环境变量，未来切换配置源
（YAML / 数据库）只需修改这三个函数，无需改动调用方。

为保持向后兼容，模块级常量（如 ``DEEPSEEK_API_KEY``）仍保留，可继续直接访问。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# 项目根目录。全部路径基于此推导，保证换工作目录运行时不会生成散落的 chroma_db。
BASE_DIR = Path(__file__).resolve().parent

# .env 不存在时静默跳过，缺失的必填项交由 validate_config() 统一报错。
load_dotenv(BASE_DIR / ".env")


# ------------------------------- 路径 -------------------------------
KNOWLEDGE_DIR = BASE_DIR / "knowledge"        # 知识源 Markdown 目录
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"   # Chroma 向量库持久化目录
CONFIG_DIR = BASE_DIR / "config"
DIMENSIONS_FILE = CONFIG_DIR / "dimensions.yaml"

COLLECTION_NAME = "touliao_knowledge"         # Chroma 集合名，灌库与检索须一致


# ------------------------------- 模型 -------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # 与项目 A 保持一致

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Rerank 重排序开关。true=向量召回后精排；false=纯向量检索（便于对比测试）。
# 注意：即使为 true，Rerank 模型加载失败也会自动降级为纯向量检索。
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").strip().lower() in ("true", "1", "yes", "on")

# BGE 中文模型的检索指令前缀。
# 仅在 query 侧拼接，文档侧不加 —— 这是 BGE 的非对称编码要求，两侧都加会掉点。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# --------------------------- 切片与检索 ---------------------------
CHUNK_SIZE = 1024      # 单个切块的字符上限，超过则二次切分
CHUNK_OVERLAP = 128    # 二次切分时的重叠字符数
TOP_K = 4              # 检索返回的片段数

# 多轮对话保留的最近轮数（1 轮 = 1 问 1 答 = 2 条消息），超过部分从最旧丢弃。
MAX_HISTORY_TURNS = 3

# Query Rewriting 开关：多轮对话时，把指代性 query（如「那第二条呢」）补全成独立 query 再检索。
# true=启用（默认）；false=关闭，便于对比测试。单轮对话（history 为空）始终不触发改写。
ENABLE_QUERY_REWRITE = True


# --------------------------------------------------------------------------- #
# 统一配置入口（未来切换配置源只需改这里）
# --------------------------------------------------------------------------- #
def get_llm_config() -> dict:
    """LLM（DeepSeek）配置。"""
    return {
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_MODEL,
        "base_url": DEEPSEEK_BASE_URL,
    }


def get_rerank_config() -> dict:
    """Rerank 重排序配置。"""
    return {
        "enabled": ENABLE_RERANK,
        "model_name": RERANK_MODEL_NAME,
    }


def get_retrieval_config() -> dict:
    """向量检索与嵌入模型配置。"""
    return {
        "top_k": TOP_K,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_device": EMBEDDING_DEVICE,
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def resolve_knowledge_path(name: str) -> Path:
    """把 knowledge/ 目录下的相对文件名补全为绝对路径。

    dimensions.yaml 里的 label_mapping 只存文件名（如 ``05-工艺类型码对照表.md``），
    由本函数统一补全为 ``PROJECT_ROOT / knowledge / 文件名``，避免依赖运行目录。
    """
    if not name:
        return KNOWLEDGE_DIR
    p = Path(name)
    return p if p.is_absolute() else KNOWLEDGE_DIR / p


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate_config(require_api_key: bool = True) -> None:
    """校验运行环境，供各入口脚本在启动时调用。

    Args:
        require_api_key: 是否要求 DEEPSEEK_API_KEY 存在。灌库（ingest.py）不调用
            DeepSeek，可传 False 跳过该项检查。

    Raises:
        ValueError: 必填配置缺失或 dimensions.yaml 无法解析。
    """
    if require_api_key and not DEEPSEEK_API_KEY:
        raise ValueError(
            "未找到 DEEPSEEK_API_KEY。\n"
            f"请将 {BASE_DIR / '.env.example'} 复制为 .env，"
            "并填入从 https://platform.deepseek.com/api_keys 申请的密钥。"
        )

    if not KNOWLEDGE_DIR.exists():
        raise ValueError(f"知识库目录不存在：{KNOWLEDGE_DIR}")

    if not DIMENSIONS_FILE.exists():
        raise ValueError(f"维度配置文件不存在：{DIMENSIONS_FILE}")
    try:
        with DIMENSIONS_FILE.open(encoding="utf-8") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"维度配置文件解析失败：{DIMENSIONS_FILE}\n{exc}") from exc


if __name__ == "__main__":
    validate_config()
    print("配置校验通过")
    print(f"  知识库目录 : {KNOWLEDGE_DIR}")
    print(f"  向量库目录 : {CHROMA_PERSIST_DIR}")
    print(f"  嵌入模型   : {EMBEDDING_MODEL_NAME} (device={EMBEDDING_DEVICE})")
    print(f"  Rerank 模型: {RERANK_MODEL_NAME} (enabled={ENABLE_RERANK})")
    print(f"  对话模型   : {DEEPSEEK_MODEL}")
    print(f"  HF 镜像    : {os.getenv('HF_ENDPOINT') or '未设置（走 huggingface.co 官方源）'}")
