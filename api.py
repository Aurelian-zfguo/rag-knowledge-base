"""B 项目检索服务：向外部系统（项目 A）提供知识检索 HTTP 接口。

启动方式（二选一）::

    uvicorn api:app --host 127.0.0.1 --port 8000
    python api.py

接口
----
- ``GET  /health``              健康检查，返回 ``{"status": "ok"}``
- ``POST /retrieve``            语义检索，请求体 ``{"query": "...", "top_k": 3}``

``/retrieve`` 返回结构::

    {
      "query": "预大率超标",
      "results": [
        {"content": "...", "source": "07-补投管控措施.md", "header": "第 1 条", "score": 0.52}
      ]
    }

score 为距离值，越小越相关。依赖前置条件：先运行 ``python ingest.py`` 完成灌库。
"""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  须最先导入以加载 .env
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from src.embeddings import build_embeddings  # noqa: E402
from src.request_logger import init_db, log_request  # noqa: E402
from src.reranker import preload as preload_reranker  # noqa: E402
from src.retriever import search  # noqa: E402

# 检索超时与兜底响应的固定话术
SEARCH_TIMEOUT_SECONDS = 30.0
BUSY_REPLY = "检索服务繁忙，请稍后重试"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化日志库，并**同步**预加载 embedding 与 Rerank 模型。

    同步加载（而非后台 to_thread）是为了避免首次 /retrieve 请求因模型尚未就绪
    而超时降级：启动阶段一次性把 embedding + Rerank 都加载完，之后所有请求走完整
    链路。代价是容器启动时间增加约 10-30 秒。
    """
    init_db()  # 创建 logs/ 目录与日志表（幂等）
    # 预加载 embedding 模型（懒加载单例，此处触发下载/加载并缓存）
    build_embeddings()
    if config.get_rerank_config()["enabled"]:
        # 阻塞直到模型加载完成；加载失败会自动降级为纯向量检索，不影响启动。
        preload_reranker()
    yield


app = FastAPI(title="RAG 知识库检索服务", version="1.0.0", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索问题")
    top_k: int = Field(default=config.get_retrieval_config()["top_k"], ge=1, le=20,
                       description="返回片段数")

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value):
        """去除首尾空白；纯空白字符串触发 422 校验失败。"""
        if isinstance(value, str):
            return value.strip()
        return value


class RetrievedChunk(BaseModel):
    content: str
    source: str
    header: str
    score: float


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/retrieve")
async def retrieve(payload: RetrieveRequest):
    """语义检索。

    正常返回 ``{query, results}``；检索超时（10 秒）或内部异常时，仍返回
    HTTP 200，但响应体固定为 ``{error, results: []}``，保证调用方不崩溃。
    每次请求（含超时/异常）都写入 SQLite 日志。
    """
    start = time.perf_counter()

    try:
        # search() 是同步阻塞调用，丢到线程池执行才能被 asyncio.timeout 打断
        hits = await asyncio.wait_for(
            asyncio.to_thread(search, payload.query, payload.top_k),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_request(payload.query, payload.top_k, elapsed_ms, "timeout", 0)
        return JSONResponse(status_code=200, content={"error": BUSY_REPLY, "results": []})
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_request(payload.query, payload.top_k, elapsed_ms, "error", 0)
        return JSONResponse(status_code=200, content={"error": BUSY_REPLY, "results": []})

    elapsed_ms = (time.perf_counter() - start) * 1000
    results = [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", ""),
            header=doc.metadata.get("header", ""),
            score=score,
        )
        for doc, score in hits
    ]
    log_request(payload.query, payload.top_k, elapsed_ms, "success", len(results))
    return {"query": payload.query, "results": [r.model_dump() for r in results]}


if __name__ == "__main__":
    import uvicorn

    # 固定监听 127.0.0.1，仅本机可访问，不对外暴露。
    uvicorn.run("api:app", host="127.0.0.1", port=8000)
