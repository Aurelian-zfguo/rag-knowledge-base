# 项目B（RAG 知识库）模块职责与数据流

> 本文档说明项目 B 的模块划分、数据流、配置位置，以及 Rerank 重排序的接入与降级逻辑。
> 适用对象：后续维护、扩展该项目的开发人员。

---

## 1. 核心模块与职责

项目根目录：`D:/Python/Claude Code/Programs/RAG知识库`

### 1.1 入口文件

| 文件 | 职责 |
|---|---|
| `app.py` | **Streamlit 问答界面**。接收用户问题，调用 `qa_chain.answer_question()`，渲染答案、图表（`st.image`）与来源引用（可展开）。 |
| `api.py` | **FastAPI 检索服务**。对外提供 `/health`、`/retrieve` 两个 HTTP 接口，供项目 A（PCB 校验 Agent）通过 HTTP 解耦调用。含参数校验、超时/异常兜底、SQLite 请求日志；启动时（lifespan）后台预加载 Rerank 模型。 |
| `ingest.py` | **灌库脚本**（CLI）。把 `knowledge/*.md` 切块、向量化，写入本地 Chroma 持久化库。可重复运行（每次先 `reset_collection`）。 |
| `src/request_logger.py` | **请求日志**（标准库 sqlite3）。`init_db()` 建 `logs/` 与日志表；`log_request()` 写入一条请求日志，失败静默、不阻塞接口。 |

### 1.2 向量检索

| 文件 | 职责 |
|---|---|
| `src/retriever.py` | **两阶段检索核心**。`search(query, top_k)`：① 向量召回 Top-N 候选；② 可选 Rerank 精排输出 Top-k。同时负责打开 Chroma 集合（`load_vectorstore`，带缓存）。 |
| `src/embeddings.py` | **嵌入模型封装**（`BAAI/bge-small-zh-v1.5`）。集中管理 BGE 的非对称编码：文档侧不加前缀（`encode_kwargs`），查询侧加检索指令前缀（`query_encode_kwargs`），单例缓存。 |
| `src/reranker.py` | **Rerank 重排序封装**（`BAAI/bge-reranker-v2-m3`）。懒加载 + 单例 + 降级；`preload()` 供 api.py 启动时预加载。 |

### 1.3 结构化数据

| 文件 | 职责 |
|---|---|
| `src/data_tool.py` | **Excel 数据查询与图表**。`preprocess_history_data()` 把宽表展开为长表（含缓存）；`query_excel()` 理解意图（单值/分类对比）并生成柱状图/折线图。 |
| `config/dimensions.yaml` | **可分组维度配置**。定义 surface / layer / process_type / customer_level 等维度的列名、展示名、触发关键词、label 映射文件。新增维度只需加配置，无需改代码。 |

### 1.4 其他支撑模块

| 文件 | 职责 |
|---|---|
| `config.py` | **配置中心 + 统一配置入口**。加载 `.env`，集中管理路径、模型名、切片/检索参数、`ENABLE_RERANK` 开关；提供 `get_llm_config()` / `get_rerank_config()` / `get_retrieval_config()` 三个统一入口（当前读环境变量，未来切配置源只改这三处）；`resolve_knowledge_path()` 补全知识库相对路径；`validate_config()` 校验环境与 dimensions.yaml。 |
| `src/qa_chain.py` | **问答链路编排 + 意图路由**。路由优先级：① 关键词命中（从 `data_tool.build_data_keywords()` 聚合自 dimensions.yaml）→ 走 data_tool；② 未命中 → 大模型三分类（data/rag/other）兜底；③ data 走 data_tool，rag/other/异常走 RAG。返回统一结构 `{answer, sources, image_path}`。 |
| `src/llm_client.py` | **DeepSeek 调用客户端**。直接 requests 调 OpenAI 兼容接口 `/chat/completions`，带重试；与项目 A 的 `client.py` 保持同一套约定。默认参数走 `config.get_llm_config()`。 |
| `src/loader.py` | **文档加载与切片**（方案 B：标题优先）。用 `MarkdownHeaderTextSplitter` 按 `##` 切分，超长块才用 `RecursiveCharacterTextSplitter` 二次切分。 |

### 1.5 数据与知识源

| 路径 | 说明 |
|---|---|
| `knowledge/*.md` | 制度/基础知识文档（9 个主题），是 RAG 检索的**唯一事实来源**。 |
| `data/CUX_历史报废率查询报表.xlsx` | 历史报废率宽表（业务数据，不入库、不入 git）。 |
| `chroma_db/` | Chroma 向量持久化目录（由 `knowledge/*.md` 灌库再生）。 |
| `static/` | 数据查询生成的图表（运行时产物）。 |
| `tests/` | 单元测试（data_tool / qa_chain / retriever）。 |

---

## 2. 整体数据流

### 2.1 用户提问 → 返回答案（Streamlit 主链路）

```
用户提问（app.py）
        │
        ▼
 qa_chain.answer_question(query)
        │
        ├── ① 关键词命中？（关键词由 data_tool.build_data_keywords() 聚合自 dimensions.yaml）
        │        │
        │        ▼ YES → 走 data_tool.query_excel(query, factory)
        │        │        ├─ preprocess_history_data()  读 Excel / parquet 缓存 → 长表
        │        │        ├─ 按工厂筛选
        │        │        ├─ 意图识别（单值 / 分类对比 / 工厂整体情况，维度来自 dimensions.yaml）
        │        │        └─ matplotlib 出图 → 返回 {text, image_path}
        │        │           （text 为 None → 继续往下走 RAG）
        │
        ├── ② 关键词未命中 → 大模型三分类（data/rag/other）
        │        ├─ data → 走 data_tool
        │        └─ rag / other / LLM 异常 → 走 RAG
        │
        ▼ 走 RAG（制度文档检索）
          retriever.search(query, top_k=4)
                 ├─ ① 向量召回：embeddings.embed_query → Chroma Top-N
                 ├─ ② Rerank 精排（可选，见 §4）
                 └─ 返回 Top-k 文档片段
                 │
          qa_chain 拼 Prompt → llm_client.chat(DeepSeek)
                 │
         返回 {answer, sources, image_path}
        │
        ▼
 app.py 渲染：文字 st.markdown / 图片 st.image / 来源引用展开
```

### 2.2 API 检索链路（供项目 A 调用，无 LLM 生成）

```
项目 A 归因模块
        │  POST /retrieve {"query": "...", "top_k": 4}
        ▼
api.py 的 retrieve()
        ├─ Pydantic 校验（query 去空白非空、top_k 1-20；失败 422）
        ├─ asyncio.wait_for(asyncio.to_thread(retriever.search), timeout=10s)
        │     ├─ 正常 → {query, results:[{content, source, header, score}]}
        │     └─ 超时/异常 → HTTP 200 {error:"检索服务繁忙，请稍后重试", results:[]}
        └─ 写一条日志 → src/request_logger → logs/request_logs.db
```

> 注意：`/retrieve` **只做检索**，不调用 DeepSeek 生成答案。答案生成只在 Streamlit 主链路（`qa_chain`）发生。

### 2.3 灌库链路（离线，一次/更新知识后）

```
ingest.py
   │
   ├─ loader.load_and_split()    knowledge/*.md → 按 ## 切块（带 metadata）
   ├─ embeddings.embed_documents()  文档侧编码（不加 BGE 前缀）
   ├─ Chroma.reset_collection()     清空旧集合（保证可重复灌库）
   └─ Chroma.add_documents()        写入持久化 → chroma_db/
```

---

## 3. 配置点分布

### 3.1 `.env`（密钥与开关，不入 git）

| 配置 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填）。 |
| `DEEPSEEK_MODEL` | 可选，默认 `deepseek-chat`。 |
| `HF_ENDPOINT` | HuggingFace 镜像地址，国内下载模型用。 |
| `ENABLE_RERANK` | Rerank 开关，`true`/`false`（默认 true）。 |

### 3.2 `config.py`（默认值与派生常量 + 统一配置入口）

| 配置 | 说明 |
|---|---|
| `KNOWLEDGE_DIR` / `CHROMA_PERSIST_DIR` / `COLLECTION_NAME` | 知识目录 / 向量库目录 / 集合名。 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1`。 |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DEVICE` | `bge-small-zh-v1.5` / `cpu`。 |
| `RERANK_MODEL_NAME` | `bge-reranker-v2-m3`。 |
| `BGE_QUERY_INSTRUCTION` | BGE 检索指令前缀（仅查询侧）。 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | 切片与检索参数。 |
| `ENABLE_RERANK` | 从 `.env` 读取后解析为布尔值。 |

**统一配置入口（未来切换配置源只改这三个函数）：**

| 函数 | 返回 |
|---|---|
| `get_llm_config()` | `{api_key, model, base_url}` |
| `get_rerank_config()` | `{enabled, model_name}` |
| `get_retrieval_config()` | `{top_k, collection_name, embedding_model, embedding_device, query_instruction, chunk_size, chunk_overlap}` |
| `resolve_knowledge_path(name)` | 把知识库相对文件名补全为 `PROJECT_ROOT/knowledge/name` 绝对路径 |
| `validate_config(require_api_key=True)` | 校验 API key / knowledge 目录 / dimensions.yaml 可解析 |

### 3.3 `config/dimensions.yaml`（可分组维度配置）

| 配置 | 说明 |
|---|---|
| `dimensions[].key` | 维度标识（surface / layer / process_type / customer_level …）。 |
| `dimensions[].column` | 长表中对应列名。 |
| `dimensions[].display_name` | 图表/文字展示名。 |
| `dimensions[].keywords` | 触发该维度的关键词（需配合通用对比词）。 |
| `dimensions[].exclusive_keywords` | 专属关键词，出现即触发，无需通用对比词。 |
| `dimensions[].label_mapping` | 可选，**仅存 knowledge/ 目录下的文件名**（如 `05-工艺类型码对照表.md`），由 `config.resolve_knowledge_path()` 补全绝对路径，自动解析「码→中文名」。 |
| `dimensions[].label_suffix` | 可选，横轴标签后缀（如层数显示「02层」）。 |

**扩展示例**：新增「客户等级」维度只需在 `dimensions.yaml` 加一条，无需改 `data_tool.py` 代码（前提是长表里有对应列）。新增维度的关键词会自动进入 `build_data_keywords()`，意图路由也无需改代码。

---

## 4. Rerank 接入位置与降级逻辑

### 4.1 接入位置

Rerank 只在 **`src/retriever.py` 的 `search()` 第二阶段**接入：

```
retriever.search()
   ├─ ① 向量召回 candidates = Chroma.similarity_search_with_score(query, k=max(10, top_k*2))
   ├─ ② if config.ENABLE_RERANK and reranker.is_available():
   │        return reranker.rerank(query, candidates, top_k)
   └─ ③ 否则：return candidates[:top_k]   ← 纯向量检索（降级）
```

- 模型封装在 `src/reranker.py`，`CrossEncoder("BAAI/bge-reranker-v2-m3")`。
- 分数口径：CrossEncoder 相似度（越大越相关）→ 转成**负数**（越小越相关），与 Chroma 的 L2 距离口径保持一致。

### 4.2 加载策略

- **懒加载 + 单例**：首次调用 `rerank()` 或 `preload()` 才加载（约 1.2GB，几秒）。
- **api.py 预加载**：lifespan 钩子在服务启动时 `asyncio.to_thread(preload_reranker)` 后台加载，不阻塞启动。
- **Streamlit 路径**：不预加载，首次检索懒加载，之后命中单例缓存。

### 4.3 降级逻辑（三层）

| 触发条件 | 行为 |
|---|---|
| `config.ENABLE_RERANK = False` | 直接走纯向量检索（`candidates[:top_k]`）。 |
| 模型加载失败（OOM / 文件损坏 / 网络） | `reranker._available = False`，之后每次调用原样返回候选前 top_k，不抛异常。 |
| 重排过程抛异常 | `reranker.rerank()` 内部 try/except 捕获，回退候选前 top_k。 |

**修改指引**：
- 要换 Rerank 模型 → 改 `src/reranker.py` 顶部 `RERANK_MODEL_NAME`（或 `config.get_rerank_config()["model_name"]`）。
- 要调召回候选数 → 改 `src/retriever.py` 里 `recall_k = max(10, top_k * 2)`。
- 要关/开 Rerank → 改 `.env` 的 `ENABLE_RERANK`，或直接改 `config.py` 里的解析默认值。

---

## 5. API 健壮性与请求日志

`api.py` 的 `/retrieve` 接口做了三步健壮性加固：

| 步骤 | 内容 | 位置 |
|---|---|---|
| ① 参数校验 | `RetrieveRequest`：`query` 必填非空（先 strip 再 `min_length=1`），`top_k` 默认 4、范围 1-20；失败 FastAPI 自动返回 HTTP 422 | `api.py` |
| ② 超时/异常兜底 | `asyncio.wait_for(asyncio.to_thread(search), timeout=10)`；超时或任何异常都返回 HTTP 200 `{error:"检索服务繁忙，请稍后重试", results:[]}` | `api.py` |
| ③ SQLite 日志 | 每次请求写一条日志（含 timeout/error），字段 `timestamp/query/top_k/response_time_ms/status/result_count` | `src/request_logger.py` |

关键点：
- `retriever.search()` 是**同步阻塞**调用，必须 `asyncio.to_thread()` 丢线程池，`asyncio.timeout` 才能生效。
- 日志用标准库 `sqlite3`，每次独立连接 + 短超时，**失败静默**，不阻塞接口。
- 日志库 `logs/request_logs.db`，服务启动时 `init_db()` 自动建表（幂等）。
- 超时/异常兜底返回 HTTP 200 是为了保证调用方（项目 A）不会因 B 内部错误而崩溃——项目 A 的 `knowledge.py` 本身也有降级逻辑，两者共同保证解耦。

---

## 6. 快速参考

| 想做的事 | 命令 / 位置 |
|---|---|
| 启动问答界面 | `streamlit run app.py` |
| 启动检索 API（供 A 调用） | `python api.py` 或 `uvicorn api:app --port 8000` |
| 首次/更新知识后灌库 | `python ingest.py` |
| 环境校验 | `python config.py` |
| 数据查询测试 | `python -c "from src.data_tool import query_excel; print(query_excel('金州厂各工艺类型的报废率对比','金州厂'))"` |
| 运行全部测试 | `python -m unittest discover -s tests` |
| 新增可分组维度 | 编辑 `config/dimensions.yaml`（关键词自动同步到路由） |
| 切换 Rerank | `.env` 的 `ENABLE_RERANK` |
| 查看请求日志 | `logs/request_logs.db`（SQLite） |
