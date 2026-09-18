# RAG 知识库（项目 B）

基于 LangChain + Chroma + Streamlit 的 PCB 投料/余数管理知识库问答系统。支持制度文档检索（RAG）与结构化 Excel 数据查询（含图表），并提供 FastAPI 检索服务供项目 A 联动调用。

## 技术栈

- **嵌入模型**：本地 `BAAI/bge-small-zh-v1.5`
- **重排序**：本地 `BAAI/bge-reranker-v2-m3`（可开关）
- **向量库**：Chroma（本地持久化）
- **大模型**：DeepSeek API
- **前端**：Streamlit
- **检索服务**：FastAPI

## 主要模块

| 模块 | 职责 |
|---|---|
| `app.py` | Streamlit 问答界面 |
| `api.py` | FastAPI 检索服务（供项目 A 调用） |
| `ingest.py` | 灌库脚本（知识文档 → 向量库） |
| `src/retriever.py` | 两阶段检索（向量召回 + Rerank） |
| `src/qa_chain.py` | 意图路由 + 问答链路 |
| `src/data_tool.py` | Excel 数据查询与图表 |
| `config/dimensions.yaml` | 可分组维度配置 |

详细架构见 `docs/项目B_模块职责与数据流.md`，量化评估见 `docs/项目B_RAGAS评估.md`。

### 项目文档（docs/ 目录）

| 文档 | 说明 |
|---|---|
| `项目B_模块职责与数据流.md` | 模块职责、数据流、配置点、Rerank 接入与降级 |
| `项目B_RAGAS评估.md` | RAGAS 量化评估闭环（发现 bug → 修复 → 指标提升） |
| `Docker容器化部署指南.md` | 镜像构建、一键启动、验证、卷/环境变量、运维命令 |
| `Docker容器化踩坑记录.md` | Docker 安装到运行全过程的踩坑记录与根因分析 |

---

## 本地运行（不依赖 Docker）

```bash
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell（cmd 用 activate.bat）

# 2. 先装 CPU 版 torch（避免拉 2GB+ GPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 装其余依赖
pip install -r requirements.txt

# 4. 配置环境变量
copy .env.example .env               # 然后填入 DEEPSEEK_API_KEY

# 5. 灌库（首次/更新知识文档后）
python ingest.py

# 6. 启动问答界面
streamlit run app.py
```

---

## Docker 部署

### 前置要求

- Docker Desktop（含 WSL 2，已验证）
- Docker Desktop 内存建议 ≥ 8GB（两个容器各加载一套 embedding + Rerank 模型）

### 一条命令启动

```bash
docker compose up -d
```

> 首次启动需先构建镜像（见下），或直接：
> ```bash
> docker build -t rag-kb:latest .
> docker compose up -d
> ```

### 服务地址

| 服务 | 地址 | 说明 |
|---|---|---|
| FastAPI 检索服务 | http://localhost:8000 | `/health`、`/retrieve` |
| Streamlit 问答界面 | http://localhost:8501 | 浏览器打开 |

### 验证

```bash
curl http://localhost:8000/health          # 期望 {"status":"ok"}
```

完整部署说明见 `docs/Docker容器化部署指南.md`，踩坑记录见 `docs/Docker容器化踩坑记录.md`。

---

## RAGAS 评估

项目已接入 RAGAS 量化评估体系，对 15 条样本（覆盖制度定义/管控措施/编码规则/工艺类型）完成评估：

| 指标 | 得分 |
|---|---|
| Faithfulness（忠实度） | 1.00 |
| Answer Relevancy（答案相关性） | 0.87 |
| Context Precision（上下文精度） | 0.93 |

完整报告见 `evaluation_report.md`，评估闭环记录见 `docs/项目B_RAGAS评估.md`。

重新评估：

```bash
python scripts/evaluate_ragas.py          # 全量 15 条
python scripts/evaluate_ragas.py --limit 3  # 冒烟测试
```

---

## 其他入口

| 想做什么 | 命令 |
|---|---|
| 灌库 | `python ingest.py` |
| 环境校验 | `python config.py` |
| 数据查询测试 | `python -c "from src.data_tool import query_excel; print(query_excel('金州厂各工艺类型的报废率对比','金州厂'))"` |
| RAGAS 评估 | `python scripts/evaluate_ragas.py` |
| 运行测试 | `python -m unittest discover -s tests` |
