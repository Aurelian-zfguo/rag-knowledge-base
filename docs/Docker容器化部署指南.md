# Docker 容器化部署指南

> 项目 B（RAG 知识库）的 Docker 部署说明。包含镜像构建、一键启动、验证、数据卷、环境变量、常用运维命令。

---

## 1. 镜像构建流程

### 构建命令

```bash
cd "D:/Python/Claude Code/Programs/RAG知识库"
docker build -t rag-kb:latest .
```

- `-t rag-kb:latest`：给镜像命名 + 打标签（`rag-kb` 是名，`latest` 是版本标签）。
- 最后的 `.`：构建上下文目录，Docker 读这里的 `Dockerfile`，并按 `.dockerignore` 排除不需要的文件。

构建完成后确认：

```bash
docker images
# 应看到 rag-kb:latest，SIZE 约 3.2GB（CPU 版 torch）
```

### Dockerfile 关键设计（见项目根 `Dockerfile`）

| 要点 | 说明 |
|---|---|
| `FROM python:3.14-slim` | Debian slim 基础镜像，glibc 兼容性最好（不用 alpine，避免 torch/pandas 无 musl wheel） |
| 先 `pip install torch --index-url .../cpu` | **关键**：先装 CPU 版 torch（约 200MB），否则默认拉 CUDA 版（2GB+） |
| 先 `COPY requirements.txt` 再 `pip install` | 利用 Docker 分层缓存，改代码不重装依赖 |
| `COPY . .` 放最后 | `.dockerignore` 生效，排除 `.env`/`.venv`/`data`/`chroma_db` 等 |
| `EXPOSE 8000 8501` | 声明端口（真正映射靠 `-p` 或 compose） |
| `CMD ["uvicorn", "api:app", ...]` | 默认启动 FastAPI |

---

## 2. 一键启动（docker compose）

```bash
docker compose up -d
```

启动两个服务（共享同一镜像 `rag-kb:latest`）：

| 服务 | 端口 | 作用 |
|---|---|---|
| `rag-api` | 8000 | FastAPI 检索服务 |
| `rag-ui` | 8501 | Streamlit 问答界面 |

`docker-compose.yml` 要点：

- 两个服务共享镜像，各自加载一套模型（内存 4-6GB+，Docker Desktop 建议 ≥8GB）。
- 都挂载 `chroma_db`/`data`/`static` 三个卷。
- 都通过 `env_file: .env` 注入环境变量。
- `rag-ui` 用 `command` 覆盖镜像默认 CMD，改启动 Streamlit。
- 自定义网络 `rag-net`，两个服务可互访。
- `restart: unless-stopped`：意外崩溃自动重启，手动 stop 不重启。

---

## 3. 验证步骤

```bash
# 1. 健康检查
curl http://localhost:8000/health
# 期望：{"status":"ok"}

# 2. 检索接口
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"补投超过投料标准多少需要审批","top_k":4}'
# 期望：返回 results 数组（含 content/source/header/score）

# 3. Streamlit 界面
# 浏览器打开 http://localhost:8501
```

> 首次启动容器会同步预加载 embedding + Rerank 模型，`Application startup complete` 会延后 10-30 秒出现，属正常。之后所有请求（含首次 /retrieve）都直接返回结果，不会超时降级。

---

## 4. 数据卷挂载说明

| 卷 | 为什么挂载而不是打进镜像 |
|---|---|
| `./chroma_db:/app/chroma_db` | 向量库是「知识文档 → 灌库」的**派生数据**。打进镜像会导致：① 镜像膨胀；② 更新知识文档后镜像内向量库过期；③ 容器删除数据丢失。挂载后容器读写宿主机同一份向量库，持久且可更新。 |
| `./data:/app/data` | 业务数据（Excel）+ parquet 缓存，敏感且体积大。挂载让容器读宿主机数据，数据不进镜像。 |
| `./static:/app/static` | 数据查询生成的图表运行时产物。挂载后图表落到宿主机，容器重启不丢。 |

---

## 5. 环境变量注入说明

**为什么用 `env_file: .env`（或 `--env-file .env`），而不是打进镜像？**

- `.env` 含真实 `DEEPSEEK_API_KEY`，是敏感密钥。
- 镜像一旦构建出来，可能被推送/分发/留存，密钥打进镜像 = 永久泄露。
- 正确做法：镜像里只留 `.env.example` 模板，**运行时**把真密钥注入为容器环境变量。

`.env` 里配置（至少需要）：

```ini
DEEPSEEK_API_KEY=sk-xxx
HF_ENDPOINT=https://hf-mirror.com
ENABLE_RERANK=true
```

---

## 6. 常用运维命令

```bash
# 启动 / 停止 / 重启
docker compose up -d          # 启动（后台）
docker compose down           # 停止并移除容器（保留卷数据）
docker compose restart        # 重启所有服务
docker compose up -d --build  # 重新构建镜像并启动

# 查看状态
docker compose ps             # 服务状态
docker images                 # 镜像列表

# 查看日志
docker compose logs -f        # 跟踪所有服务日志
docker compose logs -f rag-api  # 只看 FastAPI 日志
docker compose logs --tail 50 rag-ui  # 看 Streamlit 最近 50 行

# 进入容器排查
docker compose exec rag-api bash
# 进容器后可：ls /app、python config.py、ls /app/chroma_db 等
```

---

## 7. 常见问题

| 现象 | 排查 |
|---|---|
| 构建拉取 `python:3.14-slim` 超时 | Docker Hub 网络问题，配置 registry-mirrors 镜像加速 |
| pip 依赖下载慢 | 构建时加国内 pip 源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 容器秒退 | `docker compose logs rag-api` 看启动报错 |
| `/retrieve` 返回「向量库为空」 | 宿主机先 `python ingest.py` 灌库，或进容器灌库 |
| 端口被占用 | 改 compose 里的端口映射，如 `9000:8000` |
