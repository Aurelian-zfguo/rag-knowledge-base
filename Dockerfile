# 基础镜像：Debian slim 精简版，带 Python 3.14
# 选 slim 而非完整版/alpine：体积小，且官方预编译 wheel 兼容性好
FROM python:3.14-slim

# 设置工作目录：后续所有命令都在 /app 下执行
WORKDIR /app

# 设置环境变量：
#   PYTHONUNBUFFERED=1  让 Python 输出不缓冲，docker logs 能实时看到日志
#   PIP_NO_CACHE_DIR=1  关闭 pip 缓存，减小镜像体积
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 先单独安装 CPU 版 torch（关键步骤）：
#   直接用 pip install torch 会拉 CUDA 版（2GB+），容器无 GPU 纯属浪费。
#   用 --index-url 指向 PyTorch 官方 CPU 专用源，只下载 CPU 版（约 200MB）。
#   必须先于 requirements.txt 安装，让后续 pip 看到 torch 已满足、不再升级成 GPU 版。
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 只 COPY requirements.txt（而非整个项目）：
#   利用 Docker 分层缓存——只要 requirements.txt 不变，这层就不会重新执行，
#   改了代码重构建时无需重装依赖，构建快很多。
COPY requirements.txt .

# 安装其余依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目全部文件（.dockerignore 已排除 .venv/.env/data/chroma_db 等）
COPY . .

# 声明容器对外暴露的端口（文档性质，实际映射靠 docker run -p）
EXPOSE 8000 8501

# 默认启动 FastAPI 检索服务
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
