# Docker 容器化踩坑记录

> 记录项目 B 从零安装 Docker Desktop 到容器成功运行的完整过程中遇到的所有问题、根因与解决方案，供后续复现或排查参考。

---

## 一、Docker Desktop 安装阶段

| 问题 | 安装程序提示「现有安装已是最新」，但系统找不到 Docker Desktop |
|---|---|
| 根因 | 残留注册表误判 |
| 解决 | 卸载残留 + 重启 + 命令行指定非 C 盘路径重装 |

---

## 二、WSL 2 安装阶段

| 问题 | `docker run` 报 500 Internal Server Error |
|---|---|
| 根因 | WSL 2 未安装 |
| 解决 | 启用 Windows 功能 |

| 问题 | WSL 内核更新包报错「This update only applies to machines with the Windows Subsystem for Linux」 |
|---|---|
| 根因 | 安装顺序错误 + 旧版内核包检测严格 |
| 解决 | 先启用 Windows 功能 + 重启 + 从 GitHub releases 下载最新版 .msi |

| 问题 | `wsl --install -d Ubuntu` 报 WININET_E_TIMEOUT |
|---|---|
| 根因 | 国内访问 GitHub 超时 |
| 解决 | 加 `--web-download` 参数 |

| 问题 | WSL 安装完 Ubuntu 后 `wsl --list` 显示「没有已安装的分发」 |
|---|---|
| 根因 | Ubuntu 注册异常结束 |
| 解决 | `wsl --shutdown` + 重启电脑 + 用 `--no-launch` 参数重装 |

---

## 三、Docker 镜像加速器配置阶段（重点）

**问题**：`docker info` 的 Registry Mirrors 一直输出为空，多次点 Apply & restart 和彻底重启 Docker Desktop 都无效。

**尝试过的无效操作**：改 JSON 后 Apply、彻底 Quit 重开、更换镜像地址。

**最终结局**：某时刻突然开始正常显示 3 个镜像地址。

**真实原因分析**：

1. **根本原因不在 JSON 配置本身，而在 WSL 2 后端未完全就绪**——WSL 2 安装过程中，Docker Desktop 引擎虽然显示 running，但与 WSL 的连接不稳定，此时引擎读不到或读不全 daemon 配置。
2. **Apply & restart 有延迟**——点击后 UI 立即返回，但引擎实际重启需要 30-60 秒，重启完成前 `docker info` 读到的是旧配置。
3. **系统层面重启是必要的**——涉及 WSL 2 和 Windows 功能的改动，必须重启电脑，仅重启 Docker Desktop 不够。

**解决方案**：启用 Windows 功能 → 重启 → 装 WSL → 重启 → 配镜像 → 应用并等 1-2 分钟 → 再验证。

---

## 四、Docker 构建阶段

| 问题 | 构建上下文 1.74GB，构建极慢 |
|---|---|
| 根因 | 缺 `.dockerignore`，`.venv`/`chroma_db`/`data` 全被打包 |
| 解决 | 创建 `.dockerignore`（上下文降到 2.82kB） |

| 问题 | torch 默认装 GPU 版，镜像膨胀到 5GB+ |
|---|---|
| 根因 | requirements.txt 未显式声明 torch，被 sentence-transformers 间接拉 CUDA 版 |
| 解决 | Dockerfile 里先单独装 CPU 版 torch（`--index-url https://download.pytorch.org/whl/cpu`），最终镜像 3.22GB |

---

## 五、Docker 运行阶段

| 问题 | `--env-file .env` 报错找不到文件 |
|---|---|
| 根因 | 命令执行目录是 `C:\Users\zfguo>`，不是项目根目录 |
| 解决 | 先 `cd` 到项目根目录 |

| 问题 | 首次 `/retrieve` 请求超时降级 |
|---|---|
| 根因 | Rerank 模型懒加载，首次请求时模型还没加载完 |
| 解决（第一次） | lifespan 改为同步预加载 Rerank |

| 问题 | 即使同步预加载 Rerank，首次 `/retrieve` 仍超时 |
|---|---|
| 根因 | embedding 模型也是懒加载的，首次请求时才加载，累计耗时超 10 秒 |
| 解决 | lifespan 里同步预加载 embedding（`build_embeddings()`）+ 超时从 10 秒改为 30 秒 |

---

## 六、关键教训汇总

1. **顺序不能乱**：启用 Windows 功能 → 重启 → 装 WSL → 重启 → 装 Docker → 配镜像 → 应用并等待。
2. **系统级重启是必须的**，重启 Docker Desktop 不够。
3. **Apply & restart 至少等 1 分钟再验证**。
4. **`.dockerignore` 是刚需**，不做的话构建上下文能到 GB 级。
5. **torch 必须显式装 CPU 版**，不然镜像体积翻倍。
6. **所有同步模型（embedding + rerank）都要在 lifespan 预加载**，否则首次请求超时。
7. **Docker 命令的相对路径基于当前目录**，执行前先 `cd` 到项目根目录。
