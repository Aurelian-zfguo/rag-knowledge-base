"""检索请求日志：写入本地 SQLite。

轻量实现：标准库 sqlite3，不引入额外依赖。每次写入用独立连接 + 短超时，
失败静默（日志永远不应拖垮主流程）。
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LOG_DIR = _PROJECT_ROOT / "logs"
DB_PATH = LOG_DIR / "request_logs.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    result_count INTEGER NOT NULL
)
"""


def init_db() -> None:
    """创建 logs/ 目录与日志表（幂等，启动时调用一次）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def log_request(query: str, top_k: int, response_time_ms: int,
                status: str, result_count: int) -> None:
    """写入一条请求日志。任何异常静默吞掉，不影响接口。"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            conn.execute(
                "INSERT INTO request_logs "
                "(timestamp, query, top_k, response_time_ms, status, result_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    query,
                    top_k,
                    int(response_time_ms),
                    status,
                    int(result_count),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # 日志失败不影响主流程


if __name__ == "__main__":
    init_db()
    print(f"日志库已就绪：{DB_PATH}")
