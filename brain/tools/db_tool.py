"""
DB tool — query SQL databases.

SQLite works out of the box. PostgreSQL / MySQL work if their drivers
are installed (psycopg2-binary / pymysql).

SAFETY: by default only SELECT / WITH queries are allowed. Set
`allow_writes=True` to enable INSERT/UPDATE/DELETE — logged as dangerous.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from typing import Any, List, Optional
from urllib.parse import urlparse

from brain.tools import register_tool


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_READ_ONLY_PATTERN = re.compile(
    r"^\s*(SELECT|WITH|PRAGMA|EXPLAIN|SHOW|DESCRIBE)\b", re.IGNORECASE
)


def _is_read_only(sql: str) -> bool:
    # Simple defensive check — rejects any stmt not starting with a read verb.
    # Also rejects multiple statements.
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return False
    return bool(_READ_ONLY_PATTERN.match(stripped))


def _resolve_sqlite_path(uri_or_path: str) -> str:
    """Accept either 'myfile.db' (relative to brain_state/), full path, or sqlite:/// URI."""
    if uri_or_path.startswith("sqlite:///"):
        return uri_or_path[len("sqlite:///"):]
    if os.path.isabs(uri_or_path):
        return uri_or_path
    base = os.path.join(_app_root(), "brain_state", "databases")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, uri_or_path)


def _connect(connection: str):
    """
    Connect to any of: SQLite (default), PostgreSQL, MySQL.
    Connection strings:
      - sqlite: 'mydb.db' or 'sqlite:///path/to/file.db'
      - postgres: 'postgresql://user:pass@host:5432/dbname'
      - mysql: 'mysql://user:pass@host:3306/dbname'
    """
    if connection.startswith(("postgresql://", "postgres://")):
        try:
            import psycopg2  # type: ignore
        except ImportError:
            raise RuntimeError(
                "PostgreSQL requires psycopg2. Run: pip install psycopg2-binary"
            )
        return ("postgres", psycopg2.connect(connection))
    elif connection.startswith("mysql://"):
        try:
            import pymysql  # type: ignore
        except ImportError:
            raise RuntimeError(
                "MySQL requires pymysql. Run: pip install pymysql"
            )
        p = urlparse(connection)
        return ("mysql", pymysql.connect(
            host=p.hostname, port=p.port or 3306,
            user=p.username, password=p.password or "",
            database=p.path.lstrip("/"), charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        ))
    else:
        path = _resolve_sqlite_path(connection)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return ("sqlite", conn)


@register_tool(
    name="db.query",
    description="Run a read-only SQL query (SELECT/WITH/PRAGMA). Returns rows as dicts.",
    params={
        "connection": "str — sqlite filename, or postgres://... / mysql://... URI",
        "sql": "str — the SQL query (must be read-only unless allow_writes=true)",
        "params": "list — optional list of query parameters for ? placeholders",
        "limit": "int — max rows to return (default: 1000)",
    },
    required=["connection", "sql"],
    category="data",
)
def db_query(
    connection: str,
    sql: str,
    params: Optional[List[Any]] = None,
    limit: int = 1000,
) -> dict:
    if not _is_read_only(sql):
        return {
            "ok": False,
            "error": "Query rejected — not read-only. Use db.execute for writes.",
        }
    try:
        kind, conn = _connect(connection)
    except Exception as e:
        return {"ok": False, "error": f"Connection failed: {e}"}

    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params) if params else ())
        if kind == "sqlite":
            raw = cur.fetchmany(limit)
            rows = [dict(r) for r in raw]
        elif kind == "postgres":
            colnames = [d[0] for d in cur.description] if cur.description else []
            raw = cur.fetchmany(limit)
            rows = [dict(zip(colnames, r)) for r in raw]
        else:  # mysql
            rows = cur.fetchmany(limit)
        return {
            "ok": True,
            "row_count": len(rows),
            "rows": rows,
            "truncated": len(rows) == limit,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@register_tool(
    name="db.execute",
    description="Run a write SQL statement (INSERT/UPDATE/DELETE/CREATE). DANGEROUS.",
    params={
        "connection": "str — connection string",
        "sql": "str — the SQL statement",
        "params": "list — optional bound parameters",
    },
    required=["connection", "sql"],
    category="data",
    dangerous=True,
)
def db_execute(connection: str, sql: str, params: Optional[List[Any]] = None) -> dict:
    try:
        kind, conn = _connect(connection)
    except Exception as e:
        return {"ok": False, "error": f"Connection failed: {e}"}

    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params) if params else ())
        conn.commit()
        affected = cur.rowcount
        return {"ok": True, "rows_affected": affected}
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@register_tool(
    name="db.list_tables",
    description="List all tables in a database.",
    params={"connection": "str — connection string"},
    required=["connection"],
    category="data",
)
def db_list_tables(connection: str) -> dict:
    try:
        kind, conn = _connect(connection)
    except Exception as e:
        return {"ok": False, "error": f"Connection failed: {e}"}
    try:
        cur = conn.cursor()
        if kind == "sqlite":
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [r[0] for r in cur.fetchall()]
        elif kind == "postgres":
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname NOT IN ('pg_catalog','information_schema') "
                "ORDER BY tablename"
            )
            tables = [r[0] for r in cur.fetchall()]
        else:  # mysql
            cur.execute("SHOW TABLES")
            tables = [list(r.values())[0] for r in cur.fetchall()]
        return {"ok": True, "tables": tables, "count": len(tables)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            conn.close()
        except Exception:
            pass
