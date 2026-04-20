"""
File tool — sandboxed filesystem read/write.

SAFETY: All file ops are restricted to brain_state/workspace/ by default.
Set workspace=None to access anywhere (dangerous).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from brain.tools import register_tool


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _workspace() -> str:
    ws = os.path.join(_app_root(), "brain_state", "workspace")
    os.makedirs(ws, exist_ok=True)
    return ws


def _resolve(path: str, sandboxed: bool = True) -> str:
    """Resolve path within workspace unless sandboxed=False."""
    if not sandboxed:
        return os.path.abspath(path)
    if os.path.isabs(path):
        # Strip absolute prefix if it's under workspace; else reject
        ws = _workspace()
        abspath = os.path.abspath(path)
        if abspath.startswith(ws):
            return abspath
        # Treat as relative name
        path = os.path.basename(path)
    return os.path.abspath(os.path.join(_workspace(), path))


@register_tool(
    name="file.read",
    description="Read a text file (sandboxed to brain_state/workspace/).",
    params={
        "path": "str — path inside workspace",
        "max_bytes": "int — limit size (default: 1MB)",
    },
    required=["path"],
    category="filesystem",
)
def file_read(path: str, max_bytes: int = 1024 * 1024) -> dict:
    full = _resolve(path)
    if not os.path.exists(full):
        return {"ok": False, "error": f"Not found: {path}"}
    if not os.path.isfile(full):
        return {"ok": False, "error": f"Not a file: {path}"}
    size = os.path.getsize(full)
    if size > max_bytes:
        return {"ok": False, "error": f"File too large ({size} bytes > {max_bytes})"}
    try:
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        return {"ok": True, "path": full, "size_bytes": size, "content": content}
    except UnicodeDecodeError:
        return {"ok": False, "error": "Not a text file (binary content)"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@register_tool(
    name="file.write",
    description="Write a text file (sandboxed to brain_state/workspace/).",
    params={
        "path": "str — path inside workspace",
        "content": "str — file content",
        "append": "bool — append instead of overwrite",
    },
    required=["path", "content"],
    category="filesystem",
    dangerous=True,
)
def file_write(path: str, content: str, append: bool = False) -> dict:
    full = _resolve(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    mode = "a" if append else "w"
    try:
        with open(full, mode, encoding="utf-8") as f:
            f.write(content)
        return {
            "ok": True, "path": full,
            "bytes_written": len(content.encode("utf-8")),
            "mode": "append" if append else "overwrite",
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@register_tool(
    name="file.list",
    description="List files in a workspace directory.",
    params={
        "path": "str — directory (default: workspace root)",
        "recursive": "bool — descend into subdirs (default: false)",
    },
    required=[],
    category="filesystem",
)
def file_list(path: str = "", recursive: bool = False) -> dict:
    full = _resolve(path) if path else _workspace()
    if not os.path.isdir(full):
        return {"ok": False, "error": f"Not a directory: {path}"}
    results = []
    if recursive:
        for root, dirs, files in os.walk(full):
            for f in files:
                p = os.path.join(root, f)
                results.append({
                    "path": os.path.relpath(p, _workspace()),
                    "size": os.path.getsize(p),
                })
    else:
        for name in sorted(os.listdir(full)):
            p = os.path.join(full, name)
            results.append({
                "name": name,
                "is_dir": os.path.isdir(p),
                "size": os.path.getsize(p) if os.path.isfile(p) else None,
            })
    return {"ok": True, "entries": results, "count": len(results)}


@register_tool(
    name="file.delete",
    description="Delete a file (sandboxed to workspace).",
    params={"path": "str — file path"},
    required=["path"],
    category="filesystem",
    dangerous=True,
)
def file_delete(path: str) -> dict:
    full = _resolve(path)
    if not os.path.exists(full):
        return {"ok": False, "error": "Not found"}
    if not os.path.isfile(full):
        return {"ok": False, "error": "Only files can be deleted"}
    try:
        os.remove(full)
        return {"ok": True, "deleted": full}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
