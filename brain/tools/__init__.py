"""
NeuroLinked Agent Tools — V1.3
=====================================

Tool registry and base class for all agent-callable tools.

Every tool registered here becomes something the brain can CALL, not just
remember. This is what turns NeuroLinked from a memory system into an
actual agent that takes real-world actions.

Usage:
    from brain.tools import register_tool, list_tools, get_tool

    @register_tool(
        name="send_email",
        description="Send an email via SMTP",
        params={"to": "str", "subject": "str", "body": "str"}
    )
    def send_email(to: str, subject: str, body: str) -> dict:
        # ...
        return {"ok": True, "message_id": "..."}
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from .test_tool import echo

@dataclass
class ToolSpec:
    """Metadata about a registered tool."""
    name: str
    description: str
    params: Dict[str, str]                 # param_name -> type_hint (str)
    required: List[str] = field(default_factory=list)
    category: str = "general"
    fn: Optional[Callable] = None          # the actual function
    dangerous: bool = False                 # requires extra safety check
    registered_at: float = field(default_factory=time.time)

    def describe(self) -> dict:
        """JSON-serializable description for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "required": self.required,
            "category": self.category,
            "dangerous": self.dangerous,
        }


@dataclass
class ToolResult:
    """Result of invoking a tool."""
    tool: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    args: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "value": self.value if self._json_safe(self.value) else str(self.value),
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "args": self.args,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def _json_safe(v) -> bool:
        try:
            json.dumps(v)
            return True
        except Exception:
            return False


class ToolRegistry:
    """Central registry of every tool the brain can call."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._history: List[ToolResult] = []
        self._max_history = 1000

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            print(f"[TOOL] Warning — re-registering tool '{spec.name}'")
        self._tools[spec.name] = spec

    def list(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def describe_all(self) -> List[dict]:
        return [t.describe() for t in self._tools.values()]

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def call(self, name: str, args: Optional[dict] = None) -> ToolResult:
        """Invoke a tool by name. Records outcome to history (for benchmarks)."""
        args = args or {}
        spec = self._tools.get(name)
        if spec is None:
            result = ToolResult(tool=name, ok=False, error=f"Unknown tool: {name}", args=args)
            self._append_history(result)
            return result

        # Validate required args
        missing = [p for p in spec.required if p not in args]
        if missing:
            result = ToolResult(
                tool=name, ok=False,
                error=f"Missing required args: {missing}",
                args=args,
            )
            self._append_history(result)
            return result

        # Execute + time it
        t0 = time.time()
        try:
            value = spec.fn(**args)
            # If the tool returned a dict with ok=False, propagate that to the
            # outer result so callers can check `result.ok` directly.
            inner_ok = True
            inner_err = None
            if isinstance(value, dict) and "ok" in value:
                inner_ok = bool(value["ok"])
                if not inner_ok:
                    inner_err = value.get("error") or "Tool reported failure"
            result = ToolResult(
                tool=name, ok=inner_ok, value=value,
                error=inner_err,
                duration_ms=(time.time() - t0) * 1000,
                args=args,
            )
        except Exception as e:
            result = ToolResult(
                tool=name, ok=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.time() - t0) * 1000,
                args=args,
            )
            print(f"[TOOL] {name} failed:\n{traceback.format_exc()}")

        self._append_history(result)
        return result

    def history(self, limit: int = 100) -> List[dict]:
        return [r.to_dict() for r in self._history[-limit:]]

    def stats(self) -> dict:
        """Aggregate success/failure stats — feeds the benchmarks page."""
        total = len(self._history)
        if total == 0:
            return {
                "total_calls": 0,
                "success_rate": None,
                "avg_duration_ms": None,
                "by_tool": {},
            }
        ok_count = sum(1 for r in self._history if r.ok)
        by_tool: Dict[str, Dict[str, Any]] = {}
        for r in self._history:
            t = by_tool.setdefault(r.tool, {"total": 0, "ok": 0, "total_ms": 0.0})
            t["total"] += 1
            if r.ok:
                t["ok"] += 1
            t["total_ms"] += r.duration_ms

        for tname, s in by_tool.items():
            s["success_rate"] = round(s["ok"] / s["total"], 3)
            s["avg_duration_ms"] = round(s["total_ms"] / s["total"], 2)

        return {
            "total_calls": total,
            "success_rate": round(ok_count / total, 3),
            "avg_duration_ms": round(
                sum(r.duration_ms for r in self._history) / total, 2
            ),
            "by_tool": by_tool,
        }

    def _append_history(self, result: ToolResult) -> None:
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]


# --- Module-level registry (singleton) ---
_REGISTRY = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    params: Dict[str, str],
    required: Optional[List[str]] = None,
    category: str = "general",
    dangerous: bool = False,
):
    """
    Decorator to register a function as a brain-callable tool.

    @register_tool(
        name="send_email",
        description="Send an email via SMTP",
        params={"to": "str", "subject": "str", "body": "str"},
        required=["to", "subject", "body"],
    )
    def send_email(to, subject, body): ...
    """
    def decorator(fn: Callable) -> Callable:
        # Auto-detect required params from function signature if not given
        req = required
        if req is None:
            sig = inspect.signature(fn)
            req = [
                p.name for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty and p.name != "self"
            ]
        spec = ToolSpec(
            name=name,
            description=description,
            params=params,
            required=req,
            category=category,
            fn=fn,
            dangerous=dangerous,
        )
        _REGISTRY.register(spec)
        return fn
    return decorator


def list_tools() -> List[ToolSpec]:
    return _REGISTRY.list()


def describe_tools() -> List[dict]:
    return _REGISTRY.describe_all()


def get_tool(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(name)


def call_tool(name: str, args: Optional[dict] = None) -> ToolResult:
    return _REGISTRY.call(name, args)


def tool_history(limit: int = 100) -> List[dict]:
    return _REGISTRY.history(limit)


def tool_stats() -> dict:
    return _REGISTRY.stats()


# Auto-discover built-in tools when this module is imported.
# Each submodule uses @register_tool at import time.
def _load_builtin_tools():
    import importlib
    builtins = [
        "brain.tools.email_tool",
        "brain.tools.sheet_tool",
        "brain.tools.calendar_tool",
        "brain.tools.db_tool",
        "brain.tools.http_tool",
        "brain.tools.file_tool",
    ]
    loaded = []
    for modname in builtins:
        try:
            importlib.import_module(modname)
            loaded.append(modname)
        except Exception as e:
            print(f"[TOOL] Skipped {modname}: {e}")
    print(f"[TOOL] Loaded {len(loaded)} built-in tool modules")


_load_builtin_tools()
