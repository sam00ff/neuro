"""
NeuroLinked Agent — Plan → Execute → Verify → Learn loop.

This is what turns the brain from a memory system into an actual agent.
Given a goal, the agent:
  1. Plans a sequence of tool calls
  2. Executes them in order
  3. Verifies each outcome
  4. Records the outcome back into the knowledge store so the brain
     learns from what worked and what didn't

No LLM required at the core — you can drive it by handing it a pre-built
plan (useful for deterministic workflows). With an LLM plugged in via the
`planner` param, it writes its own plans.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from brain.tools import call_tool, describe_tools, tool_stats


@dataclass
class PlanStep:
    """One step in an agent plan."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    expect: Optional[str] = None      # optional check: "ok"|"row_count>0"|etc
    note: str = ""                    # human-readable reason

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentRun:
    """Full record of one agent execution — goal, plan, outcome."""
    run_id: str
    goal: str
    plan: List[PlanStep]
    results: List[dict] = field(default_factory=list)
    status: str = "pending"          # pending|running|success|partial|failed
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "plan": [s.to_dict() for s in self.plan],
            "results": self.results,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_sec": (
                round(self.completed_at - self.started_at, 2)
                if self.completed_at else None
            ),
            "error": self.error,
            "steps_total": len(self.plan),
            "steps_ok": sum(1 for r in self.results if r.get("ok")),
        }


def _verify(result: dict, expect: Optional[str]) -> bool:
    """
    Super-simple expectation matcher for deterministic plans.
    Supported:
      - None / "": just check ok=True
      - "ok": result["ok"] is True
      - "row_count>0" / "row_count>=N": numeric comparison on row_count
      - "status=200": exact field match
    """
    if not expect:
        return bool(result.get("ok"))
    expect = expect.strip()
    if expect == "ok":
        return bool(result.get("ok"))
    if "=" in expect and not any(c in expect for c in "<>!"):
        k, v = expect.split("=", 1)
        return str(result.get(k.strip())) == v.strip()
    for op in (">=", "<=", ">", "<", "!="):
        if op in expect:
            k, v = expect.split(op, 1)
            try:
                kv = float(result.get(k.strip(), 0))
                vv = float(v.strip())
            except Exception:
                return False
            return {
                ">": kv > vv, "<": kv < vv,
                ">=": kv >= vv, "<=": kv <= vv,
                "!=": kv != vv,
            }[op]
    return False


class Agent:
    """The core agent. Executes plans and records outcomes."""

    def __init__(self, knowledge_store=None):
        self.knowledge_store = knowledge_store
        self.history: List[AgentRun] = []
        self._max_history = 500
        self.planner: Optional[Callable[[str, List[dict]], List[PlanStep]]] = None

    def set_planner(self, planner_fn: Callable[[str, List[dict]], List[PlanStep]]):
        """Install an LLM-backed planner. Fn receives (goal, available_tools) -> list[PlanStep]."""
        self.planner = planner_fn

    # ---------- Execution ----------

    def run_plan(self, goal: str, plan: List[PlanStep]) -> AgentRun:
        """Execute a pre-built plan. Records every step."""
        run = AgentRun(
            run_id=str(uuid.uuid4()),
            goal=goal,
            plan=plan,
        )
        run.status = "running"

        all_ok = True
        for i, step in enumerate(plan):
            result = call_tool(step.tool, step.args).to_dict()
            passed = _verify(result, step.expect)
            result["step_index"] = i
            result["step_note"] = step.note
            result["expectation"] = step.expect
            result["passed"] = passed
            run.results.append(result)
            if not passed:
                all_ok = False
                # Keep executing the rest (can improve later with dependency DAG)

        run.completed_at = time.time()
        run.status = "success" if all_ok else (
            "partial" if any(r.get("passed") for r in run.results) else "failed"
        )
        self._record(run)
        return run

    def run_goal(self, goal: str) -> AgentRun:
        """Given a goal, plan + execute. Requires a planner to be installed."""
        if self.planner is None:
            run = AgentRun(
                run_id=str(uuid.uuid4()),
                goal=goal, plan=[],
                status="failed",
                error="No planner installed. Use set_planner() or call run_plan() with a pre-built plan.",
                completed_at=time.time(),
            )
            self._record(run)
            return run
        try:
            plan = self.planner(goal, describe_tools())
        except Exception as e:
            run = AgentRun(
                run_id=str(uuid.uuid4()),
                goal=goal, plan=[],
                status="failed",
                error=f"Planner crashed: {type(e).__name__}: {e}",
                completed_at=time.time(),
            )
            self._record(run)
            return run
        return self.run_plan(goal, plan)

    # ---------- Memory integration ----------

    def _record(self, run: AgentRun):
        self.history.append(run)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]
        # Persist outcome into knowledge store so the brain learns from it
        if self.knowledge_store is not None:
            try:
                summary = (
                    f"[AGENT RUN] goal={run.goal!r} status={run.status} "
                    f"steps={len(run.plan)} ok={sum(1 for r in run.results if r.get('passed'))}"
                )
                self.knowledge_store.store(
                    text=summary,
                    source="agent",
                    tags=["agent", "run", run.status],
                )
            except Exception:
                # Don't break the agent if the knowledge store has issues
                pass

    # ---------- Query ----------

    def get_run(self, run_id: str) -> Optional[AgentRun]:
        for r in self.history:
            if r.run_id == run_id:
                return r
        return None

    def recent(self, limit: int = 20) -> List[dict]:
        return [r.to_dict() for r in self.history[-limit:]]

    def stats(self) -> dict:
        """Aggregate agent performance — feeds benchmarks."""
        if not self.history:
            return {
                "total_runs": 0,
                "success_rate": None,
                "avg_duration_sec": None,
                "by_status": {},
                "tool_stats": tool_stats(),
            }
        by_status: Dict[str, int] = {}
        total_duration = 0.0
        complete_count = 0
        for r in self.history:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            if r.completed_at:
                total_duration += r.completed_at - r.started_at
                complete_count += 1
        success = by_status.get("success", 0)
        return {
            "total_runs": len(self.history),
            "success_rate": round(success / len(self.history), 3),
            "avg_duration_sec": round(total_duration / complete_count, 2) if complete_count else None,
            "by_status": by_status,
            "tool_stats": tool_stats(),
        }


# Convenience constructor that wires the agent to the knowledge store
def build_default_agent():
    """Create an agent bound to the brain's knowledge store (if available)."""
    try:
        from brain.knowledge_store import KnowledgeStore
        ks = KnowledgeStore()
    except Exception:
        ks = None
    return Agent(knowledge_store=ks)


# Helper — parse a plain-JSON plan into PlanSteps
def plan_from_json(plan_json: List[dict]) -> List[PlanStep]:
    return [
        PlanStep(
            tool=s["tool"],
            args=s.get("args", {}),
            expect=s.get("expect"),
            note=s.get("note", ""),
        )
        for s in plan_json
    ]
